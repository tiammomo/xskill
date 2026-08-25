"""重活进程：Milvus 对账 + 脏用户推荐预计算（与 Web GIL 隔离）。"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("xskill.recommend.heavy_worker")


VECTOR_RECONCILE_INTERVAL_SECONDS = 24 * 60 * 60
VECTOR_SYNC_ALGORITHM = "catalog-vector-dirty-v1"

# 全量对账批处理循环里，每处理这么多行才查一次内存占用——resource.getrusage
# 是系统调用，逐行查会有可感知开销；这个粒度下预算超限最多晚发现这么多行。
_MEMORY_BUDGET_CHECK_STRIDE = 20


def current_rss_mb() -> float:
    """当前进程的峰值常驻内存（MiB）。

    ``resource.getrusage(...).ru_maxrss`` 单位在 Linux 上是 KiB，
    macOS 上是字节——两个平台的内核实现从来没统一过，这里按平台换算成
    统一的 MiB。Windows 没有 ``resource`` 模块，返回 0.0（不阻断，只是
    这项观测在 Windows 上不可用；team server 目前只跑在 Linux 容器里）。
    """
    try:
        import resource
    except ImportError:  # Windows
        return 0.0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024.0 if sys.platform == "linux" else peak / (1024.0 * 1024.0)


def _load_catalog_rows(
    db_path: Path,
    *,
    catalog_keys: Optional[list[str]] = None,
) -> list[dict]:
    from xskill.pipeline.registry import pooled_connection

    with pooled_connection(db_path) as conn:
        where = ""
        params: list = []
        if catalog_keys is not None:
            if not catalog_keys:
                return []
            placeholders = ",".join("?" for _ in catalog_keys)
            where = f"WHERE c.catalog_key IN ({placeholders})"
            params.extend(catalog_keys)
        rows = conn.execute(
            f"""
            SELECT c.catalog_key, c.name, c.source, c.description,
                   c.content_sha, c.skill_id, c.distributable,
                   CASE WHEN l.state='retired' THEN 1 ELSE 0 END AS retired
            FROM skills_catalog AS c
            LEFT JOIN skill_lifecycle AS l ON l.skill_name=c.name
            {where}
            """,  # noqa: S608 -- placeholders carry all external values
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def _drain_incremental_batch(
    *, db_path: Path, index, embed, limit: int,
) -> dict:
    """消费一批「单个 key 真的变了」的自然脏项：无条件重新 embed + upsert。

    这条路径的事件本来就是内容变化触发的（见 catalog_store.py 的写入
    逻辑），复用旧向量的判断在这里没有意义，不必再查一次 ``index.get``。
    """
    from xskill.recommend.skill_vector_store import indexable_catalog_rows
    from xskill.recommend.vector_dirty import (
        catalog_vector_event_is_current,
        clear_catalog_vector_dirty,
        list_catalog_vector_dirty,
    )

    events = list_catalog_vector_dirty(db_path=db_path, limit=limit)
    rows = _load_catalog_rows(
        db_path,
        catalog_keys=[event["catalog_key"] for event in events],
    )
    rows_by_key = {row["catalog_key"]: row for row in rows}
    stats = {"upserted": 0, "deleted": 0, "skipped": 0, "deferred": 0}
    for event in events:
        key = event["catalog_key"]
        generation = int(event["generation"])
        row = rows_by_key.get(key)
        wanted = indexable_catalog_rows([row]) if row is not None else []
        if wanted:
            target = wanted[0]
            vector = embed(target["description"])
            if not catalog_vector_event_is_current(
                key, generation, db_path=db_path,
            ):
                stats["deferred"] += 1
                continue
            index.upsert(
                key,
                vector,
                content_sha=target["content_sha"],
                source=target.get("source") or "",
                name=target.get("name") or "",
            )
            stats["upserted"] += 1
        else:
            if not catalog_vector_event_is_current(
                key, generation, db_path=db_path,
            ):
                stats["deferred"] += 1
                continue
            index.delete(key)
            stats["deleted"] += 1
        if not clear_catalog_vector_dirty(key, generation, db_path=db_path):
            stats["deferred"] += 1
    return stats


def _drain_full_sweep_batch(
    *,
    db_path: Path,
    index,
    embed,
    limit: int,
    force_upsert: bool,
    memory_budget_mb: Optional[float] = None,
) -> dict:
    """消费一批「全量对账」播种出来的脏项。

    与 ``_drain_incremental_batch`` 的区别：这批事件里绝大多数内容根本
    没变（播种时不分青红皂白把整份可索引目录都标了一遍），逐条无脑
    re-embed 会把 bootstrap/periodic 对账变成一次性全量重付 embedding
    成本——所以这里先看 content_sha 是否真的变了，没变就跳过或复用旧
    向量，只有 ``force_upsert``（模型切换/无持久索引）时才放弃这个判断，
    因为旧向量是用不同模型算的，复用是错的。

    ``memory_budget_mb``：给定时每隔
    ``_MEMORY_BUDGET_CHECK_STRIDE`` 行查一次本进程峰值 RSS，超预算就
    停止消费这一批剩下的行——已处理的行照常清脏表，没处理到的行留在
    脏表里，下一轮接着来（issue #328：单轮内存有界，而不是等一批全处理
    完才发现已经超了）。
    """
    from xskill.recommend.skill_vector_store import indexable_catalog_rows
    from xskill.recommend.vector_dirty import (
        catalog_vector_event_is_current,
        clear_catalog_vector_dirty,
        list_catalog_vector_dirty,
    )

    events = list_catalog_vector_dirty(db_path=db_path, limit=limit)
    rows = _load_catalog_rows(
        db_path,
        catalog_keys=[event["catalog_key"] for event in events],
    )
    rows_by_key = {row["catalog_key"]: row for row in rows}
    stats = {
        "upserted": 0, "deleted": 0, "skipped": 0, "deferred": 0,
        "budget_aborted": False,
    }
    # 这一批的 catalog 行（含正文）在上面已经整批读进来了——批次加载本身
    # 就可能把预算吃光，尤其是 limit 配得偏大时；加载完立刻查一次，不等
    # 逐行循环里的下一次检查点才发现已经超了（issue #328 review）。
    if (
        memory_budget_mb is not None
        and events
        and current_rss_mb() > memory_budget_mb
    ):
        logger.warning(
            "vector full sweep aborted before processing: rss over budget "
            "(%.0f MiB budget) right after loading this batch's %s rows; "
            "none processed, all stay dirty for next round",
            memory_budget_mb, len(events),
        )
        stats["budget_aborted"] = True
        return stats
    for processed, event in enumerate(events):
        if (
            memory_budget_mb is not None
            # 步长为了减少 getrusage 系统调用频率；不能跳过第 0 条——批次
            # 比步长还小时（如 10 条 vs 步长 20），跳过第 0 条会导致这批
            # 从头到尾一次都不检查，budget 形同虚设（issue #328 review）。
            and processed % _MEMORY_BUDGET_CHECK_STRIDE == 0
            and current_rss_mb() > memory_budget_mb
        ):
            logger.warning(
                "vector full sweep aborted mid-batch: rss over budget "
                "(%.0f MiB budget), processed %s/%s this batch; "
                "remaining rows stay dirty for next round",
                memory_budget_mb, processed, len(events),
            )
            stats["budget_aborted"] = True
            break
        key = event["catalog_key"]
        generation = int(event["generation"])
        row = rows_by_key.get(key)
        wanted = indexable_catalog_rows([row]) if row is not None else []
        if wanted:
            target = wanted[0]
            cur = None if force_upsert else index.get(key)
            same_content = (
                cur is not None and cur.get("content_sha") == target["content_sha"]
            )
            same_meta = (
                same_content
                and (cur.get("source") or "") == (target.get("source") or "")
                and (cur.get("name") or "") == (target.get("name") or "")
            )
            if same_meta:
                if not catalog_vector_event_is_current(
                    key, generation, db_path=db_path,
                ):
                    stats["deferred"] += 1
                    continue
                stats["skipped"] += 1
                if not clear_catalog_vector_dirty(key, generation, db_path=db_path):
                    stats["deferred"] += 1
                continue
            vector = cur["vector"] if same_content else embed(target["description"])
            if not catalog_vector_event_is_current(
                key, generation, db_path=db_path,
            ):
                stats["deferred"] += 1
                continue
            index.upsert(
                key,
                vector,
                content_sha=target["content_sha"],
                source=target.get("source") or "",
                name=target.get("name") or "",
            )
            stats["upserted"] += 1
        else:
            if not catalog_vector_event_is_current(
                key, generation, db_path=db_path,
            ):
                stats["deferred"] += 1
                continue
            index.delete(key)
            stats["deleted"] += 1
        if not clear_catalog_vector_dirty(key, generation, db_path=db_path):
            stats["deferred"] += 1
    return stats


def run_vector_sync(
    *,
    db_path: Path,
    index,
    embed,
    model_fingerprint: str,
    force_full: bool = False,
    now: float | None = None,
    limit: int = 256,
    memory_budget_mb: Optional[float] = None,
    sweep_key: Optional[str] = None,
) -> dict:
    """优先消费增量队列；首次/模型变化/低频周期触发全量对账。

    全量对账本身也走 ``catalog_vector_dirty`` 这张脏表分批消费
    （issue #328）：没有积压时先播种（把全部可索引 ``catalog_key`` 标脏），
    之后每次调用只处理至多 ``limit`` 条，天然把「一次性全量重建」拆成
    多轮——不再需要在没装持久索引（Milvus Lite）时一次性把整份 catalog
    的正文和向量都摊进内存，用多轮换峰值内存可控；进度（本轮处理量、
    剩余量）由返回值的 ``remaining`` 字段体现，调用方负责写进日志/状态
    文件。持久索引的分批结果会落盘在同一个 db 文件里，跨轮次自然累积；
    没有持久索引（内存兜底）时每轮的索引对象在子进程退出后被丢弃，
    「多轮」只保证内存峰值有界、检索覆盖率会持续被下一轮的批次覆盖，
    并不会跨轮累积成一份完整索引——这是没有持久存储时物理上做不到的，
    不是本函数的缺陷。

    ``sweep_key``：判断「这个全量对账目标是否已经播种过」用的稳定标识，
    缺省等于 ``model_fingerprint``。两者分开是因为 ``model_fingerprint``
    里可能拼了索引实例标识（``_vector_index_identity``）——持久索引按
    db 文件的 dev/inode，重建后正确触发重新 bootstrap；但没有持久索引
    时索引标识是 ``id(index)``，每个子进程都是全新对象，若直接拿它当
    播种去重键，会导致「已经播种过」永远判定为假、每轮都重新播种，
    抹掉多轮攒下的脏表进度。调用方（``run_recommend_heavy_once``）传入
    不含索引实例标识的稳定部分。
    """
    from xskill.recommend.vector_dirty import (
        catalog_vector_reconcile_reason,
        count_catalog_vector_dirty,
        finish_catalog_vector_reconcile,
        get_sweep_seeded_fingerprint,
        mark_sweep_seeded,
        seed_full_catalog_vector_sweep,
    )

    effective_sweep_key = sweep_key if sweep_key is not None else model_fingerprint

    reason = "ephemeral" if force_full else catalog_vector_reconcile_reason(
        model_fingerprint,
        db_path=db_path,
        now=now,
        interval_seconds=VECTOR_RECONCILE_INTERVAL_SECONDS,
    )
    seed_info = None
    if reason:
        # 播种去重看的是「这个 sweep_key 是否已经播种过」，不是「脏表是否
        # 为空」——脏表非空可能只是普通 catalog 编辑产生的有机脏项，跟
        # 有没有播种过全量对账是两回事，用队列是否为空来判断会导致这类
        # 有机脏项存在时全量对账被误判成「已经播种过」而永久跳过
        # （issue #328 review）。sweep_key 变化（比如模型中途又切换）会
        # 重新播种一遍，让新目标下这份索引重新覆盖全部 catalog，不会把
        # 老模型的向量和新模型的混在一起当成同一次对账完成。
        if get_sweep_seeded_fingerprint(db_path=db_path) != effective_sweep_key:
            seed_info = seed_full_catalog_vector_sweep(
                db_path=db_path, existing_index_keys=index.list_keys(),
            )
            mark_sweep_seeded(effective_sweep_key, db_path=db_path)

    if reason:
        stats = _drain_full_sweep_batch(
            db_path=db_path,
            index=index,
            embed=embed,
            limit=limit,
            force_upsert=reason in {"model_changed", "ephemeral"},
            memory_budget_mb=memory_budget_mb,
        )
        remaining = count_catalog_vector_dirty(db_path=db_path)
        if remaining == 0:
            finish_catalog_vector_reconcile(
                {},
                model_fingerprint=model_fingerprint,
                reconciled_at=time.time() if now is None else now,
                db_path=db_path,
            )
        else:
            logger.info(
                "vector full sweep in progress: reason=%s upserted=%s "
                "deleted=%s skipped=%s remaining=%s",
                reason, stats["upserted"], stats["deleted"], stats["skipped"],
                remaining,
            )
        result = {**stats, "mode": "full", "reason": reason, "remaining": remaining}
        if seed_info is not None:
            result["total_indexable"] = seed_info["total_indexable"]
        return result

    stats = _drain_incremental_batch(
        db_path=db_path, index=index, embed=embed, limit=limit,
    )
    return {**stats, "mode": "incremental", "reason": ""}


def _skill_name_from_index(vector_index, catalog_key: str) -> str:
    row = vector_index.get(catalog_key)
    if row:
        name = (row.get("name") or "").strip()
        if name:
            return name
        if row.get("source") == "skillhub" and ":" in catalog_key:
            return catalog_key.split(":", 1)[-1]
    if ":" in catalog_key:
        return catalog_key.split(":", 1)[-1]
    return catalog_key


def compute_recommend_for_user(
    user_key: str,
    *,
    db_path: Path,
    vector_index,
    top_k: int = 20,
    profile_centers: Optional[list[list[float]]] = None,
) -> list[str]:
    """用画像中心向量在索引里 search；无中心则写空推荐（sync 侧走 ranked/ux）。"""
    from xskill.recommend.recommend_store import save_recommend_slots

    if not profile_centers:
        save_recommend_slots(user_key, [], fingerprint="no_profile", db_path=db_path)
        return []

    # 每个中心独立召回，再按中心轮询取一个未出现过的技能。直接把第一个
    # 中心的结果填满 top_k 会让后续兴趣永远没有机会进入推荐槽位。
    center_hits = [
        vector_index.search(center, top_k=top_k)
        for center in profile_centers
    ]
    positions = [0] * len(center_hits)
    names: list[str] = []
    seen: set[str] = set()
    source_centers: list[int] = []
    while len(names) < top_k:
        progress = False
        for center_index, hits in enumerate(center_hits):
            while positions[center_index] < len(hits):
                catalog_key, _score = hits[positions[center_index]]
                positions[center_index] += 1
                name = _skill_name_from_index(vector_index, catalog_key)
                if name in seen:
                    continue
                seen.add(name)
                names.append(name)
                source_centers.append(center_index)
                progress = True
                break
            if len(names) >= top_k:
                break
        if not progress:
            break
    fingerprint = (
        f"centers={len(profile_centers)};fusion=round_robin_v1;"
        f"sources={','.join(map(str, source_centers))}"
    )
    save_recommend_slots(
        user_key, names, fingerprint=fingerprint, db_path=db_path,
    )
    return names


def _user_key_for_client(engine, client_id: str) -> str:
    reg = getattr(engine, "client_registry", None)
    if reg is not None:
        try:
            name = reg.user_name_for(client_id)
            if name:
                return name
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    return client_id


def _client_id_for_user_key(engine, user_key: str) -> str:
    """推荐表键 → 画像键（client_id）。"""
    if not user_key:
        return user_key
    reg = getattr(engine, "client_registry", None)
    if reg is None:
        return user_key
    try:
        for row in reg.list():
            cid = row["client_id"]
            if cid == user_key:
                return cid
            try:
                if reg.user_name_for(cid) == user_key:
                    return cid
            except Exception:  # pylint: disable=broad-exception-caught
                continue
    except Exception:  # pylint: disable=broad-exception-caught
        logger.debug("client_id resolve failed for %s", user_key, exc_info=True)
    return user_key


def _load_profile_centers(engine, client_id: str) -> Optional[list[list[float]]]:
    try:
        user = engine.load_client_user(client_id, include_recommended=False)
        ci = user.client_interest
        if ci is None or ci.feature_tensor is None:
            return None
        return [list(map(float, row)) for row in ci.feature_tensor]
    except Exception:  # pylint: disable=broad-exception-caught
        logger.debug("profile centers unavailable for %s", client_id, exc_info=True)
        return None


def process_dirty_recommends(
    *,
    db_path: Path,
    vector_index,
    engine,
    limit: int = 32,
) -> int:
    from xskill.recommend.recommend_store import (
        clear_recommend_dirty,
        list_dirty_user_keys,
        save_recommend_slots,
    )

    keys = list_dirty_user_keys(limit=limit, db_path=db_path)
    done = 0
    for user_key in keys:
        try:
            client_id = _client_id_for_user_key(engine, user_key)
            centers = _load_profile_centers(engine, client_id)
            compute_recommend_for_user(
                user_key,
                db_path=db_path,
                vector_index=vector_index,
                profile_centers=centers,
            )
            done += 1
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("recommend dirty failed user_key=%s", user_key)
            clear_recommend_dirty(user_key, db_path=db_path)
            save_recommend_slots(user_key, [], fingerprint="error", db_path=db_path)
    return done


def _embed_fn_from_engine(engine):
    """从引擎 embed_client 构造 embed(text)->list[float]；不可用则 None。"""
    client = getattr(engine, "embed_client", None)
    if client is None or not hasattr(client, "encode"):
        return None

    def _embed(text: str) -> list[float]:
        vec = client.encode(text)
        return [float(x) for x in vec]

    return _embed


def _vector_index_identity(index, vector_db_path: Path) -> str:
    """廉价识别持久索引实例；同路径数据库被重建后强制 bootstrap。"""
    stored_path = getattr(index, "db_path", None)
    if stored_path is not None:
        path = Path(stored_path).expanduser().resolve()
        try:
            stat = path.stat()
            return f"file:{path}:{stat.st_dev}:{stat.st_ino}"
        except OSError:
            return f"file:{path}:missing"
    # 显式注入的内存/测试索引只在当前进程实例内可复用。
    return (
        f"object:{type(index).__module__}.{type(index).__qualname__}:"
        f"{id(index)}:{Path(vector_db_path).expanduser().resolve()}"
    )


def _recommends_safe_to_recompute(vec_stats: dict, *, ephemeral_index: bool) -> bool:
    """本轮的索引内容够不够完整、值不值得用它重算并覆盖已有推荐结果。

    不安全时应该跳过重算，让已有推荐槽保持原样（哪怕已经过期）——过期
    但完整的一份结果，好过用不完整/不一致的索引算出一份更差的覆盖上去
    （issue #328 review）：

    - 纯增量（``mode != "full"``）：索引一直在稳态累积，随时可信。
    - 全量对账仍在进行（``remaining > 0``）：索引处于中途状态——可能是
      增量脏项还没追完，也可能是模型切换/无持久索引导致内容还没对齐到
      同一个目标，这时候算出来的结果只会是一份内部不一致的推荐，不能
      发布。
    - 全量对账在这次调用里追平（``remaining == 0``）且不是内存兜底
      （持久索引）：安全，持久索引跨轮次真正累积，此刻磁盘上的内容就是
      完整的。
    - 全量对账在这次调用里追平、且是内存兜底：只有当这次调用同时完成了
      播种和消化（``total_indexable`` 有值，即整份可索引目录一次就在这
      批里处理完）才安全——这种情况下内存里这份索引确实覆盖了全部
      catalog。如果 ``total_indexable`` 没有值，说明这是跨多轮播种的
      尾轮，内存里这份索引对象只有这一轮处理的那一小批，不是全量，
      不能发布。
    """
    if vec_stats.get("mode") != "full":
        return True
    if vec_stats.get("remaining", 0) != 0:
        return False
    if not ephemeral_index:
        return True
    return vec_stats.get("total_indexable") is not None


def run_recommend_heavy_once(
    *,
    engine,
    db_path: Path | None = None,
    vector_db_path: Path | None = None,
    memory_index=None,
    mark_catalog_dirty: bool = True,
    vector_sync_batch_limit: int = 256,
    memory_budget_mb: Optional[float] = None,
) -> dict:
    """对账向量索引并消化推荐脏队列（画像刷新由调用方先跑）。"""
    from xskill.config import XSKILL_HOME, get_registry_db_path
    from xskill.recommend.recommend_store import mark_all_recommend_dirty
    from xskill.recommend.skill_vector_store import (
        DEFAULT_DIM,
        MemorySkillVectorIndex,
        default_vector_db_path,
        fake_embed,
        open_skill_vector_index,
    )

    registry = Path(db_path) if db_path else get_registry_db_path()
    vdb = Path(vector_db_path) if vector_db_path else default_vector_db_path(XSKILL_HOME)
    embed_fn = _embed_fn_from_engine(engine)
    if embed_fn is None:
        embed_fn = lambda text: fake_embed(text, DEFAULT_DIM)  # noqa: E731
        dim = DEFAULT_DIM
        model_fingerprint = f"{VECTOR_SYNC_ALGORITHM}:fake:{dim}"
    else:
        client = engine.embed_client
        dim = int(getattr(client, "dim", 0) or 0)
        if dim <= 0:
            # 正常 EmbedClient 在 engine 构造时已 probe；仅兼容自定义 client。
            dim = len(embed_fn("dimension probe"))
        model = str(getattr(client, "model", "") or "unknown")
        model_fingerprint = f"{VECTOR_SYNC_ALGORITHM}:{model}:{dim}"
    # open_skill_vector_index：无 pymilvus 时退回内存索引并 hourly warn
    index = memory_index or open_skill_vector_index(vdb, dim=dim)
    stable_sweep_key = model_fingerprint
    model_fingerprint = (
        f"{model_fingerprint}:{_vector_index_identity(index, vdb)}"
    )
    # 生产 fallback 每次都会创建空的内存索引：force_full 让 run_vector_sync
    # 把这轮当成需要全量对账处理（分批消费脏表，不是一次性全量重建，见
    # run_vector_sync 的说明）；调用方显式传入的 memory_index 可跨 tick
    # 复用，仍走增量路径（用于测试/嵌入式调用）。
    ephemeral_index = memory_index is None and isinstance(
        index, MemorySkillVectorIndex,
    )
    # 持久索引的 sweep 必须绑定文件 identity：同路径数据库在多轮 sweep
    # 中途被替换后，要重新播种已经从旧索引队列中清掉的 key。只有每个
    # 子进程都会得到全新对象的内存 fallback 使用稳定 key，否则 id(index)
    # 每轮变化会反复播种并抹掉多轮消费进度。
    sweep_key = stable_sweep_key if ephemeral_index else model_fingerprint
    vec_stats = run_vector_sync(
        db_path=registry,
        embed=embed_fn,
        index=index,
        model_fingerprint=model_fingerprint,
        sweep_key=sweep_key,
        force_full=ephemeral_index,
        limit=vector_sync_batch_limit,
        memory_budget_mb=memory_budget_mb,
    )
    if mark_catalog_dirty and (
        vec_stats.get("upserted", 0) or vec_stats.get("deleted", 0)
    ):
        # 标脏只是排进队列，供之后「索引可信」的某一轮消费——标脏本身不
        # 涉及用这份索引算东西，随时安全，不受下面的门禁影响。
        mark_all_recommend_dirty(reason="catalog_vector_changed", db_path=registry)
    recommends_safe = _recommends_safe_to_recompute(
        vec_stats, ephemeral_index=ephemeral_index,
    )
    if recommends_safe:
        n = process_dirty_recommends(
            db_path=registry, vector_index=index, engine=engine,
        )
    else:
        # 这一轮的索引不完整/不一致（全量对账还没追平，或没有持久索引时
        # 只覆盖了这一轮处理的那一小批），不能拿它算推荐去覆盖已有槽位
        # ——已有结果哪怕过期，也好过用不完整索引算出的一份更差结果
        # （issue #328 review）。留给脏队列，等索引可信的那一轮再消费。
        n = 0
        logger.info(
            "recommend computation deferred: vector index not yet complete "
            "this round (mode=%s remaining=%s ephemeral=%s)",
            vec_stats.get("mode"), vec_stats.get("remaining"), ephemeral_index,
        )
    index_kind = "milvus" if type(index).__name__ == "MilvusLiteSkillVectorIndex" else "memory"
    return {
        "vector": vec_stats,
        "recommends": n,
        "recommends_deferred": not recommends_safe,
        "index_kind": index_kind,
        "rss_peak_mb": current_rss_mb(),
    }
