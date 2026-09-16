"""skill_manifest.py — 给一个 client 现算它该持有的 ≤100 个 skill slot（SP1）

server 端的 skill 槽位投影不存表：ranked/recommended 排序 + skill git 状态
（main_sha / staging_sha）每次 sync 现算。**唯一例外是灰度 side 决策**——由
``CanaryRouter`` 有状态记账（在线偏差最小化），因为无状态 ``pick_side`` 在
client 基数很小时会把 staging 饿死到 0。

slot 结构 = 80 ranked + 20 recommended：
- ranked      —— 按 ux_score（main 侧近 30 天均分）滑窗取高分。
- recommended —— SP3 = 用户画像质心推荐位：基于该 client 用过的 skill 的质心，
                 从候选里取 cosine 最近邻（``profile_reco.py``）。无画像
                 （冷启动）或非 team server 调用 → 退回 ux 排序往下取。

灰度归因：某 skill 有 staging 分支时，纳入/生成发起人与体验分用量最多的用户
走体验分补漏换侧（自动灰度）；其余 client 仍由 ``CanaryRouter.assign`` sticky。
无 staging → main。看板钉死的 side 覆盖自动灰度。
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable

from xskill.canary import (
    CanaryConfig,
    CanaryRouter,
    auto_canary_side,
    main_sha,
    pick_side,
    staging_sha,
)
from xskill.skill.skill import Skill
from xskill.skill.repo import SkillRepo
from xskill.team.shared.protocol import SkillSlot, SyncResponse

_logger = logging.getLogger("xskill.team.manifest")

# §5 SkillRecommendEngine 单例（team server init 时 set_recommend_engine 注入）。
# 为 None 时退回既有 RECOMMENDER 画像路径（非 team / 测试场景）。
_engine = None

# team-CS 有状态灰度分流；server 单进程单例，重启后账本清空可接受。
_ROUTER = CanaryRouter()


def repo_search_id(skill_name: str) -> str:
    """自产 skill 在 search/install 协议中的稳定虚拟 ID。"""
    digest = hashlib.sha256(skill_name.encode("utf-8")).hexdigest()
    return f"repo@{digest}"


@dataclass(frozen=True)
class _CatalogSnapshot:
    """一次 skill 仓扫描的不可变结果。"""

    skills: tuple[Skill, ...]
    refs: dict[str, tuple[str, str | None]]
    search_by_id: dict[str, Skill]
    built_at: float


@dataclass
class _CatalogEntry:
    condition: threading.Condition = field(default_factory=threading.Condition)
    snapshot: _CatalogSnapshot | None = None
    refreshing: bool = False
    generation: int = 0
    last_error: BaseException | None = None
    error_generation: int = 0


class _ManifestCatalogCache:
    """按 skill 根目录隔离的短 TTL、single-flight 仓快照缓存。

    过期刷新失败时不回退到旧快照；异常也不会替换最后一次成功结果。下次
    请求会重新尝试刷新，因此旧结果不会被无限返回。
    """

    def __init__(
        self, *, ttl_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._entries: dict[str, _CatalogEntry] = {}
        self._entries_lock = threading.Lock()

    @staticmethod
    def _key(skill_dir: Path) -> str:
        return str(skill_dir.expanduser().resolve())

    def _entry(self, skill_dir: Path) -> _CatalogEntry:
        key = self._key(skill_dir)
        with self._entries_lock:
            return self._entries.setdefault(key, _CatalogEntry())

    def get(
        self,
        skill_dir: Path,
        *,
        max_age_seconds: float | None = None,
    ) -> _CatalogSnapshot:
        entry = self._entry(skill_dir)
        with entry.condition:
            while True:
                now = self.clock()
                max_age = (
                    self.ttl_seconds if max_age_seconds is None
                    else min(self.ttl_seconds, max_age_seconds)
                )
                if (
                    entry.snapshot is not None
                    and now - entry.snapshot.built_at < max_age
                ):
                    return entry.snapshot
                if not entry.refreshing:
                    entry.refreshing = True
                    attempt_generation = entry.generation + 1
                    break
                waited_generation = entry.generation
                while entry.refreshing and entry.generation == waited_generation:
                    entry.condition.wait()
                if (
                    entry.generation > waited_generation
                    and entry.error_generation == entry.generation
                    and entry.last_error is not None
                ):
                    raise entry.last_error

        try:
            snapshot = self._scan(skill_dir)
        except BaseException as exc:
            with entry.condition:
                entry.generation = attempt_generation
                entry.last_error = exc
                entry.error_generation = attempt_generation
                entry.refreshing = False
                entry.condition.notify_all()
            raise

        with entry.condition:
            entry.snapshot = snapshot
            entry.generation = attempt_generation
            entry.last_error = None
            entry.error_generation = 0
            entry.refreshing = False
            entry.condition.notify_all()
            return snapshot

    def _scan(self, skill_dir: Path) -> _CatalogSnapshot:
        refs: dict[str, tuple[str, str | None]] = {}
        skills: list[Skill] = []
        for skill in SkillRepo(skill_dir):
            current_main = main_sha(skill.path)
            if not current_main:
                continue
            refs[skill.name] = (current_main, staging_sha(skill.path))
            skills.append(skill)
        # ranked 排序键：UX 均分来自 registry.db ux_scores，不再逐 skill 扫 jsonl
        main_refs = {name: pair[0] for name, pair in refs.items()}
        try:
            from xskill.pipeline.ux_scores_store import avg_scores_for_refs
            ux_avgs = avg_scores_for_refs(main_refs, side="main", days=30)
        except Exception:
            _logger.warning(
                "ux_scores avg lookup failed; ranked falls back to use_count",
                exc_info=True,
            )
            ux_avgs = {}

        def rank_key(skill: Skill) -> tuple[float, int]:
            return (ux_avgs.get(skill.name, 0.0), skill.use_count)

        skills.sort(key=rank_key, reverse=True)
        repo_names = {skill.name for skill in skills}
        search_by_id: dict[str, Skill] = {}
        for skill in skills:
            search_id = repo_search_id(skill.name)
            if search_id in repo_names or search_id in search_by_id:
                _logger.error(
                    "repo search id collision; skill omitted from search: %s",
                    skill.name,
                )
                continue
            search_by_id[search_id] = skill
        return _CatalogSnapshot(
            skills=tuple(skills), refs=refs,
            search_by_id=search_by_id, built_at=self.clock(),
        )

    def clear(self, skill_dir: Path | None = None) -> None:
        with self._entries_lock:
            if skill_dir is None:
                self._entries.clear()
            else:
                self._entries.pop(self._key(skill_dir), None)


_catalog_cache = _ManifestCatalogCache()


def invalidate_manifest_cache(skill_dir: Path | str | None = None) -> None:
    """显式失效 manifest 仓快照；不传路径时清空全部根目录。"""
    _catalog_cache.clear(None if skill_dir is None else Path(skill_dir))


def manifest_catalog_snapshot(
    skill_dir: Path | str,
    *,
    max_age_seconds: float | None = None,
) -> _CatalogSnapshot:
    """返回与 sync 共用的最新仓库快照，供同进程只读接口复用。"""
    return _catalog_cache.get(
        Path(skill_dir), max_age_seconds=max_age_seconds,
    )


def _reset_manifest_cache_for_tests(
    *, ttl_seconds: float = 30.0,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """测试钩子：替换缓存，以便注入时钟并隔离用例。"""
    global _catalog_cache
    _catalog_cache = _ManifestCatalogCache(ttl_seconds=ttl_seconds, clock=clock)


def set_recommend_engine(eng) -> None:
    """team server 启动时注入 ``SkillRecommendEngine`` 单例。"""
    global _engine
    _engine = eng


def get_recommend_engine():
    """返回已注入的引擎（未注入时 None）。供 api 层在 /sync 刷新用户画像用。"""
    return _engine


def _resolve_slot(
    skill: Skill | dict, client_id: str, probability: float, bucket: str,
    refs: dict[str, tuple[str, str | None]] | None = None,
    *,
    user_key: str = "",
    fill_need: int = 5,
    db_path: Path | str | None = None,
    canary_enabled: bool = True,
) -> SkillSlot | None:
    """对一个 skill 现算它对该 client 的 side + sha。"""
    if isinstance(skill, dict) and skill.get("source") == "skillhub":
        eng = get_recommend_engine()
        if eng is None or eng.skillhub.skill_path(skill["skill_id"]) is None:
            return None
        current_sha = eng.skillhub.content_sha(skill["skill_id"])
        if not current_sha:
            return None
        return SkillSlot(
            skill_name=skill["skill_id"],
            side="main",
            sha=current_sha,
            bucket=bucket,
            source="skillhub",
            display_name=skill.get("display_name"),
            source_path=skill.get("source_path"),
        )
    cached_main, cached_staging = (
        refs[skill.name] if refs is not None else
        (main_sha(skill.path) or "", staging_sha(skill.path))
    )
    if cached_staging and canary_enabled:
        from xskill.pipeline.registry import is_auto_canary_user
        origin_db = Path(db_path) if db_path else None
        if user_key and is_auto_canary_user(
                user_key, skill.name, db_path=origin_db):
            # 纳入/生成发起人与用量最多的用户：按体验分补漏换侧。
            fallback = pick_side(client_id, skill.name, probability)
            side = auto_canary_side(
                skill.path,
                main_sha=cached_main or "",
                staging_sha=cached_staging or "",
                need=max(int(fill_need), 1),
                fallback=fallback,
            )
            _ROUTER.note(
                client_id=client_id,
                skill_name=skill.name,
                side=side,
                probability=probability,
                staging_sha=cached_staging or "",
            )
        else:
            # team-CS：有状态 CanaryRouter（hash 种子/破平），避免小基数饿死 staging。
            side = _ROUTER.assign(
                client_id=client_id,
                skill_name=skill.name,
                probability=probability,
                staging_sha=cached_staging or "",
            )
        sha = cached_staging if side == "staging" else cached_main
    else:
        side = "main"
        sha = cached_main
    if not sha:
        raise RuntimeError(f"skill {skill.name!r}: cannot resolve sha for side={side}")
    return SkillSlot(skill_name=skill.name, side=side, sha=sha, bucket=bucket)


def build_manifest(
    *,
    client_id: str,
    skill_dir: Path | str,
    probability: float,
    ranked_slots: int = 80,
    total_slots: int = 100,
    traj_root: Path | str | None = None,
    prefs: dict | None = None,
    retired: set | None = None,
    telemetry_submit: Callable[[Callable[[], None]], bool] | None = None,
    user_key: str = "",
    fill_need: int | None = None,
    db_path: Path | str | None = None,
    canary_enabled: bool = True,
) -> SyncResponse:
    """为 ``client_id`` 现算 manifest。skill 总数不足 total_slots 时全发。

    只分发**已 graduate 到 main 分支**的 skill。``baby`` 分支上的 stub
    （cluster 建了目录但 SkillEditAgent 还没跑过、没正文）没有 main，本来
    就不该下发给 client——这里直接过滤掉，不是 fallback 而是正确的可分发
    集合判定。

    P2-2.4 注入顺序（含 ``prefs``/``retired`` 时）：**blocked 先排除 →
    pinned 占位 → ranked → recommended 回填**。
    - ``prefs`` = ``registry.effective_prefs`` 的合并视图（调用方现查，
      None = 无控制面语义，纯 ranked+recommended）。
    - ``retired`` = 下线 skill 集合，无条件不分发（即便被 pin）。
    - pinned 超量在**写入侧**拒绝（D8）,这里对"pin 的 skill 已不可分发"
      只跳过不报错——sync 路径绝不 500。

    slot 分段：
    - pinned —— prefs 里的钉住 skill（全局在前），占位不参与排名。
    - ranked —— 按 ux 滑窗均分降序取高分（配额 = min(ranked_slots, 余量)）。
    - recommended —— SP3 画像推荐位：基于该 client 用过的 skill 的
      质心，从「distributable 且不在 pinned/ranked、且 client 没用过」的候选
      里取 cosine 最近邻。``traj_root`` 为 None（非 team server 调用）或该
      client 没有任何带 used_skills 的 atom（冷启动、无画像）时，退回 ux
      排序往下接着取——这不是 fallback，是画像不存在时的正确定义。
    """
    skill_dir = Path(skill_dir)
    if total_slots <= 0:
        # 控制面压测和显式禁用分发的部署不需要触碰 skill 仓；否则短 TTL
        # 过期时仍会让一批无槽位请求等待一次无意义的全量扫描。
        return SyncResponse(slots=[], server_time=time.time())
    prefs = prefs or {"pinned": [], "blocked": set()}
    retired = retired or set()
    excluded = set(prefs.get("blocked") or set()) | retired

    catalog = _catalog_cache.get(skill_dir)
    distributable = [s for s in catalog.skills if s.name not in excluded]
    skills = distributable
    by_name = {s.name: s for s in distributable}

    # pinned 占位:不存在/尚不可分发(无 main)/已下线的 pin 跳过——读路径
    # 绝不 throw(D8),写入侧校验已把守常规超量。
    pinned = [by_name[n] for n in (prefs.get("pinned") or []) if n in by_name]
    pinned = pinned[:total_slots]
    pinned_names = {s.name for s in pinned}

    remaining = total_slots - len(pinned)
    non_pinned = [s for s in skills if s.name not in pinned_names]
    ranked = non_pinned[:min(ranked_slots, remaining)]
    taken_names = pinned_names | {s.name for s in ranked}
    reco_slots = max(remaining - len(ranked), 0)

    # exclude 集合并入 blocked/retired:推荐引擎有自己的候选索引(skillhub 等),
    # 不经 distributable 过滤,必须显式排除
    chosen = pinned + ranked + _pick_recommended(
        client_id=client_id,
        skill_dir=skill_dir,
        ranked=ranked,
        ranked_names=taken_names | excluded,
        ux_ordered=non_pinned,
        reco_slots=reco_slots,
        traj_root=traj_root,
        candidate_pool=list(catalog.skills),
        candidate_refs=catalog.refs,
        persist_recommendations=False,
    )

    side_overrides = prefs.get("side") or {}
    need = CanaryConfig().min_samples if fill_need is None else max(int(fill_need), 1)
    origin_db = Path(db_path) if db_path else None
    slots: list[SkillSlot] = []
    for idx, skill in enumerate(chosen):
        if idx < len(pinned):
            bucket = "pinned"
        elif idx < len(pinned) + len(ranked):
            bucket = "ranked"
        else:
            bucket = "recommended"
        slot = _resolve_slot(
            skill, client_id, probability, bucket, refs=catalog.refs,
            user_key=user_key, fill_need=need, db_path=origin_db,
            canary_enabled=canary_enabled,
        )
        if slot is not None:
            ov = side_overrides.get(slot.skill_name)
            if ov in ("main", "staging") and not (
                    isinstance(skill, dict) and skill.get("source") == "skillhub"):
                cached_main, cached_staging = (
                    catalog.refs[skill.name] if skill.name in catalog.refs else
                    (main_sha(skill.path) or "", staging_sha(skill.path))
                )
                if ov == "staging" and not cached_staging:
                    ov = "main"
                new_sha = cached_staging if ov == "staging" else cached_main
                if new_sha:
                    slot = SkillSlot(
                        skill_name=slot.skill_name,
                        side=ov,
                        sha=new_sha,
                        bucket=slot.bucket,
                        source=slot.source,
                        display_name=slot.display_name,
                        source_path=slot.source_path,
                    )
            slots.append(slot)
    # 埋点：只记画像推荐位(recommended bucket)。team server 将写入提交给
    # 独立的有界单线程 executor，避免 SQLite 写锁进入 /sync 响应路径；直接
    # 调用 build_manifest 的场景仍同步落盘，保持原有 API 行为。
    records = [
        (s.skill_name, s.side or "main", s.bucket, s.sha or "")
        for s in slots if s.bucket == "recommended"
    ]
    recorder = partial(
        _record_recommendation_telemetry,
        engine=_engine,
        client_id=client_id,
        records=records,
    )
    if telemetry_submit is None:
        recorder()
    elif not telemetry_submit(recorder):
        _logger.debug("recommendation telemetry queue full; event skipped")
    return SyncResponse(slots=slots, server_time=time.time())


def _record_recommendation_telemetry(
    *,
    engine,
    client_id: str,
    records: list[tuple[str, str, str, str]],
) -> None:
    """批量持久化推荐双向视图和曝光事件；失败不影响 sync。"""
    if not records:
        return
    if engine is not None:
        try:
            engine.reco_store.record_many(
                user_id=client_id,
                records=[(skill, side, sha) for skill, side, _bucket, sha in records],
            )
        except Exception:  # pylint: disable=broad-exception-caught
            _logger.debug("recommendation view telemetry skipped", exc_info=True)
    try:
        from xskill.pipeline.registry import record_recommendations
        record_recommendations(client_id=client_id, records=records)
    except Exception:  # pylint: disable=broad-exception-caught
        _logger.debug("recommendation exposure telemetry skipped", exc_info=True)


def _recommend_user_key(client_id: str) -> str:
    """推荐结果表键：有名用 user_name，匿名用 client_id。"""
    if _engine is not None and getattr(_engine, "client_registry", None) is not None:
        try:
            name = _engine.client_registry.user_name_for(client_id)
            if name:
                return name
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    try:
        from xskill.team.server.api import team_context
        name = team_context().client_registry.user_name_for(client_id)
        if name:
            return name
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    return client_id


def _pick_recommended(
    *,
    client_id: str,
    skill_dir: Path,
    ranked: list[Skill],
    ranked_names: set[str],
    ux_ordered: list[Skill],
    reco_slots: int,
    traj_root: Path | str | None,
    candidate_pool: list[Skill] | None = None,
    candidate_refs: dict[str, tuple[str, str | None]] | None = None,
    persist_recommendations: bool = True,
) -> list[Skill]:
    """选 ``recommended`` bucket 的 skill。

    team server 路径：只读重活进程预写的 ``client_recommend_slots``，
    **禁止**请求内 ``get_skill_for_client`` / numpy 全库乘。过期或缺失时
    返回上一份成功结果；无行则 ux 尾部补齐（与冷启动语义一致）。
    """
    del ranked, skill_dir, candidate_refs, persist_recommendations
    if reco_slots <= 0:
        return []

    ux_tail = [s for s in ux_ordered if s.name not in ranked_names]
    if traj_root is None:
        return ux_tail[:reco_slots]  # 非 team server：无 traj_root，按 ux 取

    # 预计算推荐表（重活进程写入）；stale 也返回上一份。
    try:
        from xskill.recommend.recommend_store import load_recommend_slots

        reco_names = load_recommend_slots(_recommend_user_key(client_id))
    except Exception:  # pylint: disable=broad-exception-caught
        _logger.debug("recommend slots load failed", exc_info=True)
        reco_names = None

    if reco_names:
        by_name = {s.name: s for s in ux_tail}
        for s in candidate_pool or []:
            by_name.setdefault(s.name, s)
        picked: list = []
        picked_names: set[str] = set()
        for n in reco_names:
            if n in ranked_names or n in picked_names:
                continue
            item = by_name.get(n)
            if item is None and _engine is not None:
                try:
                    item = _engine.skillhub.entry(n)
                except Exception:  # pylint: disable=broad-exception-caught
                    item = None
            if item is None:
                continue
            picked.append(item)
            picked_names.add(n)
            if len(picked) >= reco_slots:
                break
        if len(picked) < reco_slots:
            for s in ux_tail:
                if len(picked) >= reco_slots:
                    break
                if s.name not in picked_names and s.name not in ranked_names:
                    picked.append(s)
                    picked_names.add(s.name)
        return picked[:reco_slots]

    # 无预计算结果（冷启动 / 尚未跑重活）→ ux_tail，绝不现算 KNN。
    return ux_tail[:reco_slots]
