"""`xskill search traj` 的检索装配。

真正搜的是 Atom 混合检索（`xskill.utils.search.search` / `search_all`）：
向量 + BM25，命中带 traj_id。本模块只负责按工号收窄目录、把原始命中
收成 CLI / team API 共用的精简字段。不读随包假数据，不读轨迹原文。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("xskill.traj_search")

SearchOne = Callable[..., list[dict[str, Any]]]
SearchAll = Callable[..., list[dict[str, Any]]]
FindClientId = Callable[[str], str | None]
DirNameFor = Callable[[str], str]


def parse_search_names(raw: str | None) -> list[str]:
    """把 ``--name 张三,李四`` 收成去空白的工号列表，保持用户写下的顺序。"""
    names: list[str] = []
    seen: set[str] = set()
    for part in str(raw or "").split(","):
        name = part.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def format_traj_hit(raw: dict[str, Any], *, user: str = "") -> dict[str, Any]:
    """把 ``search`` 的 Atom 命中收成对外字段，去掉路径和原文。"""
    vector = raw.get("vector_similarity")
    bm25 = raw.get("bm25_score")
    if vector is None:
        score = float(bm25 or 0.0)
    else:
        score = float(vector)
    used = raw.get("used_skills") or []
    if not used:
        atom = raw.get("atom")
        if atom is not None:
            used = getattr(atom, "used_skills", None) or []
    if not isinstance(used, list):
        used = []
    sources = raw.get("sources") or []
    if not isinstance(sources, list):
        sources = []
    return {
        "traj_id": str(raw.get("traj_id") or ""),
        "atom_id": str(raw.get("atom_id") or ""),
        "intent": str(raw.get("intent") or ""),
        "summary": str(raw.get("summary") or ""),
        "score": score,
        "vector_similarity": vector,
        "bm25_score": bm25,
        "sources": [str(item) for item in sources],
        "user": user,
        "used_skills": [str(item) for item in used],
    }


def _hit_sort_key(hit: dict[str, Any]) -> tuple:
    # 有向量分的排前面；只有 BM25 的排后面。同档再按分数、traj_id。
    has_vector = 0 if hit.get("vector_similarity") is None else 1
    return (
        -has_vector,
        -float(hit.get("score") or 0.0),
        str(hit.get("traj_id") or ""),
    )


def user_label_for_dataset(dataset_dir: str | Path) -> str:
    """用 registry 的 watch 目录 label（工号目录名）当命中上的 user。"""
    from xskill.pipeline.registry import list_watch_dirs

    target = str(Path(dataset_dir).expanduser().resolve())
    for row in list_watch_dirs():
        path = row.get("path")
        if not path:
            continue
        if str(Path(path).expanduser().resolve()) == target:
            return str(row.get("label") or "")
    return ""


def resolve_named_session_dirs(
    names: list[str],
    *,
    traj_root: Path,
    find_client_id: FindClientId,
    dir_name_for: DirNameFor,
) -> tuple[list[tuple[str, Path]], list[str]]:
    """工号 → ``<traj_root>/clients/<dir>/sessions``。不认识的工号进 unknown。"""
    found: list[tuple[str, Path]] = []
    unknown: list[str] = []
    root = Path(traj_root)
    for name in names:
        client_id = find_client_id(name)
        if not client_id:
            unknown.append(name)
            continue
        try:
            dir_name = dir_name_for(client_id)
        except ValueError:
            unknown.append(name)
            continue
        found.append((name, root / "clients" / dir_name / "sessions"))
    return found, unknown


def resolve_registered_session_dirs(
    client_rows: list[dict[str, Any]],
    *,
    traj_root: Path,
    dir_name_for: DirNameFor,
) -> list[tuple[str, Path]]:
    """已注册 team client → 各自的 sessions 目录，不包含 server 本地 watch dir。"""
    found: list[tuple[str, Path]] = []
    root = Path(traj_root)
    for row in client_rows:
        client_id = str(row.get("client_id") or "").strip()
        if not client_id:
            continue
        try:
            dir_name = dir_name_for(client_id)
        except ValueError:
            logger.warning("traj search skipped unknown client %s", client_id)
            continue
        user = str(row.get("user_name") or client_id)
        found.append((user, root / "clients" / dir_name / "sessions"))
    return found


def search_indexed_trajectories(
    query: str,
    *,
    top_k: int = 5,
    dataset_dirs: list[tuple[str, Path]] | None = None,
    search_one: SearchOne | None = None,
    search_all_fn: SearchAll | None = None,
) -> list[dict[str, Any]]:
    """在指定 sessions 目录或全量 registry 上跑 Atom 混合检索。"""
    from xskill.utils.search import search as default_search_one
    from xskill.utils.search import search_all as default_search_all

    one = search_one or default_search_one
    all_fn = search_all_fn or default_search_all
    limit = max(1, int(top_k))
    merged: list[dict[str, Any]] = []

    if dataset_dirs is None:
        raw_hits = all_fn(query_text=query, top_k=limit)
        for raw in raw_hits:
            dataset = raw.get("dataset_dir") or ""
            user = str(raw.get("user") or "") or user_label_for_dataset(dataset)
            merged.append(format_traj_hit(raw, user=user))
        merged.sort(key=_hit_sort_key)
        return merged[:limit]

    for user, path in dataset_dirs:
        directory = Path(path)
        if not directory.is_dir():
            continue
        try:
            raw_hits = one(
                dataset_dir=directory,
                query_text=query,
                top_k=limit,
            )
        except Exception:
            logger.warning("traj search skipped directory %s", directory, exc_info=True)
            continue
        for raw in raw_hits:
            merged.append(format_traj_hit(raw, user=user))
    merged.sort(key=_hit_sort_key)
    return merged[:limit]
