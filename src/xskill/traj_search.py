"""`xskill search traj` 的检索实现。

当前用随包发布的 mock 轨迹目录做关键词打分，方便 /xskill 里演示
「按指令找相关轨迹」的用法。team server 真检索接上后，只换
``search_trajectories`` 的数据源即可。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9_\u4e00-\u9fff]+", re.IGNORECASE)


def bundled_mock_catalog_path() -> Path:
    from importlib.resources import files

    return Path(str(files("xskill") / "data" / "mock_trajectories.json"))


def load_mock_trajectories(path: Path | None = None) -> list[dict[str, Any]]:
    catalog_path = path or bundled_mock_catalog_path()
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    rows = payload.get("trajectories")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict) and row.get("traj_id")]


def _tokens(text: str) -> list[str]:
    return [part.lower() for part in _TOKEN_RE.findall(text or "")]


def _haystack(row: dict[str, Any]) -> str:
    tags = row.get("tags") or []
    tag_text = " ".join(str(tag) for tag in tags)
    return " ".join(
        str(row.get(key) or "")
        for key in (
            "traj_id", "title", "summary", "skill_used",
            "ecosystem", "user", "status",
        )
    ) + " " + tag_text


def score_trajectory(query: str, row: dict[str, Any]) -> float:
    """按查询词在标题 / 摘要 / 标签里的命中比例打 0–1 分。"""
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    hay = set(_tokens(_haystack(row)))
    if not hay:
        return 0.0
    hits = sum(1 for token in query_tokens if token in hay)
    return hits / len(query_tokens)


def search_trajectories(
    query: str,
    *,
    top_k: int = 5,
    catalog: list[dict[str, Any]] | None = None,
    catalog_path: Path | None = None,
) -> list[dict[str, Any]]:
    """返回按 score 降序的轨迹命中。默认读捆绑 mock 目录。"""
    rows = catalog if catalog is not None else load_mock_trajectories(catalog_path)
    scored: list[dict[str, Any]] = []
    for row in rows:
        score = score_trajectory(query, row)
        if score <= 0:
            continue
        scored.append({
            "traj_id": row.get("traj_id"),
            "score": round(float(score), 3),
            "status": row.get("status") or "-",
            "skill_used": row.get("skill_used") or "-",
            "ecosystem": row.get("ecosystem") or "-",
            "user": row.get("user") or "-",
            "title": row.get("title") or "",
            "summary": row.get("summary") or "",
            "tags": list(row.get("tags") or []),
            "source": "mock",
        })
    scored.sort(key=lambda item: (-item["score"], str(item["traj_id"])))
    limit = max(1, int(top_k))
    return scored[:limit]
