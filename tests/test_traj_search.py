"""`xskill search traj`：捆绑 mock 目录 + CLI 分流。"""
from __future__ import annotations

import json
from types import SimpleNamespace

from xskill import cli
from xskill.traj_search import score_trajectory, search_trajectories


def _args(**overrides) -> SimpleNamespace:
    base = {
        "terms": ["traj", "memory", "leak"],
        "top_k": 5,
        "json": False,
        "download": False,
        "team": False,
        "local": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_score_counts_query_tokens_in_title_and_tags():
    row = {
        "traj_id": "traj_demo",
        "title": "Diagnose a Python process leaking memory",
        "summary": "tracemalloc found a cache",
        "tags": ["python", "memory", "leak"],
    }
    assert score_trajectory("memory leak", row) == 1.0
    assert score_trajectory("memory network", row) == 0.5
    assert score_trajectory("compose", row) == 0.0


def test_search_ranks_memory_leak_above_unrelated():
    hits = search_trajectories("python memory leak", top_k=3)
    assert hits
    assert hits[0]["traj_id"] == "traj_cc_alice_memleak"
    assert hits[0]["source"] == "mock"
    assert hits[0]["score"] > 0
    ids = [hit["traj_id"] for hit in hits]
    assert "traj_codex_dan_compose" not in ids


def test_search_respects_top_k():
    hits = search_trajectories("memory", top_k=1)
    assert len(hits) == 1


def test_search_empty_query_scores_nothing():
    assert search_trajectories("   ", top_k=5) == []


def test_cli_search_traj_prints_hits(capsys):
    rc = cli.cmd_search(_args())
    assert rc == 0
    out = capsys.readouterr().out
    assert "mock traj search" in out
    assert "traj_cc_alice_memleak" in out
    assert "python-memory-debug" in out


def test_cli_search_traj_json(capsys):
    rc = cli.cmd_search(_args(json=True, terms=["traj", "alembic"]))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["traj_id"] == "traj_cursor_carol_migration"
    assert payload[0]["source"] == "mock"


def test_cli_search_traj_missing_query_errors(capsys):
    rc = cli.cmd_search(_args(terms=["traj"]))
    assert rc == 2
    err = capsys.readouterr().err
    assert "xskill search traj <query>" in err


def test_cli_search_traj_no_hit(capsys):
    rc = cli.cmd_search(_args(terms=["traj", "qwertyuiop"]))
    assert rc == 0
    assert "无匹配" in capsys.readouterr().out


def test_cli_search_traj_ignores_download(capsys):
    rc = cli.cmd_search(_args(download=True, terms=["traj", "auth"]))
    assert rc == 0
    captured = capsys.readouterr()
    assert "轨迹检索忽略" in captured.err
    assert "traj_cc_erin_auth_retry" in captured.out


def test_cli_search_without_traj_prefix_does_not_use_mock(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "cmd_search_traj", lambda args: called.append("traj") or 0)
    monkeypatch.setattr(cli, "cmd_search_hub", lambda args: called.append("hub") or 0)
    monkeypatch.setattr(cli, "_cmd_search_local", lambda args: called.append("local") or 0)
    monkeypatch.setattr("xskill.runtime.role", lambda: "standalone")

    rc = cli.cmd_search(_args(terms=["docker", "compose"]))

    assert rc == 0
    assert called == ["local"]


def test_bundled_skill_documents_search_traj():
    from xskill.ecosystems.bundled_guide import bundled_xskill_source

    skill_md = (bundled_xskill_source() / "SKILL.md").read_text(encoding="utf-8")
    assert "xskill search traj" in skill_md
