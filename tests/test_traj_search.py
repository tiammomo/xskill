"""`xskill search traj`：Atom 混合检索装配 + CLI / team 分流。

不读随包假目录。检索要么注入与生产同形的 search 命中，要么用
AtomTaskStore + FakeEmbed 走真实 HybridSearch。
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from xskill import cli
from xskill.pipeline.atom import AtomTask, AtomTaskStore
from xskill.team.server import api as server_api
from xskill.team.server.client_registry import ClientRegistry
from xskill.traj_search import (
    format_traj_hit,
    parse_search_names,
    resolve_named_session_dirs,
    resolve_registered_session_dirs,
    search_indexed_trajectories,
)
from xskill.utils.search import HybridSearch
from tests.test_atom_task_store import _FakeEmbed

TOKEN = "secret-token"


def _args(**overrides) -> SimpleNamespace:
    base = {
        "terms": ["traj", "django", "migration"],
        "top_k": 5,
        "json": False,
        "download": False,
        "team": False,
        "local": False,
        "name": "",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _atom_hit(**overrides) -> dict:
    row = {
        "atom_id": "atom_t_0001",
        "traj_id": "traj_cc_alice_memleak",
        "intent": "diagnose a python process leaking memory",
        "summary": "tracemalloc found a cache",
        "sources": ["vector", "keyword"],
        "vector_similarity": 0.91,
        "bm25_score": 4.2,
        "used_skills": ["python-memory-debug"],
        "md_path": "/secret/server/traj_cc_alice_memleak.md",
        "dataset_dir": "/secret/server/clients/alice/sessions",
        "raw_segment": "MUST_NOT_LEAK",
    }
    row.update(overrides)
    return row


class _Response:
    def __init__(self, status_code: int, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def json(self):
        return self._json_data


class _TrajHttp:
    def __init__(self, status_code=200, payload=None, error=None):
        self.status_code = status_code
        self.payload = payload if payload is not None else {
            "results": [format_traj_hit(_atom_hit(), user="alice")],
            "count": 1,
            "meta": {"unknown_names": [], "corpus_empty": False},
        }
        self.error = error
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, path: str, **kwargs):
        self.calls.append((path, kwargs.get("params")))
        if self.error is not None:
            raise self.error
        return _Response(self.status_code, self.payload)


def _seed_sessions(root: Path, *, traj_id: str, intent: str,
                   used_skills: list[str] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    store = AtomTaskStore(root=root)
    atom = AtomTask(
        atom_id=f"atom_{traj_id}_0001",
        traj_id=traj_id,
        offset_start=0,
        offset_end=10,
        intent=intent,
        summary=intent,
        tags=[],
        used_skills=used_skills or [],
        ux_score=None,
        pre_atom_id=None,
        post_atom_id=None,
        context_prefix="",
        raw_segment="raw session text must not appear in hits",
    )
    store.save(atom)
    store.rebuild_vector_index(_FakeEmbed())
    return root


def _hybrid_search_one(dataset_dir, query_text, top_k=5, **_kwargs):
    """与 ``utils.search.search`` 同形，embed 用测试 FakeEmbed。"""
    store = AtomTaskStore(root=Path(dataset_dir))
    hits = HybridSearch(store, _FakeEmbed()).search(query_text, top_k=top_k)
    out = []
    for hit in hits:
        atom = store.load(hit["atom_id"])
        hit["traj_id"] = atom.traj_id
        hit["md_path"] = str(Path(dataset_dir) / f"{atom.traj_id}.md")
        hit["intent"] = atom.intent
        hit["summary"] = atom.summary
        hit["used_skills"] = list(atom.used_skills or [])
        out.append(hit)
    return out


# ── 装配 ──────────────────────────────────────────────────────────


def test_parse_search_names_keeps_order_and_dedupes():
    assert parse_search_names(" 张三, 李四,张三, ") == ["张三", "李四"]
    assert parse_search_names("") == []
    assert parse_search_names(None) == []


def test_format_traj_hit_drops_path_and_raw_text():
    hit = format_traj_hit(_atom_hit(), user="alice")
    assert hit["traj_id"] == "traj_cc_alice_memleak"
    assert hit["atom_id"] == "atom_t_0001"
    assert hit["user"] == "alice"
    assert hit["used_skills"] == ["python-memory-debug"]
    assert hit["score"] == pytest.approx(0.91)
    assert "md_path" not in hit
    assert "dataset_dir" not in hit
    assert "raw_segment" not in hit
    assert "MUST_NOT_LEAK" not in json.dumps(hit)


def test_format_prefers_vector_score_else_bm25():
    vector = format_traj_hit(_atom_hit(vector_similarity=0.2, bm25_score=9.0))
    assert vector["score"] == pytest.approx(0.2)
    bm25_only = format_traj_hit(_atom_hit(
        vector_similarity=None, bm25_score=9.0, sources=["keyword"],
    ))
    assert bm25_only["score"] == pytest.approx(9.0)


def test_search_indexed_merges_named_dirs_and_caps_top_k(tmp_path):
    alice = tmp_path / "alice" / "sessions"
    bob = tmp_path / "bob" / "sessions"
    alice.mkdir(parents=True)
    bob.mkdir(parents=True)
    seen: list[str] = []

    def search_one(*, dataset_dir, query_text, top_k):
        seen.append(str(dataset_dir))
        assert query_text == "memory leak"
        assert top_k == 1
        label = Path(dataset_dir).parent.name
        return [_atom_hit(
            traj_id=f"traj_{label}",
            atom_id=f"atom_{label}",
            vector_similarity=0.4 if label == "alice" else 0.9,
            user=label,
        )]

    hits = search_indexed_trajectories(
        "memory leak",
        top_k=1,
        dataset_dirs=[("alice", alice), ("bob", bob)],
        search_one=search_one,
    )
    assert [hit["traj_id"] for hit in hits] == ["traj_bob"]
    assert hits[0]["user"] == "bob"
    assert set(seen) == {str(alice), str(bob)}


def test_search_indexed_puts_bm25_only_after_vector_hits(tmp_path):
    directory = tmp_path / "sessions"
    directory.mkdir()

    def search_one(*, dataset_dir, query_text, top_k):
        return [
            _atom_hit(
                traj_id="traj_kw",
                atom_id="atom_kw",
                vector_similarity=None,
                bm25_score=99.0,
                sources=["keyword"],
            ),
            _atom_hit(
                traj_id="traj_vec",
                atom_id="atom_vec",
                vector_similarity=0.11,
                bm25_score=0.1,
            ),
        ]

    hits = search_indexed_trajectories(
        "q",
        top_k=5,
        dataset_dirs=[("u", directory)],
        search_one=search_one,
    )
    assert [hit["traj_id"] for hit in hits] == ["traj_vec", "traj_kw"]


def test_search_indexed_skips_missing_dir_and_search_errors(tmp_path):
    missing = tmp_path / "gone" / "sessions"
    broken = tmp_path / "broken" / "sessions"
    broken.mkdir(parents=True)

    def search_one(*, dataset_dir, query_text, top_k):
        raise RuntimeError("embed down")

    assert search_indexed_trajectories(
        "q",
        dataset_dirs=[("gone", missing), ("broken", broken)],
        search_one=search_one,
    ) == []


def test_search_indexed_all_uses_injected_search_all():
    def search_all_fn(*, query_text, top_k):
        assert query_text == "auth retry"
        assert top_k == 3
        return [_atom_hit(user="alice")]

    hits = search_indexed_trajectories(
        "auth retry", top_k=3, search_all_fn=search_all_fn,
    )
    assert hits[0]["traj_id"] == "traj_cc_alice_memleak"
    assert "md_path" not in hits[0]


def test_resolve_named_session_dirs_keeps_unknown(tmp_path):
    found, unknown = resolve_named_session_dirs(
        ["alice", "ghost"],
        traj_root=tmp_path,
        find_client_id={"alice": "cid-a"}.get,
        dir_name_for={"cid-a": "alice"}.get,
    )
    assert found == [("alice", tmp_path / "clients" / "alice" / "sessions")]
    assert unknown == ["ghost"]


def test_resolve_named_session_dirs_dir_lookup_failure_is_unknown(tmp_path):
    def dir_name_for(_client_id: str) -> str:
        raise ValueError("unknown client_id")

    found, unknown = resolve_named_session_dirs(
        ["alice"],
        traj_root=tmp_path,
        find_client_id={"alice": "cid-a"}.get,
        dir_name_for=dir_name_for,
    )
    assert found == []
    assert unknown == ["alice"]


def test_resolve_registered_session_dirs_only_uses_team_clients(tmp_path):
    found = resolve_registered_session_dirs(
        [
            {"client_id": "cid-a", "user_name": "alice"},
            {"client_id": "cid-b", "user_name": ""},
        ],
        traj_root=tmp_path,
        dir_name_for={"cid-a": "alice", "cid-b": "cid-b"}.get,
    )
    assert found == [
        ("alice", tmp_path / "clients" / "alice" / "sessions"),
        ("cid-b", tmp_path / "clients" / "cid-b" / "sessions"),
    ]


def test_hybrid_search_one_returns_atom_fields_not_raw(tmp_path):
    sessions = _seed_sessions(
        tmp_path / "sessions",
        traj_id="traj_cc_alice_memleak",
        intent="fix django migration conflict",
        used_skills=["alembic-half"],
    )
    hits = _hybrid_search_one(sessions, "django migration", top_k=5)
    assert hits
    assert hits[0]["traj_id"] == "traj_cc_alice_memleak"
    assert "django" in hits[0]["intent"]
    assert hits[0]["used_skills"] == ["alembic-half"]
    formatted = format_traj_hit(hits[0], user="alice")
    assert "raw session text" not in json.dumps(formatted)
    assert "md_path" not in formatted


# ── CLI ───────────────────────────────────────────────────────────


def test_cli_search_traj_local_prints_hits(monkeypatch, capsys):
    monkeypatch.setattr("xskill.runtime.role", lambda: "standalone")

    def fake_search(query, *, top_k=5, **_kwargs):
        assert query == "django migration"
        assert top_k == 5
        return [format_traj_hit(_atom_hit(), user="alice")]

    monkeypatch.setattr(
        "xskill.traj_search.search_indexed_trajectories", fake_search,
    )
    monkeypatch.setattr(
        "xskill.pipeline.registry.all_index_paths", lambda: [Path("/x")],
    )
    rc = cli.cmd_search(_args())
    assert rc == 0
    out = capsys.readouterr()
    assert "traj_cc_alice_memleak" in out.out
    assert "alice" in out.out
    assert "diagnose a python process" in out.out
    assert "MUST_NOT_LEAK" not in out.out
    assert "/secret/server" not in out.out


def test_cli_search_traj_local_json(monkeypatch, capsys):
    monkeypatch.setattr("xskill.runtime.role", lambda: "standalone")
    monkeypatch.setattr(
        "xskill.traj_search.search_indexed_trajectories",
        lambda query, **_kw: [format_traj_hit(_atom_hit(), user="alice")],
    )
    rc = cli.cmd_search(_args(json=True, terms=["traj", "alembic"]))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["traj_id"] == "traj_cc_alice_memleak"
    assert "md_path" not in payload[0]
    assert "raw_segment" not in payload[0]


def test_cli_search_traj_local_warns_and_ignores_name(monkeypatch, capsys):
    monkeypatch.setattr("xskill.runtime.role", lambda: "standalone")
    monkeypatch.setattr(
        "xskill.traj_search.search_indexed_trajectories",
        lambda query, **_kw: [],
    )
    monkeypatch.setattr("xskill.pipeline.registry.all_index_paths", lambda: [])
    rc = cli.cmd_search(_args(name="alice", terms=["traj", "q"]))
    assert rc == 0
    captured = capsys.readouterr()
    assert "仅 team" in captured.err
    assert "轨迹索引尚未建成" in captured.out


def test_cli_search_traj_missing_query_errors(capsys):
    rc = cli.cmd_search(_args(terms=["traj"]))
    assert rc == 2
    assert "xskill search traj <query>" in capsys.readouterr().err


def test_cli_search_traj_ignores_download(monkeypatch, capsys):
    monkeypatch.setattr("xskill.runtime.role", lambda: "standalone")
    monkeypatch.setattr(
        "xskill.traj_search.search_indexed_trajectories",
        lambda query, **_kw: [format_traj_hit(_atom_hit(), user="alice")],
    )
    monkeypatch.setattr(
        "xskill.pipeline.registry.all_index_paths", lambda: [Path("/x")],
    )
    rc = cli.cmd_search(_args(download=True, terms=["traj", "auth"]))
    assert rc == 0
    captured = capsys.readouterr()
    assert "轨迹检索忽略" in captured.err
    assert "traj_cc_alice_memleak" in captured.out


def test_cli_search_without_traj_prefix_stays_on_skill_search(monkeypatch):
    called = []
    monkeypatch.setattr(
        cli, "cmd_search_traj", lambda args: called.append("traj") or 0,
    )
    monkeypatch.setattr(
        cli, "cmd_search_hub", lambda args: called.append("hub") or 0,
    )
    monkeypatch.setattr(
        cli, "_cmd_search_local", lambda args: called.append("local") or 0,
    )
    monkeypatch.setattr("xskill.runtime.role", lambda: "standalone")
    rc = cli.cmd_search(_args(terms=["docker", "compose"]))
    assert rc == 0
    assert called == ["local"]


def test_cli_search_traj_dispatches_team_when_client(monkeypatch):
    called = []
    monkeypatch.setattr("xskill.runtime.role", lambda: "client")
    monkeypatch.setattr(
        cli, "_cmd_search_traj_team",
        lambda query, **kw: called.append(("team", query, kw.get("names"))) or 0,
    )
    monkeypatch.setattr(
        cli, "_cmd_search_traj_local",
        lambda query, **kw: called.append(("local", query)) or 0,
    )
    rc = cli.cmd_search(_args(name="alice,bob", terms=["traj", "发票"]))
    assert rc == 0
    assert called == [("team", "发票", ["alice", "bob"])]


def test_cli_search_traj_team_prints_and_forwards_names(capsys):
    http = _TrajHttp()
    rc = cli.cmd_search_traj(
        _args(team=True, name="alice,ghost", terms=["traj", "memory"]),
        http=http,
        headers={"X-Xskill-Token": TOKEN},
    )
    assert rc == 0
    assert http.calls[0][0] == "/api/v1/team/trajectories/search"
    assert http.calls[0][1] == {
        "query": "memory", "limit": 5, "names": "alice,ghost",
    }
    out = capsys.readouterr().out
    assert "traj_cc_alice_memleak" in out


def test_cli_search_traj_team_unknown_names_warn(capsys):
    http = _TrajHttp(payload={
        "results": [],
        "count": 0,
        "meta": {"unknown_names": ["ghost"], "corpus_empty": False},
    })
    rc = cli.cmd_search_traj(
        _args(team=True, terms=["traj", "q"]),
        http=http,
        headers={},
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "未识别工号 ghost" in captured.err
    assert "轨迹无匹配" in captured.out


def test_cli_search_traj_team_old_server(capsys):
    http = _TrajHttp(status_code=404)
    rc = cli.cmd_search_traj(
        _args(team=True, terms=["traj", "q"]),
        http=http,
        headers={},
    )
    assert rc == 1
    assert "升级 server" in capsys.readouterr().err


def test_cli_search_traj_team_network_error(capsys):
    http = _TrajHttp(error=httpx.ConnectError("refused"))
    rc = cli.cmd_search_traj(
        _args(team=True, terms=["traj", "q"]),
        http=http,
        headers={},
    )
    assert rc == 1
    assert "无法连接 team server" in capsys.readouterr().err


def test_cli_search_traj_unconnected_errors(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        "xskill.config.get_team_client_state_path",
        lambda: tmp_path / "missing.json",
    )
    rc = cli.cmd_search_traj(_args(team=True, terms=["traj", "q"]))
    assert rc == 1
    assert "未连接 team server" in capsys.readouterr().err


# ── team HTTP ─────────────────────────────────────────────────────


def _make_team_app(tmp_path: Path):
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    traj_root = tmp_path / "team_traj"
    registry = ClientRegistry(tmp_path / "clients.db")
    server_api.init_team_context(
        join_token=TOKEN,
        client_registry=registry,
        skill_dir=skill_dir,
        traj_root=traj_root,
        register_dir=lambda path, label: None,
    )
    app = FastAPI()
    app.include_router(server_api.router)
    return TestClient(app), traj_root, registry


def _register(client: TestClient, user_name: str) -> dict:
    response = client.post(
        "/api/v1/team/register",
        json={"token": TOKEN, "user_name": user_name, "hostname": "h"},
    )
    assert response.status_code == 200
    return {
        "X-Xskill-Token": TOKEN,
        "X-Xskill-Client": response.json()["client_id"],
    }


def test_team_traj_search_requires_auth(tmp_path):
    client, _traj_root, _reg = _make_team_app(tmp_path)
    assert client.get(
        "/api/v1/team/trajectories/search", params={"query": "q"},
    ).status_code == 401


def test_team_traj_search_rejects_empty_query(tmp_path):
    client, _traj_root, _reg = _make_team_app(tmp_path)
    headers = _register(client, "alice")
    response = client.get(
        "/api/v1/team/trajectories/search",
        params={"query": "  "},
        headers=headers,
    )
    assert response.status_code == 400


def test_team_traj_search_named_dir_uses_real_hybrid_search(tmp_path, monkeypatch):
    client, traj_root, registry = _make_team_app(tmp_path)
    headers = _register(client, "alice")
    client_id = registry.find_by_user_name("alice")
    sessions = traj_root / "clients" / registry.dir_name_for(client_id) / "sessions"
    _seed_sessions(
        sessions,
        traj_id="traj_cc_alice_memleak",
        intent="fix django migration conflict",
        used_skills=["alembic-half"],
    )
    monkeypatch.setattr(
        "xskill.utils.search.create_embed_client", lambda _cfg=None: _FakeEmbed(),
    )
    monkeypatch.setattr("xskill.utils.search.load_config", lambda: {})
    response = client.get(
        "/api/v1/team/trajectories/search",
        params={"query": "django migration", "limit": 5, "names": "alice,ghost"},
        headers=headers,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["meta"]["unknown_names"] == ["ghost"]
    assert payload["count"] >= 1
    hit = payload["results"][0]
    assert hit["traj_id"] == "traj_cc_alice_memleak"
    assert hit["user"] == "alice"
    assert hit["used_skills"] == ["alembic-half"]
    assert "md_path" not in hit
    assert "raw_segment" not in hit
    dumped = json.dumps(payload)
    assert "raw session text" not in dumped
    assert str(traj_root) not in dumped


def test_team_traj_search_unknown_names_do_not_fail(tmp_path, monkeypatch):
    client, _traj_root, _reg = _make_team_app(tmp_path)
    headers = _register(client, "alice")

    def boom(*_args, **_kwargs):
        raise AssertionError("should not search when every name is unknown")

    monkeypatch.setattr("xskill.utils.search.search", boom)
    monkeypatch.setattr("xskill.utils.search.search_all", boom)
    response = client.get(
        "/api/v1/team/trajectories/search",
        params={"query": "q", "names": "ghost"},
        headers=headers,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["results"] == []
    assert payload["meta"]["unknown_names"] == ["ghost"]
    assert payload["meta"]["corpus_empty"] is False


def test_team_traj_search_all_scopes_to_registered_team_dirs(tmp_path, monkeypatch):
    client, traj_root, registry = _make_team_app(tmp_path)
    headers = _register(client, "alice")
    client_id = registry.find_by_user_name("alice")
    sessions = traj_root / "clients" / registry.dir_name_for(client_id) / "sessions"
    sessions.mkdir(parents=True)

    def search_one(*, dataset_dir, query_text, top_k, **_kwargs):
        assert query_text == "auth retry"
        assert Path(dataset_dir) == sessions
        return [_atom_hit(user="alice")]

    monkeypatch.setattr("xskill.utils.search.search", search_one)
    monkeypatch.setattr(
        "xskill.utils.search.search_all",
        lambda **_kwargs: pytest.fail("team search must not scan global registry"),
    )
    response = client.get(
        "/api/v1/team/trajectories/search",
        params={"query": "auth retry", "limit": 5},
        headers=headers,
    )
    assert response.status_code == 200
    hit = response.json()["results"][0]
    assert hit["traj_id"] == "traj_cc_alice_memleak"
    assert "md_path" not in hit


def test_bundled_skill_documents_real_traj_search_not_mock():
    from xskill.ecosystems.bundled_guide import bundled_xskill_source

    skill_md = (bundled_xskill_source() / "SKILL.md").read_text(encoding="utf-8")
    assert "xskill search traj" in skill_md
    assert "mock" not in skill_md.lower()


def test_package_has_no_bundled_mock_catalog():
    data = Path(__file__).resolve().parents[1] / "src" / "xskill" / "data"
    assert not (data / "mock_trajectories.json").exists()
