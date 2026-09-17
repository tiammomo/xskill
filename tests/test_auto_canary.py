"""自动灰度：纳入/生成发起人 ∪ 用量最多的用户，按体验分补漏换侧。"""
from __future__ import annotations

import subprocess
from pathlib import Path

from xskill.canary import append_ux_score, fill_deficit_side, staging_sha
from xskill.pipeline.registry import (
    auto_canary_users,
    clear_auto_canary_cache_for_tests,
    is_auto_canary_user,
    pooled_connection,
    record_skill_origin,
    register_dir,
    skill_origin_user,
)
from xskill.pipeline.ux_scores_store import insert_ux_score
from xskill.team.server.skill_manifest import _ROUTER, build_manifest


def _git(args, cwd):
    subprocess.run(
        ["git"] + args, cwd=str(cwd), capture_output=True, text=True, check=True,
    )


def _make_skill(root: Path, name: str, *, with_staging: bool = False) -> Path:
    d = root / name
    d.mkdir(parents=True)
    _git(["init", "-q"], d)
    _git(["checkout", "-q", "-b", "main"], d)
    _git(["config", "user.email", "t@t"], d)
    _git(["config", "user.name", "t"], d)
    (d / "SKILL.md").write_text(f"# {name}\n", encoding="utf-8")
    _git(["add", "."], d)
    _git(["commit", "-q", "-m", "v1"], d)
    if with_staging:
        _git(["checkout", "-q", "-b", "staging"], d)
        (d / "SKILL.md").write_text(f"# {name} staging\n", encoding="utf-8")
        _git(["add", "."], d)
        _git(["commit", "-q", "-m", "staging"], d)
        _git(["checkout", "-q", "main"], d)
    return d


def _seed_usage(db: Path, *, skill: str, user: str, n: int, wd: Path) -> None:
    register_dir(wd, label=user, db_path=db)
    with pooled_connection(db) as conn:
        wd_id = conn.execute(
            "SELECT id FROM watch_dirs WHERE path=?",
            (str(wd.resolve()),),
        ).fetchone()[0]
        for i in range(n):
            conn.execute(
                "INSERT INTO trajectories(watch_dir_id, filename, user_key)"
                " VALUES (?,?,?)",
                (wd_id, f"{user}-t{i}.md", user),
            )
        conn.commit()
    for i in range(n):
        insert_ux_score(
            {
                "skill_name": skill,
                "side": "main",
                "commit_sha": "abc",
                "score": 8.0,
                "scored_at": "2026-01-01T00:00:00+00:00",
                "traj_id": f"{user}-t{i}",
                "atom_id": "",
            },
            db_path=db,
        )
    clear_auto_canary_cache_for_tests()


def test_fill_deficit_side_order():
    assert fill_deficit_side(
        staging_n=0, main_n=10, need=5, fallback="main") == "staging"
    assert fill_deficit_side(
        staging_n=5, main_n=1, need=5, fallback="staging") == "main"
    assert fill_deficit_side(
        staging_n=5, main_n=5, need=5, fallback="main") == "main"


def test_origin_first_writer_wins(tmp_path):
    db = tmp_path / "r.db"
    register_dir(tmp_path / "wd", label="t", db_path=db)
    assert record_skill_origin(
        skill_name="s1", user_key="alice", source="import", db_path=db)
    assert not record_skill_origin(
        skill_name="s1", user_key="bob", source="generate", db_path=db)
    assert skill_origin_user("s1", db_path=db) == "alice"


def test_auto_canary_users_union_origin_and_top_usage(tmp_path):
    db = tmp_path / "r.db"
    record_skill_origin(
        skill_name="s1", user_key="alice", source="import", db_path=db)
    _seed_usage(
        db, skill="s1", user="bob", n=3, wd=tmp_path / "bob-wd")
    _seed_usage(
        db, skill="s1", user="carol", n=1, wd=tmp_path / "carol-wd")
    users = auto_canary_users("s1", db_path=db)
    assert users == {"alice", "bob"}
    assert is_auto_canary_user("alice", "s1", db_path=db)
    assert is_auto_canary_user("bob", "s1", db_path=db)
    assert not is_auto_canary_user("carol", "s1", db_path=db)


def test_manifest_origin_user_fills_staging(tmp_path, monkeypatch):
    db = tmp_path / "r.db"
    monkeypatch.setattr("xskill.config.get_registry_db_path", lambda: db)
    register_dir(tmp_path / "wd", label="t", db_path=db)
    record_skill_origin(
        skill_name="s1", user_key="alice", source="import", db_path=db)
    skills = tmp_path / "skills"
    skills.mkdir()
    _make_skill(skills, "s1", with_staging=True)
    _ROUTER.reset()
    alice = build_manifest(
        client_id="c-alice", skill_dir=skills, probability=0,
        ranked_slots=80, total_slots=100, user_key="alice", db_path=db,
    )
    bob = build_manifest(
        client_id="c-bob", skill_dir=skills, probability=0,
        ranked_slots=80, total_slots=100, user_key="bob", db_path=db,
    )
    assert alice.slots[0].side == "staging"
    assert bob.slots[0].side == "main"
    _ROUTER.reset()


def test_manifest_top_usage_user_fills_staging(tmp_path, monkeypatch):
    db = tmp_path / "r.db"
    monkeypatch.setattr("xskill.config.get_registry_db_path", lambda: db)
    _seed_usage(db, skill="s1", user="bob", n=4, wd=tmp_path / "bob-wd")
    skills = tmp_path / "skills"
    skills.mkdir()
    _make_skill(skills, "s1", with_staging=True)
    _ROUTER.reset()
    bob = build_manifest(
        client_id="c-bob", skill_dir=skills, probability=0,
        ranked_slots=80, total_slots=100, user_key="bob", db_path=db,
    )
    alice = build_manifest(
        client_id="c-alice", skill_dir=skills, probability=0,
        ranked_slots=80, total_slots=100, user_key="alice", db_path=db,
    )
    assert bob.slots[0].side == "staging"
    assert alice.slots[0].side == "main"
    _ROUTER.reset()


def test_manifest_pin_side_overrides_auto_canary(tmp_path, monkeypatch):
    db = tmp_path / "r.db"
    monkeypatch.setattr("xskill.config.get_registry_db_path", lambda: db)
    register_dir(tmp_path / "wd", label="t", db_path=db)
    record_skill_origin(
        skill_name="s1", user_key="alice", source="import", db_path=db)
    skills = tmp_path / "skills"
    skills.mkdir()
    _make_skill(skills, "s1", with_staging=True)
    _ROUTER.reset()
    resp = build_manifest(
        client_id="c-alice", skill_dir=skills, probability=0,
        ranked_slots=80, total_slots=100, user_key="alice", db_path=db,
        prefs={"pinned": ["s1"], "blocked": set(), "side": {"s1": "main"}},
    )
    assert resp.slots[0].side == "main"
    _ROUTER.reset()


def test_canary_disabled_skips_auto_canary_lookup(tmp_path, monkeypatch):
    db = tmp_path / "r.db"
    monkeypatch.setattr("xskill.config.get_registry_db_path", lambda: db)
    register_dir(tmp_path / "wd", label="t", db_path=db)
    record_skill_origin(
        skill_name="s1", user_key="alice", source="import", db_path=db)
    skills = tmp_path / "skills"
    skills.mkdir()
    _make_skill(skills, "s1", with_staging=True)

    def forbidden_lookup(*_args, **_kwargs):
        raise AssertionError("canary disabled must not query auto-canary users")

    monkeypatch.setattr(
        "xskill.pipeline.registry.is_auto_canary_user", forbidden_lookup)
    _ROUTER.reset()
    resp = build_manifest(
        client_id="c-alice", skill_dir=skills, probability=1.0,
        ranked_slots=80, total_slots=100, user_key="alice", db_path=db,
        canary_enabled=False,
    )
    assert resp.slots[0].side == "main"
    _ROUTER.reset()


def test_auto_canary_switches_to_main_when_staging_full(tmp_path, monkeypatch):
    db = tmp_path / "r.db"
    monkeypatch.setattr("xskill.config.get_registry_db_path", lambda: db)
    register_dir(tmp_path / "wd", label="t", db_path=db)
    record_skill_origin(
        skill_name="s1", user_key="alice", source="import", db_path=db)
    skills = tmp_path / "skills"
    skills.mkdir()
    dest = _make_skill(skills, "s1", with_staging=True)
    s_sha = staging_sha(dest)
    for i in range(2):
        append_ux_score(
            dest, traj_id=f"st{i}", skill_name="s1", side="staging",
            commit_sha=s_sha or "", score=7, reasons="r",
        )
    _ROUTER.reset()
    resp = build_manifest(
        client_id="c-alice", skill_dir=skills, probability=0,
        ranked_slots=80, total_slots=100, user_key="alice", db_path=db,
        fill_need=2,
    )
    assert resp.slots[0].side == "main"
    _ROUTER.reset()
