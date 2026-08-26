"""捆绑 /xskill 指南：探测生态后装进对应 skill 目录。"""
from __future__ import annotations

from pathlib import Path

import pytest

from xskill.ecosystems.bundled_guide import install_bundled_xskill_guide


@pytest.fixture
def bundled_skill(tmp_path, monkeypatch):
    root = tmp_path / "pkgroot"
    skill = root / "data" / "skill" / "xskill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# xskill\n", encoding="utf-8")
    monkeypatch.setattr("importlib.resources.files", lambda pkg: root)
    return skill


@pytest.fixture
def install_recorder(monkeypatch):
    installed = []
    for eco in ("claude_code", "codex", "nga3", "opencode", "ngagent",
                "openclaw", "cursor", "trae", "deepseek_harness"):
        def _make(eco_name):
            def _fake(skill_path, target_root=None, side="main"):
                installed.append((eco_name, Path(skill_path), target_root, side))
                return Path(skill_path) / "SKILL.md"
            return _fake
        monkeypatch.setattr(
            f"xskill.ecosystems.install_to_{eco}",
            _make(eco),
        )
    return installed


def test_installs_into_detected_ecosystems(
        bundled_skill, install_recorder, monkeypatch, tmp_path, capsys):
    home = tmp_path / "home"
    monkeypatch.setattr(
        "xskill.ecosystems.detect_known_ecosystems",
        lambda home_root=None: [
            {"ecosystem": "claude_code", "source": "/x", "bridge": "/y"},
            {"ecosystem": "cursor", "source": "/c", "bridge": "/d"},
        ],
    )

    installed = install_bundled_xskill_guide(target_root=home)

    assert installed == ["claude_code", "cursor"]
    assert [row[0] for row in install_recorder] == ["claude_code", "cursor"]
    assert all(row[1] == bundled_skill for row in install_recorder)
    assert all(row[2] == home for row in install_recorder)
    out = capsys.readouterr().out
    assert "/xskill" in out
    assert "claude_code" in out
    assert "cursor" in out


def test_missing_bundle_skips_without_calling_installers(
        tmp_path, install_recorder, monkeypatch, capsys):
    empty = tmp_path / "empty-pkg"
    empty.mkdir()
    monkeypatch.setattr("importlib.resources.files", lambda pkg: empty)
    called = []
    monkeypatch.setattr(
        "xskill.ecosystems.detect_known_ecosystems",
        lambda home_root=None: called.append(True) or [],
    )

    installed = install_bundled_xskill_guide()

    assert installed == []
    assert install_recorder == []
    assert called == []
    assert "捆绑的 xskill skill 缺失" in capsys.readouterr().err


def test_unknown_or_failing_ecosystem_does_not_abort(
        bundled_skill, install_recorder, monkeypatch, capsys):
    def _boom(skill_path, target_root=None, side="main"):
        raise RuntimeError("disk full")

    monkeypatch.setattr(
        "xskill.ecosystems.install_to_cursor", _boom,
    )
    monkeypatch.setattr(
        "xskill.ecosystems.detect_known_ecosystems",
        lambda home_root=None: [
            {"ecosystem": "unknown_agent", "source": "/u", "bridge": "/v"},
            {"ecosystem": "cursor", "source": "/c", "bridge": "/d"},
            {"ecosystem": "codex", "source": "/x", "bridge": "/y"},
        ],
    )

    installed = install_bundled_xskill_guide()

    assert installed == ["codex"]
    assert [row[0] for row in install_recorder] == ["codex"]
    err = capsys.readouterr().err
    assert "cursor" in err
    assert "disk full" in err


def test_bundled_skill_documents_generate():
    from xskill.ecosystems.bundled_guide import bundled_xskill_source

    skill_md = (bundled_xskill_source() / "SKILL.md").read_text(encoding="utf-8")
    assert "xskill generate" in skill_md
    assert "xskill import" in skill_md
    assert "xskill connect" in skill_md
    assert "name: xskill" in skill_md


def test_no_detected_ecosystems_prints_skip(
        bundled_skill, install_recorder, monkeypatch, capsys):
    monkeypatch.setattr(
        "xskill.ecosystems.detect_known_ecosystems",
        lambda home_root=None: [],
    )

    installed = install_bundled_xskill_guide()

    assert installed == []
    assert install_recorder == []
    assert "未检测到已知 agent 生态" in capsys.readouterr().out
