"""把随包发布的 /xskill 使用指南装进用户本机已探测到的 agent skill 目录。

``xskill init`` 与 ``xskill connect`` 共用这一份实现：探测到 Claude Code /
Codex / Cursor 等生态后，把 ``xskill/data/skill/xskill`` 装到对应 skill
根目录，agent 里就可以 /xskill 查 generate、search 等用法。

安装失败只打 warning，不抛给调用方——连不上 skill 目录不该挡住 connect。
"""
from __future__ import annotations

import sys
from pathlib import Path

from xskill import ecosystems as _ecosystems

# 调用时再按生态 id 取 ecosystems 上的 install_to_*，方便测试打桩。
_INSTALLER_ATTR_BY_ECO = {
    "claude_code": "install_to_claude_code",
    "codex": "install_to_codex",
    "nga3": "install_to_nga3",
    "opencode": "install_to_opencode",
    "ngagent": "install_to_ngagent",
    "openclaw": "install_to_openclaw",
    "cursor": "install_to_cursor",
    "trae": "install_to_trae",
    "deepseek_harness": "install_to_deepseek_harness",
}


def _installer_for(eco: str):
    attr = _INSTALLER_ATTR_BY_ECO.get(eco)
    if attr is None:
        return None
    return getattr(_ecosystems, attr, None)


def bundled_xskill_source() -> Path:
    """wheel / 可编辑安装里随包发布的 xskill 指南目录。"""
    from importlib.resources import files

    return Path(str(files("xskill") / "data" / "skill" / "xskill"))


def install_bundled_xskill_guide(
    target_root: Path | str | None = None,
) -> list[str]:
    """把 /xskill 指南装进 ``target_root`` 下已探测到的生态。

    返回成功装上的生态 id 列表。捆绑目录缺失或某个生态安装失败时打印
    warning，不抛异常。
    """
    root = Path(target_root).expanduser().resolve() if target_root else None
    skill_source = bundled_xskill_source()
    if not (skill_source / "SKILL.md").is_file():
        print(
            f"warning: 捆绑的 xskill skill 缺失（{skill_source}），跳过装 skill",
            file=sys.stderr,
        )
        return []

    installed_ecosystems: list[str] = []
    for detection in _ecosystems.detect_known_ecosystems(home_root=root):
        install_fn = _installer_for(detection["ecosystem"])
        if install_fn is None:
            continue
        try:
            install_fn(skill_source, target_root=root, side="main")
            installed_ecosystems.append(detection["ecosystem"])
        except Exception as install_error:  # noqa: BLE001
            print(
                f"warning: 装到 {detection['ecosystem']} 失败：{install_error}",
                file=sys.stderr,
            )
    if installed_ecosystems:
        print(
            f"已把 xskill 使用指南装进 {'/'.join(installed_ecosystems)} 的 "
            f"skill 目录，在对应 agent 里可直接 /xskill 查用法。"
        )
    else:
        print(
            "未检测到已知 agent 生态（claude_code/codex/opencode/cursor/… "
            "均未发现），跳过装 skill。"
        )
    return installed_ecosystems
