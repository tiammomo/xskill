---
name: xskill
description: >-
  How to run, connect, generate, upgrade, debug, and search with the xskill
  CLI — the team skill-distribution and trajectory-collection daemon. Use
  when a user asks how to install or join an xskill team server, run
  `xskill generate` to write or rewrite a skill from team trajectories,
  search trajectories with `xskill search traj`, upgrade xskill, fix a
  stuck/black-window/copy-mode install, run xskill in the background on
  Windows/macOS/Linux/WSL, search or share team skills, import an existing
  skill, or read the dashboard.
---

# xskill

xskill is a thin client + background daemon that (1) mounts your team's shared
Skills into every AI-agent tool you use (Claude Code, Codex, OpenCode, Cursor,
Trae, DeepSeek Harness) and (2) quietly collects your agent trajectories and
syncs them to a team server. You join a server once, then it keeps skills in
sync and auto-updates itself.

`xskill connect` (and `xskill init`) installs this guide into every detected
agent skill directory, so you can invoke `/xskill` from Claude Code, Codex,
Cursor, and the other supported tools.

All state lives under `~/.xskill/`:

| File | What |
|---|---|
| `~/.xskill/team_client.json` | connection identity (server_url, client_id, join_token) — survives restarts |
| `~/.xskill/connect_daemon.json` | current background daemon (pid / host task) — used by `status`/`stop` |
| `~/.xskill/logs/xskill.*.log` | split logs (one file per component) |
| `~/.xskill/skill/` | the local skill repo everything is mounted from |

## Getting started

`xskill connect` is the everyday join command. After a successful handshake it
installs this guide, then starts the background daemon:

```bash
xskill connect <host:port> --token <join-token> --name <employee-id>
```

The join token is printed by the server operator when they run
`xskill serve --server`. Reconnects can omit the address and token; they reuse
`~/.xskill/team_client.json`. Pass `--no-skill` only when you want the
connection without refreshing this guide.

`xskill init` is the interactive wrapper: it installs this guide first, then
asks for server / token / name and calls `connect` (without installing twice).

```bash
xskill init                 # interactive: prompts for server / token / name
xskill init <host:port> --token <join-token> --name <employee-id> --yes
```

## Generate or rewrite a Skill

`generate` does not invent a Skill from scratch. It reads trajectories the team
server already allows this client to see, and the instruction only describes
what to create or change. The job may wait for a free SkillEdit seat; the CLI
streams queue and run logs. When it finishes, the Skill is committed to main
and pinned on the initiator's recommendation list.

```bash
xskill generate "创建一个排查 Python 内存泄漏的 Skill，包含常用诊断命令"
xskill generate "改写现有的 python-memory-debug Skill，补充 Windows 排查步骤"
xskill generate --name alice,bob "根据这些用户的成功案例生成数据库迁移 Skill"
```

`--name` is a comma-separated list of employee ids the agent should read first.
Omit it and the agent may search every trajectory the server authorizes. If the
CLI says the server is too old, ask the operator to upgrade the team server.

## Import an existing Skill

`import` takes a local Skill folder (or a parent that contains several) into
the team's own repo. This is not `upload`: import becomes a first-party Skill
on main, while upload only lands in the user's SkillHub share.

```bash
xskill import ./my-skill
xskill import ./skills-parent --json
```

## Searching trajectories

`xskill search traj` finds trajectories related to a natural-language query.
The first shipping form reads a bundled mock catalog so agents can practice
the command without a live index. Each hit is marked `source=mock`.

```bash
xskill search traj memory leak
xskill search traj "alembic migration" -k 3
xskill search traj auth retry --json
```

Human output is `score<TAB>status<TAB>skill<TAB>ecosystem<TAB>traj_id` plus
the title. Use a hit's `traj_id` and summary when you later call
`xskill generate` and want the instruction to name the evidence.

When the team-server trajectory search lands, this same command will keep
its argv; only the catalog behind it changes.

## Searching & sharing team skills

```bash
xskill search <query...>       # search team skills; returns metadata only
xskill search traj <query...>  # search trajectories (bundled mock catalog)
xskill search <query...> --download  # legacy 10-slot LRU download + auto-install
xskill download <skill-id>     # persist one result; interactively select harnesses
xskill download <skill-id> --agent claude-code --agent codex -y  # for agents/scripts
xskill search auth retry -k 3  # top-3 results (max 10)
xskill upload ./my-skill       # package a SKILL.md folder and share to the team
xskill dashboard               # print a passwordless link into the server dashboard
xskill stats                   # token usage & estimated cost
```

Already connected clients search SkillHub. Standalone machines search the local
library (semantic index, falling back to BM25). `search --download` is the old
rolling-slot path; prefer `download <skill-id>` to keep a Skill permanently.

## Upgrading

```bash
pip install -U xskill        # manual upgrade to the latest PyPI release
xskill update                # check PyPI now, upgrade + restart if newer exists
```

The daemon auto-checks PyPI every hour and upgrades itself when a newer
version is out; disable with `xskill connect --no-auto-update`. Behind a
corporate proxy, add `--use-proxy` (default is direct connection, bypassing the
SWG proxy). Background hosting differs per platform — Windows uses a Scheduled
Task, Linux/WSL uses a systemd user service. See
[references/platforms.md](references/platforms.md).

## Debugging

```bash
xskill connect --foreground          # run the daemon loop in the foreground, live logs
xskill connect --foreground --debug  # + verbose logging
```

| Symptom | Command | Expected |
|---|---|---|
| Is it running / connected? | `xskill status` | prints background task + pid, or "not running" |
| Skills not updating | `xskill connect --foreground` | watch the reconcile loop; look for `copy-mode` / `fell back to copy` warnings |
| Stop it | `xskill stop` | tears down the Scheduled Task / systemd unit |
| Restart clean | `xskill stop` then `xskill connect` | re-handshakes and re-daemonizes |

Logs are at `~/.xskill/logs/xskill.*.log`. See
[references/troubleshooting.md](references/troubleshooting.md) for the
black-window, copy-mode, and can't-connect issues in detail.

## Where skills land per tool

| Tool | Install dir | Mount |
|---|---|---|
| Claude Code | `~/.claude/skills/<name>/` | symlink → junction (Win) → copy |
| Codex | `~/.agents/skills/<name>/` (shared) | same |
| OpenCode | `~/.agents/skills/<name>/` (shared) | same |
| Cursor | `~/.cursor/skills/<name>/` | same |
| Trae | IDE workspace / `~/.trae-cn` (auto-detected) | same |
| DeepSeek Harness | `~/.dsh/skills/<name>/` | same |

symlink/junction installs are live — server updates show up instantly and your
edits round-trip back. A copy-mode install is a snapshot: it logs a warning,
updates do not propagate live, and local edits do not round-trip. Copy mode
only happens when neither symlink nor junction can be made (cross-drive, or
non-NTFS). On Windows, enable Developer Mode so symlinks work — see
[references/platforms.md](references/platforms.md).
