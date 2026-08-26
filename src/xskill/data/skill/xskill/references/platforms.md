# xskill per-platform notes

## Background hosting

`xskill connect` (and `xskill init`) daemonize by handing the run loop to the
OS's native service facility, then return. The foreground loop they run is
`xskill connect --foreground`.

| Platform | Background mechanism | Notes |
|---|---|---|
| Windows | Scheduled Task | starts at logon; console windows are suppressed |
| Linux / WSL | systemd **user** service | needs `loginctl enable-linger` for run-at-boot without an active login (xskill sets this up) |
| macOS | foreground fallback | if no native backend is available, `connect` runs the loop in the foreground |

Manage the background task:

```bash
xskill start     # install as a background service (autostart + crash-restart)
xskill status    # show task + pid
xskill stop      # stop and remove the background task
```

## Skill mount behavior

xskill mounts each skill with a three-tier fallback:

1. **symlink** — Linux, macOS, and Windows with Developer Mode. Live: server
   updates appear instantly, and your edits round-trip back to the source repo.
2. **directory junction** (`mklink /J`) — Windows without Developer Mode. NTFS
   reparse point; behaves like a symlink to readers, but only within one volume.
3. **copy** — only when neither of the above works (cross-drive, or non-NTFS). A
   snapshot: updates don't propagate live and edits don't round-trip. Logs a
   warning.

### Windows: prefer Developer Mode

Enabling Developer Mode grants the symlink privilege, so installs use tier 1
(fully live) instead of junctions or copies. It also avoids the junction code
path entirely.

Settings → Privacy & security → For developers → **Developer Mode: On**.

## Install directories per tool

| Tool | Discovery dir |
|---|---|
| Claude Code | `~/.claude/skills/` |
| Codex | `~/.agents/skills/` (shared user scope) |
| OpenCode | `~/.agents/skills/` (shared user scope) |
| Cursor | `~/.cursor/skills/` |
| Trae | IDE workspace storage, `~/.trae-cn`, or CLI trajectory dir (auto-detected) |
| DeepSeek Harness | `~/.dsh/skills/` |

Codex, OpenCode, and OpenClaw share `~/.agents/skills/`, so installing a skill
once makes it visible to all three.
