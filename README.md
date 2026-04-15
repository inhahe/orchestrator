# Orchestrator

A customized interactive shell around the [Claude Agent SDK](https://pypi.org/project/claude-agent-sdk/). Wraps `claude` (Claude Code) with a richer terminal UI, autonomous-continue loop, live task/bg panels, context-size controls, API-stall detection, and session management that drops straight onto Claude Code's on-disk session layout.

For the full flag + slash-command reference, see [REFERENCE.md](REFERENCE.md).

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.11+ and the `claude` CLI (installed and logged-in) on your PATH — the SDK launches it under the hood.

## Run

```bash
python orchestrator.py
```

On Windows, double-click `run.cmd` to launch inside Windows Terminal.

Start with a specific message:

```bash
python orchestrator.py -p "review the diff in main.py"
```

Pick a previous session interactively:

```bash
python orchestrator.py --resume
```

## What you get

- **Live bottom toolbar**: session id, plan, model, context (capped at the real window), turns, effort/thinking mode, active-tool counts, todo progress, per-task live rows, per-bg-task live rows.
- **`/show N [-K]`** to expand any tool call (Bash gets a `\`/show N [-K]\`` hint printed alongside each invocation).
- **`/think N`** to view any extended-thinking block.
- **`/bg`, `/bg N`, `/bg N K`** for background-task summary / detail / tail.
- **`/tasks`** for every non-Bash tool run this turn.
- **`/autocompact`**, **`/max-context`**, **`/clear`** (full context wipe), **`/cls`** (screen clear), **`/rename`**, **`/export`**, **`/effort`**, **`/model`**, and more — see REFERENCE.md.
- **Auto-continue loop** (`--auto-continue`) that drives Claude autonomously with a `[WAITING]`-aware back-off and burst-limit brake.
- **API stall detection** with Statuspage.io polling (`https://status.claude.com/api/v2/summary.json`) and auto-resume once the service reports healthy.
- **Rolling-window context trim** (`--max-context-tokens N`) as an alternative to summary-based compaction — forks the session JSONL to keep only the tail.
- **Picker-based session resume** with per-project two-step selection, auto-cwd switching, and live session-title editing synced with Claude Code's own `custom-title`/`ai-title` JSONL records.

## File layout

| Path               | Purpose                                             |
|--------------------|-----------------------------------------------------|
| `orchestrator.py`  | Single-file entry point. All logic lives here.      |
| `REFERENCE.md`     | Full CLI-flag / slash-command / toolbar reference.  |
| `requirements.txt` | Python dependencies.                                |
| `run.cmd`          | Windows helper: opens in Windows Terminal.          |

Session transcripts, settings, and custom-titles all live in Claude Code's existing directory — `~/.claude/projects/<sanitized-cwd>/<session-id>.jsonl` — so anything you do in this tool is visible to plain `claude --continue` and vice versa.
