"""Long-running Claude orchestrator with a Claude-Code-like terminal UI.

Parity with Claude Code:
  * `permission_mode=bypassPermissions` by default -> Claude runs Read, Write,
    Edit, Glob, Grep, Bash (incl. `run_in_background=true`), BashOutput,
    KillShell, NotebookEdit, WebFetch, WebSearch, Task, Skill, TodoWrite
    with no prompting.
  * Background-shell completion notifications the CLI injects between turns
    are surfaced inline as [notice]/unknown-message blocks -> Claude sees them
    in his next turn and you see them scroll by too.
  * `setting_sources=["user","project","local"]` pulls in your skills scope
    the way the CLI does.
  * `.mcp.json` in cwd is auto-loaded (or pass `--mcp-config PATH`).
  * Resumes the last session in cwd by default (`--no-continue` to start fresh).

Install:
    pip install claude-agent-sdk prompt_toolkit

Run:
    python orchestrator.py                         # resume last session, interactive
    python orchestrator.py --no-continue           # fresh session, interactive
    python orchestrator.py --auto-continue         # autonomous (orchestrator drives Claude)
    python orchestrator.py --resume                # interactive session picker
    python orchestrator.py --resume <session-id>   # resume a specific session
    python orchestrator.py -p "plan and ship X"    # with an initial prompt
    python orchestrator.py --effort max --model claude-opus-4-6
    python orchestrator.py --mcp-config ./my-mcp.json
    python orchestrator.py --disallowed-tool WebFetch --disallowed-tool WebSearch

Slash commands (tab-complete at the prompt):
    /help                          list commands
    /status                        session id / context / cost / usage
    /interrupt  /i                 stop the current turn
    /compact                       force a /compact right now
    /effort <auto|low|medium|high|max>  change effort (reconnects, keeps session; auto = no override)
    /model  <name>                 change model (reconnects, keeps session)
    /rename <name>                 set a custom title for this session (shown in --resume picker)
    /auto   [on|off|toggle]        enable/disable autonomous continue prompting
    /burst  N [T]                  set continue-burst limit (and window seconds); no arg = show
    /export [path]                 save the current conversation as markdown
    /tools                         list every active tool call and background task
    /show [N ...]                  expand a collapsed tool call (input + output) by its [#N] tag
    /todos  /plan                  show Claude's current TodoWrite plan
    /quit   /exit                  graceful exit (waits up to ~10s for the CLI to flush)
    /quit!  /exit!                 force-kill immediately (may lose last in-flight message)

Keys:
    Enter         submit
    Ctrl-C        while Claude is working -> interrupt that turn
                  at an empty prompt       -> exit
                  with text in the buffer  -> clear the buffer
    Ctrl-D        exit
    Up / Down     history
    Tab           complete slash commands
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

try:  # Extended-thinking blocks — present on recent SDKs.
    from claude_agent_sdk import ThinkingBlock  # type: ignore
except ImportError:  # pragma: no cover — older SDK
    ThinkingBlock = None  # type: ignore[assignment]

try:  # Native session rename — present on recent SDKs.
    from claude_agent_sdk import rename_session as _sdk_rename_session  # type: ignore
except ImportError:  # pragma: no cover — older SDK
    _sdk_rename_session = None  # type: ignore[assignment]
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style

EFFORT_LEVELS = ("low", "medium", "high", "max")
# "auto" is not a real API value; it means "don't pass effort, let the model
# pick its default" (which is typically 'high' for Opus/Sonnet 4.6).
EFFORT_CHOICES = ("auto",) + EFFORT_LEVELS

CONTINUE_PROMPT = (
    'If you need input from me before continuing, pause and include the literal '
    'token "[WAITING]" in your reply; otherwise, continue working.'
)
WAITING_SENTINEL = "[WAITING]"

DEFAULT_COMPACT_THRESHOLD = 160_000
# Compact threshold for 1M-context models — leave headroom for the reply.
DEFAULT_COMPACT_THRESHOLD_1M = 950_000


def _fmt_duration(seconds: float) -> str:
    """Compact duration: `4.2s`, `1m 23s`, `1h 4m 5s`, `18h 3m`."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h >= 1:
        return f"{h}h {m}m {s}s" if h < 10 else f"{h}h {m}m"
    return f"{m}m {s}s"


def _fmt_tok(n: int) -> str:
    """Compact token-count format: `42`, `3k`, `175k`, `1M`, `1.25M`."""
    if n >= 1_000_000:
        whole = n / 1_000_000
        return f"{whole:.0f}M" if whole >= 10 or whole == int(whole) else f"{whole:.2f}M"
    if n >= 1_000:
        return f"{n // 1000}k"
    return str(n)


def _model_context_window(model: str | None) -> int:
    """Best-effort context-window size (tokens) for the given model id.

    1M-context variants (id contains `[1m]` / `-1m` / ends in `1m`) → 1M.
    Everything else → 200k (the default window for Claude 3+/4+ models).
    When model is None we don't know what the CLI will pick; use 200k as
    a safe lower bound so the displayed ctx~ doesn't silently overrun.
    """
    if model and ("[1m]" in model or "-1m" in model or model.endswith("1m")):
        return 1_000_000
    return 200_000


def _default_compact_at(model: str | None) -> int:
    """Pick a sensible auto-compact trigger based on the selected model.

    1M-context variants (model id contains `[1m]` or `1m`) get a much larger
    threshold so you actually use the extra window. Everything else stays at
    the conservative default."""
    if model and ("[1m]" in model or "-1m" in model or model.endswith("1m")):
        return DEFAULT_COMPACT_THRESHOLD_1M
    return DEFAULT_COMPACT_THRESHOLD
CONTINUE_RESPONSE_DELAY_SECONDS = 2.0
CONTINUE_BURST_LIMIT = 3
CONTINUE_BURST_WINDOW_SECONDS = 180.0

SLASH_COMMANDS = [
    "/help",
    "/status",
    "/cost",
    "/cwd",
    "/clear",
    "/cls",
    "/interrupt",
    "/i",
    "/compact",
    "/effort",
    "/model",
    "/rename",
    "/auto",
    "/burst",
    "/export",
    "/tools",
    "/tasks",
    "/bg",
    "/background",
    "/show",
    "/think",
    "/autocompact",
    "/max-context",
    "/todos",
    "/plan",
    "/quit",
    "/exit",
    "/quit!",
    "/exit!",
]

STYLE = Style.from_dict(
    {
        "prompt": "ansibrightcyan bold",
        "bottom-toolbar": "bg:#222222 #aaaaaa",
        "bottom-toolbar.busy": "bg:#884400 #ffffff bold",
        "claude": "ansigreen",
        "tool": "ansiblue",
        "tool-err": "ansired",
        "dim": "ansibrightblack",
        "warn": "ansiyellow",
        "sys": "ansimagenta",
        "err": "ansired bold",
    }
)


@dataclass
class State:
    session_id: str | None = None
    session_title: str | None = None  # cached title from JSONL custom-title/ai-title
    context_tokens: int = 0
    total_cost_usd: float = 0.0
    turns: int = 0
    last_usage: dict[str, Any] = field(default_factory=dict)
    effort: str | None = None
    model: str | None = None
    busy: bool = False
    last_result_subtype: str | None = None
    last_compact_trigger: str | None = None
    # Set by the compact_boundary handler; consumed (and cleared) by
    # worker_loop after each turn so it can short-circuit the decision tree.
    compact_during_last_turn: bool = False
    # Turn number on which the most recent compact_boundary fired. Used as
    # a cooldown: the auto-compact check skips itself until at least
    # compact_cooldown_turns have elapsed. Stops us looping on /compact
    # when the inflated cumulative usage stays high post-compact.
    last_compact_turn: int | None = None
    # When the orchestrator is parked because something needs you:
    #   "waiting"  -> Claude emitted [WAITING] (red WAITING in toolbar)
    #   "burst"    -> burst-limit brake fired, Claude was spinning
    #                 (magenta STALLED in toolbar)
    #   None       -> nothing demanding attention (idle/dim gray)
    # Cleared at run_turn start.
    needs_user_attention: str | None = None
    recent_turn_ends: deque[float] = field(default_factory=deque)
    # Foreground tools currently in flight: tool_use_id -> {name, input, started_at, seq}
    active_tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Background Bash shells / Task subagents: task_id -> {name, started_at, task_type}
    background_tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Capped history of tool calls + results, used by /show <N>.
    tool_history: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=200)
    )
    next_tool_seq: int = 1
    # Tool seqs that have appeared in the live panel during this turn (or the
    # most recent one, once it ends). Reset at the START of each turn. Excludes
    # Bash since Bash scrolls inline. Drives /tasks.
    current_turn_tool_seqs: list[int] = field(default_factory=list)
    # Every background task started this turn, whether still running or
    # completed. Drives /bg. Reset at the START of each turn. Dict by
    # task_id so task_notification can update the matching entry in place.
    current_turn_bg: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Monotonic counter for bg task seq numbers. Never resets — matches the
    # way tool seqs work, so /bg N stays stable across turns.
    next_bg_seq: int = 1
    # Sliding-window timestamps of recent api_retry events. Used by the
    # API-stall detector to flip `needs_user_attention` to "api-error"
    # when failures get dense.
    api_retry_times: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    # Last Statuspage.io status indicator we read, if any. One of:
    # "none" / "minor" / "major" / "critical" / None. Shown in the toolbar
    # when != "none".
    api_status_indicator: str | None = None
    api_status_description: str | None = None
    # How we entered the current API stall. "status" = status feed flagged
    # it, so a return to "operational" is a valid resume signal. "heuristic"
    # = we tripped the retry-density threshold while status was clean, so
    # we require status to *become* bad and then clear before auto-resume.
    api_stall_source: str | None = None
    # Latched True once the poller observes a non-operational status since
    # the stall began. Gates heuristic-stall recovery.
    api_stall_saw_bad: bool = False
    # Capped history of thinking blocks, used by /think <N>.
    thinking_history: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=200)
    )
    next_thinking_seq: int = 1
    # Latest TodoWrite snapshot (Claude's plan). Each item: {content, status, activeForm}
    current_todos: list[dict[str, Any]] = field(default_factory=list)
    # Subscription detection + rate-limit readout. Flipped on permanently the
    # first time a rate_limit_event arrives with a subscription-bucket type.
    # Until then the toolbar shows equivalent API cost as a usage gauge.
    is_subscription: bool = False
    subscription_plan: str | None = None  # "pro", "max", etc. (from credentials.json)
    # The model actually in use per the CLI, extracted from AssistantMessage
    # metadata. Distinct from `model` (the user-pinned override). Fills in
    # after the first assistant reply; until then the window-sizing / label
    # logic falls back to `(auto)` + a 200k safe default.
    active_model: str | None = None
    # Mirror of --inline-all-tools so the rendering helpers (panel + inline)
    # don't have to reach into argparse state.
    inline_all_tools: bool = False
    # Mirror of --show-edits ("off" | "compact" | "full"). Non-"off" means
    # Edit renders inline and should be omitted from the live-tasks panel.
    show_edits: str = "off"
    # Mirror of --show-thinking so the toolbar can surface it without
    # reaching into argparse state.
    show_thinking: bool = False
    # Toolbar panel visibility (mirrors --tasks-panel / --bg-panel flags).
    show_tasks_panel: bool = True
    show_bg_panel: bool = True
    rate_limit_util: float | None = None
    rate_limit_label: str | None = None


_SUBSCRIPTION_RL_TYPES = frozenset(
    {"five_hour", "seven_day", "seven_day_opus", "seven_day_sonnet"}
)


def _detect_subscription() -> bool:
    """Best-effort: on subscription unless an API-mode env var is set.

    Claude Code uses ANTHROPIC_API_KEY for direct API access and
    CLAUDE_CODE_USE_BEDROCK / CLAUDE_CODE_USE_VERTEX for enterprise cloud
    routing — all three bypass the subscription. Anything else means we're
    signed in via `claude login` (subscription)."""
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return False
    for v in ("CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX"):
        val = os.environ.get(v, "").strip().lower()
        if val and val not in ("0", "false", "no"):
            return False
    return True


def _detect_subscription_plan() -> str | None:
    """Read the plan name ("pro", "max", ...) from Claude Code's OAuth
    credentials file. Returns None if the file isn't present (e.g. creds
    live in a system keychain) or can't be parsed."""
    base = os.environ.get("CLAUDE_CONFIG_DIR")
    path = Path(base if base else Path.home() / ".claude") / ".credentials.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    oauth = data.get("claudeAiOauth") if isinstance(data, dict) else None
    if not isinstance(oauth, dict):
        return None
    plan = oauth.get("subscriptionType")
    return plan if isinstance(plan, str) and plan else None

_RL_TYPE_LABEL = {
    "five_hour": "5h",
    "seven_day": "7d",
    "seven_day_opus": "7d opus",
    "seven_day_sonnet": "7d sonnet",
}


def _apply_rate_limit_info(state: "State", info: Any) -> None:
    """Update subscription flag + live utilization from a rate_limit_info blob.

    Accepts either a dict (SystemMessage path) or an SDK object (RateLimitEvent
    path). Subscription detection is sticky: once true, stays true."""
    if info is None:
        return
    if isinstance(info, dict):
        rl_type = info.get("rate_limit_type")
        util = info.get("utilization")
    else:
        rl_type = getattr(info, "rate_limit_type", None)
        util = getattr(info, "utilization", None)
    if rl_type in _SUBSCRIPTION_RL_TYPES:
        state.is_subscription = True
    if isinstance(util, (int, float)):
        state.rate_limit_util = float(util)
    if isinstance(rl_type, str):
        state.rate_limit_label = _RL_TYPE_LABEL.get(rl_type, rl_type)


def classify(line: str) -> tuple[str, str]:
    s = line.strip()
    if not s:
        return "empty", ""
    if not s.startswith("/"):
        return "message", s
    parts = s[1:].split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    if cmd in ("i", "interrupt"):
        return "interrupt", ""
    if cmd in ("q", "quit", "exit"):
        return "quit", ""
    if cmd in ("quit!", "exit!", "q!"):
        return "force-quit", ""
    if cmd == "compact":
        return "compact", ""
    if cmd == "status":
        return "status", ""
    if cmd == "help":
        return "help", ""
    if cmd == "effort":
        val = arg.lower()
        if val in EFFORT_CHOICES:
            return "effort", val
        return "error", f"effort must be one of {', '.join(EFFORT_CHOICES)}"
    if cmd == "model":
        if arg:
            return "model", arg
        return "error", "usage: /model <name>"
    if cmd == "rename":
        return "rename", arg  # arg may be empty (means "show current title")
    if cmd in ("auto", "auto-continue"):
        val = arg.lower()
        if val in ("on", "true", "1", "yes", "enable"):
            return "auto", "on"
        if val in ("off", "false", "0", "no", "disable"):
            return "auto", "off"
        if val in ("", "toggle"):
            return "auto", ""  # toggle (or show if no current state info)
        return "error", "usage: /auto [on|off|toggle]"
    if cmd == "burst":
        return "burst", arg  # "" = show; "N" = set count; "N T" = set both
    if cmd == "export":
        return "export", arg  # arg = path; empty = default filename
    if cmd == "tools":
        return "tools", ""
    if cmd in ("tasks", "task"):
        return "tasks", ""
    if cmd in ("bg", "background", "bgtasks"):
        return "bg", arg  # "" = summary list; "N" = detail; "N K" = detail + tail K lines
    if cmd in ("todos", "todo", "plan"):
        return "todos", ""
    if cmd == "show":
        return "show", arg  # "" = last few; "N [N2 N3 ...]" = those entries
    if cmd in ("think", "thinking", "thought"):
        return "think", arg  # "" = last few; "N [N2 ...]" = those thinking blocks
    if cmd in ("autocompact", "auto-compact"):
        return "autocompact", arg  # "" = show; "on"/"off" = toggle; "N" = set threshold
    if cmd in ("max-context", "maxcontext", "max-ctx"):
        return "max-context", arg  # "" = show; "off" = unlimited; "N" = set cap
    if cmd == "clear":
        return "clear-context", ""
    if cmd == "cls":
        return "clear-screen", ""
    if cmd in ("cost", "cwd"):
        return "status", ""
    # Unknown slash — send raw (the CLI may have skills that handle it).
    return "message", s


def brief_args(d: dict[str, Any], limit: int = 110) -> str:
    s = ", ".join(f"{k}={v!r}" for k, v in d.items())
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _print_unified_diff(old: str, new: str, *, indent: str = "    ", max_lines: int = 80) -> None:
    import difflib

    lines = list(
        difflib.unified_diff(
            old.splitlines(), new.splitlines(), lineterm="", n=2
        )
    )
    # Drop the first two header lines (--- / +++) since we print the file path separately.
    body = [ln for ln in lines if not ln.startswith("---") and not ln.startswith("+++")]
    if not body:
        print(f"{indent}\033[90m(no textual change)\033[0m")
        return
    for ln in body[:max_lines]:
        if ln.startswith("+"):
            print(f"{indent}\033[32m{ln}\033[0m")
        elif ln.startswith("-"):
            print(f"{indent}\033[31m{ln}\033[0m")
        elif ln.startswith("@@"):
            print(f"{indent}\033[36m{ln}\033[0m")
        else:
            print(f"{indent}{ln}")
    if len(body) > max_lines:
        print(f"{indent}\033[90m... [+{len(body) - max_lines} more diff lines]\033[0m")


def _msg_fields(msg: Any) -> dict[str, Any]:
    """Get a SystemMessage's payload. Python SDK usually uses `.data`; fall back to object attrs."""
    data = getattr(msg, "data", None)
    if isinstance(data, dict) and data:
        return data
    if hasattr(msg, "__dict__"):
        return {k: v for k, v in vars(msg).items() if not k.startswith("_")}
    return {}


def render_unknown_message(msg: Any, state: "State | None" = None) -> None:
    """Shape-detect top-level messages that aren't dedicated Python SDK classes."""
    cls_name = type(msg).__name__
    # RateLimitEvent: top-level Message type (NOT a SystemMessage subclass).
    if cls_name == "RateLimitEvent":
        info = getattr(msg, "rate_limit_info", None)
        status = getattr(info, "status", "?") if info is not None else "?"
        util = getattr(info, "utilization", None) if info is not None else None
        rl_type = (
            getattr(info, "rate_limit_type", None) if info is not None else None
        )
        if state is not None:
            _apply_rate_limit_info(state, info)
        color = (
            "\033[31m"
            if status == "rejected"
            else "\033[33m"
            if status == "allowed_warning"
            else "\033[90m"
        )
        bits = [f"status={status}"]
        if isinstance(util, (int, float)):
            bits.append(f"{util * 100:.0f}% used")
        if rl_type:
            bits.append(str(rl_type))
        print(f"{color}[rate-limit: {' · '.join(bits)}]\033[0m")
        return
    # tool_progress: { type: 'tool_progress', tool_name, elapsed_time_seconds, ... }
    tool_name = getattr(msg, "tool_name", None)
    elapsed = getattr(msg, "elapsed_time_seconds", None)
    if tool_name and isinstance(elapsed, (int, float)):
        print(f"\033[33m  [... {tool_name} running for {elapsed:.1f}s]\033[0m")
        return
    # partial assistant message (streaming chunk) — suppress to avoid duplicate text
    raw_type = getattr(msg, "type", None)
    if raw_type in ("stream_event", "partial_assistant"):
        return
    # Generic: show class name + the first useful fields we can find.
    hint = (
        getattr(msg, "summary", None)
        or getattr(msg, "message", None)
        or getattr(msg, "text", None)
        or getattr(msg, "content", None)
    )
    if isinstance(hint, str) and hint.strip():
        print(f"\033[35m[{cls_name}] {hint.strip()}\033[0m")
        return
    attrs = (
        {k: v for k, v in vars(msg).items() if not k.startswith("_")}
        if hasattr(msg, "__dict__")
        else {}
    )
    print(f"\033[35m[{cls_name}] {brief_args(attrs)}\033[0m")


_BG_STATUS_COLORS = {
    "completed": "\033[32m",
    "failed": "\033[31m",
    "stopped": "\033[33m",
    "cancelled": "\033[33m",
}
_BG_STATUS_MARKERS = {
    "completed": "✓",
    "failed": "✗",
    "stopped": "⏹",
    "cancelled": "⏹",
}


def _emit_bg_completion(
    state: "State",
    task_id_full: str,
    status: str,
    *,
    summary: str | None = None,
    out_file: str | None = None,
    usage: dict[str, Any] | None = None,
) -> bool:
    """Single rendering path for background-task completion, shared by
    `task_notification` and `task_updated`. Dedupes: the first handler to
    fire for a given task wins, the second becomes a no-op. Returns True
    iff a line was printed."""
    task_id = task_id_full[:8] if task_id_full else "?"
    turn_entry = state.current_turn_bg.get(task_id_full)
    bg_entry = state.background_tasks.get(task_id_full)
    # Dedupe: if turn_entry already has ended_at set, a prior handler (the
    # other of task_notification/task_updated for the same task) already
    # rendered this completion. Skip the duplicate.
    if turn_entry is not None and turn_entry.get("ended_at") is not None:
        return False
    # No tracker entry at all means the task started before we did or came
    # through a channel we didn't record — still skip; no seq to show.
    if turn_entry is None and bg_entry is None:
        return False
    seq = None
    name = None
    if turn_entry is not None:
        seq = turn_entry.get("seq")
        name = turn_entry.get("name")
    elif bg_entry is not None:
        seq = bg_entry.get("seq")
        name = bg_entry.get("name")
    state.background_tasks.pop(task_id_full, None)
    if turn_entry is not None:
        turn_entry["ended_at"] = time.monotonic()
        turn_entry["status"] = status
        if summary is not None:
            turn_entry["summary"] = summary
        if out_file is not None:
            turn_entry["output_file"] = out_file
        if usage is not None:
            turn_entry["usage"] = usage
    color = _BG_STATUS_COLORS.get(status, "\033[35m")
    marker = _BG_STATUS_MARKERS.get(status, "•")
    seq_tag = f"[#{seq}] " if isinstance(seq, int) else ""
    label = (name or summary or "(unnamed)").replace("\n", " ")
    # If this bg task originated from a Bash tool call, append the
    # command (inline if it fits, size hint otherwise).
    cmd_suffix = ""
    tu_id = turn_entry.get("tool_use_id") if turn_entry else (
        bg_entry.get("tool_use_id") if bg_entry else None
    )
    if tu_id:
        for h in state.tool_history:
            if h.get("tool_use_id") == tu_id and h.get("name") == "Bash":
                orig_cmd = (h.get("input") or {}).get("command", "") or ""
                if orig_cmd.strip():
                    lines = orig_cmd.splitlines() or [orig_cmd]
                    base = (
                        f"{color}[bg {marker} {seq_tag}{task_id} -- "
                        f"{status}]\033[0m {label}"
                    )
                    if len(lines) <= 1:
                        trial = f"{base}  \033[96m`{orig_cmd}`\033[0m"
                        if _visible_len(trial) <= _term_width():
                            cmd_suffix = f"  \033[96m`{orig_cmd}`\033[0m"
                    if not cmd_suffix:
                        cmd_suffix = (
                            f"  \033[90m({_cmd_size_hint(orig_cmd)})\033[0m"
                        )
                break
    print(
        f"{color}[bg {marker} {seq_tag}{task_id} -- {status}]\033[0m "
        f"{label}{cmd_suffix}"
    )
    if out_file:
        print(f"    \033[90moutput: {out_file}\033[0m")
    if usage:
        dur = usage.get("duration_ms")
        dur_s = f" {dur / 1000:.1f}s" if isinstance(dur, (int, float)) else ""
        print(
            f"    \033[90musage: {usage.get('total_tokens', '?')} tok, "
            f"{usage.get('tool_uses', '?')} tool uses{dur_s}\033[0m"
        )
    return True


def render_system_message(msg: SystemMessage, state: "State") -> None:
    """Dispatch on SystemMessage.subtype — mirrors the SDK message union."""
    sub = msg.subtype
    d = _msg_fields(msg)

    if sub == "init":
        new_sid = d.get("session_id")
        if not new_sid:
            return
        old_sid = state.session_id
        old_title = state.session_title
        state.session_id = new_sid
        # First check disk for this (possibly fresh) session id.
        disk_title = _read_session_title(new_sid)
        if disk_title:
            state.session_title = disk_title
            return
        # Disk has no title for the new id. If we already had a title in
        # memory (seeded at startup or set this session via /rename),
        # carry it forward — this covers the case where continue/resume
        # forks a new session id, leaving the old custom-title record
        # attached to the previous JSONL.
        if old_title and new_sid != old_sid:
            state.session_title = old_title
            # Best-effort persist: append a custom-title record to the new
            # session's JSONL once it exists. The new JSONL may not be
            # written yet at init time; fail silently if so — a future
            # /rename will re-persist.
            try:
                _write_session_title(new_sid, old_title)
            except (OSError, ValueError):
                pass
        # else: no title anywhere — leave whatever we had (often None).
        return

    if sub == "compact_boundary":
        meta = d.get("compact_metadata") or {}
        trigger = meta.get("trigger", "?")
        pre = meta.get("pre_tokens", 0)
        state.last_compact_trigger = trigger
        state.compact_during_last_turn = True
        state.last_compact_turn = state.turns
        # Resident context just shrank; reset the counter so the display
        # and the auto-compact check stop reflecting the pre-compact
        # cumulative I/O. The compact turn's own ResultMessage will be
        # suppressed from overwriting this (see ResultMessage handling).
        state.context_tokens = 0
        print(f"\033[35m[compacted -- {trigger} -- was ~{pre} tok]\033[0m")
        return

    if sub == "api_retry":
        attempt = d.get("attempt", 0)
        max_r = d.get("max_retries", 0)
        delay_ms = d.get("retry_delay_ms", 0)
        status = d.get("error_status")
        err = d.get("error")
        err_txt = ""
        if isinstance(err, dict):
            err_txt = err.get("message") or err.get("error") or ""
        # Timestamp tracking moved to the orchestrator so it can filter
        # rate-limit errors (which aren't symptoms of real service issues)
        # before counting toward the stall heuristic.
        print(
            f"\033[33m[api retry {attempt}/{max_r} in {delay_ms}ms "
            f"status={status} {err_txt}]\033[0m"
        )
        return

    if sub == "rate_limit_event":
        info = d.get("rate_limit_info") or d
        status = info.get("status") if isinstance(info, dict) else None
        util = info.get("utilization") if isinstance(info, dict) else None
        _apply_rate_limit_info(state, info)
        print(f"\033[33m[rate-limit status={status} util={util}]\033[0m")
        return

    if sub == "task_notification":
        # Background Task / Bash run_in_background completion. Same
        # rendering path as task_updated — whichever fires first for a
        # given task_id wins; the other becomes a silent no-op.
        task_id_full = d.get("task_id") or ""
        status = d.get("status", "?")
        summary = (d.get("summary") or "").strip()
        usage = d.get("usage") or {}
        out_file = d.get("output_file", "")
        _emit_bg_completion(
            state,
            task_id_full,
            status,
            summary=summary or None,
            out_file=out_file or None,
            usage=usage or None,
        )
        return

    if sub == "task_started":
        task_id_full = d.get("task_id") or ""
        task_id = task_id_full[:8] if task_id_full else "?"
        task_type = d.get("task_type", "?")
        name = d.get("name") or d.get("description") or ""
        if task_id_full:
            started = time.monotonic()
            seq = state.next_bg_seq
            state.next_bg_seq += 1
            state.background_tasks[task_id_full] = {
                "seq": seq,
                "name": name or task_type,
                "task_type": task_type,
                "started_at": started,
                "tool_use_id": d.get("tool_use_id"),
            }
            state.current_turn_bg[task_id_full] = {
                "seq": seq,
                "name": name or task_type,
                "task_type": task_type,
                "started_at": started,
                "tool_use_id": d.get("tool_use_id"),
                "ended_at": None,
                "status": None,
                "summary": None,
                "output_file": None,
            }
        # Skip the "started" print for plain background-bash launches —
        # the user already saw the Bash tool call render and the
        # task_notification on completion will be informative on its own.
        # Other task types (Task subagents, etc.) still get the print.
        if task_type != "local_bash":
            print(
                f"\033[35m[bg-task {task_id} started -- {task_type}]\033[0m {name}"
            )
        return

    if sub == "task_progress":
        return  # high-frequency; suppress unless debugging

    if sub == "hook_started":
        name = d.get("hook_name", "?")
        evt = d.get("hook_event", "?")
        print(f"\033[35m[hook {name} on {evt}]\033[0m")
        return

    if sub == "hook_response":
        name = d.get("hook_name", "?")
        outcome = d.get("outcome", "?")
        exit_code = d.get("exit_code")
        stderr = (d.get("stderr") or "").strip()
        tag = {"success": "\033[32m", "error": "\033[31m", "cancelled": "\033[33m"}.get(
            outcome, "\033[35m"
        )
        suffix = f" exit={exit_code}" if exit_code is not None else ""
        print(f"{tag}[hook {name} -- {outcome}{suffix}]\033[0m")
        if stderr and outcome != "success":
            for ln in stderr.splitlines()[:5]:
                print(f"    \033[31m{ln}\033[0m")
        return

    if sub == "hook_progress":
        return  # high-frequency

    if sub == "local_command_output":
        content = (d.get("content") or "").strip()
        if content:
            print(f"\033[36m[local cmd]\033[0m {content}")
        return

    if sub == "status":
        perm = d.get("permissionMode")
        status_payload = d.get("status") or {}
        if perm:
            print(f"\033[90m[status perm={perm}]\033[0m")
        elif isinstance(status_payload, dict) and status_payload:
            print(f"\033[90m[status] {brief_args(status_payload)}\033[0m")
        return

    if sub == "session_state_changed":
        state_val = d.get("state")
        if state_val == "requires_action":
            sys.stdout.write("\a")
            sys.stdout.flush()
            print("\033[33m[session requires action]\033[0m")
        # 'idle' and 'running' are too chatty; skip
        return

    if sub == "auth_status":
        print(f"\033[90m[auth] {brief_args(d)}\033[0m")
        return

    if sub in ("files_persisted", "files_persisted_event"):
        succeeded = d.get("succeeded") or []
        failed = d.get("failed") or []
        print(
            f"\033[90m[files persisted: {len(succeeded)} ok, {len(failed)} failed]\033[0m"
        )
        return

    if sub == "prompt_suggestion":
        return  # UI hint; irrelevant for our TUI

    if sub == "task_updated":
        # Patch-style update to a running background task. Usually carries
        # incremental state (`progress`, `end_time`, etc.). We only surface
        # terminal states; everything else is silent state-sync.
        task_id_full = d.get("task_id") or ""
        patch = d.get("patch") if isinstance(d.get("patch"), dict) else {}
        status = patch.get("status")
        if status not in ("completed", "failed", "stopped", "cancelled"):
            return
        _emit_bg_completion(state, task_id_full, status)
        return

    # Unknown subtype — surface it so nothing is silently dropped.
    print(f"\033[35m[system/{sub}] {brief_args(d)}\033[0m")


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


# Matches the CLI's habit of surfacing transport/API errors as assistant
# text after it exhausts its own retry budget, e.g.
#   API Error: 500 {"type":"error","error":{"type":"api_error",...}}
#   API Error: 529 Overloaded
# Captures the numeric status (if present) so we can feed it into the
# stall heuristic like a synthetic api_retry.
_ASSISTANT_API_ERROR_RE = re.compile(
    r"API\s*Error:?\s*(\d{3})?", re.IGNORECASE
)


def _visible_len(s: str) -> int:
    """Length of `s` after stripping ANSI color/formatting escapes."""
    return len(_ANSI_RE.sub("", s))


def _cmd_size_hint(cmd: str) -> str:
    """Compact '(N chars)' / '(N chars, M lines)' hint for a command body."""
    lines = cmd.splitlines() or [cmd]
    if len(lines) > 1:
        return f"{len(cmd)} chars, {len(lines)} lines"
    return f"{len(cmd)} chars"


def _term_width(default: int = 100) -> int:
    try:
        return shutil.get_terminal_size(fallback=(default, 24)).columns
    except (OSError, ValueError):
        return default


def _task_summary_line(name: str, inp: dict[str, Any]) -> str:
    """One-line plain-text summary of a tool call's inputs, for /tasks."""
    if name == "Bash":
        cmd = (inp.get("command") or "").splitlines()
        head = cmd[0] if cmd else ""
        if len(head) > 100:
            head = head[:97] + "..."
        return head
    if name == "Grep":
        pat = inp.get("pattern", "")
        path = inp.get("path", ".")
        filt = []
        if inp.get("glob"):
            filt.append(f"glob={inp['glob']}")
        if inp.get("type"):
            filt.append(f"type={inp['type']}")
        tail = f" ({', '.join(filt)})" if filt else ""
        return f"/{pat}/ in {path}{tail}"
    if name == "Glob":
        return f"{inp.get('pattern', '?')} in {inp.get('path', '.')}"
    if name == "Read":
        return str(inp.get("file_path", "?"))
    if name == "Edit":
        return f"{inp.get('file_path', '?')}"
    if name == "Write":
        return f"{inp.get('file_path', '?')}"
    if name == "WebFetch":
        return str(inp.get("url", "?"))
    if name == "WebSearch":
        return str(inp.get("query", "?"))
    if name == "Task":
        subtype = inp.get("subagent_type", "?")
        desc = (inp.get("description") or "").strip()
        return f"[{subtype}] {desc}"
    if name == "TodoWrite":
        todos = inp.get("todos", []) or []
        return f"({len(todos)} items)"
    if name == "NotebookEdit":
        return str(inp.get("notebook_path", "?"))
    return ""


def render_tool_use(
    block: ToolUseBlock,
    *,
    show_full_commands: bool = False,
    seq: int | None = None,
    inline_all: bool = False,
    edits_mode: str = "off",
) -> None:
    """Default behavior is *Bash-only* inline rendering; everything else
    lives in the live-tasks panel until it completes. Pass inline_all=True
    (i.e. --inline-all-tools) to restore the classic scrolling log where
    every tool call prints here.

    `edits_mode` ("off"|"compact"|"full") lets Edit render inline even when
    `inline_all` is False — useful when you want file-change activity in
    your scrollback but don't need every Read/Grep there too. `inline_all`
    forces Edit to "full"."""
    name = block.name
    effective_edits = "full" if inline_all else edits_mode
    if name == "Edit" and effective_edits != "off":
        pass  # fall through to render
    elif name != "Bash" and not inline_all:
        return
    inp = block.input or {}
    # Leading-position seq prefix so the [#N] tag lines up with the
    # identically-formatted tags on thinking and bg rows. Each type is
    # disambiguated by the command you use (/show, /think, /bg) — no
    # letter prefix needed.
    if seq is not None:
        if name == "Bash":
            seq_prefix = f"\033[90m[#{seq} -- `/show {seq} [-K]`]\033[0m "
        else:
            seq_prefix = f"\033[90m[#{seq}]\033[0m "
    else:
        seq_prefix = ""
    seq_tag = ""  # kept for legacy placeholders below; always empty now
    if name == "Edit":
        path = inp.get("file_path", "?")
        replace_all = inp.get("replace_all", False)
        tag = "Edit (replace_all)" if replace_all else "Edit"
        old = inp.get("old_string", "") or ""
        new = inp.get("new_string", "") or ""
        removed = len(old.splitlines()) or (1 if old else 0)
        added = len(new.splitlines()) or (1 if new else 0)
        if effective_edits == "compact":
            print(
                f"  {seq_prefix}\033[34mtool \033[1m{tag}\033[0m {path}  "
                f"\033[90m(\033[32m+{added}\033[0m \033[31m-{removed}\033[0m"
                f"\033[90m lines)\033[0m"
            )
        else:
            print(f"  {seq_prefix}\033[34mtool \033[1m{tag}\033[0m {path}")
            _print_unified_diff(old, new)
    elif name == "Write":
        path = inp.get("file_path", "?")
        content = inp.get("content", "") or ""
        line_count = len(content.splitlines())
        print(
            f"  {seq_prefix}\033[34mtool \033[1mWrite\033[0m {path} "
            f"\033[90m({line_count} lines, {len(content)} chars)\033[0m"
        )
        preview = content.splitlines()[:10]
        for ln in preview:
            print(f"    \033[32m+{ln}\033[0m")
        if line_count > 10:
            print(f"    \033[90m... [+{line_count - 10} more lines]\033[0m")
    elif name == "NotebookEdit":
        path = inp.get("notebook_path", "?")
        cell_id = inp.get("cell_id", "")
        mode = inp.get("edit_mode", "replace")
        print(f"  {seq_prefix}\033[34mtool \033[1mNotebookEdit\033[0m {path} cell={cell_id} mode={mode}")
        src = inp.get("new_source", "") or ""
        for ln in src.splitlines()[:12]:
            print(f"    \033[32m+{ln}\033[0m")
    elif name == "Bash":
        cmd = inp.get("command", "") or ""
        bg = bool(inp.get("run_in_background"))
        desc = inp.get("description", "")
        tag = "Bash (background)" if bg else "Bash"
        base = f"  {seq_prefix}\033[34mtool \033[1m{tag}\033[0m"
        if desc:
            base += f" \033[90m— {desc}\033[0m"
        # Inline the command itself when it's a single line that fits on
        # the terminal. Otherwise fall back to a size hint. Keeps short
        # Bash calls fully visible without the user having to /show.
        stripped = cmd.strip()
        cmd_lines = cmd.splitlines() or ([cmd] if cmd else [])
        term_w = _term_width()
        shown_inline = False
        if stripped and len(cmd_lines) <= 1:
            # Wrap the command in bright-cyan backticks so it's visually
            # distinct from the dim description and the leading [#N] tag.
            trial = f"{base}  \033[96m`{cmd}`\033[0m"
            if _visible_len(trial) <= term_w:
                print(trial)
                shown_inline = True
        if not shown_inline:
            if stripped:
                print(
                    f"{base}  \033[90m({_cmd_size_hint(cmd)})\033[0m"
                )
            else:
                print(base)
        # --show-full-commands prints every line regardless of the inline
        # fit heuristic (useful when you want verbatim even for short cmds).
        if show_full_commands and not shown_inline:
            for ln in cmd_lines or [cmd]:
                print(f"    \033[36m$\033[0m {ln}")
    elif name == "BashOutput":
        shell_id = inp.get("bash_id") or inp.get("shell_id") or "?"
        print(f"  {seq_prefix}\033[34mtool \033[1mBashOutput\033[0m shell={shell_id}")
    elif name == "KillShell":
        shell_id = inp.get("shell_id") or inp.get("bash_id") or "?"
        print(f"  {seq_prefix}\033[34mtool \033[1mKillShell\033[0m shell={shell_id}")
    elif name == "TodoWrite":
        todos = inp.get("todos", []) or []
        print(f"  {seq_prefix}\033[34mtool \033[1mTodoWrite\033[0m \033[90m({len(todos)} items)\033[0m")
        markers = {"completed": "\033[32m✓\033[0m", "in_progress": "\033[33m→\033[0m", "pending": "\033[90m·\033[0m"}
        for t in todos:
            m = markers.get(t.get("status", "pending"), "?")
            content = t.get("content", "") or t.get("activeForm", "")
            print(f"    {m} {content}")
    elif name == "Task":
        desc = inp.get("description", "")
        subtype = inp.get("subagent_type", "general-purpose")
        print(f"  {seq_prefix}\033[34mtool \033[1mTask\033[0m \033[90m[{subtype}]\033[0m {desc}")
    elif name == "Read":
        path = inp.get("file_path", "?")
        offset = inp.get("offset")
        limit = inp.get("limit")
        tail = f" offset={offset} limit={limit}" if offset or limit else ""
        print(f"  {seq_prefix}\033[34mtool \033[1mRead\033[0m {path}{tail}")
    elif name == "Grep":
        pattern = inp.get("pattern", "")
        path = inp.get("path", ".")
        print(f"  {seq_prefix}\033[34mtool \033[1mGrep\033[0m /{pattern}/ in {path}")
    elif name == "Glob":
        pattern = inp.get("pattern", "")
        path = inp.get("path", ".")
        print(f"  {seq_prefix}\033[34mtool \033[1mGlob\033[0m {pattern} in {path}")
    elif name == "WebFetch":
        url = inp.get("url", "?")
        print(f"  {seq_prefix}\033[34mtool \033[1mWebFetch\033[0m {url}")
    elif name == "WebSearch":
        q = inp.get("query", "?")
        print(f"  {seq_prefix}\033[34mtool \033[1mWebSearch\033[0m {q!r}")
    else:
        print(f"  {seq_prefix}\033[34mtool \033[1m{name}\033[0m({brief_args(inp)})")


def summarize_tool_result(block: ToolResultBlock) -> str:
    content = block.content
    if isinstance(content, list):
        parts = [str(item.get("text", "")) for item in content if isinstance(item, dict)]
        text = "\n".join(p for p in parts if p)
    else:
        text = str(content or "")
    text = text.strip()
    if not text:
        return "(empty)"
    if len(text) > 1500:
        head = text[:1000]
        tail = text[-300:]
        return f"{head}\n... [+{len(text) - 1300} chars]\n{tail}"
    return text


def _extract_context_tokens(
    usage: dict[str, Any] | None,
    model_usage: dict[str, Any] | None,
) -> int:
    """Sum input + cache_read + cache_creation tokens across whichever
    shape the CLI gave us. Anthropic-API snake_case in `usage`, CLI's
    camelCase keyed by model in `model_usage` — try both."""
    if isinstance(usage, dict) and usage:
        total = (
            usage.get("input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
        )
        if total:
            return total
        total = (
            usage.get("inputTokens", 0)
            + usage.get("cacheReadInputTokens", 0)
            + usage.get("cacheCreationInputTokens", 0)
        )
        if total:
            return total
    if isinstance(model_usage, dict) and model_usage:
        total = 0
        for mu in model_usage.values():
            if isinstance(mu, dict):
                total += (
                    mu.get("inputTokens", 0)
                    + mu.get("cacheReadInputTokens", 0)
                    + mu.get("cacheCreationInputTokens", 0)
                )
        if total:
            return total
    return 0


def _humanize_size(text: str) -> str:
    lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
    if lines == 0 and text:
        lines = 1
    chars = len(text)
    char_str = f"{chars / 1000:.1f}k chars" if chars >= 1000 else f"{chars} chars"
    return f"{lines} line{'s' if lines != 1 else ''}, {char_str}"


def _render_tool_result(
    text: str, *, is_error: bool, show_full: bool, seq: int | None = None
) -> None:
    """Print a tool result. With show_full=True, dump the entire (possibly
    truncated by summarize_tool_result) content indented under a marker.
    Default is suppressed: a dim size indicator on success, a red one-liner
    on error — full content lives in the JSONL transcript and /export.
    `seq` (when provided) prints `[#N]` so the user can `/show N` later."""
    seq_prefix = f"\033[90m[#{seq}]\033[0m " if seq is not None else ""
    if show_full:
        marker = (
            f"\033[31mtool-err:\033[0m"
            if is_error
            else f"\033[34m->\033[0m"
        )
        lines = text.splitlines() or [text]
        if not lines or (len(lines) == 1 and not lines[0].strip()):
            print(f"  {seq_prefix}{marker} (empty)")
            return
        print(f"  {seq_prefix}{marker} {lines[0]}")
        for ln in lines[1:]:
            print(f"     {ln}")
        return
    size = _humanize_size(text)
    if is_error:
        print(
            f"  {seq_prefix}\033[31m✗ tool error\033[0m  "
            f"\033[90m({size}; rerun with --show-tool-output to see)\033[0m"
        )
    else:
        print(f"  {seq_prefix}\033[90m→ {size}\033[0m")


# ----------------------------------------------------------------------------
# Session discovery (mirrors Claude Code's on-disk layout)
# ----------------------------------------------------------------------------
#   ~/.claude/projects/<sanitized-cwd>/<session-id>.jsonl
# `sanitized-cwd` replaces any non-alphanumeric char with `-` (per
# claude-code-mod/utils/sessionStoragePortable.ts:sanitizePath).
# Override base via $CLAUDE_CONFIG_DIR.

_PICKER_SENTINEL = "<picker>"


def _claude_projects_dir() -> Path:
    base = os.environ.get("CLAUDE_CONFIG_DIR")
    return Path(base if base else Path.home() / ".claude") / "projects"


def _rough_tokens(record: dict[str, Any]) -> int:
    """Cheap upper-bound token estimate for a JSONL record. We can't run the
    tokenizer across the whole transcript every turn, so fall back to
    chars/4 on whatever text-ish content the record carries."""
    msg = record.get("message")
    if not isinstance(msg, dict):
        return 0
    content = msg.get("content", "")
    if isinstance(content, str):
        return max(1, len(content) // 4)
    if isinstance(content, list):
        total = 0
        for c in content:
            try:
                total += len(json.dumps(c, default=str)) // 4
            except (TypeError, ValueError):
                total += 10
        return max(1, total)
    return 0


def _is_user_turn_start(record: dict[str, Any]) -> bool:
    """True iff this record marks the beginning of a *new* user turn — i.e.
    a human prompt, not a tool_result continuation. These are the only
    safe cut points for rolling-window trim: slicing mid-turn would leave
    tool_use blocks orphaned (no matching tool_result) and confuse the model."""
    if record.get("type") != "user" or record.get("isSidechain"):
        return False
    msg = record.get("message") or {}
    content = msg.get("content", "")
    if isinstance(content, str):
        return True
    if isinstance(content, list):
        return not any(
            isinstance(c, dict) and c.get("type") == "tool_result"
            for c in content
        )
    return False


_TRIM_MARKER_TYPE = "orch-trim-metadata"


def _is_trim_session_file(jsonl_path: Path) -> dict[str, Any] | None:
    """If this .jsonl starts with our trim marker, return the marker dict.
    Used by the picker to hide rolling-window intermediate files and by
    `_trim_session` to chain-delete superseded trims."""
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
    except OSError:
        return None
    if not first:
        return None
    try:
        obj = json.loads(first)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and obj.get("type") == _TRIM_MARKER_TYPE:
        return obj
    return None


def _trim_session(
    src_session_id: str,
    project_dir: Path,
    target_tokens: int,
) -> str | None:
    """Fork the tail of `src_session_id` into a new session JSONL whose rough
    token cost sits at or below `target_tokens`. Returns the new session
    UUID, or None if no trim was needed or possible.

    Cuts at user-turn boundaries to keep tool_use/tool_result pairs intact.
    Remaps every kept record's UUID and stitches the parentUuid chain so
    the first kept message becomes a new root (parentUuid=null).

    Writes a trim-marker record at the top of the new file. If the source
    is itself a trim (has that marker), deletes it on success — so only
    the current trim survives on disk. The *original* untrimmed session
    is never touched."""
    import uuid as uuid_mod

    src_path = project_dir / f"{src_session_id}.jsonl"
    if not src_path.exists():
        return None
    src_marker = _is_trim_session_file(src_path)
    root_session_id = (
        src_marker.get("rootSessionId") if src_marker else src_session_id
    )
    records: list[dict[str, Any]] = []
    try:
        with open(src_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Drop any pre-existing trim marker from the source; we'll
                # write a fresh one at the top of the new file.
                if isinstance(obj, dict) and obj.get("type") == _TRIM_MARKER_TYPE:
                    continue
                records.append(obj)
    except OSError:
        return None
    if not records:
        return None

    transcript = [r for r in records if not r.get("isSidechain")]
    turn_starts = [i for i, r in enumerate(transcript) if _is_user_turn_start(r)]
    if len(turn_starts) < 2:
        return None  # single turn — nothing safe to trim

    total_tokens = sum(_rough_tokens(r) for r in transcript)
    if total_tokens <= target_tokens:
        return None  # already fits

    # Walk backward through turn starts; find the earliest one whose tail
    # (from that turn forward) still fits under target. If none fit (because
    # even the last turn is huge), keep just the last turn and hope it works.
    cut_idx: int | None = None
    for ts in reversed(turn_starts):
        tail = sum(_rough_tokens(transcript[i]) for i in range(ts, len(transcript)))
        if tail <= target_tokens:
            cut_idx = ts
        else:
            break
    if cut_idx is None:
        cut_idx = turn_starts[-1]

    kept = transcript[cut_idx:]
    if not kept:
        return None

    new_session_id = str(uuid_mod.uuid4())
    uuid_map: dict[str, str] = {}
    for r in kept:
        if "uuid" in r:
            uuid_map[r["uuid"]] = str(uuid_mod.uuid4())

    # Preserve only safe pre-message metadata. file-history-snapshot records
    # reference trimmed message UUIDs, so we drop them rather than remap —
    # losing undo history is an acceptable tradeoff for clean rolling window.
    output_lines: list[str] = []
    # Marker so the picker can hide this file and subsequent trims can
    # chain-delete it.
    output_lines.append(
        json.dumps(
            {
                "type": _TRIM_MARKER_TYPE,
                "sessionId": new_session_id,
                "previousSessionId": src_session_id,
                "rootSessionId": root_session_id,
                "createdAt": time.time(),
            },
            separators=(",", ":"),
        )
    )
    for meta in records:
        if meta.get("type") == "permission-mode":
            m = dict(meta)
            m["sessionId"] = new_session_id
            output_lines.append(json.dumps(m, separators=(",", ":")))
            break  # one permission-mode record only

    prev_new_uuid: str | None = None
    for i, r in enumerate(kept):
        new = dict(r)
        orig_uuid = r.get("uuid")
        if orig_uuid and orig_uuid in uuid_map:
            new["uuid"] = uuid_map[orig_uuid]
        orig_parent = r.get("parentUuid")
        if i == 0:
            new["parentUuid"] = None
        elif orig_parent and orig_parent in uuid_map:
            new["parentUuid"] = uuid_map[orig_parent]
        else:
            new["parentUuid"] = prev_new_uuid
        new["sessionId"] = new_session_id
        new["isSidechain"] = False
        # Drop fields that would leak state from the source session.
        for key in ("forkedFrom", "logicalParentUuid"):
            new.pop(key, None)
        output_lines.append(json.dumps(new, separators=(",", ":")))
        prev_new_uuid = new.get("uuid") or prev_new_uuid

    dest = project_dir / f"{new_session_id}.jsonl"
    try:
        with open(dest, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines) + "\n")
    except OSError:
        return None
    # Chain-cleanup: if the source was itself a trim, remove it now that
    # we've superseded it. Never deletes the *original* untrimmed session.
    if src_marker is not None:
        try:
            src_path.unlink()
        except OSError:
            pass
    return new_session_id


def _sanitize_cwd(cwd: str) -> str:
    """Match Claude Code's sanitizePath: any non-alphanumeric becomes '-'.
    See claude-code-mod/utils/sessionStoragePortable.ts:sanitizePath."""
    import re

    return re.sub(r"[^a-zA-Z0-9]", "-", cwd)


def project_dir_for_cwd(cwd: str) -> Path:
    """Return the on-disk project directory Claude Code would use for `cwd`."""
    try:
        resolved = str(Path(cwd).resolve(strict=False))
    except OSError:
        resolved = cwd
    return _claude_projects_dir() / _sanitize_cwd(resolved)


def _normalize_path_for_compare(p: str) -> str:
    """Path-comparison normalizer. Backslash → slash, strip trailing slash,
    lowercase on Windows (case-insensitive filesystem)."""
    s = p.replace("\\", "/").rstrip("/")
    if sys.platform == "win32":
        s = s.lower()
    return s


def _sniff_session_cwd(jsonl: Path) -> str | None:
    """Read the first few records of a session JSONL and return the first
    `cwd` field present."""
    try:
        with jsonl.open(encoding="utf-8", errors="replace") as f:
            for _ in range(20):
                line = f.readline()
                if not line:
                    break
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict) and isinstance(rec.get("cwd"), str):
                    return rec["cwd"]
    except OSError:
        pass
    return None


def find_project_for_cwd(cwd: str) -> Path | None:
    """Locate an existing project directory for `cwd`. Tries the direct
    sanitize-and-check path first; falls back to scanning every project
    dir for one whose first-record `cwd` matches (case-insensitive on
    Windows). Returns None if no project on disk corresponds."""
    direct = project_dir_for_cwd(cwd)
    if direct.exists():
        return direct
    try:
        target = str(Path(cwd).resolve(strict=False))
    except OSError:
        target = cwd
    target_norm = _normalize_path_for_compare(target)
    projects = _claude_projects_dir()
    if not projects.exists():
        return None
    for project in projects.iterdir():
        if not project.is_dir():
            continue
        # Only look at one jsonl per project — they all share the same cwd.
        for jsonl in project.glob("*.jsonl"):
            stored = _sniff_session_cwd(jsonl)
            if stored and _normalize_path_for_compare(stored) == target_norm:
                return project
            break
    return None


def _find_session_dir(session_id: str) -> Path | None:
    """Locate the project dir that contains <session_id>.jsonl, scanning all."""
    projects = _claude_projects_dir()
    if not projects.exists():
        return None
    for project in projects.iterdir():
        if project.is_dir() and (project / f"{session_id}.jsonl").exists():
            return project
    return None


def _read_session_title(session_id: str) -> str | None:
    """Look up a session's display title from Claude Code's native storage:
    titles are appended as records inside the session's JSONL transcript —
    `{"type": "custom-title", "customTitle": ...}` (user rename, wins) or
    `{"type": "ai-title", "aiTitle": ...}` (Haiku auto-name). For each type
    the latest record wins; `customTitle` always beats `aiTitle`."""
    project = _find_session_dir(session_id)
    if project is None:
        return None
    jsonl = project / f"{session_id}.jsonl"
    if not jsonl.exists():
        return None
    custom_title: str | None = None
    ai_title: str | None = None
    try:
        with jsonl.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                t = rec.get("type")
                if t == "custom-title":
                    v = rec.get("customTitle")
                    if isinstance(v, str) and v.strip():
                        custom_title = v.strip()  # last write wins
                elif t == "ai-title":
                    v = rec.get("aiTitle")
                    if isinstance(v, str) and v.strip():
                        ai_title = v.strip()
    except OSError:
        return None
    return custom_title or ai_title


def _find_sessions_with_title(
    title: str, exclude_id: str | None = None
) -> list[str]:
    """Return session_ids whose effective title matches (case-insensitive),
    excluding the given id. Reads via _read_session_title so the source of
    truth (Claude Code's JSONL custom-title / ai-title records) is
    consulted directly."""
    target = title.strip().lower()
    if not target:
        return []
    matches: list[str] = []
    projects = _claude_projects_dir()
    if not projects.exists():
        return matches
    for project in projects.iterdir():
        if not project.is_dir():
            continue
        for jsonl in project.glob("*.jsonl"):
            sid = jsonl.stem
            if exclude_id and sid == exclude_id:
                continue
            t = _read_session_title(sid)
            if t and t.strip().lower() == target:
                matches.append(sid)
    return matches


def _write_session_title(session_id: str, title: str) -> None:
    """Set a session's title — appends a `custom-title` record to the
    session's JSONL transcript, the same storage Claude Code's `/rename`
    uses. Delegates to `claude_agent_sdk.rename_session` when available
    (handles UUID validation, empty-title rejection, and the directory
    search itself); falls back to a hand-rolled append on older SDKs."""
    if _sdk_rename_session is not None:
        # SDK raises ValueError on bad input or FileNotFoundError if the
        # session JSONL isn't on disk yet. Let those propagate.
        _sdk_rename_session(session_id, title)
        return
    # Fallback for SDKs without rename_session.
    project = _find_session_dir(session_id)
    if project is None:
        raise OSError(f"session {session_id} not found on disk")
    jsonl = project / f"{session_id}.jsonl"
    if not jsonl.exists():
        raise OSError(f"session jsonl missing: {jsonl}")
    record = {
        "type": "custom-title",
        "customTitle": title,
        "sessionId": session_id,
    }
    with jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _extract_text(content: Any) -> str:
    """Pull plain text out of a message-content field that may be str or list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("text"), str):
                    parts.append(block["text"])
        return "\n".join(parts)
    return ""


def _parse_session_info(jsonl: Path, project_slug: str) -> dict[str, Any] | None:
    """Read a session JSONL and pull out id, cwd, first/last user message, timestamp."""
    try:
        st = jsonl.stat()
    except OSError:
        return None
    info: dict[str, Any] = {
        "session_id": jsonl.stem,
        "project_slug": project_slug,
        "cwd": None,
        "first_user_msg": None,
        "last_user_msg": None,
        "last_timestamp": None,
        "mtime": st.st_mtime,
        "size": st.st_size,
        "msg_count": 0,
        "title": None,
    }
    # Title from JSONL `custom-title` / `ai-title` records (Claude Code's
    # native storage; same place /rename writes).
    info["title"] = _read_session_title(jsonl.stem)
    try:
        with jsonl.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                info["msg_count"] += 1
                if info["cwd"] is None and isinstance(rec.get("cwd"), str):
                    info["cwd"] = rec["cwd"]
                ts = rec.get("timestamp")
                if isinstance(ts, str):
                    info["last_timestamp"] = ts
                if rec.get("type") == "user":
                    msg = rec.get("message")
                    text = ""
                    if isinstance(msg, dict):
                        text = _extract_text(msg.get("content"))
                    elif isinstance(msg, str):
                        text = msg
                    text = text.strip()
                    # Skip pure tool-result user messages (no human text).
                    if text and not text.startswith("<bash") and not text.startswith("<tool"):
                        if info["first_user_msg"] is None:
                            info["first_user_msg"] = text
                        info["last_user_msg"] = text
    except OSError:
        return None
    return info


def list_sessions(filter_cwd: str | None = None) -> list[dict[str, Any]]:
    """Return all sessions on disk, newest first.

    If filter_cwd is given, only include sessions whose project dir matches.
    """
    projects = _claude_projects_dir()
    if not projects.exists():
        return []
    sessions: list[dict[str, Any]] = []
    for project in projects.iterdir():
        if not project.is_dir():
            continue
        for jsonl in project.glob("*.jsonl"):
            if _is_trim_session_file(jsonl) is not None:
                continue  # rolling-window intermediate file; skip
            info = _parse_session_info(jsonl, project.name)
            if info is None:
                continue
            if filter_cwd is not None and info.get("cwd") != filter_cwd:
                continue
            sessions.append(info)
    sessions.sort(key=lambda s: s.get("mtime", 0.0), reverse=True)
    return sessions


def list_projects() -> list[dict[str, Any]]:
    """Light-weight project listing for the picker. No JSONL parsing — only
    stats files and reads the first record of the newest session in each
    project (to recover the real cwd path)."""
    projects_dir = _claude_projects_dir()
    if not projects_dir.exists():
        return []
    out: list[dict[str, Any]] = []
    for project in projects_dir.iterdir():
        if not project.is_dir():
            continue
        jsonls = list(project.glob("*.jsonl"))
        if not jsonls:
            continue
        try:
            stats = [(j, j.stat().st_mtime) for j in jsonls]
        except OSError:
            continue
        newest_mtime = max(m for _, m in stats)
        # Read just enough of the newest jsonl to find the real cwd.
        cwd: str | None = None
        try:
            newest_jsonl = max(stats, key=lambda x: x[1])[0]
            with newest_jsonl.open(encoding="utf-8", errors="replace") as f:
                for _ in range(20):  # cwd is usually on the very first record
                    line = f.readline()
                    if not line:
                        break
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(rec, dict) and isinstance(rec.get("cwd"), str):
                        cwd = rec["cwd"]
                        break
        except OSError:
            pass
        out.append(
            {
                "project_dir": project,
                "project_slug": project.name,
                "cwd": cwd,
                "session_count": len(jsonls),
                "newest_mtime": newest_mtime,
            }
        )
    out.sort(key=lambda p: p["newest_mtime"], reverse=True)
    return out


async def cursor_select(
    title: str,
    text: str,
    values: list[tuple[Any, str]],
) -> Any | None:
    """Cursor-as-selection picker. Up/Down moves the highlight (no Space
    needed); Enter confirms whatever's highlighted (no tab to OK); Esc /
    Ctrl-C cancels. Scrolls when the list overflows the viewport."""
    if not values:
        return None
    from prompt_toolkit.application import Application
    from prompt_toolkit.application.current import get_app
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

    state: dict[str, Any] = {"cursor": 0, "scroll": 0, "result": None}

    def _make_row_handler(idx: int):
        def _handler(event: MouseEvent):
            if event.event_type == MouseEventType.MOUSE_DOWN:
                if event.button == MouseButton.LEFT:
                    state["cursor"] = idx
                    state["result"] = values[idx][0]
                    get_app().exit()
                    return None
            elif event.event_type == MouseEventType.SCROLL_UP:
                if state["cursor"] > 0:
                    state["cursor"] -= 1
                return None
            elif event.event_type == MouseEventType.SCROLL_DOWN:
                if state["cursor"] < len(values) - 1:
                    state["cursor"] += 1
                return None
            return NotImplemented

        return _handler

    def _render() -> list[tuple[str, str]]:
        try:
            term_h = get_app().output.get_size().rows
        except Exception:  # noqa: BLE001
            term_h = 24
        # Reserve title (1) + text (≤4) + footer (2) + a little padding.
        text_lines = (text.count("\n") + 1) if text else 0
        viewport = max(3, term_h - (4 + min(text_lines, 4)))
        # Keep cursor on screen.
        if state["cursor"] < state["scroll"]:
            state["scroll"] = state["cursor"]
        elif state["cursor"] >= state["scroll"] + viewport:
            state["scroll"] = state["cursor"] - viewport + 1
        visible = values[state["scroll"] : state["scroll"] + viewport]
        out: list[Any] = []
        for i, (_val, label) in enumerate(visible):
            actual = state["scroll"] + i
            line = str(label).rstrip("\n")
            handler = _make_row_handler(actual)
            if actual == state["cursor"]:
                out.append(("reverse", " ▶ " + line + "\n", handler))
            else:
                out.append(("", "   " + line + "\n", handler))
        out.append(("", "\n"))
        out.append(
            (
                "ansibrightblack",
                f" [{state['cursor'] + 1}/{len(values)}]  "
                f"↑↓/click navigate · wheel scrolls · Enter/click selects · Esc cancels",
            )
        )
        return out

    kb = KeyBindings()

    @kb.add("up", eager=True)
    @kb.add("c-p", eager=True)
    def _(event):  # noqa: D401
        if state["cursor"] > 0:
            state["cursor"] -= 1

    @kb.add("down", eager=True)
    @kb.add("c-n", eager=True)
    def _(event):
        if state["cursor"] < len(values) - 1:
            state["cursor"] += 1

    @kb.add("pageup", eager=True)
    def _(event):
        state["cursor"] = max(0, state["cursor"] - 10)

    @kb.add("pagedown", eager=True)
    def _(event):
        state["cursor"] = min(len(values) - 1, state["cursor"] + 10)

    @kb.add("home", eager=True)
    def _(event):
        state["cursor"] = 0

    @kb.add("end", eager=True)
    def _(event):
        state["cursor"] = len(values) - 1

    @kb.add("enter", eager=True)
    def _(event):
        state["result"] = values[state["cursor"]][0]
        event.app.exit()

    @kb.add("escape", eager=True)
    @kb.add("c-c", eager=True)
    @kb.add("c-d", eager=True)
    def _(event):
        state["result"] = None
        event.app.exit()

    body: list[Any] = [
        Window(FormattedTextControl(HTML(f"<b>{title}</b>")), height=1)
    ]
    if text:
        body.append(
            Window(
                FormattedTextControl(text),
                height=Dimension(min=1, max=4),
                wrap_lines=True,
            )
        )
    body.append(
        Window(
            FormattedTextControl(_render, focusable=True, show_cursor=False)
        )
    )
    app: Application[Any] = Application(
        layout=Layout(HSplit(body)),
        key_bindings=kb,
        full_screen=True,
        mouse_support=True,
    )
    await app.run_async()
    return state["result"]


def find_most_recent_session_for_cwd(cwd: str) -> Path | None:
    """Locate the most-recently-modified .jsonl in the project dir for cwd."""
    project = project_dir_for_cwd(cwd)
    if not project.exists():
        return None
    candidates: list[tuple[Path, float]] = []
    for jsonl in project.glob("*.jsonl"):
        try:
            candidates.append((jsonl, jsonl.stat().st_mtime))
        except OSError:
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[1])[0]


def render_session_history_text(jsonl: Path) -> tuple[int, str]:
    """Build the ANSI-colored backscroll for a session JSONL transcript.
    Returns (message_count, rendered_text). The caller is expected to write
    the whole string in one shot so the terminal doesn't scroll line-by-line
    while history is being assembled."""
    import io

    rendered = 0
    buf = io.StringIO()
    write = buf.write

    try:
        f = jsonl.open(encoding="utf-8", errors="replace")
    except OSError as e:
        return 0, f"\033[31m[history: failed to open {jsonl.name}: {e}]\033[0m\n"

    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            t = rec.get("type")
            msg = rec.get("message")
            if t == "user" and isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    text = content.strip()
                    if not (text.startswith("<bash") or text.startswith("<tool")):
                        write(f"\n\033[36m> you (history):\033[0m {text}\n")
                        rendered += 1
                elif isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        bt = block.get("type")
                        if bt == "tool_result":
                            inner = block.get("content")
                            text = (
                                inner
                                if isinstance(inner, str)
                                else _extract_text(inner)
                            )
                            text = (text or "").strip()
                            if text:
                                if len(text) > 600:
                                    text = (
                                        text[:600]
                                        + f"... [+{len(text) - 600} chars]"
                                    )
                                tag = (
                                    "\033[31m  tool-err:\033[0m"
                                    if block.get("is_error")
                                    else "\033[34m  ->\033[0m"
                                )
                                write(f"{tag} {text}\n")
                        elif bt == "text" and isinstance(block.get("text"), str):
                            text = block["text"].strip()
                            if text:
                                write(f"\n\033[36m> you (history):\033[0m {text}\n")
                                rendered += 1
            elif t == "assistant" and isinstance(msg, dict):
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                started = False
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    bt = block.get("type")
                    if bt == "text":
                        text = (block.get("text") or "").strip()
                        if not text:
                            continue
                        if not started:
                            write("\n\033[32mclaude (history):\033[0m ")
                            started = True
                        else:
                            write(" ")
                        write(text)
                    elif bt == "tool_use":
                        if started:
                            write("\n")
                            started = False
                        name = block.get("name", "?")
                        inp = block.get("input") or {}
                        write(
                            f"  \033[34mtool {name}\033[0m({brief_args(inp)})\n"
                        )
                    # thinking blocks intentionally skipped
                if started:
                    write("\n")
                rendered += 1
    return rendered, buf.getvalue()


def _render_session_markdown(jsonl: Path) -> str:
    """Convert a session JSONL transcript into a readable markdown export."""
    metadata: dict[str, Any] = {
        "session_id": jsonl.stem,
        "cwd": None,
        "first_ts": None,
        "last_ts": None,
        "title": _read_session_title(jsonl.stem),
    }
    body: list[str] = []

    def _ts(rec: dict[str, Any]) -> str:
        ts = rec.get("timestamp")
        if isinstance(ts, str):
            if metadata["first_ts"] is None:
                metadata["first_ts"] = ts
            metadata["last_ts"] = ts
            return f" — _{ts}_"
        return ""

    with jsonl.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            if metadata["cwd"] is None and isinstance(rec.get("cwd"), str):
                metadata["cwd"] = rec["cwd"]
            t = rec.get("type")
            msg = rec.get("message")
            ts_suffix = _ts(rec)

            if t == "user" and isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    body.append(f"\n## You{ts_suffix}\n\n{content.strip()}\n")
                elif isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        bt = block.get("type")
                        if bt == "tool_result":
                            inner = block.get("content")
                            text = (
                                inner
                                if isinstance(inner, str)
                                else _extract_text(inner)
                            )
                            text = (text or "").strip() or "(empty)"
                            body.append(
                                f"\n**Result:**\n\n```\n{text}\n```\n"
                            )
                        elif bt == "text" and isinstance(block.get("text"), str):
                            body.append(
                                f"\n## You{ts_suffix}\n\n{block['text'].strip()}\n"
                            )
            elif t == "assistant" and isinstance(msg, dict):
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                section_started = False
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    bt = block.get("type")
                    if bt == "text":
                        text = (block.get("text") or "").strip()
                        if not text:
                            continue
                        if not section_started:
                            body.append(f"\n## Claude{ts_suffix}\n")
                            section_started = True
                        body.append(f"\n{text}\n")
                    elif bt == "tool_use":
                        name = block.get("name", "?")
                        inp = block.get("input", {})
                        if not section_started:
                            body.append(f"\n## Claude{ts_suffix}\n")
                            section_started = True
                        try:
                            inp_text = json.dumps(inp, indent=2, default=str)
                        except (TypeError, ValueError):
                            inp_text = str(inp)
                        body.append(
                            f"\n**Tool: `{name}`**\n\n```json\n{inp_text}\n```\n"
                        )
                    # thinking blocks intentionally skipped from export

    header = ["# Claude Conversation\n"]
    if metadata["title"]:
        header.append(f"- **Title:** {metadata['title']}")
    header.append(f"- **Session:** `{metadata['session_id']}`")
    if metadata["cwd"]:
        header.append(f"- **Project:** `{metadata['cwd']}`")
    if metadata["first_ts"]:
        header.append(f"- **Started:** {metadata['first_ts']}")
    if metadata["last_ts"]:
        header.append(f"- **Last activity:** {metadata['last_ts']}")
    return "\n".join(header) + "\n\n---\n" + "".join(body)


def find_session_cwd(session_id: str) -> str | None:
    """Locate a session by id and return its recorded cwd, if present."""
    project_dir = _find_session_dir(session_id)
    if project_dir is None:
        return None
    info = _parse_session_info(
        project_dir / f"{session_id}.jsonl", project_dir.name
    )
    return info.get("cwd") if info else None


def list_sessions_for_project(project_dir: Path) -> list[dict[str, Any]]:
    """Parse all sessions in a single project dir, newest first. Skips
    rolling-window trim files — those are intermediate state, not
    first-class sessions the user should resume from the picker."""
    sessions: list[dict[str, Any]] = []
    for jsonl in project_dir.glob("*.jsonl"):
        if _is_trim_session_file(jsonl) is not None:
            continue
        info = _parse_session_info(jsonl, project_dir.name)
        if info is not None:
            sessions.append(info)
    sessions.sort(key=lambda s: s.get("mtime", 0.0), reverse=True)
    return sessions


def format_project_label(p: dict[str, Any], width: int = 100) -> str:
    """One-line label for the project picker."""
    name_source = p.get("cwd") or p.get("project_slug") or "?"
    name = Path(name_source).name if name_source != "?" else "(unknown)"
    count = p.get("session_count", 0)
    age = _format_session_age(p.get("newest_mtime") or 0.0)
    cwd = p.get("cwd") or p.get("project_slug") or ""
    head = f"{name:<28}  {count:>3} sessions  newest {age:>9}  "
    remaining = max(20, width - len(head))
    if len(cwd) > remaining:
        cwd = "…" + cwd[-(remaining - 1) :]
    return head + cwd


def _format_session_age(mtime: float) -> str:
    delta = max(0.0, time.time() - mtime)
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta / 60)}m ago"
    if delta < 86400:
        return f"{int(delta / 3600)}h ago"
    if delta < 86400 * 30:
        return f"{int(delta / 86400)}d ago"
    try:
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
    except (OSError, ValueError, OverflowError):
        return "long ago"


# ----------------------------------------------------------------------------
# Status-panel framework (drives the live bottom toolbar)
# ----------------------------------------------------------------------------
# Each "panel" is a function (state) -> str (HTML). The toolbar layout is a
# list-of-lists: outer list = lines, inner = panels joined by " | " on that
# line. Adding a new live indicator = define a panel function and append it
# to the layout. Panels returning "" are dropped from the line.

PanelFn = Callable[["State"], str]


def _panel_session(state: "State") -> str:
    if state.busy:
        busy = "<b><ansigreen>● WORKING</ansigreen></b>"
    elif state.needs_user_attention == "waiting":
        busy = "<b><ansired>● WAITING</ansired></b>"
    elif state.needs_user_attention == "burst":
        busy = "<b><ansibrightmagenta>● STALLED</ansibrightmagenta></b>"
    elif state.needs_user_attention == "api-error":
        label = "● API-STALL"
        if state.api_status_description:
            desc = _tb_escape(state.api_status_description)[:40]
            label = f"● API-STALL ({desc})"
        busy = f"<b><ansired>{label}</ansired></b>"
    else:
        busy = "idle"
    sid = (state.session_id or "(new)")[:8]
    title_part = (
        f"  <ansibrightcyan>★ {state.session_title}</ansibrightcyan>"
        if state.session_title
        else ""
    )
    if state.is_subscription:
        plan = state.subscription_plan or "sub"
        if state.rate_limit_util is not None:
            label = state.rate_limit_label or "limit"
            plan_field = (
                f"plan: {plan} {state.rate_limit_util * 100:.0f}% / {label}"
            )
        else:
            plan_field = f"plan: {plan}"
    else:
        plan_field = f"cost: ${state.total_cost_usd:.4f}"
    # Model: prefer user-pinned (--model) over the CLI-reported active model
    # (discovered from AssistantMessage.model). Shows "(auto)" only when we
    # haven't seen an AssistantMessage yet.
    effective_model = state.model or state.active_model or ""
    short_model = (
        effective_model[len("claude-"):]
        if effective_model.startswith("claude-")
        else effective_model
    )
    model_part = short_model if short_model else "(auto)"
    effort_part = state.effort if state.effort else "auto"
    think_part = "on" if state.show_thinking else "off"
    # Approximate "current resident context": cap the last turn's cumulative
    # input-tokens figure at the model's actual window. Window is shown as
    # "?" when we haven't identified the model yet (no --model pinned and
    # no AssistantMessage received) — better than silently guessing 200k.
    if effective_model:
        window = _model_context_window(effective_model)
        window_str = _fmt_tok(window)
    else:
        window = None
        window_str = "?"
    if state.context_tokens:
        resident = min(state.context_tokens, window) if window else state.context_tokens
        ctx = f"ctx: ~{_fmt_tok(resident)}/{window_str} tok"
    else:
        ctx = f"ctx: ~?/{window_str} tok"
    return (
        f"session: <b>{sid}</b>{title_part}  |  {busy}  |  "
        f"{ctx}  |  "
        f"turns: {state.turns}  |  {plan_field}  |  "
        f"model: <ansibrightcyan>{_tb_escape(model_part)}</ansibrightcyan>  |  "
        f"effort: {_tb_escape(effort_part)}  |  "
        f"think: {think_part}"
    )


def _panel_tools(state: "State") -> str:
    active = state.active_tools
    if not active:
        return "tools: -"
    unique = sorted({t["name"] for t in active.values()})
    shown = ", ".join(unique[:5])
    if len(unique) > 5:
        shown += f", +{len(unique) - 5}"
    return f"tools: <b>{len(active)}</b> {shown}"


def _panel_bg(state: "State") -> str:
    bg = state.background_tasks
    return f"bg: <b>{len(bg)}</b>" if bg else "bg: -"


def _panel_todos(state: "State") -> str:
    todos = state.current_todos
    if not todos:
        return "todos: -"
    done = sum(1 for t in todos if t.get("status") == "completed")
    in_prog_label = ""
    for t in todos:
        if t.get("status") == "in_progress":
            label = (t.get("activeForm") or t.get("content") or "").strip()
            label = label.replace("\n", " ")
            if len(label) > 50:
                label = label[:49] + "…"
            in_prog_label = f" → {label}"
            break
    return f"todos: <b>{done}/{len(todos)}</b>{in_prog_label}"


def _tb_escape(s: str) -> str:
    """Escape text for prompt_toolkit HTML — only `<`, `>`, `&` need handling."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _grep_filter_suffix(inp: dict[str, Any]) -> str:
    """Format Grep's `glob` / `type` filters (either, both, or neither) as a
    trailing annotation like `(*.py)` or `(glob=*.py, type=py)`."""
    parts: list[str] = []
    g = inp.get("glob")
    t = inp.get("type")
    if g and t:
        parts.append(f"glob={_tb_escape(str(g))}")
        parts.append(f"type={_tb_escape(str(t))}")
    elif g:
        parts.append(_tb_escape(str(g)))
    elif t:
        parts.append(f"type={_tb_escape(str(t))}")
    return f" ({', '.join(parts)})" if parts else ""


def _describe_current_sub(info: dict[str, Any]) -> str:
    """One-liner describing what a sub-tool is currently doing. Used under a
    Task row to show the live cursor across the subagent's inner work."""
    name = info.get("name", "")
    inp = info.get("input") or {}
    if name == "Grep":
        pat = _tb_escape(str(inp.get("pattern", "")))
        path = _tb_escape(str(inp.get("path", ".")))
        return f"searching /<b>{pat}</b>/ in {path}{_grep_filter_suffix(inp)}"
    if name == "Glob":
        pat = _tb_escape(str(inp.get("pattern", "")))
        path = _tb_escape(str(inp.get("path", ".")))
        return f"globbing <b>{pat}</b> in {path}"
    if name == "Read":
        path = _tb_escape(str(inp.get("file_path", "?")))
        return f"reading <b>{path}</b>"
    if name == "WebFetch":
        url = _tb_escape(str(inp.get("url", "?")))
        return f"fetching <b>{url}</b>"
    if name == "WebSearch":
        q = _tb_escape(str(inp.get("query", "?")))
        return f"web-searching <b>{q}</b>"
    if name == "Bash":
        cmd = (inp.get("command", "") or "").splitlines()[0] if inp.get("command") else ""
        return f"running: <b>{_tb_escape(cmd[:80])}</b>"
    return f"running {_tb_escape(name)}"


_LIVE_TASKS_CAP = 20  # max top-level tasks shown in the panel before overflow

_PANEL_HEADER_WIDTH = 50  # total chars in section headers, padded with "-"


def _panel_header(title: str) -> str:
    """Center a section title inside a row of `-`s, padded to
    _PANEL_HEADER_WIDTH chars so every section header lines up."""
    return f" {title} ".center(_PANEL_HEADER_WIDTH, "-")


def _panel_live_tasks(state: "State") -> list[str]:
    """Dynamic rows: one block per in-flight *top-level* tool use. Task-tool
    rows get a second line showing the current sub-tool (resulting in a
    resizable live display). Returns [] when idle or when the user opted
    into the classic scrolling log via --inline-all-tools.

    Hard-capped at _LIVE_TASKS_CAP top-level entries so a runaway burst of
    concurrent tools can't push the prompt off-screen. Overflow shows a
    trailing `… +N more (/tasks for all)` line."""
    if state.inline_all_tools or not state.show_tasks_panel:
        return []
    lines: list[str] = []
    rendered_count = 0
    overflow = 0
    header_lines = [
        _panel_header("tasks"),
        "(`/tasks`: all this turn, `/show N`: full detail)",
    ]
    for tid, info in state.active_tools.items():
        if info.get("parent_id"):
            continue  # sub-tools get rendered under their parent Task row
        name = info.get("name", "?")
        # Early filter: tool types that never render in this panel regardless
        # of cap. Keeps the cap counter honest (it only counts entries that
        # would actually appear).
        if name in ("Bash", "TodoWrite", "BashOutput", "KillShell"):
            continue
        if name == "Edit" and state.show_edits != "off":
            continue
        if rendered_count >= _LIVE_TASKS_CAP:
            overflow += 1
            continue
        rendered_count += 1
        seq = info.get("seq", "?")
        inp = info.get("input") or {}
        if name == "Task":
            subtype = _tb_escape(str(inp.get("subagent_type", "?")))
            desc = _tb_escape(str(inp.get("description") or "").strip()[:60])
            trail = info.get("sub_trail") or []
            grep_calls = [
                s for s in trail
                if s["name"] == "Grep" and (s["input"] or {}).get("pattern")
            ]
            reads = [s for s in trail if s["name"] == "Read"]
            globs = [s for s in trail if s["name"] == "Glob"]
            head = (
                f'[<b>#{seq}</b>] <ansiyellow>task</ansiyellow> '
                f'<b>{subtype}</b>: "{desc}"'
            )
            if grep_calls:
                shown = grep_calls[:3]
                pat_str = " ".join(
                    f"/{_tb_escape(s['input'].get('pattern', ''))}/"
                    f"{_grep_filter_suffix(s['input'] or {})}"
                    for s in shown
                )
                extra = len(grep_calls) - len(shown)
                head += f"  patterns: {pat_str}"
                if extra > 0:
                    head += f" +{extra}"
            if globs:
                head += f"  globs: <b>{len(globs)}</b>"
            if reads:
                head += f"  reads: <b>{len(reads)}</b>"
            lines.append(head)
            cur_id = info.get("current_sub_id")
            cur = state.active_tools.get(cur_id) if cur_id else None
            if cur is not None:
                lines.append(f"  <ansibrightblack>→</ansibrightblack> {_describe_current_sub(cur)}")
            else:
                lines.append(
                    "  <ansibrightblack>→ (subagent thinking...)</ansibrightblack>"
                )
        elif name == "Grep":
            pat = _tb_escape(str(inp.get("pattern", "")))
            path = _tb_escape(str(inp.get("path", ".")))
            lines.append(
                f"[<b>#{seq}</b>] <ansiyellow>search</ansiyellow> "
                f"/<b>{pat}</b>/ in {path}{_grep_filter_suffix(inp)}"
            )
        elif name == "Glob":
            pat = _tb_escape(str(inp.get("pattern", "")))
            path = _tb_escape(str(inp.get("path", ".")))
            lines.append(
                f"[<b>#{seq}</b>] <ansiyellow>glob</ansiyellow> "
                f"<b>{pat}</b> in {path}"
            )
        elif name == "Read":
            path = _tb_escape(str(inp.get("file_path", "?")))
            lines.append(
                f"[<b>#{seq}</b>] <ansiyellow>read</ansiyellow> {path}"
            )
        elif name == "WebFetch":
            url = _tb_escape(str(inp.get("url", "?")))
            lines.append(
                f"[<b>#{seq}</b>] <ansiyellow>fetch</ansiyellow> {url}"
            )
        elif name == "WebSearch":
            q = _tb_escape(str(inp.get("query", "?")))
            lines.append(
                f"[<b>#{seq}</b>] <ansiyellow>web</ansiyellow> {q}"
            )
        elif name == "Edit":
            # Edit under --show-edits!=off is filtered out at the top of the
            # loop; only render here when show_edits == "off".
            path = _tb_escape(str(inp.get("file_path", "?")))
            tag = "edit-all" if inp.get("replace_all") else "edit"
            lines.append(
                f"[<b>#{seq}</b>] <ansiyellow>{tag}</ansiyellow> {path}"
            )
        elif name == "Write":
            path = _tb_escape(str(inp.get("file_path", "?")))
            lines.append(
                f"[<b>#{seq}</b>] <ansiyellow>write</ansiyellow> {path}"
            )
        elif name == "NotebookEdit":
            path = _tb_escape(str(inp.get("notebook_path", "?")))
            lines.append(
                f"[<b>#{seq}</b>] <ansiyellow>nb-edit</ansiyellow> {path}"
            )
        else:
            lines.append(
                f"[<b>#{seq}</b>] <ansiyellow>tool</ansiyellow> "
                f"{_tb_escape(name)}"
            )
    if overflow > 0:
        lines.append(
            f"<ansibrightblack>… +{overflow} more task"
            f"{'s' if overflow != 1 else ''} (/tasks for all)</ansibrightblack>"
        )
    # Prepend the two-line header only when we have rows to show — otherwise
    # the header alone would add a noisy empty section.
    if lines:
        lines[0:0] = header_lines
    return lines


# Default toolbar layout. To add a panel later: write a `_panel_xxx(state)`
# function and stick it in here.
_TOOLBAR_LAYOUT: list[list[PanelFn]] = [
    [_panel_session],
    [_panel_tools, _panel_bg, _panel_todos],
]


_LIVE_BG_CAP = 20  # max bg rows in the panel before overflow


def _panel_live_bg(state: "State") -> list[str]:
    """One row per currently-running background task (bash run_in_background
    shells + Task-tool subagents). Empty when nothing's running. Suppressed
    when --inline-all-tools is set since everything is scrolling already.
    Capped at _LIVE_BG_CAP rows; overflow shows a trailing counter."""
    if state.inline_all_tools or not state.show_bg_panel:
        return []
    bg = state.background_tasks
    if not bg:
        return []
    now = time.monotonic()
    out: list[str] = [
        _panel_header("background tasks"),
        "(`/bg`: list, `/bg N`: detail, `/bg N K`: tail K lines of output)",
    ]
    overflow = 0
    rendered = 0
    for tid, info in bg.items():
        if rendered >= _LIVE_BG_CAP:
            overflow += 1
            continue
        rendered += 1
        elapsed = now - info.get("started_at", now)
        task_type = _tb_escape(str(info.get("task_type", "?")))
        raw_name = str(info.get("name") or "(unnamed)").replace("\n", " ")
        if len(raw_name) > 40:
            raw_name = raw_name[:37] + "..."
        name = _tb_escape(raw_name)
        seq = info.get("seq")
        seq_tag = f"[<b>#{seq}</b>] " if isinstance(seq, int) else ""
        out.append(
            f"{seq_tag}<b>{task_type}</b>: {name} ({elapsed:.0f}s)"
        )
    if overflow > 0:
        out.append(
            f"… +{overflow} more bg task"
            f"{'s' if overflow != 1 else ''} (/bg for all)"
        )
    return out


def _render_toolbar(state: "State") -> str:
    # Fixed status rows at the top, dynamic panels below. The terminal's
    # very-bottom row is whatever panel row happens to be last; the fixed
    # session/tools rows stay visually anchored at the top of the toolbar
    # block.
    lines: list[str] = []
    for panels in _TOOLBAR_LAYOUT:
        parts = [p(state) for p in panels]
        parts = [s for s in parts if s]
        if parts:
            lines.append(" " + "  |  ".join(parts) + " ")
    for row in _panel_live_tasks(state):
        lines.append(" " + row + " ")
    for row in _panel_live_bg(state):
        lines.append(" " + row + " ")
    # Every user-provided cell (Task desc, Grep pattern, Bash command
    # preview, API-status description, session title, etc.) can carry an
    # embedded newline that would silently render as an extra toolbar row
    # for the duration of that event. Collapse internal \n/\r per row so
    # the toolbar height only ever grows by intentional rows.
    lines = [ln.replace("\r", " ").replace("\n", " ") for ln in lines]
    return "\n".join(lines)


def format_session_label(s: dict[str, Any], width: int = 100) -> str:
    """Compact one-line per-session label (no project info; the picker
    renders that as a group header)."""
    sid = s["session_id"][:8]
    age = _format_session_age(s.get("mtime") or 0.0)
    title = s.get("title")
    if title:
        msg = f"★ {title}"
    else:
        msg = (
            s.get("last_user_msg")
            or s.get("first_user_msg")
            or "(no user message)"
        ).strip()
    msg = msg.replace("\n", " ").replace("\r", " ")
    head = f"{sid} {age:>9}  "
    remaining = max(20, width - len(head))
    if len(msg) > remaining:
        msg = msg[: remaining - 1] + "…"
    return head + msg




class Orchestrator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        # 'auto' is the sentinel for "no override" — store as None so
        # _make_options simply omits the field.
        initial_effort = None if args.effort in (None, "auto") else args.effort
        sub = _detect_subscription()
        self.state = State(
            effort=initial_effort,
            model=args.model,
            is_subscription=sub,
            subscription_plan=_detect_subscription_plan() if sub else None,
            inline_all_tools=bool(getattr(args, "inline_all_tools", False)),
            show_edits=getattr(args, "show_edits", "off") or "off",
            show_thinking=bool(getattr(args, "show_thinking", False)),
            show_tasks_panel=bool(getattr(args, "tasks_panel", True)),
            show_bg_panel=bool(getattr(args, "bg_panel", True)),
        )
        self.event_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self.turn_msg_queue: asyncio.Queue[Any] = asyncio.Queue()
        self.turn_active = asyncio.Event()
        self.interrupt_event = asyncio.Event()
        self.stop_event = asyncio.Event()
        self.client: ClaudeSDKClient | None = None
        self.dispatcher_task: asyncio.Task[None] | None = None
        self.session: PromptSession | None = None
        self._mcp_servers: dict[str, Any] | None = self._load_mcp_config()
        # Set if --resume was passed; cleared after the first connect so that
        # /effort and /model reconnects fall back to state.session_id.
        self._initial_resume_id: str | None = None
        if args.resume and args.resume != _PICKER_SENTINEL:
            self._initial_resume_id = args.resume

    def _load_mcp_config(self) -> dict[str, Any] | None:
        """Load MCP server config from --mcp-config path, or auto-detect .mcp.json in cwd."""
        import json

        path: Path | None = None
        if self.args.mcp_config:
            path = Path(self.args.mcp_config).expanduser()
            if not path.exists():
                print(f"\033[31m[warn] --mcp-config not found: {path}\033[0m")
                return None
        else:
            candidate = Path(self.args.cwd) / ".mcp.json"
            if candidate.exists():
                path = candidate
        if path is None:
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"\033[31m[warn] failed to load MCP config {path}: {e}\033[0m")
            return None
        servers = data.get("mcpServers") if isinstance(data, dict) else None
        if not isinstance(servers, dict) or not servers:
            return None
        print(f"\033[35m[sys] loaded MCP servers from {path}: {list(servers)}\033[0m")
        return servers

    # ---- prompt / UI ---------------------------------------------------

    def _keybindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("c-c")
        def _(event):  # type: ignore[no-untyped-def]
            buf = event.app.current_buffer
            if self.state.busy:
                self.interrupt_event.set()
            elif buf.text:
                buf.reset()
            else:
                self.interrupt_event.set()
                self.stop_event.set()
                event.app.exit(exception=EOFError)

        @kb.add("c-d")
        def _(event):  # type: ignore[no-untyped-def]
            self.interrupt_event.set()
            self.stop_event.set()
            event.app.exit(exception=EOFError)

        # Multi-line input: Enter submits, Alt-Enter inserts a newline.
        @kb.add("enter")
        def _(event):  # type: ignore[no-untyped-def]
            event.current_buffer.validate_and_handle()

        @kb.add("escape", "enter")
        def _(event):  # type: ignore[no-untyped-def]
            event.current_buffer.insert_text("\n")

        return kb

    def _bottom_toolbar(self):  # returns HTML; multi-line per the layout
        return HTML(_render_toolbar(self.state))

    def _prompt_message(self):
        return HTML("<prompt>> </prompt>")

    def print_help(self) -> None:
        print()
        print("Commands:")
        print("  /help                           this help")
        print("  /status  /cost  /cwd            print session info, cost, usage")
        print("  /clear                          start a fresh session (wipes context)")
        print("  /cls                            clear the screen (keeps the session)")
        print("  /interrupt  /i                  stop the current turn (or press Ctrl-C)")
        print("  /compact                        force a /compact now")
        print(f"  /effort <level>                 one of {', '.join(EFFORT_CHOICES)}  (auto = no override)")
        print("  /model <name>                   e.g. claude-opus-4-6, claude-sonnet-4-6")
        print("  /rename <name>                  set a custom title for this session")
        print("  /auto [on|off|toggle]           enable/disable autonomous continue prompting")
        print("  /burst N [T]                    set continue-burst limit (and window seconds)")
        print("  /export [path]                  save the conversation as markdown")
        print("  /tools                          list active tool calls and background tasks")
        print("  /tasks                          list every task this turn (in-flight + completed)")
        print("  /bg  /background                list background shells / Task subagents still running")
        print("  /show [N ...]                   expand collapsed tool calls by their [#N] tag")
        print("  /think [N ...]                  show full thinking blocks by their [#N] tag")
        print("  /autocompact [on|off|N]         enable/disable/set auto-compact threshold")
        print("  /max-context [off|N]            cap context at N tokens (rolling-window trim)")
        print("  /todos  /plan                   show Claude's current TodoWrite plan")
        print("  /quit  /exit                    graceful exit (waits up to ~10s for CLI flush)")
        print("  /quit! /exit!                   force exit immediately (may lose last message)")
        print("Input: Enter submits.  Alt-Enter inserts a newline (multi-line input).")
        print("Anything else is sent to Claude as a message.")
        print()

    def clear_screen(self) -> None:
        sys.stdout.write("\033[2J\033[3J\033[H")
        sys.stdout.flush()
        print(f"\033[90m[screen cleared -- session {self.state.session_id} continues]\033[0m")

    def _check_api_stall(
        self, error_status: Any = None, error_info: Any = None
    ) -> None:
        """Called from both turn-loop and async-message dispatch when an
        api_retry arrives. Policy:

          1. Kick a rate-limited one-shot Statuspage check so the *first*
             error tries the status feed before waiting for a heuristic
             threshold.
          2. Track timestamps in a sliding window; if ≥ --api-stall-limit
             retries happen within --api-stall-window seconds, enter the
             stall regardless of what Statuspage says.

        Already-stalled → no-op (the periodic poller is handling recovery).
        """
        # Params accepted for forward-compat; currently every retry counts
        # (rate_limit errors included). Filtering was tried and reverted.
        del error_status, error_info
        if self.state.needs_user_attention == "api-error":
            return
        now = time.monotonic()
        self.state.api_retry_times.append(now)
        # Rate-limited one-shot status probe: at most once every 60s while
        # not stalled. Fires on the first non-rate-limit retry even before
        # the heuristic can cross its threshold.
        last = getattr(self, "_last_status_probe_at", 0.0)
        if now - last >= 60.0 and not self.args.no_status_poll:
            self._last_status_probe_at = now
            asyncio.create_task(
                self._one_shot_status_check(), name="status-probe"
            )
        # Heuristic threshold.
        limit = int(getattr(self.args, "api_stall_limit", 0) or 0)
        if limit <= 0:
            return
        window = float(getattr(self.args, "api_stall_window", 60.0) or 60.0)
        cutoff = now - window
        while (
            self.state.api_retry_times
            and self.state.api_retry_times[0] < cutoff
        ):
            self.state.api_retry_times.popleft()
        if len(self.state.api_retry_times) >= limit:
            self._enter_api_stall(
                f"{len(self.state.api_retry_times)} retries in "
                f"{window:.0f}s (heuristic; status page didn't flag it)",
                source="heuristic",
            )

    def _enter_api_stall(self, reason: str, source: str) -> None:
        """Flip into api-error state and start the recovery poller.
        `source` must be "status" (status feed flagged it) or "heuristic"
        (retry density threshold tripped while status looked clean).
        Idempotent."""
        if self.state.needs_user_attention == "api-error":
            return
        self.state.needs_user_attention = "api-error"
        self.state.api_stall_source = source
        # For status-sourced stalls, we've already seen "bad" — the current
        # indicator *is* the signal we're waiting to clear. For heuristic
        # stalls, status was clean, so we require it to go bad before we
        # can trust a return-to-clean as a recovery signal.
        self.state.api_stall_saw_bad = source == "status"
        print(f"\033[31m[API STALL ({source}) -- {reason}]\033[0m")
        if not self.args.no_status_poll:
            self._ensure_status_poller()
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()

    async def _one_shot_status_check(self) -> None:
        """Single fetch of the Statuspage feed. If it reports an active
        problem with the Claude API or Claude Code components, enter the
        stall immediately; otherwise just log what we saw and let the
        heuristic handle the next steps."""
        url = getattr(self.args, "status_url", "") or ""
        if not url:
            return
        import urllib.request
        import urllib.error
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "claude-orchestrator"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
        except (urllib.error.URLError, OSError, ValueError) as e:
            print(f"\033[90m[status-probe] fetch failed: {e}\033[0m")
            return
        indicator, bad = self._status_summary(data)
        self.state.api_status_indicator = indicator
        self.state.api_status_description = (
            (data.get("status") or {}).get("description")
            if isinstance(data, dict)
            else None
        )
        if indicator and indicator != "none":
            desc = self.state.api_status_description or indicator
            self._enter_api_stall(
                f"Anthropic status: {desc}", source="status"
            )
            return
        if bad:
            names = ", ".join(
                (c.get("name") or "?") + "=" + (c.get("status") or "?")
                for c in bad[:3]
            )
            self._enter_api_stall(
                f"components degraded: {names}", source="status"
            )
            return
        print(
            "\033[90m[status-probe] Anthropic status is clean; treating "
            "the retry as transient (heuristic still armed)\033[0m"
        )

    @staticmethod
    def _status_summary(
        data: Any,
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Extract (top_level_indicator, non_operational_watched_components)
        from a Statuspage summary.json. Watched components are matched by
        name (case-insensitive "claude api" / "claude code")."""
        if not isinstance(data, dict):
            return None, []
        indicator = (data.get("status") or {}).get("indicator")
        components = data.get("components") or []
        watched = [
            c for c in components
            if isinstance(c, dict) and any(
                key in (c.get("name") or "").lower()
                for key in ("claude api", "claude code")
            )
        ]
        bad = [
            c for c in watched
            if c.get("status") not in (None, "operational")
        ]
        return indicator, bad

    def _ensure_status_poller(self) -> None:
        """Start a background asyncio task that hits the Statuspage feed
        and clears api-error stall when the service returns to operational.
        Idempotent."""
        existing = getattr(self, "_status_task", None)
        if existing is not None and not existing.done():
            return
        self._status_task = asyncio.create_task(
            self._status_poller_loop(), name="status-poller"
        )

    async def _status_poller_loop(self) -> None:
        interval = float(getattr(self.args, "status_poll_interval", 30.0) or 30.0)
        url = getattr(self.args, "status_url", "")
        if not url:
            return
        import urllib.request
        import urllib.error
        print(
            f"\033[36m[status-poll] watching {url} every "
            f"{interval:.0f}s; will auto-resume when operational\033[0m"
        )
        while self.state.needs_user_attention == "api-error":
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "claude-orchestrator"}
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8", "replace"))
            except (urllib.error.URLError, OSError, ValueError) as e:
                print(f"\033[90m[status-poll] fetch failed: {e}\033[0m")
                await asyncio.sleep(interval)
                continue
            indicator, bad = self._status_summary(data)
            desc = (
                (data.get("status") or {}).get("description")
                if isinstance(data, dict)
                else None
            )
            self.state.api_status_indicator = indicator
            self.state.api_status_description = desc
            status_clean = indicator in ("none",) and not bad
            status_bad = (indicator and indicator != "none") or bad
            # Latch once we've seen bad. For heuristic stalls, this is the
            # gate that unlocks auto-resume — a heuristic stall with status
            # that never goes bad stays stalled until the user resumes.
            if status_bad:
                self.state.api_stall_saw_bad = True
            if status_clean and self.state.api_stall_saw_bad:
                recent_window = float(
                    getattr(self.args, "api_stall_window", 60.0) or 60.0
                )
                cutoff = time.monotonic() - recent_window
                while (
                    self.state.api_retry_times
                    and self.state.api_retry_times[0] < cutoff
                ):
                    self.state.api_retry_times.popleft()
                if self.state.api_retry_times:
                    print(
                        "\033[36m[status-poll] status indicator clear, but "
                        "retries still in window; waiting another tick\033[0m"
                    )
                else:
                    self.state.needs_user_attention = None
                    self.state.api_stall_source = None
                    self.state.api_stall_saw_bad = False
                    print(
                        "\033[32m[status-poll] Anthropic services operational "
                        "-- resuming]\033[0m"
                    )
                    sys.stdout.write("\a")
                    sys.stdout.flush()
                    try:
                        self.event_queue.put_nowait(
                            ("wakeup", "api-status-recovered")
                        )
                    except asyncio.QueueFull:
                        pass
                    return
            elif status_clean:
                # Status is clean but we never saw it flip bad — must be a
                # heuristic stall that the status page never acknowledged.
                # Keep waiting; /q or manual input still lets you resume.
                print(
                    "\033[90m[status-poll] status clean (heuristic stall) "
                    "-- waiting for status to acknowledge the issue; "
                    "type to resume manually\033[0m"
                )
            else:
                bits = [f"indicator={indicator}"]
                if bad:
                    names = ", ".join(
                        (c.get("name") or "?") + "=" + (c.get("status") or "?")
                        for c in bad[:3]
                    )
                    bits.append(f"components: {names}")
                print(f"\033[90m[status-poll] {'; '.join(bits)}\033[0m")
            await asyncio.sleep(interval)

    async def _maybe_trim_context(self) -> bool:
        """If --max-context-tokens is set and current context exceeds it,
        rewrite the session JSONL to a trimmed copy and reconnect with
        resume=<new-id>. Returns True when a trim happened."""
        cap = getattr(self.args, "max_context_tokens", 0) or 0
        if cap <= 0 or not self.state.session_id:
            return False
        if self.state.context_tokens <= cap:
            return False
        project_dir = project_dir_for_cwd(self.args.cwd)
        # Target 85% of cap so there's headroom for the next turn's reply +
        # any output tokens before the next trim check fires.
        target = int(cap * 0.85)
        old_sid = self.state.session_id
        new_id = _trim_session(old_sid, project_dir, target)
        if new_id is None:
            print(
                f"\033[33m[sys] max-context trim skipped -- "
                f"not enough turns to cut (ctx={self.state.context_tokens}, "
                f"cap={cap})\033[0m"
            )
            return False
        print(
            f"\033[35m[sys] max-context: ctx ~{self.state.context_tokens} tok "
            f"> cap {cap} -- trimmed to {new_id[:8]} (was {old_sid[:8]})\033[0m"
        )
        await self._disconnect()
        self.state.session_id = new_id
        self.state.context_tokens = 0  # refreshes on next ResultMessage
        self.state.active_tools.clear()
        self.state.background_tasks.clear()
        self._initial_resume_id = None
        await self._connect(resume_id=new_id)
        return True

    async def clear_context(self) -> None:
        """Start a fresh session — equivalent to Claude Code's /clear.

        Disconnects, wipes in-memory session state (tokens, cost, turns,
        tool/thinking history, todos, active tools), and reconnects without
        resume/continue. The old session's JSONL stays on disk."""
        old_sid = self.state.session_id
        await self._disconnect()
        self.state.session_id = None
        self.state.session_title = None
        self.state.context_tokens = 0
        self.state.turns = 0
        self.state.total_cost_usd = 0.0
        self.state.last_usage = {}
        self.state.last_result_subtype = None
        self.state.last_compact_trigger = None
        self.state.compact_during_last_turn = False
        self.state.needs_user_attention = None
        self.state.recent_turn_ends.clear()
        self.state.active_tools.clear()
        self.state.background_tasks.clear()
        self.state.tool_history.clear()
        self.state.next_tool_seq = 1
        self.state.thinking_history.clear()
        self.state.next_thinking_seq = 1
        self.state.current_todos = []
        # Block _make_options from resuming or continuing on the next connect.
        self._initial_resume_id = None
        prev_no_continue = self.args.no_continue
        self.args.no_continue = True
        try:
            await self._connect()
        finally:
            self.args.no_continue = prev_no_continue
        old = f"(was {old_sid[:8]})" if old_sid else ""
        print(f"\033[35m[sys] context cleared -- fresh session {old}\033[0m")

    def toggle_auto_continue(self, payload: str) -> None:
        if payload == "on":
            self.args.auto_continue = True
        elif payload == "off":
            self.args.auto_continue = False
        else:
            self.args.auto_continue = not self.args.auto_continue
        if self.args.auto_continue:
            print(
                f"\033[35m[auto-continue ON  "
                f"(delay {self.args.continue_response_delay}s, "
                f"burst {self.args.continue_burst_limit}/"
                f"{self.args.continue_burst_window:.0f}s)]\033[0m"
            )
        else:
            self.state.recent_turn_ends.clear()
            print(
                "\033[35m[auto-continue OFF -- orchestrator will wait for "
                "your input after each turn]\033[0m"
            )

    def set_burst(self, payload: str) -> None:
        payload = payload.strip()
        if not payload:
            print(
                f"\033[35m[burst: limit={self.args.continue_burst_limit}, "
                f"window={self.args.continue_burst_window:.0f}s "
                f"(only used while --auto-continue is on)]\033[0m"
            )
            return
        parts = payload.split()
        try:
            n = int(parts[0])
            t = float(parts[1]) if len(parts) > 1 else None
        except (ValueError, IndexError):
            print("\033[31m[error: usage /burst N [T-seconds]]\033[0m")
            return
        if n < 0 or (t is not None and t <= 0):
            print("\033[31m[error: N must be >= 0, T must be > 0]\033[0m")
            return
        self.args.continue_burst_limit = n
        if t is not None:
            self.args.continue_burst_window = t
        self.state.recent_turn_ends.clear()
        print(
            f"\033[35m[burst: limit={n}, "
            f"window={self.args.continue_burst_window:.0f}s]\033[0m"
        )

    def set_autocompact(self, payload: str) -> None:
        p = payload.strip().lower()
        if not p:
            status = "OFF" if self.args.no_compact else f"ON (at ~{self.args.compact_at} tok)"
            print(f"\033[35m[sys] auto-compact: {status}]\033[0m")
            return
        if p in ("on", "true", "enable", "1"):
            self.args.no_compact = False
            print(
                f"\033[35m[sys] auto-compact ON "
                f"(at ~{self.args.compact_at} tok)\033[0m"
            )
            return
        if p in ("off", "false", "disable", "0"):
            self.args.no_compact = True
            print("\033[35m[sys] auto-compact OFF\033[0m")
            return
        try:
            n = int(p.replace(",", "").replace("_", ""))
        except ValueError:
            print(
                "\033[31m[error: usage /autocompact [on|off|N] where N is a "
                "token count]\033[0m"
            )
            return
        if n <= 0:
            print("\033[31m[error: /autocompact N must be positive]\033[0m")
            return
        self.args.compact_at = n
        self.args.no_compact = False
        print(f"\033[35m[sys] auto-compact threshold -> ~{n} tok (ON)\033[0m")

    def set_max_context(self, payload: str) -> None:
        p = payload.strip().lower()
        if not p:
            cur = self.args.max_context_tokens
            status = "unlimited" if not cur else f"~{cur} tok"
            print(f"\033[35m[sys] max context: {status}\033[0m")
            return
        if p in ("off", "none", "unlimited", "0"):
            self.args.max_context_tokens = 0
            print("\033[35m[sys] max context: unlimited\033[0m")
            return
        try:
            n = int(p.replace(",", "").replace("_", ""))
        except ValueError:
            print(
                "\033[31m[error: usage /max-context [off|N] where N is a "
                "token count]\033[0m"
            )
            return
        if n <= 0:
            print("\033[31m[error: /max-context N must be positive]\033[0m")
            return
        self.args.max_context_tokens = n
        print(f"\033[35m[sys] max context -> ~{n} tok (rolling-window trim)\033[0m")

    def show_thinking_detail(self, payload: str) -> None:
        history = self.state.thinking_history
        payload = payload.strip()
        if not payload:
            recent = list(history)[-3:]
            if not recent:
                print("\033[90m[no thinking blocks yet]\033[0m")
                return
            print(
                f"\033[90m[showing last {len(recent)} thinking block(s); "
                f"/think <N> [N2 ...] to pick specific ones]\033[0m"
            )
            for e in recent:
                self._print_thinking_entry(e)
            return
        try:
            nums = [int(p) for p in payload.split()]
        except ValueError:
            print(
                "\033[31m[error: usage /think [<N> ...] -- numbers come from "
                "the [#N] tags shown with each thinking preview]\033[0m"
            )
            return
        by_seq = {e["seq"]: e for e in history}
        for n in nums:
            entry = by_seq.get(n)
            if entry is None:
                print(
                    f"\033[33m[#{n} not found "
                    f"(history kept = last {history.maxlen} blocks)]\033[0m"
                )
            else:
                self._print_thinking_entry(entry)

    def _print_thinking_entry(self, e: dict[str, Any]) -> None:
        seq = e.get("seq", "?")
        text = (e.get("text") or "").strip()
        print()
        print(f"\033[1m[#{seq}] thinking\033[0m")
        if not text:
            print("  \033[90m(empty)\033[0m")
            return
        for ln in text.splitlines():
            print(f"  \033[90m{ln}\033[0m")

    def show_tool_detail(self, payload: str) -> None:
        """Usage:
          /show                 -- last 5 tool history entries, each full
          /show N               -- entry N, full input + full output
          /show N -K            -- entry N, full input + *last K lines* of output
          /show N1 N2 N3 ...    -- entries N1, N2, N3, ..., each full
        Positive ints are entry seqs; negative ints are tail-line counts
        (must follow a seq). `/show 19 18` means two entries; `/show 19
        -18` means entry 19 with the last 18 output lines."""
        history = self.state.tool_history
        payload = payload.strip()
        if not payload:
            recent = list(history)[-5:]
            if not recent:
                print("\033[90m[no tool history yet]\033[0m")
                return
            print(
                f"\033[90m[showing last {len(recent)} tool call(s); "
                f"/show N -- full; /show N -K -- tail K lines of output]\033[0m"
            )
            for e in recent:
                self._print_tool_history_entry(e)
            return
        try:
            nums = [int(p) for p in payload.split()]
        except ValueError:
            print(
                "\033[31m[error: usage /show [N [-K]] or /show N1 N2 N3 ... "
                "-- positive ints are entry seqs, -K (negative) is "
                "tail lines]\033[0m"
            )
            return
        by_seq = {e["seq"]: e for e in history}
        # Walk the list: each positive int picks an entry; a negative int
        # immediately after a positive one is that entry's tail count.
        i = 0
        while i < len(nums):
            n = nums[i]
            if n < 0:
                print(
                    f"\033[31m[error: standalone negative {n} — "
                    f"tail must follow an entry number]\033[0m"
                )
                return
            tail_k: int | None = None
            if i + 1 < len(nums) and nums[i + 1] < 0:
                tail_k = -nums[i + 1]
                i += 2
            else:
                i += 1
            entry = by_seq.get(n)
            if entry is None:
                print(
                    f"\033[33m[#{n} not found "
                    f"(history kept = last {history.maxlen} calls)]\033[0m"
                )
                continue
            self._print_tool_history_entry(entry, tail=tail_k)

    def _print_tool_history_entry(
        self, e: dict[str, Any], tail: int | None = None
    ) -> None:
        seq = e.get("seq", "?")
        name = e.get("name", "?")
        inp = e.get("input") or {}
        print()
        print(f"\033[1m[#{seq}] tool {name}\033[0m")
        if name == "Bash":
            cmd = inp.get("command", "") or ""
            desc = inp.get("description", "")
            bg = bool(inp.get("run_in_background"))
            if desc:
                print(f"  \033[90m{desc}\033[0m")
            if bg:
                print("  \033[90m(background)\033[0m")
            for ln in cmd.splitlines() or [cmd]:
                print(f"  \033[36m$\033[0m {ln}")
        elif name == "WebFetch":
            print(f"  URL: {inp.get('url', '?')}")
            if inp.get("prompt"):
                print(f"  Prompt: {inp['prompt']}")
        elif name == "WebSearch":
            print(f"  Query: {inp.get('query', '')!r}")
        elif name == "Read":
            print(f"  Path: {inp.get('file_path', '?')}")
            offset, limit = inp.get("offset"), inp.get("limit")
            if offset is not None or limit is not None:
                print(f"  Range: offset={offset} limit={limit}")
        elif name == "Edit":
            print(f"  Path: {inp.get('file_path', '?')}")
            old = inp.get("old_string", "") or ""
            new = inp.get("new_string", "") or ""
            for ln in old.splitlines():
                print(f"  \033[31m-\033[0m {ln}")
            for ln in new.splitlines():
                print(f"  \033[32m+\033[0m {ln}")
        elif name == "Write":
            print(f"  Path: {inp.get('file_path', '?')}")
            content = inp.get("content", "") or ""
            for ln in content.splitlines():
                print(f"  \033[32m+\033[0m {ln}")
        else:
            try:
                rendered = json.dumps(inp, indent=2, default=str)
            except (TypeError, ValueError):
                rendered = repr(inp)
            for ln in rendered.splitlines():
                print(f"  {ln}")
        result = e.get("result_text")
        is_err = e.get("is_error")
        if result is None:
            print("\n  \033[90m(still running)\033[0m")
        else:
            tag = (
                "\033[31mtool-err:\033[0m"
                if is_err
                else "\033[34mresult:\033[0m"
            )
            result_lines = result.splitlines() or [result]
            total = len(result_lines)
            if tail is not None and tail > 0 and total > tail:
                tail_head = (
                    f"\n  {tag} \033[90m(tail -{tail}, showing last {tail} "
                    f"of {total} lines)\033[0m"
                )
                print(tail_head)
                for ln in result_lines[-tail:]:
                    print(f"    {ln}")
            else:
                print(f"\n  {tag}")
                for ln in result_lines:
                    print(f"    {ln}")
        print()

    def show_todos(self) -> None:
        todos = self.state.current_todos
        print()
        if not todos:
            print("\033[90mNo todos yet (Claude hasn't called TodoWrite).\033[0m")
            print()
            return
        done = sum(1 for t in todos if t.get("status") == "completed")
        in_prog = sum(1 for t in todos if t.get("status") == "in_progress")
        pending = sum(1 for t in todos if t.get("status") == "pending")
        print(
            f"\033[1mClaude's plan\033[0m  "
            f"\033[32m{done} done\033[0m / "
            f"\033[33m{in_prog} in-progress\033[0m / "
            f"\033[90m{pending} pending\033[0m  "
            f"({len(todos)} total)"
        )
        markers = {
            "completed": "\033[32m✓\033[0m",
            "in_progress": "\033[33m→\033[0m",
            "pending": "\033[90m·\033[0m",
        }
        for t in todos:
            status = t.get("status", "pending")
            m = markers.get(status, "?")
            content = (
                t.get("content")
                or t.get("activeForm")
                or "(no description)"
            ).strip()
            line_color = (
                "\033[90m" if status == "completed" else
                "\033[33m" if status == "in_progress" else
                ""
            )
            print(f"  {m} {line_color}{content}\033[0m")
        print()

    def show_tasks(self) -> None:
        """List every non-Bash tool that ran (or is running) during the
        current / most-recent turn — the full set of rows that appeared in
        the live panel this turn, with each one's final status. Cleared at
        the start of each new turn, so between turns it still shows the
        turn that just ended."""
        seqs = self.state.current_turn_tool_seqs
        if not seqs:
            print(
                "\033[90m[no tasks this turn -- /tools shows in-flight "
                "state; /show N expands any completed tool]\033[0m"
            )
            return
        # Index tool_history by seq so we can look up status + input per row.
        by_seq: dict[int, dict[str, Any]] = {
            h["seq"]: h for h in self.state.tool_history
        }
        now = time.monotonic()
        print()
        print(
            f"\033[1mTasks this turn\033[0m ({len(seqs)}); "
            f"\033[90m/show N for full detail\033[0m"
        )
        for seq in seqs:
            h = by_seq.get(seq)
            if h is None:
                print(f"  \033[90m[#{seq}] (history evicted)\033[0m")
                continue
            name = h.get("name", "?")
            inp = h.get("input") or {}
            ended = h.get("ended_at")
            is_err = h.get("is_error")
            if ended is None:
                elapsed = now - h.get("started_at", now)
                status = f"\033[33m… running {elapsed:.1f}s\033[0m"
            elif is_err:
                status = "\033[31m✗ error\033[0m"
            else:
                dur = ended - h.get("started_at", ended)
                status = f"\033[32m✓\033[0m \033[90m{dur:.1f}s\033[0m"
            summary = _task_summary_line(name, inp)
            print(f"  [\033[90m#{seq}\033[0m] {status}  \033[34m{name}\033[0m  {summary}")
        print()

    def _bg_entry_index(self) -> dict[int, tuple[str, dict[str, Any]]]:
        """Build a seq → (task_id, entry) map across both trackers.
        Entries from current_turn_bg win (they may already be marked
        completed); running tasks started in prior turns are injected
        with a carryover flag for display."""
        index: dict[int, tuple[str, dict[str, Any]]] = {}
        for tid, info in self.state.current_turn_bg.items():
            seq = info.get("seq")
            if isinstance(seq, int):
                index[seq] = (tid, info)
        for tid, info in self.state.background_tasks.items():
            seq = info.get("seq")
            if not isinstance(seq, int) or seq in index:
                continue
            index[seq] = (
                tid,
                {
                    "seq": seq,
                    "name": info.get("name"),
                    "task_type": info.get("task_type"),
                    "started_at": info.get("started_at"),
                    "tool_use_id": info.get("tool_use_id"),
                    "ended_at": None,
                    "status": None,
                    "summary": None,
                    "output_file": None,
                    "carryover": True,
                },
            )
        return index

    def show_bg_tasks(self, payload: str = "") -> None:
        """`/bg` — one-liner summary of every bg task relevant to this turn
        (started this turn + any carryover still running).
        `/bg N` — full detail for task `[#N]`, including output_file and
                  any stored summary.
        `/bg N K` — same, plus tail the last K lines of the output_file."""
        parts = payload.split()
        index = self._bg_entry_index()
        if not parts:
            self._print_bg_summary(index)
            return
        try:
            n = int(parts[0])
        except ValueError:
            print(
                "\033[31m[error: usage /bg [N [K]] — N is the bg task "
                "number, K is lines to tail from output_file]\033[0m"
            )
            return
        tail_lines: int | None = None
        if len(parts) > 1:
            try:
                tail_lines = int(parts[1])
            except ValueError:
                print("\033[31m[error: K (tail lines) must be an integer]\033[0m")
                return
        match = index.get(n)
        if match is None:
            print(f"\033[33m[bg #{n} not found]\033[0m")
            return
        self._print_bg_detail(match[0], match[1], tail_lines)

    def _print_bg_summary(self, index: dict[int, tuple[str, dict[str, Any]]]) -> None:
        now = time.monotonic()
        print()
        if not index:
            print("\033[1mBackground tasks\033[0m: \033[90mnone\033[0m")
            print()
            return
        running = sum(
            1 for _, info in index.values() if info.get("ended_at") is None
        )
        done = len(index) - running
        print(
            f"\033[1mBackground tasks\033[0m ({len(index)}): "
            f"\033[33m{running} running\033[0m, "
            f"\033[32m{done} completed\033[0m  "
            f"\033[90m/bg N for detail, /bg N K to tail K lines\033[0m"
        )
        for seq in sorted(index):
            tid, info = index[seq]
            task_type = info.get("task_type", "?")
            name = info.get("name") or "(unnamed)"
            started = info.get("started_at", now)
            ended = info.get("ended_at")
            if ended is None:
                elapsed = now - started
                status = f"\033[33m… {elapsed:.0f}s\033[0m"
            else:
                dur = ended - started
                st = info.get("status") or "completed"
                marker = {
                    "completed": "\033[32m✓\033[0m",
                    "failed": "\033[31m✗\033[0m",
                    "stopped": "\033[33m⏹\033[0m",
                }.get(st, f"\033[90m[{st}]\033[0m")
                status = f"{marker} \033[90m{dur:.0f}s\033[0m"
            carry = " \033[90m(carryover)\033[0m" if info.get("carryover") else ""
            short_name = name.replace("\n", " ")
            if len(short_name) > 60:
                short_name = short_name[:57] + "..."
            print(
                f"  [\033[90m#{seq}\033[0m] {status}  "
                f"\033[34m{task_type}\033[0m: {short_name}  "
                f"\033[35m{tid[:8]}\033[0m{carry}"
            )
        print()

    def _print_bg_detail(
        self,
        tid: str,
        info: dict[str, Any],
        tail_lines: int | None,
    ) -> None:
        seq = info.get("seq", "?")
        task_type = info.get("task_type", "?")
        name = info.get("name") or "(unnamed)"
        started = info.get("started_at")
        ended = info.get("ended_at")
        now = time.monotonic()
        print()
        print(f"\033[1m[#{seq}] {task_type}: {name}\033[0m")
        print(f"  task_id: \033[35m{tid}\033[0m")
        tu_id = info.get("tool_use_id")
        if tu_id:
            print(f"  tool_use_id: \033[90m{tu_id}\033[0m")
            # Cross-ref tool_history for the original tool input (the
            # full Bash command, or the Task's prompt/description).
            for h in self.state.tool_history:
                if h.get("tool_use_id") == tu_id:
                    t_seq = h.get("seq")
                    t_name = h.get("name")
                    inp = h.get("input") or {}
                    print(
                        f"  originating tool: [#{t_seq}] {t_name} "
                        f"(see /show {t_seq})"
                    )
                    if t_name == "Bash":
                        cmd = inp.get("command", "") or ""
                        for ln in cmd.splitlines() or [cmd]:
                            print(f"    \033[36m$\033[0m {ln}")
                    elif t_name == "Task":
                        desc = inp.get("description", "")
                        prompt = inp.get("prompt", "")
                        sub = inp.get("subagent_type", "?")
                        print(f"    subagent: {sub}")
                        if desc:
                            print(f"    desc: {desc}")
                        if prompt:
                            head = prompt.strip().splitlines()[0][:200]
                            print(f"    prompt: {head}")
                    break
        if started is not None:
            if ended is None:
                print(f"  status: \033[33mrunning {now - started:.1f}s\033[0m")
            else:
                st = info.get("status") or "completed"
                print(
                    f"  status: \033[34m{st}\033[0m "
                    f"\033[90m({ended - started:.1f}s)\033[0m"
                )
        usage = info.get("usage") or {}
        if usage:
            dur = usage.get("duration_ms")
            dur_s = f" {dur / 1000:.1f}s" if isinstance(dur, (int, float)) else ""
            print(
                f"  usage: {usage.get('total_tokens', '?')} tok, "
                f"{usage.get('tool_uses', '?')} tool uses{dur_s}"
            )
        summary = info.get("summary")
        if summary:
            print("  summary:")
            for ln in summary.strip().splitlines():
                print(f"    {ln}")
        out_file = info.get("output_file")
        if out_file:
            print(f"  output_file: \033[36m{out_file}\033[0m")
            if tail_lines is not None and tail_lines > 0:
                self._print_output_tail(out_file, tail_lines)
        elif tail_lines is not None:
            print("  \033[33m[no output_file recorded for this task]\033[0m")
        print()

    def _print_output_tail(self, path: str, n: int) -> None:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError as e:
            print(f"  \033[31m[tail failed: {e}]\033[0m")
            return
        tail = lines[-n:] if len(lines) > n else lines
        total = len(lines)
        print(
            f"  \033[1mtail\033[0m -{n}  "
            f"\033[90m(showing {len(tail)} of {total} lines)\033[0m"
        )
        for ln in tail:
            print(f"    {ln.rstrip()}")

    def show_tools(self) -> None:
        active = self.state.active_tools
        bg = self.state.background_tasks
        now = time.monotonic()
        print()
        if active:
            print(f"\033[1mActive tool calls\033[0m ({len(active)}):")
            for tu_id, info in active.items():
                elapsed = now - info.get("started_at", now)
                name = info.get("name", "?")
                inp = info.get("input", {}) or {}
                seq = info.get("seq")
                seq_tag = f" \033[90m[#{seq}]\033[0m" if seq is not None else ""
                if name == "Bash":
                    cmd = inp.get("command", "") or ""
                    bg_tag = " [background]" if inp.get("run_in_background") else ""
                    desc = inp.get("description", "")
                    head = (
                        f"  \033[34m[{tu_id[:8]}]\033[0m{seq_tag} Bash{bg_tag}  "
                        f"\033[90m-- {elapsed:.1f}s\033[0m"
                    )
                    if desc:
                        head += f"\n    \033[90m{desc}\033[0m"
                    print(head)
                    for ln in cmd.splitlines() or [cmd]:
                        print(f"    \033[36m$\033[0m {ln}")
                elif name == "WebFetch":
                    url = inp.get("url", "")
                    print(
                        f"  \033[34m[{tu_id[:8]}]\033[0m{seq_tag} WebFetch  "
                        f"\033[90m-- {elapsed:.1f}s\033[0m"
                    )
                    print(f"    \033[36m→\033[0m {url}")
                    prompt_str = inp.get("prompt", "")
                    if prompt_str:
                        print(f"    \033[90m{prompt_str}\033[0m")
                elif name == "WebSearch":
                    q = inp.get("query", "")
                    print(
                        f"  \033[34m[{tu_id[:8]}]\033[0m{seq_tag} WebSearch  "
                        f"\033[90m-- {elapsed:.1f}s\033[0m"
                    )
                    print(f"    \033[36m?\033[0m {q!r}")
                else:
                    inp_brief = brief_args(inp, limit=200)
                    print(
                        f"  \033[34m[{tu_id[:8]}]\033[0m{seq_tag} {name}({inp_brief})  "
                        f"\033[90m-- {elapsed:.1f}s\033[0m"
                    )
        else:
            print("Active tool calls: \033[90mnone\033[0m")
        print()
        if bg:
            print(f"\033[1mBackground tasks\033[0m ({len(bg)}):")
            for tid, info in bg.items():
                elapsed = now - info.get("started_at", now)
                print(
                    f"  \033[35m[{tid[:8]}]\033[0m "
                    f"{info.get('task_type', '?')}: "
                    f"{info.get('name') or '(unnamed)'}  "
                    f"\033[90m-- running {elapsed:.1f}s\033[0m"
                )
        else:
            print("Background tasks: \033[90mnone\033[0m")
        print()

    def export_session(self, path_arg: str) -> None:
        sid = self.state.session_id
        if not sid:
            print(
                "\033[33m[no active session id yet -- /export needs at least "
                "one completed turn]\033[0m"
            )
            return
        project_dir = _find_session_dir(sid)
        if project_dir is None:
            print(
                f"\033[31m[session {sid[:8]} not found on disk; can't export]\033[0m"
            )
            return
        jsonl = project_dir / f"{sid}.jsonl"
        if not jsonl.exists():
            print(f"\033[31m[session file not found: {jsonl}]\033[0m")
            return

        out_path: Path
        path_arg = path_arg.strip()
        if path_arg:
            out_path = Path(path_arg).expanduser()
            if out_path.is_dir():
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                out_path = out_path / f"claude-{sid[:8]}-{stamp}.md"
        else:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            out_path = Path(self.args.cwd) / f"claude-{sid[:8]}-{stamp}.md"

        try:
            markdown = _render_session_markdown(jsonl)
        except Exception as e:  # noqa: BLE001
            print(f"\033[31m[export failed while rendering: {e}]\033[0m")
            return
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(markdown, encoding="utf-8")
        except OSError as e:
            print(f"\033[31m[export failed while writing: {e}]\033[0m")
            return
        print(
            f"\033[35m[exported -> {out_path.resolve()} "
            f"({out_path.stat().st_size} bytes)]\033[0m"
        )

    def rename_session(self, new_title: str) -> None:
        sid = self.state.session_id
        if not sid:
            print(
                "\033[33m[no active session id yet -- run /rename after the "
                "first turn so the session has been created]\033[0m"
            )
            return
        new_title = new_title.strip()
        if not new_title:
            current = _read_session_title(sid)
            if current:
                print(f"\033[35m[current title: {current}]\033[0m")
            else:
                print("\033[33m[no title set; usage: /rename <name>]\033[0m")
            return
        duplicates = _find_sessions_with_title(new_title, exclude_id=sid)
        if duplicates:
            print(
                f"\033[33m[warning: title '{new_title}' is already used by "
                f"{len(duplicates)} other session(s):]\033[0m"
            )
            for d in duplicates[:5]:
                print(f"    \033[90m{d[:8]}\033[0m")
            if len(duplicates) > 5:
                print(f"    \033[90m... +{len(duplicates) - 5} more\033[0m")
            print(
                "\033[33m  (renaming anyway; the picker shows session ids "
                "alongside titles so you can still tell them apart)\033[0m"
            )
        try:
            _write_session_title(sid, new_title)
        except (OSError, ValueError) as e:
            print(f"\033[31m[rename failed: {e}]\033[0m")
            return
        self.state.session_title = new_title
        print(f"\033[35m[renamed session {sid[:8]} -> '{new_title}']\033[0m")

    def print_status(self) -> None:
        print()
        print(f"  session id   : {self.state.session_id}")
        print(f"  session name : {self.state.session_title or '(unnamed)'}")
        if self.state.session_id:
            disk_title = _read_session_title(self.state.session_id)
            proj = _find_session_dir(self.state.session_id)
            print(f"  title on disk: {disk_title or '(none)'}")
            print(f"  project dir  : {proj or '(not found on disk)'}")
        # Cwd resolution diagnostics
        cwd_proj = find_project_for_cwd(self.args.cwd)
        cwd_expected = project_dir_for_cwd(self.args.cwd)
        print(f"  cwd          : {Path(self.args.cwd).resolve(strict=False)}")
        print(f"  cwd -> proj  : {cwd_proj or '(none on disk)'}")
        if cwd_proj is not None and cwd_proj != cwd_expected:
            print(
                f"  expected at  : {cwd_expected}  "
                f"\033[33m(mismatch — case/normalization differs)\033[0m"
            )
        print(f"  turns        : {self.state.turns}")
        print(f"  context      : ~{self.state.context_tokens} tokens")
        print(f"  cost total   : ${self.state.total_cost_usd:.4f}")
        print(f"  effort       : {self.state.effort or 'default'}")
        print(f"  model        : {self.state.model or 'default'}")
        print(f"  last result  : {self.state.last_result_subtype}")
        print(f"  last usage   : {self.state.last_usage}")
        print()

    # ---- SDK plumbing --------------------------------------------------

    def _make_options(self, resume_id: str | None = None) -> ClaudeAgentOptions:
        kwargs: dict[str, Any] = {
            "permission_mode": self.args.permission_mode,
            "cwd": self.args.cwd,
            # Load skills / CLAUDE.md-linked config from user + project + local scopes
            # so the SDK behaves like the CLI.
            "setting_sources": ["user", "project", "local"],
        }
        resuming = False
        effective_resume = resume_id or self._initial_resume_id
        if effective_resume:
            kwargs["resume"] = effective_resume
            resuming = True
        elif not self.args.no_continue:
            kwargs["continue_conversation"] = True
            resuming = True
        if self.state.effort:
            kwargs["effort"] = self.state.effort
        if self.state.model:
            kwargs["model"] = self.state.model
        if self.args.allowed_tool:
            kwargs["allowed_tools"] = list(self.args.allowed_tool)
        if self.args.disallowed_tool:
            kwargs["disallowed_tools"] = list(self.args.disallowed_tool)
        if self.args.append_system_prompt:
            kwargs["append_system_prompt"] = self.args.append_system_prompt
        if self._mcp_servers is not None:
            kwargs["mcp_servers"] = self._mcp_servers
        # Note on history replay: the CLI's --replay-user-messages flag is
        # NOT for historical playback (it just echoes inputs we send back at
        # us). Claude Code's TUI loads the session JSONL from disk and
        # renders it itself; we do the same in run() before connecting,
        # gated by --no-replay.
        return ClaudeAgentOptions(**kwargs)

    async def _connect(self, resume_id: str | None = None) -> None:
        options = self._make_options(resume_id=resume_id)
        # _initial_resume_id is consumed on the first connect; subsequent
        # reconnects (from /effort, /model) use state.session_id instead.
        self._initial_resume_id = None
        self.client = ClaudeSDKClient(options=options)
        await self.client.connect()
        # Start the persistent SDK message dispatcher.
        self.dispatcher_task = asyncio.create_task(
            self._message_dispatcher(), name="msg-dispatcher"
        )

    async def _disconnect(self) -> None:
        if self.dispatcher_task is not None:
            self.dispatcher_task.cancel()
            try:
                await self.dispatcher_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self.dispatcher_task = None
        if self.client is not None:
            try:
                await self.client.disconnect()
            except Exception as e:  # noqa: BLE001
                print(f"[warn] disconnect error: {e!r}")
            self.client = None

    async def _reconnect(self) -> None:
        print(
            f"[sys] reconnecting "
            f"(effort={self.state.effort or 'default'}, "
            f"model={self.state.model or 'default'}, "
            f"resume={self.state.session_id})"
        )
        await self._disconnect()
        await self._connect(resume_id=self.state.session_id)

    # ---- resume-cwd switching -----------------------------------------

    def _maybe_switch_cwd_for_resume(self) -> None:
        """When resuming, if the session's recorded cwd differs from --cwd,
        switch over so file ops, MCP detection, and the history file all
        line up with what Claude actually remembers."""
        sid = self._initial_resume_id
        if not sid:
            return
        session_cwd = find_session_cwd(sid)
        if not session_cwd:
            return
        try:
            current = Path(self.args.cwd).resolve(strict=False)
            target = Path(session_cwd).resolve(strict=False)
        except OSError:
            current = Path(self.args.cwd)
            target = Path(session_cwd)
        if current == target:
            return
        if not target.exists():
            print(
                f"\033[33m[note: session's recorded cwd does not exist on this machine]"
                f"\n\033[33m  recorded: {target}"
                f"\n\033[33m  staying in: {current}\033[0m"
            )
            return
        print(
            f"\033[35m[switching cwd to session's recorded directory]"
            f"\n\033[35m  from: {current}"
            f"\n\033[35m  to:   {target}\033[0m"
        )
        self.args.cwd = str(target)
        # Reload MCP from the new cwd in case .mcp.json differs.
        self._mcp_servers = self._load_mcp_config()

    # ---- session picker -----------------------------------------------

    async def _pick_session(self) -> str | None:
        """Two-step picker. First choose a project (cheap to list — only
        stats files), then choose a session within that project (parses
        only that project's JSONLs). Esc at the session step goes back to
        the project step."""
        projects = list_projects()
        if not projects:
            print("\033[33m[no sessions found in ~/.claude/projects/]\033[0m")
            return None

        while True:
            # ---- Step 1: project ---------------------------------------
            if len(projects) == 1:
                chosen_project = projects[0]
            else:
                project_values = [
                    (p["project_slug"], format_project_label(p)) for p in projects
                ]
                try:
                    chosen_slug = await cursor_select(
                        title="Resume — pick a project",
                        text="Most-recently-used project first.",
                        values=project_values,
                    )
                except Exception as e:  # noqa: BLE001
                    print(
                        f"\033[33m[picker UI failed ({e}); falling back to text]\033[0m"
                    )
                    return self._pick_session_textfallback(projects)
                if chosen_slug is None:
                    return None
                chosen_project = next(
                    p for p in projects if p["project_slug"] == chosen_slug
                )

            # ---- Step 2: session within project ------------------------
            sessions = list_sessions_for_project(chosen_project["project_dir"])
            if not sessions:
                print(
                    f"\033[33m[no parseable sessions in {chosen_project['project_slug']}]\033[0m"
                )
                if len(projects) == 1:
                    return None
                continue

            project_name = Path(
                chosen_project.get("cwd") or chosen_project["project_slug"]
            ).name
            session_values = [
                (s["session_id"], format_session_label(s)) for s in sessions
            ]
            try:
                chosen = await cursor_select(
                    title=f"Resume — {project_name} ({len(sessions)} sessions)",
                    text="Esc here returns to the project list.",
                    values=session_values,
                )
            except Exception as e:  # noqa: BLE001
                print(
                    f"\033[33m[picker UI failed ({e}); falling back to text]\033[0m"
                )
                return self._pick_sessions_in_project_textfallback(sessions)
            if chosen is None:
                if len(projects) == 1:
                    return None
                continue  # back to project picker
            print(f"\033[35m[resuming session {chosen[:8]}]\033[0m")
            return chosen

    def _pick_session_textfallback(
        self, projects: list[dict[str, Any]]
    ) -> str | None:
        print("\nProjects (most-recent first):")
        for i, p in enumerate(projects):
            print(f"  [{i:>2}] {format_project_label(p)}")
        print()
        try:
            raw = input("Project number (Enter to cancel): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not raw:
            return None
        try:
            chosen_project = projects[int(raw)]
        except (ValueError, IndexError):
            print("\033[31m[invalid selection; cancelled]\033[0m")
            return None
        sessions = list_sessions_for_project(chosen_project["project_dir"])
        if not sessions:
            print("\033[33m[no parseable sessions in that project]\033[0m")
            return None
        return self._pick_sessions_in_project_textfallback(sessions)

    def _pick_sessions_in_project_textfallback(
        self, sessions: list[dict[str, Any]]
    ) -> str | None:
        print("\nSessions (newest first):")
        for i, s in enumerate(sessions):
            print(f"  [{i:>3}] {format_session_label(s)}")
        print()
        try:
            raw = input("Session number to resume (Enter to cancel): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not raw:
            return None
        try:
            return sessions[int(raw)]["session_id"]
        except (ValueError, IndexError):
            print("\033[31m[invalid selection; cancelled]\033[0m")
            return None

    # ---- SDK message dispatcher (persistent reader) --------------------

    async def _message_dispatcher(self) -> None:
        """Continuously read SDK messages. Route to the turn queue while a
        turn is active; otherwise treat as a between-turns async event."""
        if self.client is None:
            return
        try:
            async for msg in self.client.receive_messages():
                if self.turn_active.is_set():
                    await self.turn_msg_queue.put(msg)
                else:
                    self._handle_async_message(msg)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            print(
                f"\033[31m[dispatcher error: {type(e).__name__}: {e}]\033[0m"
            )

    def _handle_async_message(self, msg: Any) -> None:
        """Render a between-turns message and, for wakeup-worthy events
        (background-task completion, requires_action), push a 'wakeup' onto
        event_queue so any pending wait returns CONTINUE_PROMPT."""
        if isinstance(msg, SystemMessage):
            render_system_message(msg, self.state)
            if msg.subtype == "api_retry":
                d = _msg_fields(msg)
                self._check_api_stall(
                    error_status=d.get("error_status"),
                    error_info=d.get("error"),
                )
        elif isinstance(msg, AssistantMessage):
            m = getattr(msg, "model", None)
            if m:
                self.state.active_model = m
            for block in msg.content:
                if isinstance(block, TextBlock):
                    print(f"\033[32mclaude (async):\033[0m {block.text}")
                elif isinstance(block, ToolUseBlock):
                    render_tool_use(
                        block,
                        show_full_commands=self.args.show_full_commands,
                        inline_all=self.args.inline_all_tools,
                        edits_mode=self.args.show_edits,
                    )
        elif isinstance(msg, UserMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                print(f"\033[35m[notice] {content.strip()}\033[0m")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, ToolResultBlock):
                        # Match in-turn behavior: only Bash results print inline.
                        tool_name = None
                        for h in reversed(self.state.tool_history):
                            if h["tool_use_id"] == block.tool_use_id:
                                tool_name = h.get("name")
                                break
                        if tool_name == "Bash" or self.args.inline_all_tools:
                            _render_tool_result(
                                summarize_tool_result(block),
                                is_error=bool(block.is_error),
                                show_full=self.args.show_tool_output,
                            )
        elif isinstance(msg, ResultMessage):
            # Stray ResultMessage outside a turn (rare). Capture session id.
            if msg.session_id:
                self.state.session_id = msg.session_id
        else:
            render_unknown_message(msg, self.state)

        reason = self._wakeup_reason(msg)
        if reason is not None:
            try:
                self.event_queue.put_nowait(("wakeup", reason))
            except asyncio.QueueFull:
                pass

    def _record_turn_end(self) -> None:
        """Track turn-end timestamps so we can detect a continue-burst."""
        if self.args.continue_burst_limit <= 0:
            return
        now = time.monotonic()
        self.state.recent_turn_ends.append(now)
        cutoff = now - self.args.continue_burst_window
        while self.state.recent_turn_ends and self.state.recent_turn_ends[0] < cutoff:
            self.state.recent_turn_ends.popleft()

    def _is_continue_burst(self) -> bool:
        """True iff Claude has finished N turns within the last T seconds."""
        if self.args.continue_burst_limit <= 0:
            return False
        return len(self.state.recent_turn_ends) >= self.args.continue_burst_limit

    @staticmethod
    def _wakeup_reason(msg: Any) -> str | None:
        if isinstance(msg, SystemMessage):
            sub = msg.subtype
            d = _msg_fields(msg)
            if sub == "task_notification":
                tid = (d.get("task_id") or "?")[:8]
                status = d.get("status", "?")
                return f"bg-task {tid} {status}"
            if sub == "session_state_changed" and d.get("state") == "requires_action":
                return "session requires action"
        return None

    # ---- turn driver ---------------------------------------------------

    async def _interrupt_watcher(self) -> None:
        await self.interrupt_event.wait()
        if self.client is not None:
            try:
                await self.client.interrupt()
            except Exception as e:  # noqa: BLE001
                print(f"[warn] interrupt error: {e!r}")

    async def run_turn(self, prompt_text: str) -> tuple[str, bool]:
        assert self.client is not None
        # Drain anything left in the turn queue from the prior between-turns
        # window — handle as async events instead of mistaking them for this
        # turn's response.
        while not self.turn_msg_queue.empty():
            try:
                stale = self.turn_msg_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._handle_async_message(stale)

        self.interrupt_event.clear()
        self.state.busy = True
        self.state.needs_user_attention = None
        self.state.api_stall_source = None
        self.state.api_stall_saw_bad = False
        self.state.compact_during_last_turn = False
        self.state.current_turn_tool_seqs = []
        self.state.current_turn_bg = {}
        turn_started = time.monotonic()
        self.turn_active.set()

        preview = prompt_text if len(prompt_text) <= 160 else prompt_text[:157] + "..."
        print()
        print(f"\033[36m> you:\033[0m {preview}")
        print()

        watcher = asyncio.create_task(self._interrupt_watcher())
        assistant_parts: list[str] = []
        in_text = False
        api_error_reported = False  # one orchestrator notice per turn

        try:
            await self.client.query(prompt_text)
            while True:
                msg = await self.turn_msg_queue.get()
                if isinstance(msg, SystemMessage):
                    if in_text and msg.subtype not in ("init",):
                        print()
                        in_text = False
                    render_system_message(msg, self.state)
                    if msg.subtype == "api_retry":
                        d = _msg_fields(msg)
                        self._check_api_stall(
                            error_status=d.get("error_status"),
                            error_info=d.get("error"),
                        )
                elif isinstance(msg, AssistantMessage):
                    m = getattr(msg, "model", None)
                    if m:
                        self.state.active_model = m
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            if not in_text:
                                sys.stdout.write("\033[32mclaude:\033[0m ")
                                in_text = True
                            sys.stdout.write(block.text)
                            sys.stdout.flush()
                            assistant_parts.append(block.text)
                        elif isinstance(block, ToolUseBlock):
                            if in_text:
                                sys.stdout.write("\n")
                                in_text = False
                            parent_id = getattr(msg, "parent_tool_use_id", None)
                            seq = self.state.next_tool_seq
                            self.state.next_tool_seq += 1
                            started = time.monotonic()
                            self.state.active_tools[block.id] = {
                                "name": block.name,
                                "input": block.input,
                                "started_at": started,
                                "seq": seq,
                                "parent_id": parent_id,
                                "sub_trail": [] if block.name == "Task" else None,
                                "current_sub_id": None,
                            }
                            # Track this seq in the current-turn list unless
                            # it's (a) Bash (scrolls inline) or (b) a sub-tool
                            # under a Task (rolled up under its parent row).
                            if block.name != "Bash" and not parent_id:
                                self.state.current_turn_tool_seqs.append(seq)
                            # If this is a sub-tool under an active Task, record it
                            # so the live-tasks panel can show aggregated patterns
                            # and the currently-running sub-operation.
                            if parent_id and parent_id in self.state.active_tools:
                                parent = self.state.active_tools[parent_id]
                                trail = parent.get("sub_trail")
                                if trail is None:
                                    trail = []
                                    parent["sub_trail"] = trail
                                trail.append(
                                    {
                                        "name": block.name,
                                        "input": block.input,
                                        "seq": seq,
                                        "tool_use_id": block.id,
                                    }
                                )
                                parent["current_sub_id"] = block.id
                            self.state.tool_history.append(
                                {
                                    "seq": seq,
                                    "tool_use_id": block.id,
                                    "name": block.name,
                                    "input": block.input,
                                    "started_at": started,
                                    "result_text": None,
                                    "is_error": None,
                                    "ended_at": None,
                                }
                            )
                            # Capture Claude's plan from TodoWrite snapshots.
                            if block.name == "TodoWrite":
                                todos = (block.input or {}).get("todos")
                                if isinstance(todos, list):
                                    self.state.current_todos = list(todos)
                            render_tool_use(
                                block,
                                show_full_commands=self.args.show_full_commands,
                                seq=seq,
                                inline_all=self.args.inline_all_tools,
                                edits_mode=self.args.show_edits,
                            )
                        elif ThinkingBlock is not None and isinstance(block, ThinkingBlock):
                            if in_text:
                                sys.stdout.write("\n")
                                in_text = False
                            full_text = block.thinking or ""
                            seq = self.state.next_thinking_seq
                            self.state.next_thinking_seq += 1
                            self.state.thinking_history.append(
                                {
                                    "seq": seq,
                                    "text": full_text,
                                    "started_at": time.time(),
                                }
                            )
                            if self.args.show_thinking:
                                print(
                                    f"\033[90m  [#{seq} -- /think {seq}] "
                                    f"think: {full_text.strip()}\033[0m"
                                )
                            else:
                                # No snippet in the compact form — leave it to
                                # /think N to fetch the full text, same way
                                # Bash gets its command body via /show N.
                                print(
                                    f"\033[90m  [#{seq} -- /think {seq}] "
                                    f"(thinking)\033[0m"
                                )
                    # End-of-AssistantMessage: flush a newline if we left
                    # an unterminated streamed-text line, otherwise
                    # patch_stdout buffers it and the prompt redraws on
                    # top of the partial line.
                    if in_text:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        in_text = False
                    # The CLI sometimes emits transport/API errors as
                    # *assistant text* after its own retry budget runs out.
                    # The text is real Claude output (keep it on screen)
                    # but the orchestrator also needs to know — for the
                    # stall heuristic + status-check path — that something
                    # went wrong, so synthesize an api_retry-equivalent.
                    if not api_error_reported:
                        m = _ASSISTANT_API_ERROR_RE.search(
                            "".join(assistant_parts)
                        )
                        if m:
                            api_error_reported = True
                            status_code = m.group(1)
                            status_part = (
                                f" status={status_code}"
                                if status_code
                                else ""
                            )
                            print(
                                f"\033[31m[orchestrator] detected API error "
                                f"in assistant text{status_part} -- "
                                f"treating as api_retry for stall "
                                f"detection\033[0m"
                            )
                            self._check_api_stall(
                                error_status=status_code, error_info=None
                            )
                elif isinstance(msg, UserMessage):
                    content = msg.content
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, ToolResultBlock):
                                if in_text:
                                    sys.stdout.write("\n")
                                    in_text = False
                                active = self.state.active_tools.pop(
                                    block.tool_use_id, None
                                )
                                # If this was a sub-tool of a Task, clear the
                                # parent's "currently running" marker so the
                                # live-tasks panel stops showing it.
                                if active and active.get("parent_id"):
                                    parent = self.state.active_tools.get(
                                        active["parent_id"]
                                    )
                                    if (
                                        parent
                                        and parent.get("current_sub_id")
                                        == block.tool_use_id
                                    ):
                                        parent["current_sub_id"] = None
                                # Backup cleanup: if this is the result of a
                                # Task tool use, purge any background_tasks
                                # entry bound to it. Covers the case where the
                                # CLI doesn't emit a matching task_notification
                                # (observed: bg[N] counter leaking across Task
                                # calls).
                                if active and active.get("name") == "Task":
                                    leaked = [
                                        tid
                                        for tid, bg in self.state.background_tasks.items()
                                        if bg.get("tool_use_id") == block.tool_use_id
                                    ]
                                    for tid in leaked:
                                        self.state.background_tasks.pop(tid, None)
                                seq = active.get("seq") if active else None
                                tool_name = active.get("name") if active else None
                                text = summarize_tool_result(block)
                                is_err = bool(block.is_error)
                                # Update history.
                                for h in reversed(self.state.tool_history):
                                    if h["tool_use_id"] == block.tool_use_id:
                                        h["result_text"] = text
                                        h["is_error"] = is_err
                                        h["ended_at"] = time.monotonic()
                                        break
                                # Match the tool-use side: Bash always prints
                                # inline; others only with --inline-all-tools.
                                if (
                                    tool_name == "Bash"
                                    or self.args.inline_all_tools
                                ):
                                    _render_tool_result(
                                        text,
                                        is_error=is_err,
                                        show_full=self.args.show_tool_output,
                                        seq=seq,
                                    )
                    elif isinstance(content, str) and content.strip():
                        # System-injected user messages — e.g. background-shell
                        # completion notifications the CLI inserts into context.
                        if in_text:
                            print()
                            in_text = False
                        print(f"\033[35m[notice] {content.strip()}\033[0m")
                elif isinstance(msg, ResultMessage):
                    if in_text:
                        print()
                        in_text = False
                    self.state.last_result_subtype = msg.subtype
                    if msg.session_id:
                        self.state.session_id = msg.session_id
                    self.state.turns += 1
                    if msg.total_cost_usd is not None:
                        self.state.total_cost_usd += msg.total_cost_usd
                    usage = msg.usage or {}
                    self.state.last_usage = usage
                    model_usage = getattr(msg, "model_usage", None)
                    # On a turn where compact_boundary fired, the final
                    # usage blob still sums the pre-compact API calls and
                    # would clobber the 0 we set at the boundary. Leave
                    # context_tokens at the post-compact value instead.
                    if not self.state.compact_during_last_turn:
                        self.state.context_tokens = _extract_context_tokens(
                            usage, model_usage
                        )
                    # Re-read disk title every turn so Haiku's auto-named
                    # ai-title (added asynchronously by Claude Code) and any
                    # cross-tool /rename land in our cached title without
                    # waiting for a reconnect.
                    if self.state.session_id:
                        disk_title = _read_session_title(self.state.session_id)
                        if disk_title:
                            self.state.session_title = disk_title
                    cost_part = ""
                    if not self.state.is_subscription:
                        cost = msg.total_cost_usd or 0.0
                        cost_part = f"${cost:.4f} -- "
                    elapsed = time.monotonic() - turn_started
                    print(
                        f"\033[90m[turn done -- {msg.subtype} -- "
                        f"{cost_part}ctx~{self.state.context_tokens} tok -- "
                        f"{_fmt_duration(elapsed)}]\033[0m"
                    )
                    break
                else:
                    # Forward-compatible fallback — anything not matched above
                    # (tool_progress, auth_status, partial streaming chunks,
                    # future message types) gets shape-detected and rendered.
                    if in_text:
                        print()
                        in_text = False
                    render_unknown_message(msg, self.state)
        finally:
            self.turn_active.clear()
            self.state.busy = False
            # Drop any foreground tool tracker entries left over from an
            # interrupted turn (background tasks survive — they keep running).
            self.state.active_tools.clear()
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass

        interrupted = self.interrupt_event.is_set()
        self.interrupt_event.clear()
        return "".join(assistant_parts), interrupted

    # ---- input loop ----------------------------------------------------

    async def input_loop(self) -> None:
        assert self.session is not None
        try:
            while not self.stop_event.is_set():
                try:
                    line = await self.session.prompt_async()
                except EOFError:
                    break
                except KeyboardInterrupt:
                    continue  # should not happen with our c-c binding, be safe
                if self.stop_event.is_set():
                    break
                if line is None:
                    continue
                kind, payload = classify(line)
                if kind == "empty":
                    continue
                if kind == "interrupt":
                    self.interrupt_event.set()
                    if not self.state.busy:
                        print("\033[33m[nothing to interrupt]\033[0m")
                    continue
                if kind == "error":
                    print(f"\033[31m[error] {payload}\033[0m")
                    continue
                if kind == "quit":
                    # Also interrupt so the worker ends quickly if mid-turn.
                    if self.state.busy:
                        self.interrupt_event.set()
                        print(
                            "\033[33m[shutting down (interrupting current "
                            "turn; up to ~10s for the CLI to flush the "
                            "session file)...]\033[0m"
                        )
                    else:
                        print(
                            "\033[33m[shutting down (up to ~5s for the CLI "
                            "to flush the session file)...]\033[0m"
                        )
                    await self.event_queue.put((kind, payload))
                    break
                if kind == "force-quit":
                    print(
                        "\033[31m[force-quit: killing CLI subprocess "
                        "immediately -- last in-flight message may be "
                        "lost from the JSONL]\033[0m"
                    )
                    sys.stdout.flush()
                    import os
                    os._exit(0)
                await self.event_queue.put((kind, payload))
        finally:
            self.stop_event.set()
            try:
                self.event_queue.put_nowait(("quit", ""))
            except asyncio.QueueFull:
                pass

    # ---- main worker loop ---------------------------------------------

    async def _drain_between_turns(
        self,
    ) -> tuple[list[str], bool, bool]:
        """Empty the event queue. Returns (injected_messages, reconnect_needed, quit)."""
        injected: list[str] = []
        reconnect_needed = False
        quit_requested = False
        while not self.event_queue.empty():
            kind, payload = self.event_queue.get_nowait()
            if kind == "quit":
                quit_requested = True
            elif kind == "message":
                injected.append(payload)
            elif kind == "compact":
                injected.append("/compact")
            elif kind == "wakeup":
                # Async event arrived during/right at end of a turn; drop —
                # the next turn's response will already include it via the SDK.
                pass
            elif kind == "status":
                self.print_status()
            elif kind == "help":
                self.print_help()
            elif kind == "clear-screen":
                self.clear_screen()
            elif kind == "clear-context":
                await self.clear_context()
            elif kind == "rename":
                self.rename_session(payload)
            elif kind == "auto":
                self.toggle_auto_continue(payload)
            elif kind == "burst":
                self.set_burst(payload)
            elif kind == "export":
                self.export_session(payload)
            elif kind == "tools":
                self.show_tools()
            elif kind == "tasks":
                self.show_tasks()
            elif kind == "bg":
                self.show_bg_tasks(payload)
            elif kind == "show":
                self.show_tool_detail(payload)
            elif kind == "think":
                self.show_thinking_detail(payload)
            elif kind == "autocompact":
                self.set_autocompact(payload)
            elif kind == "max-context":
                self.set_max_context(payload)
            elif kind == "todos":
                self.show_todos()
            elif kind == "effort":
                self.state.effort = None if payload == "auto" else payload
                print(f"\033[35m[sys] effort -> {payload}\033[0m")
                reconnect_needed = True
            elif kind == "model":
                self.state.model = payload
                print(f"\033[35m[sys] model -> {payload}\033[0m")
                reconnect_needed = True
        return injected, reconnect_needed, quit_requested

    async def _await_user_or_quit(self, timeout: float | None = None) -> str | None:
        """Wait for a user message. Returns the next prompt, or None on quit."""
        while True:
            if timeout is None:
                kind, payload = await self.event_queue.get()
            else:
                try:
                    kind, payload = await asyncio.wait_for(
                        self.event_queue.get(), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    return None  # caller decides what to do on timeout
            if kind == "quit":
                self.stop_event.set()
                return None
            if kind == "message":
                return payload
            if kind == "compact":
                return "/compact"
            if kind == "wakeup":
                sys.stdout.write("\a")
                sys.stdout.flush()
                print(f"\033[36m[wakeup -- {payload}]\033[0m")
                return CONTINUE_PROMPT
            if kind == "status":
                self.print_status()
            elif kind == "help":
                self.print_help()
            elif kind == "clear-screen":
                self.clear_screen()
            elif kind == "clear-context":
                await self.clear_context()
            elif kind == "rename":
                self.rename_session(payload)
            elif kind == "auto":
                self.toggle_auto_continue(payload)
            elif kind == "burst":
                self.set_burst(payload)
            elif kind == "export":
                self.export_session(payload)
            elif kind == "tools":
                self.show_tools()
            elif kind == "tasks":
                self.show_tasks()
            elif kind == "bg":
                self.show_bg_tasks(payload)
            elif kind == "show":
                self.show_tool_detail(payload)
            elif kind == "think":
                self.show_thinking_detail(payload)
            elif kind == "autocompact":
                self.set_autocompact(payload)
            elif kind == "max-context":
                self.set_max_context(payload)
            elif kind == "todos":
                self.show_todos()
            elif kind == "effort":
                self.state.effort = None if payload == "auto" else payload
                print(f"\033[35m[sys] effort -> {payload}\033[0m")
                await self._reconnect()
            elif kind == "model":
                self.state.model = payload
                print(f"\033[35m[sys] model -> {payload}\033[0m")
                await self._reconnect()
            # loop and keep waiting

    async def worker_loop(self) -> None:
        try:
            await self._connect()

            if self.args.initial_prompt:
                next_prompt: str | None = self.args.initial_prompt
            else:
                print("\033[90m(type a first message, or /help)\033[0m")
                next_prompt = await self._await_user_or_quit()

            while next_prompt is not None and not self.stop_event.is_set():
                try:
                    text, interrupted = await self.run_turn(next_prompt)
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001
                    print(f"\033[31m[error: {type(e).__name__}: {e}]\033[0m")
                    if self.args.auto_reconnect:
                        print("\033[33m[sys] auto-reconnecting...\033[0m")
                        try:
                            await self._reconnect()
                        except Exception as e2:  # noqa: BLE001
                            print(f"\033[31m[reconnect failed: {e2}]\033[0m")
                            next_prompt = await self._await_user_or_quit()
                            continue
                        next_prompt = CONTINUE_PROMPT
                        continue
                    next_prompt = await self._await_user_or_quit()
                    continue

                injected, reconnect_needed, quit_requested = await self._drain_between_turns()
                if quit_requested:
                    self.stop_event.set()
                    break
                if reconnect_needed:
                    await self._reconnect()

                if injected:
                    next_prompt = "\n".join(injected)
                    continue

                if interrupted:
                    sys.stdout.write("\a")
                    sys.stdout.flush()
                    print("\033[33m(interrupted -- your turn)\033[0m")
                    next_prompt = await self._await_user_or_quit()
                    continue

                # API-stall mode: wait for the status poller to push a
                # wakeup (or for you to type something). Blocks before any
                # further work regardless of --auto-continue — including
                # post-compact auto-continue, which would otherwise charge
                # ahead during an outage.
                if self.state.needs_user_attention == "api-error":
                    # If a compact just finished during the stall, still
                    # clear that flag so we don't re-trigger on the next
                    # wakeup.
                    if self.state.compact_during_last_turn:
                        self.state.compact_during_last_turn = False
                        self.state.recent_turn_ends.clear()
                    print(
                        "\033[36m[API-stalled -- waiting for Anthropic "
                        "services to recover]\033[0m"
                    )
                    next_prompt = await self._await_user_or_quit()
                    continue

                # If a compact_boundary fired during this turn (manual or
                # auto), short-circuit: the context_tokens reading from
                # this turn's ResultMessage still reflects the
                # pre-compact cached usage, so the threshold check below
                # would loop us back into another /compact. Also clear
                # the burst tracker so a maintenance turn doesn't trip
                # the "stop nudging" brake. Then auto-continue (or wait,
                # if --auto-continue is off).
                if self.state.compact_during_last_turn:
                    self.state.compact_during_last_turn = False
                    self.state.recent_turn_ends.clear()
                    if self.args.auto_continue:
                        print(
                            "\033[90m[post-compact: auto-continuing]\033[0m"
                        )
                        next_prompt = CONTINUE_PROMPT
                    else:
                        print(
                            "\033[90m[post-compact: waiting for your input]\033[0m"
                        )
                        next_prompt = await self._await_user_or_quit()
                    continue

                # Rolling-window trim takes priority when configured: if it
                # fires we've already reconnected onto a shorter session, so
                # don't also trigger an auto-compact on top of that.
                if await self._maybe_trim_context():
                    next_prompt = await self._await_user_or_quit()
                    continue
                if (
                    not self.args.no_compact
                    and self.state.context_tokens >= self.args.compact_at
                ):
                    cooldown = int(
                        getattr(self.args, "compact_cooldown_turns", 3) or 0
                    )
                    last = self.state.last_compact_turn
                    if (
                        cooldown > 0
                        and last is not None
                        and self.state.turns - last < cooldown
                    ):
                        remaining = cooldown - (self.state.turns - last)
                        print(
                            f"\033[90m[ctx ~{self.state.context_tokens} tok "
                            f">= {self.args.compact_at}, but compacted "
                            f"{self.state.turns - last} turn(s) ago -- "
                            f"skipping re-compact ({remaining} more turn(s) "
                            f"of cooldown)]\033[0m"
                        )
                    else:
                        print(
                            f"\033[35m[ctx ~{self.state.context_tokens} tok "
                            f">= {self.args.compact_at}; compacting]\033[0m"
                        )
                        next_prompt = "/compact"
                        continue

                # Without --auto-continue, the orchestrator just waits for
                # your input after every turn (like a normal interactive
                # session). The [WAITING] / burst-limit / response-delay
                # mechanics only matter when we're driving Claude
                # autonomously.
                if not self.args.auto_continue:
                    next_prompt = await self._await_user_or_quit()
                    continue

                self._record_turn_end()
                burst = self._is_continue_burst() and WAITING_SENTINEL not in text
                if WAITING_SENTINEL in text or burst:
                    if burst:
                        print(
                            f"\033[33m[continue burst limit hit "
                            f"({self.args.continue_burst_limit} turns within "
                            f"{self.args.continue_burst_window:.0f}s without [WAITING]); "
                            f"backing off]\033[0m"
                        )
                        self.state.needs_user_attention = "burst"
                    else:
                        self.state.needs_user_attention = "waiting"
                    self.state.recent_turn_ends.clear()
                    sys.stdout.write("\a")
                    sys.stdout.flush()
                    print(
                        "\033[36m[Claude is waiting -- your turn "
                        "(or async wakeup on bg-task / requires-action)]\033[0m"
                    )
                    next_prompt = await self._await_user_or_quit()
                    continue

                print(
                    f"\033[90m[idle -- auto-continuing in {self.args.continue_response_delay:.1f}s; "
                    f"type to interject]\033[0m"
                )
                grace_prompt = await self._await_user_or_quit(timeout=self.args.continue_response_delay)
                if grace_prompt is None and self.stop_event.is_set():
                    break
                next_prompt = grace_prompt if grace_prompt is not None else CONTINUE_PROMPT
        finally:
            await self._disconnect()

    # ---- entry point ---------------------------------------------------

    async def run(self) -> None:
        # Resolve --resume FIRST (before PromptSession & history file are
        # created and before we connect), so a session whose recorded cwd
        # differs from --cwd can switch us cleanly.
        if self.args.resume == _PICKER_SENTINEL:
            chosen = await self._pick_session()
            if chosen is None:
                print("\033[33m[no session chosen — exiting]\033[0m")
                return
            self._initial_resume_id = chosen
        if self._initial_resume_id:
            self._maybe_switch_cwd_for_resume()

        # Tell the user when this cwd has no prior project on disk — the
        # CLI will create ~/.claude/projects/<sanitized-cwd>/ on first turn.
        # find_project_for_cwd() does the smart match (direct sanitize, then
        # falls back to scanning project dirs for matching recorded cwd) so
        # a Windows case mismatch doesn't make a returning user look new.
        existing_project = find_project_for_cwd(self.args.cwd)
        is_new_project = existing_project is None
        new_project_dir = project_dir_for_cwd(self.args.cwd)

        # Pick which session's history to render into the backscroll.
        # Explicit --resume id wins; otherwise the most-recent in the cwd's
        # project dir (matching what continue_conversation=True will resume).
        history_jsonl: Path | None = None
        if not self.args.no_replay:
            if self._initial_resume_id:
                proj = _find_session_dir(self._initial_resume_id)
                if proj is not None:
                    candidate = proj / f"{self._initial_resume_id}.jsonl"
                    if candidate.exists():
                        history_jsonl = candidate
            elif not self.args.no_continue:
                history_jsonl = find_most_recent_session_for_cwd(self.args.cwd)

        history_path = Path(self.args.cwd) / ".orchestrator_history"
        completer = WordCompleter(SLASH_COMMANDS, ignore_case=True, sentence=False)
        self.session = PromptSession(
            message=self._prompt_message,
            history=FileHistory(str(history_path)),
            key_bindings=self._keybindings(),
            completer=completer,
            complete_while_typing=False,
            bottom_toolbar=self._bottom_toolbar,
            style=STYLE,
            enable_history_search=True,
            multiline=True,  # Alt-Enter newline, Enter submit (see keybindings)
            refresh_interval=0.5,  # keep toolbar's busy/ctx/cost fields fresh
        )

        # Set the terminal title so long sessions are easy to find.
        try:
            title = f"Claude Orchestrator -- {Path(self.args.cwd).resolve().name}"
            sys.stdout.write(f"\033]0;{title}\a")
            sys.stdout.flush()
        except Exception:  # noqa: BLE001
            pass

        tool_summary = (
            "all Claude Code tools (Read, Write, Edit, Glob, Grep, Bash "
            "incl. background, BashOutput, KillShell, NotebookEdit, "
            "WebFetch, WebSearch, Task, Skill, TodoWrite)"
        )
        if self.args.allowed_tool:
            tool_summary = f"allowed: {', '.join(self.args.allowed_tool)}"
        if self.args.disallowed_tool:
            tool_summary += f"  (disallowed: {', '.join(self.args.disallowed_tool)})"

        print("=" * 78)
        print(" Claude Orchestrator")
        print("  - type + Enter to send; Tab completes /commands")
        print("  - Ctrl-C: interrupt turn / clear input / exit at empty prompt")
        print("  - Ctrl-D: exit")
        if self.args.auto_continue:
            ac_summary = (
                f"auto-continue ON (delay {self.args.continue_response_delay}s, "
                f"burst {self.args.continue_burst_limit}/"
                f"{self.args.continue_burst_window:.0f}s)"
            )
        else:
            ac_summary = "auto-continue OFF (interactive — wait for your input each turn)"
        compact_summary = (
            "auto-compact OFF"
            if self.args.no_compact
            else f"auto-compact at ~{self.args.compact_at} tokens"
        )
        max_ctx = getattr(self.args, "max_context_tokens", 0) or 0
        max_ctx_summary = (
            f"max-context ~{max_ctx} tok" if max_ctx > 0 else "max-context unlimited"
        )
        print(
            f"  - {compact_summary}"
            f"  |  {max_ctx_summary}"
            f"  |  perm={self.args.permission_mode}"
        )
        print(f"  - {ac_summary}")
        resume_summary = "fresh session"
        if not self.args.no_continue:
            resume_summary = (
                "resuming last session (replaying history)"
                if not self.args.no_replay
                else "resuming last session (no replay)"
            )
        print(
            f"  - cwd={self.args.cwd}"
            f"  |  {resume_summary}"
            f"  |  effort={self.state.effort or 'default'}"
            f"  |  model={self.state.model or 'default'}"
        )
        print(f"  - tools: {tool_summary}")
        if self._mcp_servers:
            print(f"  - mcp servers: {', '.join(self._mcp_servers)}")
        if is_new_project:
            print(
                f"  - new project: no Claude Code sessions exist yet for this "
                f"cwd; one will be\n"
                f"    created at {new_project_dir}"
            )
        print("=" * 78)

        # Render conversation history before the prompt so the resumed
        # session has visible backscroll (matching `claude --continue`).
        # Build the entire history in memory and write it in a single
        # syscall so the terminal doesn't appear to scroll line-by-line
        # while we're populating it.
        if history_jsonl is not None:
            n, history_text = render_session_history_text(history_jsonl)
            buffered = (
                f"\033[90m[loading history from {history_jsonl.name} ...]\033[0m\n"
                f"{history_text}"
                f"\033[90m[end of history -- {n} message(s)]\033[0m\n"
            )
            sys.stdout.write(buffered)
            sys.stdout.flush()

        # Seed session id + title in state now (before connecting) so the
        # bottom toolbar shows real info instead of "(new)" while we wait
        # for the SDK init message. Both --resume <id> and --continue can
        # be resolved here without any SDK round-trip.
        seed_sid: str | None = None
        if self._initial_resume_id:
            seed_sid = self._initial_resume_id
        elif history_jsonl is not None:
            seed_sid = history_jsonl.stem
        if seed_sid:
            self.state.session_id = seed_sid
            self.state.session_title = _read_session_title(seed_sid)

        # Loud red warning about the unattended-mode footgun.
        if self.args.permission_mode == "bypassPermissions":
            print(
                "\033[1;31m"
                "!!  WARNING: bypass-permissions mode is ON.\n"
                "!!  Claude can run ANY local command (Bash, including "
                "`run_in_background=true`),\n"
                "!!  read/write/edit/delete files ANYWHERE on disk that your "
                "user account has access to\n"
                "!!  (Read/Write/Edit take absolute paths — cwd is the default "
                "for Bash & Glob/Grep, not a\n"
                "!!  sandbox), fetch ANY URL, and run web searches — all "
                "without asking you. Don't paste\n"
                "!!  prompts you don't trust. Use --permission-mode default "
                "(or acceptEdits) if you want\n"
                "!!  approval prompts."
                "\033[0m"
            )

        input_task = asyncio.create_task(self.input_loop(), name="input")
        worker_task = asyncio.create_task(self.worker_loop(), name="worker")

        done, pending = await asyncio.wait(
            {input_task, worker_task}, return_when=asyncio.FIRST_COMPLETED
        )
        self.stop_event.set()
        for t in pending:
            t.cancel()
        for t in pending:
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        for t in done:
            exc = t.exception()
            if exc is not None and not isinstance(exc, asyncio.CancelledError):
                print(f"\033[31m[fatal: {type(exc).__name__}: {exc}]\033[0m")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Long-running Claude orchestrator with a Claude-Code-like TUI."
    )
    ap.add_argument("--initial-prompt", "-p", default=None, help="First message to send.")
    ap.add_argument(
        "--no-continue",
        action="store_true",
        help="Start a fresh session instead of resuming the most recent one in cwd.",
    )
    ap.add_argument(
        "--no-replay",
        action="store_true",
        help="When resuming, do NOT replay prior user/assistant messages into "
        "the backscroll. Default is to replay (matching `claude --continue`'s "
        "behavior of showing the conversation history on resume).",
    )
    ap.add_argument(
        "--resume",
        nargs="?",
        const=_PICKER_SENTINEL,
        default=None,
        metavar="SESSION_ID",
        help="Resume a specific session by id. Pass --resume with no value to "
        "open an interactive picker listing all sessions in "
        "~/.claude/projects/ (project dir, last user message, age). Overrides "
        "--continue. Combine with --no-replay if you don't want history scrollback.",
    )
    ap.add_argument(
        "--effort",
        choices=list(EFFORT_CHOICES),
        default=None,
        help="Thinking effort level. 'auto' (or omit) means don't pass an "
        "effort parameter; the model uses its own default (typically 'high').",
    )
    ap.add_argument("--model", default=None, help='e.g. "claude-opus-4-6", "claude-sonnet-4-6".')
    ap.add_argument("--cwd", default=".", help="Working directory Claude operates in.")
    ap.add_argument(
        "--compact-at",
        type=int,
        default=None,
        help=f"Force /compact when context tokens exceed this. When omitted, "
        f"derived from the model: {DEFAULT_COMPACT_THRESHOLD_1M} for 1M-context "
        f"variants, {DEFAULT_COMPACT_THRESHOLD} otherwise.",
    )
    ap.add_argument(
        "--auto-compact",
        dest="no_compact",
        action="store_false",
        help="Enable the orchestrator's auto-compact check (off by default — "
        "the CLI's own auto-compact handles context-window pressure just "
        "fine). With this flag, the orchestrator also injects /compact "
        "when context_tokens >= --compact-at at a turn boundary. Useful "
        "for turn-boundary predictability when using a tight threshold.",
    )
    ap.add_argument(
        "--no-compact",
        dest="no_compact",
        action="store_true",
        default=True,
        help="(Default.) Disable the orchestrator's auto-compact entirely "
        "and leave compaction to the CLI. You can still run /compact or "
        "/clear manually, or pair with --max-context-tokens to cap "
        "context via rolling-window trim instead.",
    )
    ap.add_argument(
        "--compact-cooldown-turns",
        type=int,
        default=3,
        help="After an auto-compact fires, skip the compact check for this "
        "many turns. Default 3. Prevents a re-compact loop when the last "
        "turn's cumulative I/O stays inflated above --compact-at even "
        "though the real resident context just shrank.",
    )
    ap.add_argument(
        "--max-context-tokens",
        type=int,
        default=0,
        help="Cap the session's context window at ~N tokens by trimming the "
        "oldest turns before each query (rolling window, no summarization). "
        "0 = disabled. Truly an alternative to auto-compact — typically "
        "pair with --no-compact. Trimming respects tool_use/tool_result "
        "pair boundaries, so history stays coherent.",
    )
    ap.add_argument(
        "--auto-continue",
        action="store_true",
        help="Enable the autonomous orchestrator behavior: after each turn, "
        "automatically nudge Claude with the continue prompt unless he "
        "emitted [WAITING]. Default OFF — by default the orchestrator just "
        "waits for your input after each turn (like a normal interactive "
        "session). The --continue-response-delay and --continue-burst-* "
        "flags only have effect when this is enabled.",
    )
    ap.add_argument(
        "--continue-response-delay",
        type=float,
        default=CONTINUE_RESPONSE_DELAY_SECONDS,
        help="Only with --auto-continue. Seconds to wait after Claude "
        "finishes a turn (and is NOT [WAITING]) before sending the next "
        "auto-continue prompt. Doubles as the grace window during which "
        "you can interject — anything you type in this window is sent "
        "instead of the auto-continue.",
    )
    ap.add_argument(
        "--continue-burst-limit",
        type=int,
        default=CONTINUE_BURST_LIMIT,
        metavar="N",
        help="Only with --auto-continue. Safety brake against runaway "
        "auto-continue. If Claude finishes this many turns within "
        "--continue-burst-window seconds without emitting [WAITING], "
        "treat it as [WAITING] and stop nudging until you (or an async "
        "wakeup) intervene. Set to 0 to disable.",
    )
    ap.add_argument(
        "--continue-burst-window",
        type=float,
        default=CONTINUE_BURST_WINDOW_SECONDS,
        metavar="SECONDS",
        help="Time window for --continue-burst-limit (default 180s / 3min).",
    )
    ap.add_argument(
        "--permission-mode",
        choices=["bypassPermissions", "acceptEdits", "default", "plan"],
        default="bypassPermissions",
        help="Tool permission mode. Default bypassPermissions lets Claude run "
        "any tool (Read/Write/Edit/Bash/etc.) with no prompts, matching an "
        "unattended Claude Code session.",
    )
    ap.add_argument(
        "--allowed-tool",
        action="append",
        default=[],
        metavar="NAME",
        help="Restrict Claude to the given tools (repeatable). Omit to allow all "
        "built-in tools (Read, Write, Edit, Glob, Grep, Bash, BashOutput, "
        "KillShell, NotebookEdit, WebFetch, WebSearch, Task, Skill, TodoWrite).",
    )
    ap.add_argument(
        "--disallowed-tool",
        action="append",
        default=[],
        metavar="NAME",
        help="Block specific tools by name (repeatable).",
    )
    ap.add_argument(
        "--append-system-prompt",
        default=None,
        metavar="TEXT",
        help="Extra instructions appended to Claude's system prompt.",
    )
    ap.add_argument(
        "--mcp-config",
        default=None,
        metavar="PATH",
        help="Path to an MCP servers JSON file (same shape as .mcp.json / settings.mcpServers). "
        "If omitted, .mcp.json in cwd is auto-loaded if present.",
    )
    ap.add_argument(
        "--show-thinking",
        action="store_true",
        help="Print the full text of extended-thinking blocks "
        "(default shows a one-line collapsed snippet).",
    )
    ap.add_argument(
        "--show-full-commands",
        action="store_true",
        help="Print Bash commands inline. Default hides them entirely (only "
        "the tool header + description is shown); the full command is "
        "always available via /tools while it's running, /show N "
        "afterwards, and in /export.",
    )
    ap.add_argument(
        "--show-tool-output",
        action="store_true",
        help="Print full tool result content inline (Bash output, Read "
        "contents, Grep matches, etc.). Default suppresses it — long Bash "
        "outputs and file reads can fill the screen — and shows just "
        "`→ N lines, K chars` for success or `✗ tool error -- ...` for "
        "failures. Full content is always preserved in the JSONL "
        "transcript and visible via /export.",
    )
    ap.add_argument(
        "--show-tool-everything",
        action="store_true",
        help="Convenience: implies BOTH --show-full-commands AND "
        "--show-tool-output. Use when you want maximum visibility into "
        "what Claude is doing.",
    )
    ap.add_argument(
        "--inline-all-tools",
        action="store_true",
        help="Render every tool call inline with [#N] tags, like Bash does, "
        "instead of the transient live panel. Useful when you want a full "
        "scrollable log of activity. The live panel is still available for "
        "any tool running when this is off.",
    )
    ap.add_argument(
        "--tasks-panel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the live tasks panel in the toolbar while non-Bash tools "
        "are running. Pass --no-tasks-panel to hide the panel entirely "
        "(still accessible via /tasks / /show). Default on.",
    )
    ap.add_argument(
        "--bg-panel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the live background-tasks panel in the toolbar while bg "
        "shells / Task subagents are running. Pass --no-bg-panel to hide "
        "(still accessible via /bg). Default on.",
    )
    ap.add_argument(
        "--show-edits",
        choices=("off", "compact", "full"),
        default="off",
        help="How Edit tool calls render. off (default): live panel only, "
        "use /show N for detail. compact: scroll inline as a one-liner "
        "`edit path (+A -R lines) [#N]`. full: scroll inline with the full "
        "unified diff. --inline-all-tools overrides this to 'full'.",
    )
    ap.add_argument(
        "--api-stall-limit",
        type=int,
        default=5,
        help="Enter API-stall mode after N api_retry events within "
        "--api-stall-window seconds. Default 5. Set to 0 to disable.",
    )
    ap.add_argument(
        "--api-stall-window",
        type=float,
        default=60.0,
        help="Sliding window (seconds) for --api-stall-limit. Default 60.",
    )
    ap.add_argument(
        "--status-url",
        default="https://status.claude.com/api/v2/summary.json",
        help="Anthropic Statuspage.io summary feed. While in API-stall mode "
        "the orchestrator polls this and auto-resumes when the Claude API "
        "and Claude Code components return to 'operational'.",
    )
    ap.add_argument(
        "--status-poll-interval",
        type=float,
        default=30.0,
        help="How often (seconds) to hit the status feed while stalled. "
        "Default 30. Statuspage.io is CDN-cached, don't set below ~15.",
    )
    ap.add_argument(
        "--no-status-poll",
        action="store_true",
        help="Don't poll the Anthropic status page when API-stalled. You'll "
        "have to type something to retry; otherwise the orchestrator waits "
        "on the bell like a [WAITING] state.",
    )
    ap.add_argument(
        "--auto-reconnect",
        action="store_true",
        help="If a turn fails (e.g. the CLI subprocess crashes), reconnect "
        "with resume=<session id> and auto-continue instead of waiting for you. "
        "Recommended for unattended multi-hour runs.",
    )
    args = ap.parse_args()
    # --show-tool-everything is a convenience that flips both detail flags.
    if args.show_tool_everything:
        args.show_full_commands = True
        args.show_tool_output = True
    # Fill compact-at from the model when the user didn't specify.
    if args.compact_at is None:
        args.compact_at = _default_compact_at(args.model)
    return args


async def _amain() -> None:
    args = parse_args()
    orch = Orchestrator(args)
    with patch_stdout(raw=True):
        await orch.run()


def _enable_windows_ansi() -> None:
    """Enable ANSI / VT processing so colors work in cmd.exe and PowerShell 5.1.

    No-op on non-Windows and on consoles that already have VT enabled. Prefers
    colorama (maintained, handles edge cases); falls back to a direct Win32
    SetConsoleMode call so the orchestrator still works without the dep.
    """
    if sys.platform != "win32":
        return
    try:
        import colorama  # type: ignore

        colorama.just_fix_windows_console()
        return
    except ImportError:
        pass
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        ENABLE_VT = 0x0004
        for handle_id in (-11, -12):  # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
            h = kernel32.GetStdHandle(handle_id)
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(h, ctypes.byref(mode)):
                kernel32.SetConsoleMode(h, mode.value | ENABLE_VT)
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    _enable_windows_ansi()
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
