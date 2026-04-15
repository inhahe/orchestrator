# Orchestrator reference

Companion reference for `orchestrator.py`. Covers every CLI flag, every in-session
slash command, keyboard bindings, and the behavioral rules the orchestrator layers
on top of the Claude Agent SDK.

---

## Default model

The orchestrator does **not** pin a model. When `--model` is omitted, no `model`
field is passed to `ClaudeAgentOptions`, so the underlying Claude Code CLI picks
it. The CLI's selection logic (from `utils/model/model.ts:getDefaultMainLoopModelSetting`):

| User tier                                          | Default model          |
|----------------------------------------------------|------------------------|
| Max subscriber                                     | Opus (1M if enabled)   |
| Team Premium                                       | Opus (1M if enabled)   |
| Anthropic internal (`USER_TYPE=ant`)               | Opus 1M                |
| Everyone else (PAYG, Pro, Enterprise, Team Std.)   | Sonnet 4.6             |

If `model` is set in `~/.claude/settings.json` or `.claude/settings.json`, that
overrides the tier-based default. Pass `--model claude-opus-4-6` (or similar)
to pin explicitly.

---

## CLI flags

### Session

- `--initial-prompt TEXT`, `-p TEXT`
  First message to send on startup. If omitted, the orchestrator waits at the
  prompt for you to type the first message.

- `--no-continue`
  Start a fresh session instead of resuming the most recent one in `--cwd`.
  Default behavior is to resume (equivalent to CLI `claude --continue`).

- `--resume [SESSION_ID]`
  Resume a specific session by its UUID. Pass `--resume` with no value to
  open an **interactive two-step picker**:
  1. **Project picker** — lists every project under
     `~/.claude/projects/*/` that has at least one session. Each row:
     project name, session count, age of newest session, full cwd path.
     Sorted most-recently-used project first. This step is fast — only
     stats files; no JSONL parsing. (If only one project exists, this step
     is skipped automatically.)
  2. **Session picker** — after picking a project, parses just that
     project's JSONLs and shows them newest first. Each row: 8-char
     session id, age (`12m ago`, `3d ago`, …), and either the custom
     title (`★ <title>`, set via `/rename`) or the last user message.
     Enter resumes, Esc goes back to the project picker.

  Both pickers use a custom cursor-as-selection widget: **moving the
  cursor immediately selects** (no Space needed), **Enter confirms the
  current row** (no Tab to OK), Esc/Ctrl-C cancels. PgUp/PgDn jumps by
  10, Home/End jumps to ends. **Mouse: left-click on a row resumes that
  session immediately; mouse-wheel moves the cursor.** Hold Shift while
  dragging to drop out of mouse-capture and select text the normal
  terminal way. The widget handles its own viewport scrolling
  (`[N/M]` indicator at the bottom). Falls back to a two-step numbered
  text list if the TUI can't render. Overrides `--continue`. Combine
  with `--no-replay` to skip rendering the conversation history into
  the backscroll.

  **Auto-cwd switch.** Whether the session id comes from the picker or
  from `--resume <id>` directly, the orchestrator looks up the session's
  recorded cwd and — if it differs from `--cwd` — switches to it before
  starting. This keeps file ops, MCP detection (`.mcp.json`), and the
  per-cwd input-history file (`.orchestrator_history`) lined up with
  what Claude actually remembers. A `[switching cwd ...]` notice is
  printed. If the recorded cwd doesn't exist on the current machine
  (e.g. the project was on another box), the orchestrator stays in
  `--cwd` and prints a warning instead.

- `--no-replay`
  When resuming, do **not** print the prior conversation into the
  backscroll. Default is to replay — matching `claude --continue`. The
  orchestrator reads the session JSONL straight from disk
  (`~/.claude/projects/<slug>/<session-id>.jsonl`) and renders it before
  the prompt opens, with `(history)` tags so historical messages are
  visually distinct from live activity. (The CLI's
  `--replay-user-messages` flag is **not** what does this — its
  description in the source is "re-emit user messages from stdin back on
  stdout for acknowledgment", a different feature.) Which session gets
  rendered: the one passed via `--resume <id>` (or chosen via the
  picker), or the most-recently-modified `.jsonl` in the project dir
  for `--cwd` when relying on `--continue` defaults.

- `--cwd PATH`
  Working directory Claude operates in. Default: `.`. Everything — file ops,
  history file, auto-detected `.mcp.json`, project-scope settings — is keyed
  off this.

### Model & effort

- `--model NAME`
  e.g. `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`.
  Changeable mid-session via `/model <name>` (reconnects + resumes).

- `--effort {auto,low,medium,high,max}`
  Thinking-effort level. `low`/`medium`/`high`/`max` are passed to
  `ClaudeAgentOptions.effort`. `auto` (or omitting the flag) means **don't
  pass `effort=` at all** — the model uses its own default (typically
  `'high'` for Opus/Sonnet 4.6). Changeable mid-session via
  `/effort <level>`.

### Tools & permissions

- `--permission-mode {bypassPermissions,acceptEdits,default,plan}`
  Default: `bypassPermissions` — Claude runs any tool (Read, Write, Edit,
  Bash incl. `run_in_background=true`, BashOutput, KillShell, NotebookEdit,
  WebFetch, WebSearch, Task, Skill, TodoWrite) without prompting.
  Downgrading to `default`/`acceptEdits` exposes `session_state_changed →
  requires_action` events the orchestrator only beeps at — there is no
  approval UI, so turns will stall. Keep `bypassPermissions` unless you
  rebuild that flow.

- `--allowed-tool NAME` (repeatable)
  Whitelist. If set, **only** these tools can run. Omit to allow all
  built-ins.

- `--disallowed-tool NAME` (repeatable)
  Blacklist. Blocks specific tool names.

### Context / compaction

- `--compact-at TOKENS`
  When omitted, **derived from the model**: `950000` when the model id
  looks like a 1M-context variant (contains `[1m]` / `-1m` / ends in
  `1m`), otherwise `160000`. When context tokens (input + cache_read +
  cache_creation) exceed this, the orchestrator injects `/compact` at
  the **next turn boundary**. The CLI's own auto-compact fires at
  `effective_window - 13000` (effective = context_window − 20k reserved
  for output), so:

  | Context window    | CLI auto-compact trigger | Derived `--compact-at` |
  |-------------------|--------------------------|------------------------|
  | 200k (Sonnet)     | ~167k                    | `160000`               |
  | 1M  (Opus 1M)     | ~967k                    | `950000`               |

  The point of our threshold being below the CLI's is **turn-boundary
  predictability** — our `/compact` fires cleanly between turns, whereas
  CLI auto-compact can kick in mid-turn if a big tool dump or heavy
  thinking blows through in one call. Changeable live via `/autocompact`.

- `--no-compact`
  Disable the orchestrator's auto-compact check entirely. Context grows
  unbounded until you run `/compact` or `/clear` (now a context wipe,
  see below) manually. Typically pair with `--max-context-tokens` so
  something caps the window. Toggleable live via `/autocompact off`.

- `--max-context-tokens N`
  Default `0` (disabled). Alternative to auto-compact: when context
  exceeds `N` tokens, the orchestrator **rewrites the session** to the
  tail of the transcript (rolling window, no summarization) and
  reconnects. The cut point is always a user-turn boundary so
  tool_use/tool_result pairs stay intact. A *new* session JSONL is
  written (UUIDs remapped, parentUuid chain restitched, marker record
  at the top); the original untrimmed JSONL is never touched and stays
  resumable from the picker. On each subsequent trim, the previous
  *trimmed* file is deleted so only one trim survives on disk. Token
  accounting is approximate (chars/4 heuristic); 15% headroom is
  baked in. Settable live via `/max-context`.

- `--auto-continue`
  Off by default. Without it, the orchestrator behaves like a normal
  interactive session — after every turn it just waits for your input.
  Enable it to make the orchestrator drive Claude autonomously: after
  each turn that doesn't end with `[WAITING]`, it sends the continue
  prompt automatically. The `--continue-*` flags below have no effect
  unless this is set.

- `--continue-response-delay SECONDS`
  Only with `--auto-continue`. Default: `2.0`. After Claude finishes a
  turn (and is NOT `[WAITING]`), wait this long before sending the next
  auto-continue prompt. This is a one-shot delay measured from the end
  of each turn, not a periodic poll — a 5-minute thinking-and-tooling
  turn never queues up multiple continues. Doubles as the grace window
  during which you can interject: anything you type during this window
  is sent instead of the auto continue.

- `--continue-burst-limit N`
  Only with `--auto-continue`. Default: `3`. Safety brake against the
  failure mode where Claude *should* have emitted `[WAITING]` but didn't,
  and the orchestrator ends up nudging him in a fast loop. If `N`
  consecutive turns finish within `--continue-burst-window` seconds
  without any `[WAITING]`, the orchestrator treats the situation as if
  Claude had emitted `[WAITING]` and stops nudging until you type or an
  async wakeup arrives. Set to `0` to disable the brake.

- `--continue-burst-window SECONDS`
  Default: `180.0` (3 min). Time window paired with `--continue-burst-limit`.

No `--waiting-poll-interval` exists. While Claude is `[WAITING]` the
orchestrator blocks indefinitely on the event queue; the persistent SDK
message dispatcher wakes Claude immediately when a `task_notification`
(bg-shell completion, Task tool result) or `session_state_changed →
requires_action` arrives between turns.

### Prompts & config

- `--append-system-prompt TEXT`
  Appended to the default Claude Code system prompt (does **not** replace it —
  tool instructions remain intact).

- `--mcp-config PATH`
  JSON file with `{"mcpServers": {...}}` shape. If omitted, `.mcp.json` in
  `--cwd` is auto-loaded when present.

### Display & reliability

- `--show-thinking`
  Print the full text of extended-thinking blocks. Default shows a
  single-line collapsed snippet.

- `--show-full-commands`
  Controls whether the **body** of a Bash call (the actual shell lines)
  is printed inline. The Bash header and result line are always shown
  regardless — that's one line `tool Bash — <description> [#N]` for
  the call plus one line `→ N lines, K chars` (dim, on success) or
  `✗ tool error ...` (red, on failure) for the result. Default off:
  just those two lines. With the flag on, every line of the command
  itself is printed between them. The full command body is always
  available via `/tools` while the call is in flight, via `/show N`
  afterwards, and in `/export`.

- `--show-tool-output`
  Print full tool result content inline. Default suppresses it because
  Bash output, file reads, and big greps can fill the screen — instead
  the orchestrator prints just `→ N lines, K chars` (dim) on success or
  `✗ tool error -- N lines, K chars (rerun with --show-tool-output to
  see)` (red) on failure. Either way the full results are persisted in
  the JSONL transcript and visible via `/export`. **Note**: by default
  only Bash results render inline at all (see `--inline-all-tools`), so
  this flag mostly affects Bash unless you've opted in.

- `--show-tool-everything`
  Convenience flag: implies BOTH `--show-full-commands` AND
  `--show-tool-output`. Use when you want full visibility into what
  Claude is doing.

- `--inline-all-tools`
  Render every tool call inline with `[#N]` tags (like Bash does) instead
  of routing them through the transient live-tasks panel. Use when you
  want a scrollable activity log rather than ephemeral status rows.
  While off (default), only Bash scrolls inline; Read/Grep/WebFetch/
  Task/etc. appear in the live panel and vanish on completion.

- `--show-edits {off,compact,full}`
  Default `off`. Controls how `Edit` tool calls render:
  - `off` — in the live panel only; use `/show N` for detail.
  - `compact` — one-liner inline: `tool Edit <path> (+A -R lines) [#N]`
    with green `+A` and red `-R`.
  - `full` — inline with the complete unified diff.

  `--inline-all-tools` overrides this to `full`. When `compact` or
  `full`, Edit is suppressed from the live panel to avoid duplication.

- `--auto-reconnect`
  If a turn fails mid-stream (e.g. the CLI subprocess crashes), reconnect
  with `resume=<session-id>` and re-send the continue prompt instead of
  waiting for user input. Use for unattended multi-hour runs.

---

## In-session slash commands

Type at the prompt and press **Enter**. Tab-completes.

| Command                              | Effect                                                                      |
|--------------------------------------|-----------------------------------------------------------------------------|
| `/help`                              | List commands.                                                              |
| `/status`                            | Print session id, turns, context tokens, cost, effort, model, usage.        |
| `/cost`, `/cwd`                      | Aliases for `/status`.                                                      |
| `/clear`                             | **Clears the conversation context** (Claude Code convention). Disconnects, wipes session id, context tokens, cost, turns, tool/thinking history, todos, active tools, then reconnects without resume/continue. The old session's JSONL stays on disk and is resumable via `--resume`. |
| `/cls`                               | Clears the screen and scrollback only. Keeps the session. (Prior `/clear` behavior.) |
| `/interrupt`, `/i`                   | Stop the current turn. Same effect as pressing Ctrl-C while Claude is working. |
| `/compact`                           | Force a `/compact` on the next turn.                                        |
| `/autocompact [on\|off\|N]`           | Toggle the orchestrator's auto-compact check, or set the threshold. `on`/`off` flips the `--no-compact` state. `N` sets `--compact-at` and re-enables auto-compact if it was disabled. Bare `/autocompact` prints the current state. |
| `/max-context [off\|N]`               | Set/clear `--max-context-tokens` at runtime. `off` disables the rolling-window cap; `N` enables it at ~N tokens. Bare `/max-context` prints the current cap. |
| `/effort <auto\|low\|medium\|high\|max>` | Change thinking effort (`auto` = no override). Disconnects + reconnects with `resume=<session-id>`, so the conversation persists. |
| `/model <name>`                      | Change model. Same reconnect-with-resume behavior.                          |
| `/rename <name>`                     | Set a custom title for the current session. Stored as a `{"type":"custom-title","customTitle":"...","sessionId":"..."}` record appended to the session's JSONL — **same format and location Claude Code uses**, so a rename in either tool shows up in the other. Shown in the `--resume` picker as `★ <name>` and in the bottom toolbar. The reader also picks up Claude Code's auto-generated `ai-title` records (lower priority than `custom-title`). `/rename` with no argument prints the current title. |
| `/auto [on\|off\|toggle]`             | Enable, disable, or flip the `--auto-continue` behavior live. Bare `/auto` toggles. |
| `/burst N [T]`                       | Update the continue-burst safety brake. `/burst 5` sets the count to 5, `/burst 5 60` sets count=5 and window=60s. Bare `/burst` shows current values. Resets the in-flight burst tracker. Only matters when `--auto-continue` is on. |
| `/export [path]`                     | Save the current session's transcript as markdown. With no path, writes `claude-<id8>-<YYYYMMDD-HHMMSS>.md` in cwd. With a directory path, writes the auto-named file inside it. With a file path, writes there exactly. Includes session id, project, timestamps, all user/assistant text, and tool calls/results in fenced code blocks. Thinking blocks are skipped. |
| `/tools`                             | List every tool currently in flight (with id, `[#N]` tag, name, args summary, elapsed time) and every background task running (with id, type, name, runtime). Counts also appear live in the bottom toolbar as `tools[N]: ...names | bg[N]`. |
| `/tasks`                             | List every **non-Bash** tool that ran (or is running) during the current turn — the full set of rows that appeared in the live panel this turn, each with status (`✓` / `✗` / `… running Xs`), duration, `[#N]`, and a one-line input summary. Cleared at the *start* of the next turn, so between turns it still shows the turn that just ended. Bash is excluded since it scrolls inline. |
| `/show [N ...]`                      | Expand collapsed tool calls. Every tool call is numbered `[#N]` (visible in result indicators and in `/tools`). `/show 42` prints `[#42]`'s input and output in full; `/show 42 43 44` prints several. Bare `/show` shows the last 5. History is kept for the last 200 calls per session. |
| `/think [N ...]`                     | Expand extended-thinking blocks by their `[#T<N>]` tag. Each thinking block emitted during the turn gets a numbered preview like `[#T3] (thinking) …`; `/think 3` prints the full text, `/think 3 7` prints several, bare `/think` shows the last 3. Capped at the last 200 blocks per session. |
| `/todos`  `/plan`                    | Show Claude's current `TodoWrite` plan, with `✓` for completed, `→` for in-progress, `·` for pending, and per-status counts. The plan summary (`todos[done/total] -> in-progress label`) also appears live on line 2 of the bottom toolbar; this command shows the full list. |
| `/quit`, `/exit`                     | Graceful exit. Interrupts the current turn (if any), then closes the CLI subprocess. The SDK transport waits up to ~5s for graceful shutdown after sending EOF on stdin so the JSONL transcript flushes the final assistant message — then SIGTERM (5s more) and SIGKILL if the subprocess is wedged. Worst case ~10s, typically <1s when idle. |
| `/quit!`, `/exit!`                   | **Force exit.** Calls `os._exit(0)` immediately — skips the SDK's graceful subprocess-close, so the last in-flight assistant message may not make it into the JSONL. Use when you're sure you're idle and don't want to wait. |

**Also sent through to the CLI** (handled by the Claude Code CLI itself,
not by the orchestrator): anything else starting with `/`. `/compact` reaches
the CLI this way. `/resume`, `/init`, `/memory`, `/agents`, `/mcp`,
`/doctor`, `/output-style`, `/plan`, `/fast`, `/login`, and `#memory-append`
syntax are **not** dispatched — they'll be interpreted by Claude as text.

---

## Keyboard bindings

| Key             | Behavior                                                                 |
|-----------------|--------------------------------------------------------------------------|
| `Enter`         | Submit the current buffer.                                               |
| `Alt-Enter`     | Insert a newline (multi-line input).                                     |
| `Ctrl-C`        | While Claude is working → interrupt the turn. With text in the buffer → clear the buffer. At an empty prompt → exit. |
| `Ctrl-D`        | Exit.                                                                    |
| `Up` / `Down`   | History (persisted to `.orchestrator_history` in cwd).                   |
| `Tab`           | Complete slash commands.                                                 |

Bracketed paste is supported automatically — pasting multi-line text inserts
all lines without triggering a submit.

---

## Behavioral rules

**After-turn behavior.** Without `--auto-continue` (the default), the orchestrator simply waits for your input after every turn — a normal interactive session. The decision tree below applies only when `--auto-continue` is set.

**Auto-continue loop.** With `--auto-continue`, after every turn the orchestrator decides what to do next, in this order:

1. User injected messages during the turn → send them.
2. User hit `/interrupt` or Ctrl-C → wait for user input.
3. Context tokens ≥ `--compact-at` → send `/compact`.
4. Claude's reply contained `[WAITING]` **— or the continue-burst limit fired** (more than `--continue-burst-limit` turns finished within `--continue-burst-window` seconds with no `[WAITING]`, suggesting Claude is spinning) → ring the terminal bell and wait. The wait returns when **(a)** you type something, or **(b)** an async wakeup arrives (the persistent SDK reader saw a `task_notification` or `session_state_changed → requires_action` event and pushed a `wakeup` onto the event queue, which sends the continue prompt). If he still needs you after a wakeup, he just re-emits `[WAITING]`.
5. No user input within `--continue-response-delay` seconds → send the continue prompt: *"If you need input from me before continuing, pause and include the literal token `[WAITING]` in your reply; otherwise, continue working."*

**Session resume.** `--continue` is implicit. Mid-session `/effort` or `/model`
disconnect and reconnect with `resume=<session-id>`, so conversation state
survives.

**Reconnect on crash.** Off by default. Enable with `--auto-reconnect` for
unattended runs.

**Terminal bell (`\a`).** Rung when Claude emits `[WAITING]`, on interrupt,
and on `session_state_changed → requires_action`. Useful for noticing a
multi-hour run that wants you.

**Terminal title.** Set on startup via OSC-0 to `Claude Orchestrator -- <cwd>`.

---

## Always-on status (bottom toolbar)

The bottom toolbar refreshes 2×/sec. It's at least two lines; **the
live-tasks panel adds extra rows** on demand (one per in-flight
non-Bash tool, plus a sub-line under each `Task` subagent row). The
toolbar resizes as tools start and complete.

```
session abc12345 | WORKING | ctx~12345 tok | turns 7 | max 42% / 5h
tools[3]: Bash, Read, Grep | bg[1] | todos[5/12] -> implementing burst limiter
 task[#17] explore: "find usages of foo"  patterns: /foo/ /bar/(*.py)  reads: 2
   → searching /bar/ in src/
 search[#19] /baz/ in src/
 read[#20] src/handler.py
```

- **Line 1**: session id, status (`WORKING` / `WAITING` / `STALLED` /
  `idle`), context tokens, turn count, and one of:
  - **Subscription users**: `<plan>` (dim), e.g. `max` / `pro` / `sub`,
    upgraded to `<plan> NN% / <window>` once a `rate_limit_event`
    arrives (subscription buckets: `five_hour` → `5h`, `seven_day` →
    `7d`, `seven_day_opus` → `7d opus`, `seven_day_sonnet` →
    `7d sonnet`). No rate_limit_event fires until you're near a
    threshold, so the plain plan name is what you'll usually see.
  - **API users** (`ANTHROPIC_API_KEY` / Bedrock / Vertex env set):
    `$X.XXXX` equivalent API cost.
- **Line 2**:
  - `tools[N]: ...names` — every foreground tool call currently in flight (deduped names, capped at 5; `+M` for overflow). `tools: -` when none.
  - `bg[N]` — count of background-shell / Task-tool jobs currently running. `bg: -` when none.
  - `todos[done/total] -> <in-progress label>` — Claude's `TodoWrite` plan summary; the in-progress item's `activeForm` (or `content`) is shown trailing.
- **Live-tasks rows** (dynamic; suppressed by `--inline-all-tools`):
  - `task[#N] <subtype>: "desc"  patterns: /a/(*.py) /b/  reads: K`
    for Task subagents, plus `  → <current sub-op>` on a second line
    (updates as the subagent moves between inner Grep/Read/WebFetch/Bash
    calls — the "current file being searched" effect).
  - `search[#N] /pattern/ in <path> (glob=*.py, type=py)` — bare Grep.
  - `glob[#N] <pattern> in <path>` — bare Glob.
  - `read[#N] <path>` — bare Read.
  - `fetch[#N] <url>` — bare WebFetch.
  - `web[#N] <query>` — bare WebSearch.
  - `edit[#N] <path>` / `write[#N] <path>` — file mutations (unless
    `--show-edits` routes Edit inline instead).
  - Bash is excluded — it scrolls inline with its own `[#N]` tag.

For full detail use `/tools` (per-tool elapsed time + args; per-bg-task
type and runtime), `/tasks` (full turn's task list with status), or
`/todos` (full plan with `✓` / `→` / `·` markers).

---

## Terminal compatibility (Windows)

At startup the orchestrator calls `colorama.just_fix_windows_console()` to
enable ANSI / VT processing on stdout/stderr. This makes colors work in:

- **Windows Terminal** (recommended — gives true color, Unicode, tabs, GPU redraw)
- **PowerShell 7** and **Windows PowerShell 5.1**
- **`cmd.exe`** on Windows 10 1607+
- **VS Code terminal** and other embedded terminals

If `colorama` isn't installed, the orchestrator falls back to a direct Win32
`SetConsoleMode` call, so it still works — installing colorama is preferred
because it also handles older Windows editions and edge-case stdout
redirections.

`PowerShell` vs `cmd.exe` makes no functional difference to the orchestrator;
ANSI support comes from the *terminal* (console host), not the shell.

---

## New-project detection

When the orchestrator starts in a cwd that has no existing
`~/.claude/projects/<sanitized-cwd>/` directory, the startup banner gets an
extra line:

```
  - new project: no Claude Code sessions exist yet for this cwd; one will be
    created at C:/Users/you/.claude/projects/D--my-projects-foo
```

The directory itself is created by the underlying CLI on the first turn —
nothing extra to do; this is purely a heads-up so you know which on-disk
slug your sessions will live under.

---

## Files & config the orchestrator reads

| Path                                   | Purpose                                                   |
|----------------------------------------|-----------------------------------------------------------|
| `<cwd>/.orchestrator_history`          | Readline-style input history (managed by prompt_toolkit). |
| `<cwd>/.mcp.json`                      | Auto-loaded MCP server config, unless overridden.         |
| `~/.claude/` + `<cwd>/.claude/`        | Claude Code settings/skills/agents/hooks/commands.        |
| `<cwd>/CLAUDE.md` and parent-dir ones  | Loaded by the CLI into the system prompt.                 |

`setting_sources=["user","project","local"]` is always passed, so skills
discovered in any of those scopes are picked up.
