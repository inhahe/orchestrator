# Orchestrator reference

Companion reference for `orchestrator.py`. Covers every CLI flag, every in-session
slash command, keyboard bindings, and the behavioral rules the orchestrator layers
on top of the Claude Agent SDK.

---

## Authentication

The orchestrator **cannot initiate a login flow** — the Claude Code CLI's
interactive OAuth flow can't run when its stdin/stdout are piped through
the SDK, and implementing the OAuth callback locally would duplicate the
CLI's logic. Instead, the orchestrator checks at startup whether any
credentials are available:

- `ANTHROPIC_API_KEY` env var set → API mode, OK
- `CLAUDE_CODE_USE_BEDROCK` / `CLAUDE_CODE_USE_VERTEX` set → enterprise
  cloud routing, OK
- `~/.claude/.credentials.json` exists with a valid `accessToken` →
  subscription OAuth, OK
- None of the above → print a clear error telling the user to run
  `claude login` (or set an API key) and exit cleanly

So the expected workflow for first-time subscription users is:
**`claude login`** (in a normal terminal), then start the orchestrator.

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

- `--resume [SESSION_ID | NAME]`
  Resume a specific session. Accepts either a **UUID** (used as-is) or a
  **session name** (matched case-insensitively against custom-title /
  ai-title records in all session JSONLs). If exactly one session matches
  the name, it's used automatically; if multiple match, a disambiguation
  picker is shown; if none match, the orchestrator exits with an error.
  Pass `--resume` with no value to open an **interactive two-step picker**:
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
  `/effort <level>`. `effort` only controls thinking *budget*, not
  whether thinking happens at all — see `--no-thinking` for that.

- `--no-thinking`
  Disable extended thinking entirely at the API level. Sends
  `thinking={"type":"disabled"}` to the SDK so Claude skips the
  reasoning phase. Useful for fast short replies where deliberation
  would just add latency/cost with no quality gain. `--effort`
  becomes a no-op while this is set. Toggle at runtime with
  `/thinking on` / `/thinking off` / `/thinking toggle` — each
  reconnects (preserves session). Toolbar shows `thinking: on`
  (default) or `thinking: off`.

### Tools & permissions

- `--permission-mode {bypassPermissions,acceptEdits,default,plan}`
  Default: `bypassPermissions` — Claude runs any tool (Read, Write, Edit,
  Bash incl. `run_in_background=true`, BashOutput, KillShell, NotebookEdit,
  WebFetch, WebSearch, Task, Skill, TodoWrite) without prompting. With
  any other mode, the orchestrator wires up the SDK's `can_use_tool`
  callback and prompts inline when Claude tries to run a tool that
  needs approval:
  ```
  [permission] Claude wants to run:
    Bash — <description>
    reply 'y' to allow, 'n' (or anything else) to deny, 'a' to allow and remember this tool
  ```
  Type `y` / `yes` / `allow` to approve, `a` / `always` to approve
  (remembering per-tool is a future enhancement; currently same as
  `y`), anything else to deny. The turn blocks until you respond.
  A bell rings (event: `requires-action`) so you notice the prompt
  during long turns. `plan` mode currently has no local UI beyond
  this approval prompt.

- `--allowed-tool NAME` (repeatable)
  Whitelist. If set, **only** these tools can run. Omit to allow all
  built-ins.

- `--disallowed-tool NAME` (repeatable)
  Blacklist. Blocks specific tool names.

### Context / compaction

- `--compact-at TOKENS`
  When omitted, **derived from the model**: `950000` when the model has
  a 1M context window (Opus 4+ models, or any id containing `[1m]` /
  `-1m` / ending in `1m`), otherwise `160000`. When context tokens
  (input + cache_read + cache_creation) exceed this, the orchestrator
  injects `/compact` at
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

- `--no-compact` *(default)*
  Disable the orchestrator's auto-compact check entirely. Context grows
  until the CLI's own auto-compact fires, or you run `/compact` / `/clear`
  manually. This is the **default**: the CLI already handles window
  pressure, so the orchestrator layer is opt-in now.

- `--auto-compact`
  Opt in to the orchestrator's auto-compact check. When on, a `/compact`
  is injected at the next turn boundary whenever `context_tokens >=
  --compact-at`. Reasons to opt in:
  - **Turn-boundary predictability** — our `/compact` fires cleanly
    between turns, whereas the CLI's mid-turn compact can kick in
    partway through a big tool dump.
  - **Keep resident context smaller than the model's window.** A tight
    `--compact-at` saves money for API users (smaller `input_tokens`
    per turn = directly lower bill) and saves rate-limit budget for
    subscription users (smaller per-turn usage = more turns before
    the 5h/7d window fills).

  Toggleable live via `/autocompact on`.

- `--compact-cooldown-turns N` (default `3`)
  After an auto-compact fires, skip the check for this many turns. Stops
  a re-compact loop when the last turn's cumulative-I/O reading stays
  inflated above `--compact-at` even though real resident context just
  shrank.

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

- `--continue-prompt TEXT`
  Override the text sent to Claude on each auto-continue turn. Default
  includes instructions about `[WAITING]`/`[DONE]` tokens. Use
  `/continue-prompt` at runtime to view or change it.

No `--waiting-poll-interval` exists. While Claude is `[WAITING]` the
orchestrator blocks indefinitely on the event queue; the persistent SDK
message dispatcher wakes Claude immediately when a `task_notification`
(bg-shell completion, Task tool result) or `session_state_changed →
requires_action` arrives between turns.

### API-stall detection

- `--api-stall-limit N` (default `5`)
  Enter API-stall mode after N `api_retry` events arrive within the
  sliding `--api-stall-window` seconds. Set to `0` to disable the
  heuristic entirely. Rate-limit and real errors both count.

- `--api-stall-window SEC` (default `60`)
  Sliding window (seconds) paired with `--api-stall-limit`.

- `--status-url URL` (default `https://status.claude.com/api/v2/summary.json`)
  Anthropic Statuspage.io `summary.json` feed. Hit (a) *once* on the
  first real api_retry (rate-limited to once per 60s while not
  stalled), and (b) *periodically* while stalled.

- `--status-poll-interval SEC` (default `30`)
  How often to hit the status feed while API-stalled. Statuspage.io is
  CDN-cached; don't set below ~15.

- `--no-status-poll`
  Skip status-feed polling entirely. Stalls wait for manual input and
  ring the bell on entry.

### Live panels

- `--tasks-panel` / `--no-tasks-panel` (default **off**)
  Show a live tasks panel in the toolbar for in-flight and recently-
  completed tools. Off by default since `--show-tasks compact` (also
  default) already prints tool activity to the scroll. Use
  `--tasks-panel` with a high `--panel-grace` (e.g. 15) to move tool
  activity from scroll to toolbar.

- `--bg-panel` / `--no-bg-panel` (default **on**)
  Show the live background-tasks panel while bg shells / Task subagents
  are running. With `--no-bg-panel`, `/bg` still lists and tails tasks.

- `--todos-panel` / `--no-todos-panel` (default **off**)
  Show a live todos panel in the toolbar with one row per `TodoWrite`
  item — `✓` for completed (dimmed), `→` for in-progress, `·` for
  pending. Panel caps at 50 rows with a `… +N more (/todos for all)`
  overflow line. Off by default since the compact `todos: N/N` badge on
  the status line plus `/todos` for the full list in scrollback usually
  suffices. Turn on when you want Claude's full plan visible at all
  times.

- `--panel-delay SECS` (default `0.0`)
  Seconds a tool must be running before it appears in the toolbar
  panels. Useful to reduce noise from sub-second ops (Read, Grep, Glob).
  `0` shows immediately — the grace period (below) prevents flicker by
  keeping tasks visible for a minimum duration.

- `--panel-grace SECS` (default `10.0`)
  Minimum seconds a task stays visible in the toolbar panel after first
  appearing. If a task completes before this grace period it shows a
  done marker (✓) until the period elapses, so you can see what ran.
  Higher values (10-15s) give a useful activity summary when
  `--tasks-panel` is on. `0` disables.

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
  available via `/tools` while the call is in flight, via `/show tN`
  afterwards, and in `/export`.

- `--show-tool-output`
  Print full tool result content inline. Default suppresses it because
  Bash output, file reads, and big greps can fill the screen — instead
  the orchestrator prints just `→ N lines, K chars` (dim) on success or
  `✗ tool error -- N lines, K chars (rerun with --show-tool-output to
  see)` (red) on failure. Either way the full results are persisted in
  the JSONL transcript and visible via `/export`. **Also governs history
  replay**: with the flag on, resumed sessions print the last ≤600
  chars of each tool_result; with it off, history shows the same
  `→ N lines, K chars` summary. **Note**: by default only Bash results
  render inline at all (see `--inline-all-tools`), so this flag mostly
  affects Bash unless you've opted in.

- `--show-tool-everything`
  Convenience flag: implies BOTH `--show-full-commands` AND
  `--show-tool-output`. Use when you want full visibility into what
  Claude is doing.

- `--inline-all-tools`
  Render every tool call inline with `[#N]` tags (instead of the
  transient live panel). Controls **visibility** only — detail level is
  driven by `--show-tasks` / `--show-edits` separately. With defaults
  (both `compact`) you get one-liner headers for every tool. Combine with
  `--show-tasks full` (and/or `--show-edits full`) when you also want
  detail (Edit diffs, Write/NotebookEdit previews, full TodoWrite plan,
  MCP arg dumps).

- `--show-tasks {off,compact,full,full+output}` (default `compact`)
  Print non-Bash tool activity to the scrolling log.
  - `compact` — one-liner per tool start and result (`Grep /pat/ path`
    followed by `→ N lines, K chars`). Tool names in bold blue,
    parameters in dim gray.
  - `full` — tool start with detail (Edit diffs, Write previews,
    full TodoWrite plan, MCP arg dumps) + result summary.
  - `full+output` — everything `full` does, plus full tool output
    content (instead of just size summaries).
  - `off` — toolbar panel only (with `--tasks-panel`), nothing inline.

- `--show-edits {off,compact,full}` (default `compact`)
  Controls how `Edit` tool calls render:
  - `compact` — one-liner inline: `Edit <path>  (+A -R lines)` with
    green `+A` and red `-R`.
  - `full` — inline with the complete unified diff.
  - `off` — panel / `/show tN` only (with `--show-tasks` as fallback;
    `--show-tasks=full` then drives Edit to `full`).

  `--inline-all-tools` no longer auto-upgrades Edit to `full` — detail is
  always opt-in via `--show-edits=full` (or `--show-tasks=full`). The flag
  only forces *visibility* (Edit renders inline even when both `--show-edits`
  and `--show-tasks` are off).

- `--auto-reconnect`
  If a turn fails mid-stream (e.g. the CLI subprocess crashes), reconnect
  with `resume=<session-id>` and re-send the continue prompt instead of
  waiting for user input. Use for unattended multi-hour runs.

- `--ascii-only`
  Render status markers as ASCII (`>`, `v`, `x`, `-`) instead of Unicode
  (`▶`, `✓`, `✗`, `⏹`, `→`, `·`). Default is Unicode — every modern
  terminal/font combo (Windows Terminal, iTerm2, mainstream Linux
  emulators) renders BMP glyphs cleanly. Flip this on if the glyphs
  show up as boxes or `?` in your terminal, or when piping scrollback
  to a consumer that chokes on non-ASCII. Affects: bg-task `▶ started`
  / `▶ completed` / `✗ failed` / `⏹ stopped` markers, tool-result
  `→ N lines` arrows, TodoWrite `✓`/`→`/`·` status markers, `/tasks`
  status glyphs, the `--resume` picker's current-row `▶`, and the
  rate-limit middle-dot separator.

---

## In-session slash commands

Type at the prompt and press **Enter**. Tab-completes.

| Command                              | Effect                                                                      |
|--------------------------------------|-----------------------------------------------------------------------------|
| `/help`                              | List commands.                                                              |
| `/status`                            | Print session id, turns, context tokens, cost, effort, model, usage.        |
| `/cost`, `/cwd`                      | Aliases for `/status`.                                                      |
| `/debug`                             | Internal diagnostic dump — shows `busy`/`needs_user_attention`/`turn_active` flags, event queue and turn queue sizes, queued-prompt contents, counts of active tools / background tasks, pending permission future status, and each internal asyncio task's state (input, worker, dispatcher, status poller, rate watcher) with where it's parked if running. Use when a typed message at idle doesn't echo `you:`, or when `/exit` takes its full 25s before the watchdog fires. Runs as an immediate command, so it works even when the worker is wedged. |
| `/clear`                             | **Clears the conversation context** (Claude Code convention). Disconnects, wipes session id, context tokens, cost, turns, tool/thinking history, todos, active tools, then reconnects without resume/continue. The old session's JSONL stays on disk and is resumable via `--resume`. |
| `/cls`                               | Clears the screen and scrollback only. Keeps the session. (Prior `/clear` behavior.) |
| `/interrupt`, `/i`                   | Stop the current turn. Same effect as pressing Ctrl-C while Claude is working. |
| `/compact`                           | Force a `/compact` on the next turn.                                        |
| `/autocompact [on\|off\|N]`           | Toggle the orchestrator's auto-compact check, or set the threshold. `on`/`off` flips the `--no-compact` state. `N` sets `--compact-at` and re-enables auto-compact if it was disabled. Bare `/autocompact` prints the current state. |
| `/max-context [off\|N]`               | Set/clear `--max-context-tokens` at runtime. `off` disables the rolling-window cap; `N` enables it at ~N tokens. Bare `/max-context` prints the current cap. |
| `/continue-prompt [TEXT\|default]`    | View/change the auto-continue prompt. Bare prints the current value. `/continue-prompt default` resets to the built-in default. Any other argument replaces it. Takes effect on the next auto-continue turn. |
| `/bell [all\|none\|EVENTS]`           | View/change which events ring the terminal bell. Bare `/bell` prints the current set and lists available events. `/bell all` or `/bell none` replaces the set. Otherwise incremental: comma-separated event names with optional `on`/`off` suffix (default `on`). Examples: `/bell turn-done on` adds turn-done; `/bell bg-done off` removes bg-done; `/bell turn-done on,bg-done off` both. Use this to enable `turn-done` temporarily when stepping away from a long turn, then disable it after. |
| `/queue [N\|drop N\|clear]`           | View/manage prompts queued while Claude is busy (the live panel also shows them). Bare `/queue` lists them (numbered, multi-line shown as `+N more lines`). `/queue N` prints prompt #N's full text. `/queue drop N` removes #N (others shift). `/queue clear` empties the queue. The queue is also cleared automatically when you interrupt with Ctrl-C — the assumption is that interrupting means redirecting, so the old queue is probably stale. A pending `/compact` queued during the same turn is dropped on interrupt too (with a `[pending /compact dropped on interrupt]` notice). |
| `/panel [tasks\|bg\|todos [on\|off\|toggle]]` | Show or change visibility of the three toolbar panels at runtime. Bare `/panel` prints the current on/off state of each. `/panel <name>` toggles. `/panel <name> on` / `... off` sets explicitly. Valid names: `tasks` (foreground-tools panel, default off), `bg` (background-tasks panel, default on), `todos` (TodoWrite panel, default off). Mirrors the startup flags `--tasks-panel` / `--bg-panel` / `--todos-panel`. |
| `/effort <auto\|low\|medium\|high\|max>` | Change thinking effort (`auto` = no override). Disconnects + reconnects with `resume=<session-id>`, so the conversation persists. |
| `/effort`                            | No arg: print the current effort level and the list of available levels with short descriptions. |
| `/thinking [on\|off\|toggle]`         | Enable/disable API-level extended thinking. `on` (default) = let the SDK/model decide (budget governed by `/effort`); `off` = send `thinking={"type":"disabled"}` so Claude skips reasoning entirely. Bare `/thinking` toggles. Reconnects with `resume=<session-id>`, so conversation persists. Toolbar shows current state as `thinking: on/off`. |
| `/model <name>`                      | Change model. Same reconnect-with-resume behavior.                          |
| `/model`                             | No arg: print the current model (pinned via `--model` or CLI-picked from `AssistantMessage.model`) and a list of known model IDs (Opus 4.6, Opus 4.6[1m], Sonnet 4.6, Haiku 4.5). |
| `/connect`, `/reconnect`            | Reconnect to the Claude Code CLI subprocess. If a session is already active, disconnects and resumes the same session; if no session exists yet, starts a fresh connection. Any entries in the background-tasks panel are cleared on reconnect (the old CLI subprocess's child processes are dead; if any tasks are still alive, the new CLI will emit fresh `task_started` messages). Useful if the CLI subprocess dies or the SDK transport gets wedged — saves you from restarting the orchestrator. |
| `/rename <name>`                     | Set a custom title for the current session. Stored as a `{"type":"custom-title","customTitle":"...","sessionId":"..."}` record appended to the session's JSONL — **same format and location Claude Code uses**, so a rename in either tool shows up in the other. Shown in the `--resume` picker as `★ <name>` and in the bottom toolbar. The reader also picks up Claude Code's auto-generated `ai-title` records (lower priority than `custom-title`). `/rename` with no argument prints the current title. |
| `/auto [on\|off\|toggle]`             | Enable, disable, or flip the `--auto-continue` behavior live. Bare `/auto` toggles. |
| `/burst N [T]`                       | Update the continue-burst safety brake. `/burst 5` sets the count to 5, `/burst 5 60` sets count=5 and window=60s. Bare `/burst` shows current values. Resets the in-flight burst tracker. Only matters when `--auto-continue` is on. |
| `/export [path]`                     | Save the current session's transcript as markdown. With no path, writes `claude-<id8>-<YYYYMMDD-HHMMSS>.md` in cwd. With a directory path, writes the auto-named file inside it. With a file path, writes there exactly. Includes session id, project, timestamps, all user/assistant text, and tool calls/results in fenced code blocks. Thinking blocks are skipped. |
| `/tools`                             | List every tool currently in flight (with id, `[#N]` tag, name, args summary, elapsed time) and every background task running (with id, type, name, runtime). The bg count also appears live in the bottom toolbar as `bg[N]`. |
| `/tasks`                             | List every **non-Bash** tool that ran (or is running) during the current turn — the full set of rows that appeared in the live panel this turn, each with status (`✓` / `✗` / `… running Xs`), duration, `[#N]`, and a one-line input summary. Cleared at the *start* of the next turn, so between turns it still shows the turn that just ended. Bash is excluded since it scrolls inline. |
| `/show [tN\|bN\|kN ...]`              | **Unified viewer** — inspect tool calls, background tasks, or thinking blocks by their `[#<letter><N>]` tag. Prefix letters: `t` = tool call, `b` = background task, `k` = thinking block. A bare number (`/show 42`) defaults to `t`. `-tail K` (legacy `-K`) appended to a ref tails the last K lines of that entry's output; applies to `t` and `b` (thinking has no output stream, `-tail` is a no-op). Multi-ref mixes types freely: `/show t5 b3 k17 -tail 30 t7 -tail 10`. Bare `/show` summarises recent entries of each type. Histories capped at last 200 per type. |
| `/btw <question>`                    | Side question (mirrors Claude Code's `/btw`). Fires immediately even mid-turn — runs as a background task using an independent one-shot `query()` in a fresh isolated subprocess, so it doesn't interrupt or conflict with the active turn. Nothing is written to the main session's JSONL. Inherits current `--model` / `--effort` / `--permission-mode` / tool allow/deny. Response is prefixed `claude (btw):` and interleaves with the main turn's output in scrollback. |
| `/bg`  `/background`                 | List every background task relevant to this turn (started this turn + any still-running carryover from prior turns), each as a compact one-liner with `[#bN]`, status marker, task type, name, and short task_id. **Detail view moved to `/show bN`** — `/bg N` now errors with a redirect hint. When carryover tasks exist, a hint shows the dismiss command. |
| `/bg dismiss [N\|all]`               | Remove stale background-task entries that never received a completion notification from the SDK. With a sequence number (`/bg dismiss 3`), removes just that task. With `all` or no argument (`/bg dismiss`), clears every entry. Useful when `local_bash` tasks accumulate as "running" indefinitely because the SDK didn't send a terminal status update. |
| `/todos`  `/plan`                    | Show Claude's current `TodoWrite` plan, with `✓` for completed, `→` for in-progress, `·` for pending, and per-status counts. The plan summary (`todos[done/total] -> in-progress label`) also appears live on line 2 of the bottom toolbar; this command shows the full list. |
| `/graphify <path> [opts]`            | Build a knowledge graph from a folder using the [graphify](https://github.com/safishamsi/graphify) skill (PyPI: `graphifyy`). Sends a skill-prefixed message to Claude so it can run the graphify Python API via tool use. Subcommands: `/graphify . --update` (incremental), `/graphify query "Q"` (search graph), `/graphify path "A" "B"` (shortest path), `/graphify explain "Node"` (neighborhood), `/graphify add <url>` (fetch+ingest), `/graphify . --watch` (auto-rebuild), `/graphify . --wiki` (build wiki). Outputs go to `graphify-out/` (HTML visualization, JSON graph, audit report, Obsidian vault). **First-time setup:** `pip install graphifyy && graphify install` (installs tree-sitter parsers and the Claude Code skill file). |
| `/quit`, `/exit`                     | Graceful exit. Interrupts the current turn (if any), then closes the CLI subprocess. The SDK transport waits up to ~5s for graceful shutdown after sending EOF on stdin so the JSONL transcript flushes the final assistant message — then SIGTERM (5s more) and SIGKILL if the subprocess is wedged. Worst case ~10s, typically <1s when idle. The orchestrator enforces a hard 12s cap on the SDK disconnect and a 3s cap on the dispatcher-task cancel; a daemon-thread shutdown watchdog also guarantees `os._exit(1)` 25s after `/exit` is pressed if anything in the shutdown path is wedged, so you're never stuck at a dead prompt. |
| `/quit!`, `/exit!`                   | **Force exit.** Calls `os._exit(0)` immediately — skips the SDK's graceful subprocess-close, so the last in-flight assistant message may not make it into the JSONL. Use when you're sure you're idle and don't want to wait. |

**Unknown slash commands** produce an inline error: `unknown command /<cmd> (try /help)`.
They are _not_ forwarded to the CLI. `/compact` is the only command that the
orchestrator explicitly queues for the CLI to handle. `/resume`, `/init`,
`/memory`, `/agents`, `/mcp`, `/doctor`, `/output-style`, `/plan`, `/fast`,
`/login`, and `#memory-append` syntax are **not** dispatched — they'll be
interpreted by Claude as text.

---

## Keyboard bindings

| Key             | Behavior                                                                 |
|-----------------|--------------------------------------------------------------------------|
| `Enter`         | Submit the current buffer.                                               |
| `Shift-Enter`   | Insert a newline (multi-line input). Works on Windows Terminal and VT100 terminals that send `modifyOtherKeys` sequences. |
| `Ctrl-Enter`    | Insert a newline (same as Shift-Enter).                                  |
| `Ctrl-J`        | Insert a newline (universal fallback — always works regardless of terminal). |
| `Ctrl-Z`        | Undo the last edit in the input buffer.                                  |
| `Escape`        | Clear the input buffer instantly (no delay — fires eagerly on all platforms). Note: because Escape is eager, `Alt-Enter` (Esc then Enter) no longer works as a newline shortcut; use `Ctrl-J` instead. |
| `Ctrl-C`        | With a Shift-selection active → **copy** selected text to the system clipboard and clear the selection. While Claude is working → interrupt the turn (also clears any queued prompts and any pending `/compact` — interrupting implies redirecting, so prior intent is dropped). With text in the buffer → clear the buffer. At an empty prompt → no-op (sets the interrupt flag in case something off-turn is running, but never exits — use `/exit` for that). |
| `Ctrl-D`        | Exit.                                                                    |
| `Up` / `Down`   | Move the cursor within multi-line input. **Does not navigate history** — the input is a mini-editor. Completion-menu navigation when `/…` completions are open. History is still persisted to `.orchestrator_history` in cwd but accessed via `Ctrl-R` / `Ctrl-S` search. |
| `Backspace`     | With a Shift-selection active → delete the selected text. Otherwise delete the character before the cursor. |
| `Delete`        | With a Shift-selection active → delete the selected text. Otherwise delete the character after the cursor. |
| `Shift-Left/Right` | Character-level text selection. Highlighted text can be copied with the terminal's native copy shortcut. Any plain (non-Shift) movement clears the selection. Left/Right with an active selection collapse the cursor to the near/far edge respectively. Typing any character while a selection is active **replaces** the selection with the typed character (standard editor behavior). |
| `Shift-Up/Down` | Extend selection by one visual row (same wrap-aware math as plain Up/Down). |
| `Shift-Home/End` | Extend selection to the start/end of the current visual row.            |
| `Ctrl-Shift-Left/Right` | Word-level selection (previous word beginning / next word ending). |
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
5. No user input within `--continue-response-delay` seconds → send the continue prompt: *"If you need input from me before continuing, pause and include the literal token `[WAITING]` in your reply. If you are finished with all your tasks, include the literal token `[DONE]` instead. Otherwise, continue working."* The two sentinels are distinguishable in the toolbar: red `● waiting` (wants input) vs. bright-cyan `● done` (all tasks complete). Both pause auto-continue until you type.

**Post-compact auto-continue.** When a compact fires *during* a turn, the
orchestrator auto-continues only if the compacted turn was itself
auto-generated (the continue prompt or `/compact`).  If *you* sent the
message that triggered the compact, the orchestrator waits for your input
after the turn — your explicit message means you own the flow.

**Session resume.** `--continue` is implicit. Mid-session `/effort` or `/model`
disconnect and reconnect with `resume=<session-id>`, so conversation state
survives.

**Reconnect on crash.** Off by default. Enable with `--auto-reconnect` for
unattended runs.

**Terminal bell (`\a`).** The `--bell-on` flag selects which events ring
the terminal bell, so you can tune "attention needed" notifications.
Event names:

| Event             | When it fires                                              |
|-------------------|------------------------------------------------------------|
| `turn-done`       | Auto-continue off, a turn just ended → your turn to type   |
| `waiting`         | Claude emitted `[WAITING]` (auto-continue mode)            |
| `done`            | Claude emitted `[DONE]` — all tasks complete               |
| `stalled`         | Burst-limit brake fired (spinning in auto-continue)        |

The four "turn-completion" events above (`turn-done`, `waiting`, `done`,
`stalled`) are **deferred while background tasks are still running** —
the bell is held back and fires only when the last bg task finishes,
so you get one "needs attention" signal per logical batch of work
rather than one per turn plus one per bg completion. If you actually
want per-task bells, enable `bg-done` explicitly.
| `api-stall`       | Entering API-stall mode (Anthropic outage / rate-limit)    |
| `api-ok`          | Status poller says API is back                             |
| `interrupt`       | User Ctrl-C'd a turn                                       |
| `bg-done`         | A background task completed                                |
| `requires-action` | `session_state_changed → requires_action` from the SDK     |
| `rate-hit`        | Subscription rate limit was just hit (status → `rejected`) |
| `rate-reset`      | Subscription rate limit's `resets_at` time passed — Claude can resume |

Defaults:
`waiting,done,stalled,api-stall,requires-action,rate-hit,rate-reset`
(the attention-demanding events). `turn-done` is **off** by default because
ringing after every interactive reply gets old fast — enable it (or
use `--bell-on all`) if you typically step away during long turns.
`api-ok`, `bg-done`, and `interrupt` are also off by default; add
them if you want the confirmation bell. Use `--bell-on none` to
silence everything.

Runtime control: the `/bell` slash command does the same thing
incrementally without a restart. `/bell turn-done on` adds turn-done
to the current set; `/bell turn-done off` removes it. You can mix on
the same line: `/bell turn-done on,bg-done off`. This is the
canonical workflow for "enable bell temporarily during this one long
turn, then disable it again."

Note: whether `\a` makes an audible sound depends on your terminal —
Windows Terminal defaults to a visual flash (flash window + taskbar
icon) unless `bellStyle` is set to `"audible"` in your profile. Either
way it's a notification; it just might not have a sound.

**Subscription rate limits.** When the SDK emits a `rate_limit_event`
with `status=rejected` and a `resets_at` timestamp, the orchestrator:
- Rings the `rate-hit` bell on the transition into `rejected`, so you
  notice right away that Claude just got blocked.
- Replaces the per-bucket utilization gauge (`NN% / <window>[, ...]`)
  with `rate-limit reset: <time>` (red) so you can see when Claude
  will be available again.
- Starts a 1-Hz watcher task. When `resets_at` passes, it rings the
  `rate-reset` bell, clears the rejected state, prints
  `[rate-limit reset — ready to resume]`, and pushes a wakeup event.
- If `--auto-continue` is on, the wakeup causes the driver loop to
  resume (sends the continue prompt). If off, you just see the
  notification and can decide when to resume.

**Background-task completion.** When a bg task finishes, the orchestrator
rings the bell and prints `[wakeup -- <info>]`. It does **not** start a
new orchestrator-initiated turn — the SDK/CLI has already injected the
`<task-notification>` into Claude's context, and Claude typically
auto-responds to it. That auto-response arrives while `turn_active` is
False and renders with a `claude (async):` prefix to distinguish it from
turn-driven output. The orchestrator then waits for real user input.

**Init logging.** A line like `[init] session a9ef6988-e75` prints on
first connect. On reconnects (after `/effort` / `/model` changes,
`/clear`, or `--auto-reconnect`), if the SDK returns a different
session ID than expected, the line reads
`[init] SDK forked: expected <old>, got <new> (context was carried over)`.

**Silent-fresh-session detection.** When the orchestrator passes
`--continue` or `--resume <id>` to Claude Code, it pre-computes which
session id it expects the SDK's first init to report. In print mode
(which the SDK always uses), Claude Code's `--continue` silently starts
a *fresh* session if `loadConversationForResume` returns null — no error,
no warning. The orchestrator catches this by comparing the SDK's reported
init id to the expected one. On mismatch at first init, you see
`[!! warning] expected to resume session <X>, but Claude Code silently
started a fresh session <Y> -- prior conversation context was NOT loaded.
To recover: /quit, then re-run with --resume <X>`. Without this check,
the empty fresh session becomes the new "most recent" and every
subsequent `--continue` resumes *it* instead of the real prior session
— so a single silent failure perpetuates indefinitely.

**Terminal title.** Set on startup via OSC-0 to `Claude Orchestrator -- <cwd>`.

---

## Always-on status (bottom toolbar)

The bottom toolbar refreshes 2×/sec via a persistent prompt_toolkit
`Application` that stays alive for the entire session (prompt and
toolbar never disappear between inputs). It's at least two lines;
panels add extra rows on demand.

**Backgrounds:** the status line uses a light-gray background.
Task/bg detail rows inherit the light status background. Each panel's
header line includes a compact hint suffix (e.g. `(/bg: list, /show bN: detail)`)
rendered in dark-gray on the same line as the dash band.

**Which panels show:**
- The foreground **tasks panel** (one row per non-Bash tool in flight)
  is **off by default** (`--no-tasks-panel`). Tool activity goes to
  the scroll via `--show-tasks compact` instead. Enable with
  `--tasks-panel` if you want ephemeral status rows.
- The **background-tasks panel** is on by default (`--bg-panel`),
  showing one row per running `run_in_background=true` Bash or `Task`
  subagent.
- The **todos panel** is **off by default** (`--no-todos-panel`).
  Enable with `--todos-panel` to see one row per `TodoWrite` item
  (completed / in-progress / pending) in the toolbar at all times;
  otherwise just the `todos: N/N` badge on the status line shows.
- The **queued-prompts panel** is always on (no flag) and only appears
  when you typed one or more messages while Claude was busy. One row
  per queued message, numbered `[#N]` with the first line shown
  (truncated with `…` if too long). Manage via `/queue` (see
  Commands). The panel auto-hides when the queue drains to empty.

**No ghost rows.** The toolbar is rendered outside prompt_toolkit via
DECSTBM (Set Top and Bottom Margins) scroll-region management.  The
scroll region confines all normal terminal scrolling to the area above
the toolbar; the toolbar itself is painted into fixed bottom rows using
ANSI cursor positioning and SGR colour sequences.  Because the toolbar
is never part of prompt_toolkit's rendered area, there is no "reserved
space" that the terminal can fail to reclaim when the toolbar shrinks.

```
session abc12345 | working (1m 32s) | ctx~12345 tok | turns 7 | max 42% / 5h
bg[1] | todos[5/12] -> implementing burst limiter
 task[#17] explore: "find usages of foo"  patterns: /foo/ /bar/(*.py)  reads: 2
   → searching /bar/ in src/
 search[#19] /baz/ in src/
 read[#20] src/handler.py
```

- **Line 1**: session id, status (`connecting Ns` / `working (duration)` /
  `waiting` / `stalled` / `idle`), context tokens, turn count, and one of:
  - `working (1m 32s)` includes the live elapsed duration of the current
    turn, updated at the toolbar's refresh rate (~2 Hz). Hidden between
    turns.
  - `connecting` is shown while the SDK is starting the Claude Code CLI
    subprocess and completing the control-protocol `initialize`
    handshake. This phase can take tens of seconds on large sessions
    because Claude Code has to parse the same JSONL we're resuming.
    Anything you type during `connecting` lands in `event_queue` with
    no consumer yet — it'll be picked up once the handshake completes.
  - **Subscription users**: `plan: <name>` (e.g. `plan: max` / `pro` /
    `sub`). Once `rate_limit_event`s arrive, a **separate toolbar
    section** appears with the per-bucket utilizations: `NN% / <window>`,
    comma-separated when multiple buckets are tracked simultaneously
    (e.g. `32% / 5h, 80% / 7d`). Subscription buckets: `five_hour` →
    `5h`, `seven_day` → `7d`, `seven_day_opus` → `7d opus`,
    `seven_day_sonnet` → `7d sonnet`. Each bucket's event updates only
    its own slot, so a 5h update doesn't clobber an in-flight 7d gauge.
    Rendering order is stable (5h first, then 7d, then model-specific
    7d) regardless of which event fired most recently. No
    rate_limit_event fires until you're near a threshold, so the
    plain `plan: …` is what you'll usually see.
  - **API users** (`ANTHROPIC_API_KEY` / Bedrock / Vertex env set):
    `$X.XXXX` equivalent API cost.
- **Line 2**:
  - `bg[N]` — count of background-shell / Task-tool jobs currently running. `bg: -` when none.
  - `todos[done/total] -> <in-progress label>` — Claude's `TodoWrite` plan summary; the in-progress item's `activeForm` (or `content`) is shown trailing.
  - (No foreground-tools badge: Claude typically runs one foreground tool at a time, so the badge would just duplicate what's already inline in scrollback or in the `--tasks-panel`. If you want it back, flip `_SHOW_TOOLS_BADGE = True` in orchestrator.py — the implementation is preserved.)
- **Live-tasks rows** (dynamic; only when `--tasks-panel` is on):
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
| `<script-dir>/orchestrator-colors.conf` | Colour-scheme overrides for scrollback output, colocated with `orchestrator.py` so it's trivially findable (auto-created with defaults on first run). Format: `NAME=[fg=PARAMS] [bg=PARAMS] [bold] [$ref ...]` per line — each token optional, any order. `$ref` pulls in another entry's spec (variable or colour, case-insensitive) and later tokens override field-by-field. `$name=<spec>` defines a reusable variable. Built-in colours (`$dim`, `$path`, etc.) are pre-seeded so they're referenceable without redefinition. Example: `$accent=fg=38;5;208 bold` then `PATTERN=$accent`. Semantic names listed below. |
| `<script-dir>/orchestrator.log`        | Append-only log of uncaught asyncio exceptions. Each entry has a `=== <ISO timestamp>  [asyncio] ===` header followed by the full traceback. On screen you only see a one-line `[exception: <Type>: <message> -- logged to orchestrator.log]` notice in yellow, so a transient prompt_toolkit/SDK hiccup doesn't dump pages of traceback into the scrollback. |

**Colour names** — each drives a specific role in scrollback rendering:

| Name | Drives |
|------|--------|
| `RESET` | Reset SGR (clear all attributes). |
| `DIM` | `[#N --]` sequence tags, `k=v` suffixes (offset/limit, shell=X), size hints like `(N lines, N chars)`, grace-period markers. |
| `DIM_BOLD` | Inline slash-command hints shown alongside tool calls (e.g. `/show 17 [-tail N]`). |
| `BOLD_BLUE` | Tool names (`Edit`, `Write`, `Bash`, `Grep`, `Task`, etc.). |
| `RED` / `BOLD_RED` | Errors, removed lines in diffs, error one-liners in tool results. |
| `GREEN` | Added lines in diffs, ✓ completion markers. |
| `YELLOW` | Warnings, interrupt notice, in-progress `→` markers. |
| `BLUE` / `BOLD_BLUE` | Blue accents (tool-label foreground). |
| `MAGENTA` | System/sys messages (e.g. `[orchestrator: compacting session]`, `[sys] effort -> X`). |
| `CYAN` / `BRIGHT_CYAN` | Thinking blocks, wakeup notices, Bash `$` prompt char. |
| `BOLD` | Standalone bold (no fg/bg) for headings and emphasis. |
| `PATH` | Filesystem paths — Read/Edit/Write/NotebookEdit, Grep's search path, Glob's search path. |
| `URL` | WebFetch URL. |
| `PATTERN` | Grep regex (`/…/`), Glob pattern, WebSearch query. |
| `COMMAND` | Bash command body (in backticks). |
| `DESC` | Freeform descriptions — Bash `— description`, Task `[subtype] description`. |

`setting_sources=["user","project","local"]` is always passed, so skills
discovered in any of those scopes are picked up.
