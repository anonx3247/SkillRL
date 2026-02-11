---
phase: 01
plan: 03
subsystem: agent
tags: [deepseek, agent-loop, tool-calling, trajectory, python]

requires: [01-01, 01-02]
provides:
  - DeepSeek API client with retry logic
  - Autonomous think-act-observe agent loop
  - OpenAI function-calling tool schema for 12 ALFWorld tools
  - CLI entry point for single-task execution
  - Formatted live step display with agent index prefix
affects: [02-01]

tech-stack:
  added: [openai, tenacity]
  patterns: [async-agent-loop, tool-to-command-mapping, on-step-callback]

key-files:
  created:
    - src/agent/client.py
    - src/agent/prompts.py
    - src/agent/loop.py
    - src/agent/__init__.py
    - src/main.py
  modified:
    - src/environment/env_manager.py
    - src/environment/actions.py

decisions:
  - decision: "Use deepseek-chat model, NOT deepseek-reasoner for tool calling"
    rationale: "deepseek-reasoner doesn't support function calling"
    impact: "Model selection is critical for tool-use agent"
  - decision: "Approach A (Direct): agent loop calls env_manager.step() directly, not via MCP"
    rationale: "Simpler for single-agent execution, avoids MCP overhead"
    impact: "FastMCP server exists for future parallel use but isn't called during agent execution"
  - decision: "ALFWorld uses 'take X from Y' and 'move X to Y' commands"
    rationale: "Discovered empirically: 'take X' and 'put X in/on Y' return 'Nothing happens'"
    impact: "take tool requires from_receptacle param; put tool generates 'move' command"
  - decision: "Minimal system prompt — no strategy hints"
    rationale: "Strategies like 'go_to before interact' should be learned skills, not hardcoded"
    impact: "Agent starts naive; skill library will improve it over iterations"

metrics:
  duration: ~30min (including debugging and checkpoint verification)
  completed: 2026-02-11
---

# Phase 1 Plan 3: Agent Loop & End-to-End Execution Summary

**One-liner:** DeepSeek-powered autonomous agent loop with tool calling, trajectory capture, and formatted CLI output

## Objective

Implement the core agent loop connecting DeepSeek API to ALFWorld environment, executing think-act-observe cycles and capturing complete trajectories with a CLI entry point.

## What Was Built

### 1. DeepSeekClient (src/agent/client.py)

- AsyncOpenAI wrapper targeting api.deepseek.com
- Model: `deepseek-chat` (NOT deepseek-reasoner — no tool support)
- Tenacity retry: exponential backoff (4-60s), 5 attempts, retries on rate limit / connection / server errors
- 120s per-call timeout via asyncio.wait_for

### 2. System Prompt (src/agent/prompts.py)

- Minimal `AUTONOMOUS_AGENT_PROMPT`: lists 12 tools with signatures, 6 brief instructions
- No strategy hints — agent learns through skill library evolution

### 3. Agent Loop (src/agent/loop.py)

- `build_tools_spec()`: OpenAI function-calling schema for 12 tools
- `_map_tool_to_command()`: Maps tool names to ALFWorld command strings
- `run_task()`: Async think-act-observe loop with:
  - Step/wall-clock timeouts
  - Tool call parsing and execution
  - task_completed signal handling
  - Trajectory construction
  - `on_step` callback for live display
- `format_step()`: Formats steps with `[Agent N]` prefix for display

### 4. CLI Entry Point (src/main.py)

- `--task-index`, `--max-steps`, `--output-dir`, `--agent-index` arguments
- Live step display via on_step callback
- Formatted summary with agent prefix
- Agent/environment success discrepancy detection
- Trajectory persistence to JSONL

### 5. Bug Fixes Applied During Verification

**Python 3.14 eval() compatibility (critical):**
- Root cause: TextWorld's `EvalSymbol.derive()` uses `locals().update()` then `eval()`, but Python 3.14 changed `locals()` behavior — updates don't affect eval scope
- Fix: Replaced monkey-patch to pass variables directly to `eval()` as locals dict
- Impact: All ALFWorld observations now show real object/location names

**ALFWorld command format fixes:**
- `take`: Requires `take X from Y`, not `take X` — added `from_receptacle` parameter
- `put`: ALFWorld uses `move X to Y`, not `put X in/on Y` — changed underlying command
- `task_id`: Changed key from `"game_file"` to `"extra.gamefile"` with list un-batching
- `admissible_commands`: Added un-batching for nested list format `[[cmd1, cmd2, ...]]`

## Verified End-to-End

Task 0 (cool): Agent successfully completed "put a cool tomato in microwave" in 12 steps:
1. Searched countertops for tomato
2. Found tomato on countertop 2, took it
3. Went to fridge, cooled tomato
4. Went to microwave, opened it, placed tomato
5. Called task_completed(success=True)

Environment confirmed: done=True, score=1.0

## Commits

| Hash | Message | Files |
|------|---------|-------|
| 16fca28 | feat(01-03): implement DeepSeek client and agent loop | client.py, prompts.py, loop.py, __init__.py |
| 4e32385 | feat(01-03): create main.py CLI entry point | main.py |
| c2ec1f0 | fix(01-03): fix Python 3.14 eval bug, ALFWorld command format, and step display | env_manager.py, actions.py, loop.py, prompts.py, main.py, __init__.py |

## File Manifest

```
src/agent/
├── __init__.py     # Exports DeepSeekClient, run_task, build_tools_spec, format_step
├── client.py       # DeepSeek API wrapper with retry
├── prompts.py      # AUTONOMOUS_AGENT_PROMPT
└── loop.py         # build_tools_spec, _map_tool_to_command, format_step, run_task

src/main.py         # CLI entry point
```

## Success Criteria Met

- [x] DeepSeekClient connects to api.deepseek.com with retry logic and per-call timeout
- [x] Agent prompt instructs autonomous execution with all 12 tools listed
- [x] run_task executes think-act-observe loop for up to max_steps or wall_clock_timeout
- [x] Tool calls correctly map to ALFWorld commands via env_manager.step()
- [x] Trajectory captures every step's thought, action, action_input, observation
- [x] Trajectory includes task_id, task_type, success, total_steps, duration_seconds, failure_reason, env_done
- [x] main.py runs single task from CLI and saves trajectory to JSONL
- [x] End-to-end test produces valid trajectory with successful task completion
