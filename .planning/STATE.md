# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-11)

**Core value:** Demonstrate that a frozen LLM + an evolving skill library can drive ALFWorld performance improvements without weight updates

**Current focus:** Phase 2 - Skill System & Evaluation

## Current Position

Phase: 2 of 3 (Skill System & Evaluation)
Plan: 2 of TBD in current phase
Status: In progress
Last activity: 2026-02-11 — Completed 02-02-PLAN.md

Progress: [█████░░░░░] 50%

## Performance Metrics

**Velocity:**
- Total plans completed: 5
- Average duration: ~11 min
- Total execution time: ~1.1 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| Phase 1 | 3 | ~48 min | ~16 min |
| Phase 2 | 2 | ~10 min | ~5 min |

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Frozen model (no SFT/RL) — Tests whether skill evolution alone drives improvement
- Single model as agent + teacher — Simpler architecture, DeepSeek V3.2 Reasoner capable for both roles
- FastMCP for tool interfaces — Clean separation between model and environment/skill management
- Full re-eval each iteration — Clean performance curves, no bias from selective re-evaluation
- Flat general skill library — No hierarchy/categories, all skills are general transferable principles
- 50-step max per task — Generous but bounded (ALFWorld tasks typically take 5-30 steps)

**From 01-01:**
- Standard dataclasses for internal data (10-100x faster than Pydantic) — Performance critical for trajectory storage
- Atomic write pattern with os.replace — Cross-platform crash resilience for JSONL appends
- JSONL format for trajectories — Streaming reads and atomic appends without full file parsing

**From 01-02:**
- Module-level env_manager instance (not FastMCP lifespan) — Workaround for FastMCP bug #1115
- Return only observation strings from tools — Agent should not see privileged state (score, done flags)

**From 01-03:**
- Use deepseek-chat for tool calling, NOT deepseek-reasoner (no function calling support)
- Agent loop calls env_manager.step() directly (Approach A), not via MCP
- ALFWorld commands: `take X from Y` (not `take X`), `move X to Y` (not `put X in/on Y`)
- Minimal system prompt — no strategy hints; let skill library learn over time
- Python 3.14 breaks TextWorld locals().update() + eval() — fixed with direct eval(expr, globals, locals)

**From 02-01:**
- Skill encoding: `{name}: {principle}. {when_to_apply}` captures full semantics for embedding
- Double normalization (encode + FAISS) ensures proper cosine similarity via dot product
- Empty library returns empty list gracefully — no crashes on iteration 0
- Prompt injection via string replacement between tools and instructions
- Module-level MCP library instance follows environment server pattern

**From 02-02:**
- Each concurrent worker creates its own EnvManager (ALFWorld has global state)
- Atomic checkpoint writes using temp + fsync + os.replace + symlink pattern
- Metrics computed post-hoc from trajectories (not during execution)
- CLI restructured with subcommands for extensibility (run, evaluate)
- Iteration 0 baseline runs with empty skill library (no skills retrieved)

### Pending Todos

None.

### Blockers/Concerns

**Phase 1 - All Resolved:**
- ~~ALFWorld Python 3.14 incompatibility~~ — Works with monkey-patch
- ~~FastMCP Python SDK API~~ — Verified fastmcp>=2.14.5 works
- ~~TextWorld grammar/eval bug~~ — Fixed: pass variables directly to eval() as locals dict
- ~~ALFWorld command format~~ — Discovered: take X from Y, move X to Y
- ~~DeepSeek tool calling~~ — Verified: use deepseek-chat model, works with OpenAI function-calling format

**Known Issues (non-blocking):**
- FastMCP lifespan bug #1115 — Workaround in place (module-level env_manager)

## Session Continuity

Last session: 2026-02-11
Stopped at: Completed 02-02-PLAN.md
Resume file: None

**Next action:** Ready for iteration 0 baseline evaluation (can run manually or create Plan 02-03)
