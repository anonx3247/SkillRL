# Phase 3: Teacher & Evolution - Context

**Gathered:** 2026-02-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Autonomous multi-iteration skill evolution: teacher (same DeepSeek V3.2 model) analyzes trajectories offline, distills general skills from successes/failures, proposes library updates (add/update/remove), and iterates. W&B tracks everything live. Iteration 0 baseline (no skills) must be logged for comparison.

</domain>

<decisions>
## Implementation Decisions

### W&B dashboard layout
- Multi-panel view: success rate, average steps, and skill count — all tracked over iterations
- Iteration 0 baseline (empty skill library) must be logged as the first data point for comparison

### Per-subtask tracking
- Claude's discretion on whether to use per-subtask curves or aggregate + table

### Skill library logging
- Log skill count as a W&B metric each iteration
- Log skill names as a summary list (not full library artifacts)
- No need to upload full skill JSON as W&B artifacts

### Teacher decision logging
- Log teacher decisions as a W&B table each iteration
- Table columns: action (add/update/remove), skill name, reason
- Visible in the W&B dashboard for iteration-by-iteration analysis

### Claude's Discretion
- Teacher analysis strategy: what trajectories to analyze, depth of analysis
- Skill proposal rules: how to decide add vs update vs remove, granularity constraints
- Evolution loop design: number of iterations, convergence criteria, skill changes per iteration
- Per-subtask breakdown format (curves vs tables)
- All technical implementation details (prompts, batching, concurrency)

</decisions>

<specifics>
## Specific Ideas

- Iteration 0 baseline is critical — the whole experiment is about showing improvement over no-skills baseline
- Teacher decisions table in W&B enables post-hoc analysis of what the teacher learned and when

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-teacher-evolution*
*Context gathered: 2026-02-11*
