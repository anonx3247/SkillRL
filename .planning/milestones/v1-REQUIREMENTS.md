# Requirements Archive: v1 Frozen-Model Skill Evolution Ablation

**Archived:** 2026-02-11
**Status:** SHIPPED

This is the archived requirements specification for v1.
For current requirements, see `.planning/REQUIREMENTS.md` (created for next milestone).

---

## v1 Requirements

**Defined:** 2026-02-11
**Core Value:** Demonstrate that a frozen LLM + an evolving skill library can drive ALFWorld performance improvements without weight updates

### Environment Integration

- [x] **ENV-01**: ALFWorld environment exposed as FastMCP tool server with action tools (go_to, take, put, open, close, clean, heat, cool, use, examine, inventory) — VALIDATED
- [x] **ENV-02**: Agent receives natural language observations from environment (no privileged state info) — VALIDATED
- [x] **ENV-03**: Full 134 ALFWorld test tasks loadable across 6 subtask types (Pick, Look, Clean, Heat, Cool, Pick2) — VALIDATED

### Agent Execution

- [x] **AGT-01**: Agent runs autonomously in think → act → observe loop via DeepSeek V3.2 (api.deepseek.com, OpenAI-compatible endpoint) — VALIDATED
- [x] **AGT-02**: Agent has `task_completed(success, summary)` tool to end its run — VALIDATED
- [x] **AGT-03**: Max 50 steps per task — failure if `task_completed` not called or task not done properly — VALIDATED
- [x] **AGT-04**: Agent prompt includes retrieved skills from library when available — VALIDATED
- [x] **AGT-05**: Agent is told it's running autonomously with no user — must act independently — VALIDATED

### Trajectory Recording

- [x] **TRJ-01**: Full trajectory captured per task: each step's thought, action, observation, and final outcome — VALIDATED
- [x] **TRJ-02**: Trajectory includes step count, success/failure status, and wall-clock duration — VALIDATED
- [x] **TRJ-03**: Trajectories persisted to disk for teacher analysis — VALIDATED

### Skill Library

- [x] **SKL-01**: Flat general skill library (no hierarchy — all skills are general purpose) — VALIDATED
- [x] **SKL-02**: Skill format: name, principle, when_to_apply — VALIDATED
- [x] **SKL-03**: Teacher MCP tools: add_skill, update_skill, remove_skill for library management — VALIDATED
- [x] **SKL-04**: Semantic retrieval via sentence-transformers + faiss — TopK most relevant skills injected into agent prompt — VALIDATED
- [x] **SKL-05**: Skill library persisted to disk as JSON — VALIDATED

### Teacher & Evolution

- [x] **TCH-01**: Teacher (same DeepSeek V3.2 model) analyzes trajectories offline to distill skills — VALIDATED
- [x] **TCH-02**: Success trajectories distilled into strategic patterns (what worked and why) — VALIDATED
- [x] **TCH-03**: Failure trajectories distilled into lessons (what went wrong, what should have been done) — VALIDATED
- [x] **TCH-04**: Skill generality enforced via prompt constraints — skills must be abstract transferable principles — VALIDATED
- [x] **TCH-05**: Recursive evolution: after each evaluation iteration, teacher analyzes failures and proposes new skills — VALIDATED
- [x] **TCH-06**: Skill pruning: remove skills that aren't helping after sufficient iterations — VALIDATED

### Evaluation & Orchestration

- [x] **EVL-01**: Full 134-task re-evaluation each iteration — VALIDATED
- [x] **EVL-02**: Parallel task execution (10 concurrent workers) — VALIDATED
- [x] **EVL-03**: Metrics per task: success (binary), step count, skills retrieved — VALIDATED
- [x] **EVL-04**: Aggregate metrics per iteration: overall success rate, per-subtask success rate, avg steps (separate for successes and failures) — VALIDATED
- [x] **EVL-05**: State persistence with atomic writes — stop and resume experiment at any point — VALIDATED
- [x] **EVL-06**: Iteration 0 is baseline (no skills) for comparison — VALIDATED

### Logging & Visualization

- [x] **LOG-01**: W&B integration for live experiment logging — VALIDATED
- [x] **LOG-02**: Performance curves logged to W&B: success rate and avg steps over iterations — VALIDATED
- [x] **LOG-03**: Skill library state logged to W&B each iteration (full library contents visible in dashboard) — VALIDATED
- [x] **LOG-04**: Per-subtask success rates logged to W&B — VALIDATED

## v2 Requirements (Not in scope for v1)

### Analysis & Reproducibility

- **ANL-01**: Skill usage attribution — track which skills influenced which decisions per task
- **ANL-02**: Failure mode categorization — classify failures (early/mid/late, action error, timeout)
- **ANL-03**: Ablation configurations — no-skills baseline, random-skills control
- **ANL-04**: Skill library diff tool — compare library between iterations
- **ANL-05**: Reproducibility package — git hash, model version, seeds, full config export

### Extended Environments

- **EXT-01**: WebShop environment integration
- **EXT-02**: Search-augmented QA tasks
- **EXT-03**: Multi-model comparison (different LLMs as agent/teacher)

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENV-01 | Phase 1 | Complete |
| ENV-02 | Phase 1 | Complete |
| ENV-03 | Phase 1 | Complete |
| AGT-01 | Phase 1 | Complete |
| AGT-02 | Phase 1 | Complete |
| AGT-03 | Phase 1 | Complete |
| AGT-05 | Phase 1 | Complete |
| TRJ-01 | Phase 1 | Complete |
| TRJ-02 | Phase 1 | Complete |
| TRJ-03 | Phase 1 | Complete |
| SKL-01 | Phase 2 | Complete |
| SKL-02 | Phase 2 | Complete |
| SKL-03 | Phase 2 | Complete |
| SKL-04 | Phase 2 | Complete |
| SKL-05 | Phase 2 | Complete |
| AGT-04 | Phase 2 | Complete |
| EVL-01 | Phase 2 | Complete |
| EVL-02 | Phase 2 | Complete |
| EVL-03 | Phase 2 | Complete |
| EVL-04 | Phase 2 | Complete |
| EVL-05 | Phase 2 | Complete |
| EVL-06 | Phase 2 | Complete |
| TCH-01 | Phase 3 | Complete |
| TCH-02 | Phase 3 | Complete |
| TCH-03 | Phase 3 | Complete |
| TCH-04 | Phase 3 | Complete |
| TCH-05 | Phase 3 | Complete |
| TCH-06 | Phase 3 | Complete |
| LOG-01 | Phase 3 | Complete |
| LOG-02 | Phase 3 | Complete |
| LOG-03 | Phase 3 | Complete |
| LOG-04 | Phase 3 | Complete |

**Coverage:**
- v1 requirements: 32 total
- Shipped: 32 (100%)
- Adjusted: 0
- Dropped: 0

---

## Milestone Summary

**Shipped:** 32 of 32 v1 requirements
**Adjusted:** None — all requirements shipped as originally defined
**Dropped:** None

---
*Archived: 2026-02-11 as part of v1 milestone completion*
