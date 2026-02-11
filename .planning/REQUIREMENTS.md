# Requirements: SkillRL Frozen-Model Ablation

**Defined:** 2026-02-11
**Core Value:** Demonstrate that a frozen LLM + an evolving skill library can drive ALFWorld performance improvements without weight updates

## v1 Requirements

### Environment Integration

- [x] **ENV-01**: ALFWorld environment exposed as FastMCP tool server with action tools (go_to, take, put, open, close, clean, heat, cool, use, examine, inventory)
- [x] **ENV-02**: Agent receives natural language observations from environment (no privileged state info)
- [x] **ENV-03**: Full 134 ALFWorld test tasks loadable across 6 subtask types (Pick, Look, Clean, Heat, Cool, Pick2)

### Agent Execution

- [x] **AGT-01**: Agent runs autonomously in think → act → observe loop via DeepSeek V3.2 Reasoner (api.deepseek.com, OpenAI-compatible endpoint)
- [x] **AGT-02**: Agent has `task_completed(success, summary)` tool to end its run
- [x] **AGT-03**: Max 50 steps per task — failure if `task_completed` not called or task not done properly
- [x] **AGT-04**: Agent prompt includes retrieved skills from library when available
- [x] **AGT-05**: Agent is told it's running autonomously with no user — must act independently

### Trajectory Recording

- [x] **TRJ-01**: Full trajectory captured per task: each step's thought, action, observation, and final outcome
- [x] **TRJ-02**: Trajectory includes step count, success/failure status, and wall-clock duration
- [x] **TRJ-03**: Trajectories persisted to disk for teacher analysis

### Skill Library

- [x] **SKL-01**: Flat general skill library (no hierarchy — all skills are general purpose)
- [x] **SKL-02**: Skill format: name, principle, when_to_apply
- [x] **SKL-03**: Teacher MCP tools: add_skill, update_skill, remove_skill for library management
- [x] **SKL-04**: Semantic retrieval via sentence-transformers + faiss — TopK most relevant skills injected into agent prompt
- [x] **SKL-05**: Skill library persisted to disk as JSON

### Teacher & Evolution

- [ ] **TCH-01**: Teacher (same DeepSeek V3.2 Reasoner model) analyzes trajectories offline to distill skills
- [ ] **TCH-02**: Success trajectories distilled into strategic patterns (what worked and why)
- [ ] **TCH-03**: Failure trajectories distilled into lessons (what went wrong, what should have been done)
- [ ] **TCH-04**: Skill generality enforced via prompt constraints — skills must be abstract transferable principles, never mention task-specific details, objects, locations, or answers
- [ ] **TCH-05**: Recursive evolution: after each evaluation iteration, teacher analyzes failures and proposes new skills
- [ ] **TCH-06**: Skill pruning: remove skills that aren't helping after sufficient iterations

### Evaluation & Orchestration

- [x] **EVL-01**: Full 134-task re-evaluation each iteration
- [x] **EVL-02**: Parallel task execution (10 concurrent workers)
- [x] **EVL-03**: Metrics per task: success (binary), step count, skills retrieved
- [x] **EVL-04**: Aggregate metrics per iteration: overall success rate, per-subtask success rate, avg steps (separate for successes and failures)
- [x] **EVL-05**: State persistence with atomic writes — stop and resume experiment at any point
- [x] **EVL-06**: Iteration 0 is baseline (no skills) for comparison

### Logging & Visualization

- [ ] **LOG-01**: W&B integration for live experiment logging
- [ ] **LOG-02**: Performance curves logged to W&B: success rate and avg steps over iterations
- [ ] **LOG-03**: Skill library state logged to W&B each iteration (full library contents visible in dashboard)
- [ ] **LOG-04**: Per-subtask success rates logged to W&B

## v2 Requirements

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

## Out of Scope

| Feature | Reason |
|---------|--------|
| SFT / cold-start fine-tuning | Frozen model — no weight updates is the entire point |
| GRPO / RL training | Inference only — testing skill evolution without policy training |
| API rate limiting | DeepSeek doesn't have rate limits |
| Hierarchical skill categories | Keeping skills flat and general — no task-specific categories |
| GUI / Dashboard | W&B handles visualization, CLI for everything else |
| Task-specific skills | All skills must be general transferable principles |
| Human-in-the-loop skill editing | Fully autonomous teacher-driven evolution |
| Real-time mid-task skill modification | Skills only update between iterations |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENV-01 | Phase 1 | Done |
| ENV-02 | Phase 1 | Done |
| ENV-03 | Phase 1 | Done |
| AGT-01 | Phase 1 | Done |
| AGT-02 | Phase 1 | Done |
| AGT-03 | Phase 1 | Done |
| AGT-05 | Phase 1 | Done |
| TRJ-01 | Phase 1 | Done |
| TRJ-02 | Phase 1 | Done |
| TRJ-03 | Phase 1 | Done |
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
| TCH-01 | Phase 3 | Pending |
| TCH-02 | Phase 3 | Pending |
| TCH-03 | Phase 3 | Pending |
| TCH-04 | Phase 3 | Pending |
| TCH-05 | Phase 3 | Pending |
| TCH-06 | Phase 3 | Pending |
| LOG-01 | Phase 3 | Pending |
| LOG-02 | Phase 3 | Pending |
| LOG-03 | Phase 3 | Pending |
| LOG-04 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 32 total
- Mapped to phases: 32 (100%)
- Unmapped: 0

**Phase Distribution:**
- Phase 1 (Foundation & Agent Loop): 10 requirements
- Phase 2 (Skill System & Evaluation): 12 requirements
- Phase 3 (Teacher & Evolution): 10 requirements

---
*Requirements defined: 2026-02-11*
*Last updated: 2026-02-11 after roadmap creation*
