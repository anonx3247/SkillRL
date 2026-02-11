# SkillRL: Frozen-Model Skill Evolution Ablation

## What This Is

A Python implementation testing whether an evolving skill library alone — with zero weight updates — can drive performance improvements on ALFWorld. A frozen DeepSeek V3.2 Reasoner serves as both agent and teacher, interacting through FastMCP tool interfaces. This is the key ablation the SkillRL paper never ran: can distilled skills substitute for policy training?

## Core Value

Demonstrate that a frozen LLM + an evolving hierarchical skill library can match or approach the performance of RL-trained models on ALFWorld, proving that skill abstraction — not weight updates — is the primary driver of improvement.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] ALFWorld environment integration via FastMCP tools (go to, take, put, open, close, clean, heat, cool, use, examine, etc.)
- [ ] Agent execution loop: autonomous think → act → observe cycle, max 50 steps per task, `task_completed` tool to end run
- [ ] DeepSeek V3.2 Reasoner integration via OpenAI-compatible endpoint (api.deepseek.com)
- [ ] Same model serves as both agent and teacher (no separate models)
- [ ] Hierarchical skill library: general skills + task-specific skills (6 ALFWorld subtask types: Pick, Look, Clean, Heat, Cool, Pick2)
- [ ] Skill distillation from trajectories: success patterns → demonstrations, failure patterns → lessons
- [ ] Skill retrieval: general skills always included, task-specific skills retrieved by semantic similarity
- [ ] Teacher MCP tools: `add_skill`, `update_skill`, `remove_skill` for library management
- [ ] Skill evolution: teacher analyzes failed trajectories, proposes new skills, prunes unhelpful ones
- [ ] Skill generality enforcement: skills must be abstract transferable principles, never task-specific answers or details
- [ ] Full evaluation: all 134 ALFWorld test tasks re-run each iteration
- [ ] Multi-dimensional metrics: success rate (binary), step count (efficiency), skills used per task
- [ ] Persistent state: skill library, eval results, iteration state all saved to disk — stop/resume anytime
- [ ] Performance curve tracking: success rate and avg steps over iterations

### Out of Scope

- SFT / cold-start fine-tuning — the whole point is no weight updates
- GRPO / RL training — frozen model only
- WebShop environment — ALFWorld first, may add later
- Search-augmented QA tasks — out of scope for this ablation
- Multi-model setups — single model serves all roles
- Training infrastructure (TRL, Transformers training loops) — inference only

## Context

- **Paper reference:** SkillRL (ICML 2026 submission) in `paper/` directory. Proposes skill distillation + hierarchical skill library + recursive evolution + GRPO. This implementation tests the ablation: steps 1 (trajectory collection), 2 (skill distillation), and 5 (recursive evolution) — skipping 3 (cold-start SFT) and 4 (GRPO).
- **Key paper findings:** MemRL (frozen policy + raw memory) achieved only 21.4% on ALFWorld. SkillRL (full pipeline) achieved 89.9%. The gap between raw memory and distilled skills is the hypothesis this ablation tests.
- **ALFWorld benchmark:** 134 test tasks across 6 subtask types (Pick, Look, Clean, Heat, Cool, Pick2). Text-based household environment. Standard eval from the paper.
- **Paper hyperparameters:** K=6 for skill retrieval, initial library ~55 skills (12 general, 43 task-specific), grows to ~100 through evolution.
- **Agent execution model:** Each task runs autonomously. The model is told it's running without a user. It loops: think, act (via tools), observe. Ends by calling `task_completed` or hitting max 50 steps.
- **Skill quality constraint:** Skills must be general principles ("verify you're holding an object before placing it"), never task-specific answers ("look in cabinet 3 for the apple"). The teacher prompt must enforce this rigorously. The goal is learning better strategies, not gaming the benchmark.
- **Trajectory length matters:** A task done in 300 steps is worse than the same task done in 20 steps. Skills should improve both success rate AND efficiency.

## Constraints

- **API provider**: DeepSeek V3.2 Reasoner via api.deepseek.com (OpenAI-compatible endpoint)
- **Tool framework**: FastMCP for all model-tool interactions
- **Environment**: ALFWorld standard benchmark (134 test tasks, 6 subtask types)
- **Max steps**: 50 steps per task (configurable)
- **Language**: Python
- **No training**: Pure inference — no gradient updates, no fine-tuning

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Frozen model (no SFT/RL) | Tests whether skill evolution alone drives improvement — the key untested ablation | -- Pending |
| Single model as agent + teacher | Simpler architecture, DeepSeek V3.2 Reasoner is capable enough for both roles | -- Pending |
| FastMCP for tool interfaces | Clean separation between model and environment/skill management, standard protocol | -- Pending |
| Full re-eval each iteration | Clean performance curves, no bias from selective re-evaluation | -- Pending |
| Accumulate + prune skills | Prevents context bloat while allowing library growth | -- Pending |
| 50-step max per task | ~2-5x optimal (ALFWorld tasks take 5-30 steps), generous but bounded | -- Pending |
| Step count as eval metric | Measures skill quality beyond binary success — efficiency matters | -- Pending |

---
*Last updated: 2026-02-11 after initialization*
