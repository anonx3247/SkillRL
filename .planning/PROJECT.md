# SkillRL: Frozen-Model Skill Evolution Ablation

## What This Is

A Python system testing whether an evolving skill library alone — with zero weight updates — can drive performance improvements on ALFWorld. A frozen DeepSeek V3.2 serves as both agent and teacher, executing 134 household tasks and iteratively evolving a flat general skill library based on trajectory analysis. Includes parallel evaluation, semantic skill retrieval via FAISS, teacher-driven skill evolution, and live W&B monitoring.

## Core Value

Demonstrate that a frozen LLM + an evolving skill library can drive ALFWorld performance improvements without weight updates, proving that skill abstraction — not policy training — is the primary driver of improvement.

## Requirements

### Validated

- ✓ ALFWorld environment integration via FastMCP tools (12 action tools) — v1
- ✓ Agent execution loop: autonomous think → act → observe cycle, max 50 steps, task_completed tool — v1
- ✓ DeepSeek V3.2 integration via OpenAI-compatible endpoint — v1
- ✓ Same model serves as both agent and teacher — v1
- ✓ Flat general skill library with name, principle, when_to_apply fields — v1
- ✓ Skill distillation from trajectories: success patterns and failure lessons — v1
- ✓ Semantic skill retrieval via sentence-transformers + FAISS — v1
- ✓ Teacher MCP tools: add_skill, update_skill, remove_skill — v1
- ✓ Skill evolution: teacher analyzes trajectories, proposes updates, prunes unhelpful skills — v1
- ✓ Skill generality enforcement via prompt constraints and regex validation — v1
- ✓ Full 134-task evaluation with 10 concurrent workers — v1
- ✓ Multi-dimensional metrics: success rate, step count, per-subtask breakdowns — v1
- ✓ Persistent state with atomic writes — stop/resume anytime — v1
- ✓ W&B experiment tracking: performance curves, skill library state, teacher decisions — v1

### Active

- [ ] Skill usage attribution — track which skills influenced which decisions per task
- [ ] Failure mode categorization — classify failures (early/mid/late, action error, timeout)
- [ ] Ablation configurations — no-skills baseline, random-skills control
- [ ] Skill library diff tool — compare library between iterations
- [ ] Reproducibility package — git hash, model version, seeds, full config export

### Out of Scope

- SFT / cold-start fine-tuning — the whole point is no weight updates
- GRPO / RL training — frozen model only
- WebShop environment — ALFWorld first, may add later
- Search-augmented QA tasks — out of scope for this ablation
- Multi-model setups — single model serves all roles
- Training infrastructure (TRL, Transformers training loops) — inference only
- Hierarchical skill categories — kept flat for this ablation, simpler and sufficient

## Context

Shipped v1 with 3,315 LOC Python across 30 source files.
Tech stack: Python, DeepSeek V3.2 (deepseek-chat), FastMCP, FAISS, sentence-transformers, W&B, ALFWorld.
System ready for experiment execution: `python -m src.main evolve --max-iterations 20`.

- **Paper reference:** SkillRL (ICML 2026 submission). This implementation tests the key ablation: can distilled skills substitute for policy training?
- **Key paper findings:** MemRL (frozen + raw memory) = 21.4%, SkillRL (full pipeline) = 89.9%. This ablation tests where distilled skills alone land.
- **ALFWorld benchmark:** 134 test tasks, 6 subtask types (Pick, Look, Clean, Heat, Cool, Pick2).
- **Known issues:** FastMCP lifespan bug #1115 (workaround in place), manual Step reconstruction for Trajectory loading.

## Constraints

- **API provider**: DeepSeek V3.2 via api.deepseek.com (OpenAI-compatible endpoint)
- **Tool framework**: FastMCP for all model-tool interactions
- **Environment**: ALFWorld standard benchmark (134 test tasks, 6 subtask types)
- **Max steps**: 50 steps per task (configurable)
- **Language**: Python
- **No training**: Pure inference — no gradient updates, no fine-tuning

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Frozen model (no SFT/RL) | Tests whether skill evolution alone drives improvement | ✓ Good — system built, awaiting experiment results |
| Single model as agent + teacher | Simpler architecture, DeepSeek V3.2 capable for both | ✓ Good — works well for both roles |
| FastMCP for tool interfaces | Clean separation, standard protocol | ⚠️ Revisit — lifespan bug #1115, workaround needed |
| Full re-eval each iteration | Clean performance curves, no selection bias | ✓ Good — clean comparison between iterations |
| Flat general skill library | Simpler than hierarchical, all skills transferable | ✓ Good — avoids category overhead |
| 50-step max per task | ~2-5x optimal, generous but bounded | ✓ Good — awaiting empirical validation |
| Step count as eval metric | Measures efficiency beyond binary success | ✓ Good — captures skill quality signal |
| Standard dataclasses (not Pydantic) | 10-100x faster serialization | ✓ Good — performance critical for trajectories |
| Atomic writes (os.replace) | Cross-platform crash resilience | ✓ Good — reliable state persistence |
| deepseek-chat for tool calling | deepseek-reasoner lacks function calling | ✓ Good — necessary discovery |
| Direct env_manager.step() calls | Simpler than MCP for agent interaction | ✓ Good — reduced complexity |
| Regex validation for skill generality | Prevents task-specific overfitting | ✓ Good — catches specific objects/locations |
| Usage tracking via shared references | Safe in asyncio single-threaded execution | ✓ Good — simple and correct |

---
*Last updated: 2026-02-11 after v1 milestone*
