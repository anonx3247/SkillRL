# Milestone v1: Frozen-Model Skill Evolution Ablation

**Status:** SHIPPED 2026-02-11
**Phases:** 1-3
**Total Plans:** 9

## Overview

This milestone delivered a controlled ablation experiment testing whether skill library evolution alone — without model fine-tuning — can drive performance improvements on ALFWorld. The system uses a frozen DeepSeek V3.2 model in dual roles (agent and teacher) to execute 134 household tasks and iteratively evolve a flat general skill library based on failure analysis. The journey progresses from basic agent execution to autonomous multi-iteration skill evolution with live W&B monitoring.

## Phases

### Phase 1: Foundation & Agent Loop

**Goal**: Agent can execute individual ALFWorld tasks autonomously, logging complete trajectories.

**Depends on**: Nothing (first phase)

**Requirements**: ENV-01, ENV-02, ENV-03, AGT-01, AGT-02, AGT-03, AGT-05, TRJ-01, TRJ-02, TRJ-03

**Plans**: 3 plans

Plans:
- [x] 01-01-PLAN.md — Project scaffolding, trajectory data models, and JSONL persistence
- [x] 01-02-PLAN.md — ALFWorld environment wrapper with FastMCP tool server (12 tools)
- [x] 01-03-PLAN.md — Agent loop, DeepSeek integration, and end-to-end single-task execution

**Success Criteria** (all verified):
1. ALFWorld environment exposes 12 action tools via FastMCP
2. Agent executes think-act-observe loop autonomously on single tasks via DeepSeek V3.2, ending with task_completed or 50-step timeout
3. Full trajectory captured per task with thought, action, observation per step plus final outcome
4. Agent receives only natural language observations (no privileged state information)

### Phase 2: Skill System & Evaluation

**Goal**: Full 134-task evaluation with skill retrieval running in parallel, establishing baseline performance.

**Depends on**: Phase 1

**Requirements**: SKL-01, SKL-02, SKL-03, SKL-04, SKL-05, AGT-04, EVL-01, EVL-02, EVL-03, EVL-04, EVL-05, EVL-06

**Plans**: 2 plans

Plans:
- [x] 02-01-PLAN.md — Skill library system: data model, CRUD with atomic JSON persistence, semantic retrieval (sentence-transformers + FAISS), prompt injection
- [x] 02-02-PLAN.md — Parallel evaluation orchestrator: 134-task concurrent execution, metrics, checkpointing, CLI integration

**Success Criteria** (all verified):
1. Flat general skill library stores skills with name, principle, when_to_apply fields, persisted as JSON
2. Semantic retrieval via sentence-transformers + FAISS returns TopK most relevant skills
3. Agent prompt includes retrieved skills when available
4. Full 134-task evaluation runs with 10 concurrent workers
5. State persists to disk with atomic writes — experiment can stop and resume anytime
6. Iteration 0 baseline (no skills) establishes comparison point

### Phase 3: Teacher & Evolution

**Goal**: Autonomous multi-iteration skill evolution with teacher analyzing failures and proposing library updates, tracked live in W&B.

**Depends on**: Phase 2

**Requirements**: TCH-01, TCH-02, TCH-03, TCH-04, TCH-05, TCH-06, LOG-01, LOG-02, LOG-03, LOG-04

**Plans**: 4 plans

Plans:
- [x] 03-01-PLAN.md — Teacher analysis system: trajectory analysis via DeepSeek, skill proposals with generality enforcement
- [x] 03-02-PLAN.md — W&B experiment tracking: metrics logging, per-subtask breakdowns, teacher decisions tables
- [x] 03-03-PLAN.md — Evolution loop: evaluate-analyze-update cycle, convergence detection, CLI evolve command
- [x] 03-04-PLAN.md — Gap closure: wire usage tracking (usage_count, last_used_iteration) through retriever, orchestrator, and teacher

**Success Criteria** (all verified):
1. Teacher analyzes trajectories offline to distill skills from success patterns and failure lessons
2. Skill generality enforced via prompt constraints — skills are abstract transferable principles
3. Recursive evolution runs after each iteration
4. Skill pruning removes unhelpful skills based on usage tracking
5. W&B logs performance curves, skill library state, and per-subtask success rates

---

## Milestone Summary

**Key Decisions:**

- Frozen model (no SFT/RL) — Tests whether skill evolution alone drives improvement
- Single model as agent + teacher — Simpler architecture, DeepSeek V3.2 capable for both roles
- FastMCP for tool interfaces — Clean separation between model and environment/skill management
- Full re-eval each iteration — Clean performance curves, no bias from selective re-evaluation
- Flat general skill library — No hierarchy, all skills are general transferable principles
- Standard dataclasses (not Pydantic) — 10-100x faster for trajectory storage
- Atomic write pattern with os.replace — Cross-platform crash resilience
- Use deepseek-chat for tool calling (not deepseek-reasoner) — No function calling support in reasoner
- Agent loop calls env_manager.step() directly — Simpler than MCP for agent interaction
- Convergence detection with patience — Skip iteration 0, patience-based detection
- Usage tracking via shared object references — Safe in asyncio single-threaded execution

**Issues Resolved:**

- Python 3.14 breaks TextWorld locals().update() + eval() — Fixed with direct eval(expr, globals, locals)
- ALFWorld command format: take X from Y (not take X), move X to Y (not put X in/on Y)
- FastMCP lifespan bug #1115 — Workaround with module-level env_manager
- DeepSeek tool calling requires deepseek-chat model, not deepseek-reasoner
- ALFWorld batch indexing returns lists even with batch_size=1

**Issues Deferred:**

- FastMCP lifespan bug #1115 — Workaround in place, waiting for upstream fix
- Hierarchical skill categories — Intentionally kept flat for this ablation

**Technical Debt Incurred:**

- Module-level env_manager instance (FastMCP lifespan workaround)
- Manual Step reconstruction for Trajectory loading (no from_dict method)

---

_For current project status, see .planning/MILESTONES.md_
