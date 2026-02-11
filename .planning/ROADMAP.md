# Roadmap: SkillRL Frozen-Model Ablation

## Overview

This roadmap delivers a controlled ablation experiment testing whether skill library evolution alone — without model fine-tutuning — can drive performance improvements on ALFWorld. The system uses a frozen DeepSeek V3.2 Reasoner model in dual roles (agent and teacher) to execute 134 household tasks and iteratively evolve a flat general skill library based on failure analysis. The journey progresses from basic agent execution to autonomous multi-iteration skill evolution with live W&B monitoring.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Foundation & Agent Loop** - Environment integration, basic agent execution, trajectory logging
- [ ] **Phase 2: Skill System & Evaluation** - Skill library, semantic retrieval, full 134-task evaluation with parallel execution
- [ ] **Phase 3: Teacher & Evolution** - Teacher distillation, autonomous skill evolution, W&B monitoring

## Phase Details

### Phase 1: Foundation & Agent Loop
**Goal**: Agent can execute individual ALFWorld tasks autonomously via FastMCP tools, logging complete trajectories.

**Depends on**: Nothing (first phase)

**Requirements**: ENV-01, ENV-02, ENV-03, AGT-01, AGT-02, AGT-03, AGT-05, TRJ-01, TRJ-02, TRJ-03

**Success Criteria** (what must be TRUE):
  1. ALFWorld environment exposes 10 action tools plus task_completed via FastMCP (go_to, take, put, open, close, clean, heat, cool, use, examine, inventory)
  2. Agent executes think-act-observe loop autonomously on single tasks via DeepSeek V3.2 Reasoner, ending with task_completed tool call or 50-step timeout
  3. Full trajectory captured per task with thought, action, observation per step plus final outcome (success/failure, step count, duration)
  4. Agent receives only natural language observations (no privileged state information)

**Plans**: 3 plans

Plans:
- [ ] 01-01-PLAN.md — Project scaffolding, trajectory data models, and JSONL persistence
- [ ] 01-02-PLAN.md — ALFWorld environment wrapper with FastMCP tool server (11 tools)
- [ ] 01-03-PLAN.md — Agent loop, DeepSeek integration, and end-to-end single-task execution

### Phase 2: Skill System & Evaluation
**Goal**: Full 134-task evaluation with skill retrieval running in parallel, establishing baseline performance.

**Depends on**: Phase 1

**Requirements**: SKL-01, SKL-02, SKL-03, SKL-04, SKL-05, AGT-04, EVL-01, EVL-02, EVL-03, EVL-04, EVL-05, EVL-06

**Success Criteria** (what must be TRUE):
  1. Flat general skill library stores skills with name, principle, when_to_apply fields, persisted as JSON, manageable via add_skill/update_skill/remove_skill tools
  2. Semantic retrieval via sentence-transformers + faiss returns TopK most relevant skills for task descriptions
  3. Agent prompt includes retrieved skills when available
  4. Full 134-task evaluation runs with 10 concurrent workers, capturing success rate, per-subtask success rate, avg steps (separate for successes/failures)
  5. State persists to disk with atomic writes (skill library, trajectories, iteration checkpoints) — experiment can stop and resume anytime
  6. Iteration 0 baseline (no skills) establishes MemRL comparison point

**Plans**: TBD (1-3 plans)

Plans:
- [ ] 02-01: TBD during planning

### Phase 3: Teacher & Evolution
**Goal**: Autonomous multi-iteration skill evolution with teacher analyzing failures and proposing library updates, tracked live in W&B.

**Depends on**: Phase 2

**Requirements**: TCH-01, TCH-02, TCH-03, TCH-04, TCH-05, TCH-06, LOG-01, LOG-02, LOG-03, LOG-04

**Success Criteria** (what must be TRUE):
  1. Teacher (same DeepSeek V3.2 model) analyzes trajectories offline to distill skills from success patterns and failure lessons
  2. Skill generality enforced via prompt constraints — skills are abstract transferable principles, never mention task-specific details, objects, locations, or answers
  3. Recursive evolution runs after each iteration: teacher analyzes failures and proposes new skills or updates to existing ones
  4. Skill pruning removes unhelpful skills after sufficient iterations based on usage tracking
  5. W&B logs performance curves (success rate, avg steps over iterations), skill library state, and per-subtask success rates each iteration

**Plans**: TBD (1-3 plans)

Plans:
- [ ] 03-01: TBD during planning

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation & Agent Loop | 0/3 | Planning complete | - |
| 2. Skill System & Evaluation | 0/TBD | Not started | - |
| 3. Teacher & Evolution | 0/TBD | Not started | - |
