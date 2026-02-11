# Feature Landscape

**Domain:** Frozen-Model Skill Evolution Evaluation System
**Researched:** 2026-02-11
**Confidence:** MEDIUM (based on project requirements analysis and agent evaluation domain knowledge)

## Table Stakes

Features the system must have or the experiment cannot function.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **ALFWorld Task Execution** | Core experimental substrate | Medium | Must support 134 tasks across 6 subtask types with think-act-observe loop |
| **DeepSeek API Integration** | Both agent and teacher use same model | Low | Standard API calls with prompt management |
| **FastMCP Tool Integration** | Required for ALFWorld interaction | Medium | Must handle environment state and action validation |
| **Think-Act-Observe Loop** | Agent's core decision cycle | Medium | Max 50 steps per task with early termination via task_completed |
| **Trajectory Recording** | Teacher needs full execution history | Medium | Must capture thoughts, actions, observations, and outcomes |
| **Skill Library Storage** | Persistent hierarchical skills | Medium | General skills + 6 task-specific skill sets |
| **Skill Application** | Agent must use skills during execution | High | Context injection, retrieval logic, relevance matching |
| **Teacher Distillation** | Extract skills from failed trajectories | High | Failure analysis + abstract principle extraction |
| **Full Re-evaluation** | 134 tasks per iteration for fair comparison | Medium | Deterministic task initialization, consistent evaluation |
| **Binary Success Metric** | Task completion indicator | Low | Clear success/failure per task |
| **Step Count Tracking** | Efficiency measure | Low | Count actions from start to completion/failure |
| **State Persistence** | Stop/resume capability | Medium | Checkpoint skills, iteration state, metrics history |
| **Iteration Management** | Track evolution cycles | Low | Iteration counter, metadata per cycle |
| **Skill Pruning** | Remove unhelpful skills | Medium | Criteria for identifying underperforming skills |

## Differentiators

Features that elevate research quality and publication potential.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Skill Usage Attribution** | Track which skills influenced decisions | High | Per-task skill utilization, contribution analysis |
| **Performance Curve Visualization** | Show learning over iterations | Low | Success rate and efficiency plots over time |
| **Skill Evolution Provenance** | Track skill creation, modification, pruning history | Medium | Lineage tracking for analysis |
| **Parallel Task Execution** | Speed up 134-task evaluation | Medium | Concurrent ALFWorld instances with proper isolation |
| **Skill Clustering Analysis** | Identify skill redundancy/patterns | High | Semantic similarity, usage correlation |
| **Failure Mode Categorization** | Classify why tasks fail | Medium | Action errors, timeout, invalid states, etc. |
| **Skill Effectiveness Metrics** | Beyond usage: measure actual impact | High | Counterfactual analysis or A/B per skill |
| **Ablation Study Support** | Compare variations (no skills, only general, only task-specific) | Medium | Configuration presets for controlled experiments |
| **Reproducibility Package** | Seeds, versions, full config for replication | Low | Timestamp, model version, library hash, random seeds |
| **Skill Abstraction Validation** | Verify skills don't encode task-specific solutions | Medium | Pattern matching against task IDs, specific object names |
| **Detailed Execution Logs** | Full diagnostic trails for debugging | Low | Structured logging with levels and searchability |
| **Skill Library Diff Tool** | Compare library changes between iterations | Low | Show added/removed/modified skills |
| **Checkpoint Rollback** | Revert to previous iteration if needed | Medium | Version control for skill library states |
| **Custom Metric Plugins** | Extensible evaluation framework | Medium | Hook architecture for new metrics |
| **Multi-Model Comparison** | Test with different LLMs as agent/teacher | Medium | Model abstraction layer, consistent prompting |

## Anti-Features

Features to explicitly NOT build. Common mistakes for this type of system.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Task-Specific Skill Validation** | Skills MUST be abstract; validating per-task defeats purpose | Trust teacher's distillation; validate abstraction level only |
| **Real-Time Skill Modification** | Mid-task skill updates break reproducibility | Only update skills between iterations after full re-eval |
| **Human-in-the-Loop Skill Editing** | Introduces subjective bias, not scalable | Fully autonomous teacher-driven evolution |
| **Partial Evaluation** | Sampling tasks biases results | Always full 134-task evaluation per iteration |
| **Task-Specific Answer Caching** | Defeats the learning measurement | Force fresh execution each iteration |
| **Gradient-Based Skill Updates** | This is frozen-model; no fine-tuning | Discrete skill library modifications only |
| **Complex Skill Voting/Ensembles** | Overcomplicates agent; obscures individual skill value | Simple relevance-based skill retrieval |
| **GUI/Dashboard** | Scope creep for research prototype | CLI with file-based outputs (CSV, JSON, plots) |
| **Multi-User/Multi-Tenancy** | Single researcher, single experiment focus | Local single-process or simple job queue |
| **Advanced Skill Versioning** | Git-like branches/merges add complexity | Linear iteration history is sufficient |

## Feature Dependencies

```
Core Execution Layer:
  ALFWorld Integration → FastMCP Tools → Think-Act-Observe Loop
                                       ↓
                              Trajectory Recording
                                       ↓

Skill Management Layer:
  Trajectory Recording → Teacher Distillation → Skill Library Storage
                                              ↓
                                         Skill Pruning
                                              ↓
  Skill Library Storage → Skill Application → Agent Execution

Evaluation Layer:
  Agent Execution → Full Re-evaluation → Metrics (Success, Steps, Usage)
                                       ↓
                              Performance Tracking
                                       ↓
                              Iteration Management

Persistence Layer:
  State Persistence (Skills + Metrics + Iteration State)
```

**Critical Dependencies:**
- Skill Application depends on Skill Library Storage
- Teacher Distillation depends on Trajectory Recording
- Full Re-evaluation depends on Think-Act-Observe Loop
- Performance Tracking depends on consistent Metrics

## MVP Recommendation

For MVP, prioritize (in order):

1. **ALFWorld + FastMCP Integration** (foundational)
2. **Think-Act-Observe Loop** (core agent)
3. **Trajectory Recording** (capture data)
4. **Skill Library Storage** (persistent hierarchical structure)
5. **Teacher Distillation** (skill extraction from failures)
6. **Skill Application** (agent uses skills)
7. **Full Re-evaluation** (134 tasks, deterministic)
8. **Basic Metrics** (success rate, step count)
9. **State Persistence** (stop/resume)
10. **Iteration Management** (evolution cycles)
11. **Skill Pruning** (remove unhelpful skills)

Defer to post-MVP:
- **Skill Usage Attribution**: High complexity, adds after basic loop works
- **Parallel Task Execution**: Optimization, not core functionality
- **Skill Clustering**: Analysis feature, not required for basic experiment
- **Failure Mode Categorization**: Nice-to-have analysis
- **Ablation Study Support**: Add once baseline works
- **Multi-Model Comparison**: Future work, focus on DeepSeek first

## Feature Complexity Analysis

### High Complexity (3+ days each)
- **Skill Application**: Relevance matching, context injection, prompt engineering
- **Teacher Distillation**: Failure analysis, abstraction principle extraction, quality control
- **Skill Usage Attribution**: Causality tracking through decision chain
- **Skill Effectiveness Metrics**: Counterfactual or A/B comparison infrastructure

### Medium Complexity (1-3 days each)
- **ALFWorld Task Execution**: Environment setup, task loading, state management
- **Think-Act-Observe Loop**: Control flow, step limits, termination logic
- **Trajectory Recording**: Structured data capture, memory management
- **Skill Library Storage**: Hierarchical data structure, I/O operations
- **Full Re-evaluation**: Task orchestration, result aggregation
- **State Persistence**: Checkpoint format, serialization
- **Skill Pruning**: Effectiveness criteria, library modification
- **Parallel Task Execution**: Process/thread management, isolation
- **Skill Evolution Provenance**: Metadata tracking, history storage

### Low Complexity (<1 day each)
- **DeepSeek API Integration**: HTTP requests, response parsing
- **Binary Success Metric**: Boolean flag per task
- **Step Count Tracking**: Simple counter
- **Iteration Management**: Integer counter, metadata dict
- **Performance Curve Visualization**: Plotting library usage
- **Reproducibility Package**: Config file export
- **Detailed Execution Logs**: Logging framework configuration
- **Skill Library Diff Tool**: Text comparison utility

## Implementation Risks

### Risk 1: Skill Abstraction Quality
**Challenge**: Ensuring teacher extracts truly abstract skills, not task-specific solutions
**Mitigation**:
- Explicit prompt constraints to teacher
- Post-generation validation (pattern matching for task IDs, specific objects)
- Manual spot-checking early iterations

### Risk 2: Skill Library Explosion
**Challenge**: Library grows unboundedly, degrading agent performance
**Mitigation**:
- Aggressive pruning based on usage and success correlation
- Skill similarity detection to merge redundant entries
- Max library size constraint per category

### Risk 3: Evaluation Time Scaling
**Challenge**: 134 tasks × 50 steps × multiple iterations takes hours
**Mitigation**:
- Parallel execution (differentiator, but consider for MVP if too slow)
- ALFWorld task timeout optimization
- Caching environment initializations

### Risk 4: State Persistence Bugs
**Challenge**: Corruption or incomplete saves break long-running experiments
**Mitigation**:
- Atomic writes (write to temp, then move)
- Validation on load
- Backup previous checkpoint before overwriting

### Risk 5: Prompt Context Length
**Challenge**: Skill library grows beyond model context window
**Mitigation**:
- Skill retrieval (only relevant subset in prompt)
- Hierarchical injection (general first, then task-specific)
- Context length monitoring and alerts

## MVP Success Criteria

The MVP is complete when:
- [x] Agent can execute all 134 ALFWorld tasks with think-act-observe loop
- [x] Trajectories are captured with full decision history
- [x] Teacher can extract skills from failed trajectories
- [x] Skill library persists hierarchically (general + 6 task-specific)
- [x] Agent applies skills during execution (injected in prompt)
- [x] Full 134-task re-evaluation completes per iteration
- [x] Success rate and step count metrics are tracked
- [x] System can stop and resume from checkpoint
- [x] Skill pruning removes underperforming skills
- [x] At least 3 full iterations run without manual intervention

**Not required for MVP:**
- Performance visualizations (can plot manually from CSV)
- Skill usage attribution (track in later version)
- Parallel execution (acceptable if <24hrs per iteration)
- Advanced ablation configurations (manual config edits acceptable)

## Sources

**Confidence Level: MEDIUM**
- Project requirements from provided context (HIGH confidence)
- ALFWorld benchmark structure (MEDIUM confidence - standard features)
- Agent evaluation best practices (MEDIUM confidence - domain knowledge from training)
- SkillRL paper mechanics (MEDIUM confidence - inferred from description)

**Verification needed:**
- ALFWorld API specifics via official documentation
- FastMCP integration patterns via documentation
- DeepSeek API rate limits and context length
- State-of-the-art skill library structures in recent papers

**Assumptions made:**
- ALFWorld supports 134 tasks across 6 subtask types (from project context)
- FastMCP provides sufficient tooling for ALFWorld (from project context)
- DeepSeek V3.2 Reasoner has adequate context length for skill library
- Teacher and agent roles can use same model with different prompts
