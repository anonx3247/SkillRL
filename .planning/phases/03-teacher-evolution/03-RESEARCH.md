# Phase 3: Teacher & Evolution - Research

**Researched:** 2026-02-11
**Domain:** LLM-as-Teacher + Iterative Skill Evolution + Experiment Tracking
**Confidence:** MEDIUM

## Summary

Phase 3 implements autonomous multi-iteration skill evolution where DeepSeek V3.2 acts as both agent and teacher. The teacher analyzes trajectory data offline, distills general transferable skills from success/failure patterns, and proposes library updates (add/update/remove). W&B (Weights & Biases) provides experiment tracking infrastructure for real-time visualization of performance curves and skill library evolution across iterations.

**Core architecture:** Evolution loop runs full 134-task evaluations, collects trajectories, teacher analyzes batch via DeepSeek API, proposes skill changes, updates library, repeats. W&B logs metrics (success rate, avg steps, skill count), per-subtask breakdowns, and teacher decisions as tables for post-hoc analysis.

**Primary recommendation:** Use W&B Python SDK (wandb>=0.24.2) for metric logging and tables, DeepSeek V3.2 (deepseek-chat model) for teacher analysis with existing retry/timeout infrastructure, and implement convergence detection via plateau monitoring (no improvement over N consecutive iterations) to avoid infinite loops.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| wandb | >=0.24.2 | Experiment tracking, metric logging, visualization | Industry standard for ML experiment tracking, rich dashboard support, table/media logging |
| openai | >=1.0 | DeepSeek API client (OpenAI-compatible) | Already in use, DeepSeek uses OpenAI-compatible API at api.deepseek.com |
| tenacity | (current) | Retry logic for API calls | Already in use, handles rate limits and transient failures |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tqdm | >=4.66 | Progress bars for batch processing | Already in use, useful for visualizing teacher analysis progress over trajectories |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| wandb | MLflow, TensorBoard | W&B has superior table support for logging teacher decisions and richer dashboard interactivity |
| DeepSeek V3.2 | GPT-4o, Claude | DeepSeek already in use, cost-effective, strong reasoning capabilities |

**Installation:**
```bash
pip install wandb>=0.24.2
# openai, tenacity, tqdm already in dependencies
```

**Note:** pyproject.toml already includes openai>=1.0, tenacity, tqdm>=4.66. Only wandb needs to be added.

## Architecture Patterns

### Recommended Project Structure
```
src/
├── teacher/              # New module for Phase 3
│   ├── __init__.py
│   ├── analyzer.py       # Trajectory analysis logic
│   ├── prompts.py        # Teacher system prompts
│   └── proposer.py       # Skill change proposals
├── evolution/            # New module for Phase 3
│   ├── __init__.py
│   ├── loop.py           # Main evolution orchestrator
│   └── convergence.py    # Stopping criteria detection
└── logging/              # New module for Phase 3
    ├── __init__.py
    └── wandb_logger.py   # W&B integration
```

### Pattern 1: Evolution Loop with Convergence Detection
**What:** Iterative evaluation → analysis → update cycle with plateau-based stopping
**When to use:** Multi-iteration learning systems where performance should improve over time

**Example:**
```python
# Conceptual structure (not production code)
import wandb

def run_evolution(max_iterations: int = 20, patience: int = 5):
    """Run evolution loop with early stopping."""
    wandb.init(project="skillrl-evolution", config={
        "max_iterations": max_iterations,
        "patience": patience,
    })

    best_success_rate = 0.0
    no_improvement_count = 0

    for iteration in range(max_iterations):
        # 1. Run evaluation (existing orchestrator)
        metrics = orchestrator.run_iteration(iteration)

        # 2. Log to W&B
        wandb.log({
            "iteration": iteration,
            "success_rate": metrics.overall_success_rate,
            "avg_steps": metrics.avg_steps_success,
            "skill_count": len(skill_library),
        })

        # 3. Convergence check
        if metrics.overall_success_rate > best_success_rate:
            best_success_rate = metrics.overall_success_rate
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f"Early stopping: no improvement for {patience} iterations")
            break

        # 4. Teacher analysis & skill updates
        if iteration < max_iterations - 1:
            proposals = teacher.analyze_and_propose(trajectories)
            apply_proposals(skill_library, proposals)

    wandb.finish()
```

### Pattern 2: Batch Trajectory Analysis via LLM
**What:** Teacher LLM processes multiple trajectories in parallel to extract patterns
**When to use:** When analyzing large volumes of structured execution traces

**Example:**
```python
# Based on TrajTune pattern - batch analysis with error metrics
async def analyze_trajectories_batch(
    trajectories: list[Trajectory],
    client: DeepSeekClient,
    batch_size: int = 10,
) -> list[SkillProposal]:
    """Analyze trajectories in batches to avoid context limits."""
    proposals = []

    for i in range(0, len(trajectories), batch_size):
        batch = trajectories[i:i+batch_size]

        # Separate successes and failures
        successes = [t for t in batch if t.success]
        failures = [t for t in batch if not t.success]

        # Analyze success patterns
        if successes:
            success_analysis = await analyze_success_patterns(successes, client)
            proposals.extend(success_analysis)

        # Analyze failure patterns
        if failures:
            failure_analysis = await analyze_failure_patterns(failures, client)
            proposals.extend(failure_analysis)

    return proposals
```

### Pattern 3: W&B Table Logging for Teacher Decisions
**What:** Log structured data (teacher decisions) as interactive tables in W&B dashboard
**When to use:** When decisions need post-hoc analysis and iteration-by-iteration comparison

**Example:**
```python
# Source: https://docs.wandb.ai/ref/python/data-types/table/
import wandb

def log_teacher_decisions(iteration: int, proposals: list[SkillProposal]):
    """Log teacher decisions as W&B table."""
    # Create table with columns
    table = wandb.Table(columns=["action", "skill_name", "reason"])

    # Add rows for each proposal
    for proposal in proposals:
        table.add_data(
            proposal.action,  # "add", "update", or "remove"
            proposal.skill_name,
            proposal.reason,
        )

    # Log table to W&B
    wandb.log({f"iteration_{iteration}_decisions": table})
```

### Pattern 4: Per-Subtask Metrics Logging
**What:** Track success rates broken down by task type over iterations
**When to use:** When aggregate metrics hide important subgroup performance differences

**Example:**
```python
def log_subtask_metrics(iteration: int, metrics: AggregateMetrics):
    """Log per-subtask success rates to W&B."""
    # Log aggregate metrics
    wandb.log({
        "iteration": iteration,
        "success_rate": metrics.overall_success_rate,
        "avg_steps_success": metrics.avg_steps_success,
        "avg_steps_failure": metrics.avg_steps_failure,
    })

    # Log per-subtask as separate metrics (creates multiple curves)
    for task_type, rate in metrics.per_subtask_success_rate.items():
        wandb.log({
            "iteration": iteration,
            f"success_rate/{task_type}": rate,
        })

    # Alternative: Log as table for compact view
    subtask_table = wandb.Table(
        columns=["task_type", "success_rate"],
        data=[[k, v] for k, v in metrics.per_subtask_success_rate.items()]
    )
    wandb.log({f"iteration_{iteration}_subtasks": subtask_table})
```

### Anti-Patterns to Avoid
- **Creating new W&B run per iteration:** Use single run with iteration as x-axis via `wandb.log({"iteration": N, ...})`
- **Uploading full skill library as artifacts every iteration:** Log skill count as metric and names as summary list; full library is in git
- **Synchronous teacher analysis:** Use async batch processing to analyze trajectories in parallel
- **No convergence detection:** Always implement early stopping to prevent infinite loops or wasted compute

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Experiment tracking dashboard | Custom web UI for metrics | wandb.init() + wandb.log() | W&B provides live updating graphs, table visualization, run comparison, and sharing out of the box |
| Metric comparison across runs | Manual CSV logging + plotting | W&B projects and runs | Automatic aggregation, filtering, and sorting in web UI |
| API rate limit handling | Manual sleep/retry logic | tenacity library (already in use) | Exponential backoff, jitter, retry conditions already implemented in DeepSeekClient |
| Trajectory batch processing | Sequential for-loop over trajectories | asyncio.gather() with semaphore | Existing pattern in evaluation orchestrator handles concurrency correctly |
| Convergence detection | Manual threshold tuning | Plateau detection with patience parameter | Standard early stopping pattern from ML training |
| Skill embedding recomputation | Re-embed all skills every iteration | Incremental indexing | FAISS supports adding vectors without full rebuild (though full rebuild is cheap for small libraries) |

**Key insight:** W&B handles all experiment tracking complexity (versioning, visualization, comparison, sharing). Don't build custom logging infrastructure.

## Common Pitfalls

### Pitfall 1: Context Window Overflow in Teacher Analysis
**What goes wrong:** Sending all 134 trajectories (potentially 1000s of steps) to teacher LLM in one prompt exceeds context limits and causes API errors.

**Why it happens:** DeepSeek V3.2 has a 64K token context window. Full trajectories with all steps can easily exceed this, especially with task descriptions and observations.

**How to avoid:**
- Batch trajectories (10-20 at a time) and analyze in parallel
- Compress trajectory representation: send only (task_type, success, total_steps, failure_reason) for high-level analysis
- For detailed analysis, sample representative trajectories (e.g., hardest failures, surprising successes)
- Monitor token usage via API response and adjust batch size dynamically

**Warning signs:**
- API errors mentioning token limits
- Slow teacher responses (large context = slow processing)
- Teacher producing generic advice (too much information to process)

### Pitfall 2: Task-Specific Skills Despite Generality Prompts
**What goes wrong:** Teacher generates skills that mention specific objects, locations, or task details (e.g., "When looking for a tomato, check fridge first") instead of general principles.

**Why it happens:** Without explicit constraints and examples, LLMs naturally extract specific patterns from specific examples.

**How to avoid:**
- Strong prompt engineering with negative examples: "NEVER mention specific objects (tomato, mug), locations (countertop 1), or task details"
- Post-process skill proposals: regex check for common ALFWorld objects/locations, reject if found
- Few-shot examples in teacher prompt showing good (abstract) vs bad (specific) skills
- Iterative refinement: if skill is task-specific, ask teacher to generalize it

**Warning signs:**
- Skills contain numbers (e.g., "cabinet 3")
- Skills mention ALFWorld-specific objects (see AUTONOMOUS_AGENT_PROMPT tool list)
- Skills only apply to one task type
- New skills don't retrieve for diverse tasks (low reuse)

### Pitfall 3: Skill Library Bloat Without Pruning
**What goes wrong:** Library grows unbounded with redundant or unused skills, degrading retrieval quality and confusing the agent.

**Why it happens:** Teacher always proposes additions, never removals (unless explicitly prompted). Redundant skills naturally emerge from analyzing similar trajectories.

**How to avoid:**
- Implement usage tracking (already in Skill model: usage_count, last_used_iteration)
- Pruning rule: after N iterations (e.g., 5), remove skills with usage_count=0
- Deduplication: before adding, check if similar skill exists (cosine similarity > 0.9)
- Teacher pruning phase: periodically analyze library for redundancy and propose removals

**Warning signs:**
- Skill count grows linearly with iterations
- Retrieval returns very similar skills
- Skills with usage_count=0 after multiple iterations
- Performance plateaus despite library growth

### Pitfall 4: W&B Run Not Properly Finished
**What goes wrong:** W&B run shows as "running" forever in dashboard, or final metrics aren't logged.

**Why it happens:** Exception during evolution loop or forgetting to call `wandb.finish()` at end.

**How to avoid:**
- Use context manager: `with wandb.init(...) as run:` (auto-finishes even on exception)
- Or explicit try/finally: `try: ... finally: wandb.finish()`
- Log iteration completion explicitly: `wandb.log({"iteration_complete": iteration})`
- Test with short runs (2-3 iterations) before full evolution

**Warning signs:**
- W&B dashboard shows run as "running" after script exits
- Missing data points in metric curves
- Cannot compare runs because some are "incomplete"

### Pitfall 5: No Iteration 0 Baseline
**What goes wrong:** Cannot measure improvement because first logged iteration already has skills from prior work or random initialization.

**Why it happens:** Forgetting that the whole experiment depends on showing improvement over no-skills baseline.

**How to avoid:**
- ALWAYS run iteration 0 with empty skill library before starting evolution
- Verify library is empty before iteration 0: `assert len(skill_library) == 0`
- Log iteration 0 metrics prominently in W&B dashboard
- Document baseline performance in experiment notes

**Warning signs:**
- First W&B data point is iteration 1
- Cannot answer "what's the improvement vs baseline?"
- Unclear if skills actually help or just add noise

### Pitfall 6: Teacher Hallucinating Tool Calls or Repeating Actions
**What goes wrong:** Teacher analysis includes references to non-existent tools or proposes skills based on misread trajectories.

**Why it happens:** LLMs can hallucinate when analyzing structured data like trajectories. Repetitive actions in trajectories can mislead pattern detection.

**How to avoid:**
- Include tool schema in teacher prompt (copy from AUTONOMOUS_AGENT_PROMPT)
- Validate teacher proposals: check skill references real tools/concepts
- Filter trajectories: exclude trivial failures (timeout on step 1) from analysis
- TrajTune-style error metrics: detect repetitive actions, hallucinated tools before analysis

**Warning signs:**
- Skills reference tools not in agent's toolset
- Skills describe impossible action sequences
- Teacher extracts patterns from failed early-timeout trajectories

## Code Examples

Verified patterns from official sources:

### W&B Experiment Initialization
```python
# Source: https://docs.wandb.ai/tutorials/experiments/
import wandb

# Initialize with project name and hyperparameters
config = {
    "max_iterations": 20,
    "patience": 5,
    "top_k_skills": 3,
    "max_steps": 50,
}

with wandb.init(project="skillrl-evolution", config=config) as run:
    # Evolution loop here
    for iteration in range(config["max_iterations"]):
        # Run iteration...
        wandb.log({"iteration": iteration, "success_rate": rate})

    # Automatic finish on context exit
```

### Multi-Metric Logging Per Iteration
```python
# Source: https://docs.wandb.ai/guides/track/log/
# Log multiple metrics in single call (creates single step)
wandb.log({
    "iteration": iteration,
    "success_rate": metrics.overall_success_rate,
    "avg_steps": metrics.avg_steps_success,
    "skill_count": len(skill_library),
})

# For per-subtask curves, use namespaced keys
for task_type, rate in metrics.per_subtask_success_rate.items():
    wandb.log({
        "iteration": iteration,
        f"success_rate/{task_type}": rate,
    })
```

### W&B Table for Teacher Decisions
```python
# Source: https://docs.wandb.ai/ref/python/data-types/table/
import wandb

# Create table with column names
decisions_table = wandb.Table(columns=["action", "skill_name", "reason"])

# Add rows for each teacher decision
for proposal in teacher_proposals:
    decisions_table.add_data(
        proposal.action,      # "add", "update", or "remove"
        proposal.skill_name,
        proposal.reason,
    )

# Log table with iteration-specific key
wandb.log({f"iteration_{iteration}_decisions": decisions_table})
```

### Summary Metrics (Persistent Across Iterations)
```python
# Source: https://docs.wandb.ai/guides/track/log/
# Set summary values that persist (not tied to step)
wandb.run.summary["best_success_rate"] = best_success_rate
wandb.run.summary["best_iteration"] = best_iteration
wandb.run.summary["total_skills_created"] = total_skills_created
wandb.run.summary["final_skill_count"] = len(skill_library)

# Useful for run comparison in W&B UI
```

### Async Batch Processing with Existing Infrastructure
```python
# Pattern already in evaluation/orchestrator.py
import asyncio

async def analyze_batch(trajectories: list[Trajectory], batch_size: int = 10):
    """Analyze trajectories in parallel batches."""
    semaphore = asyncio.Semaphore(batch_size)
    client = DeepSeekClient()  # Existing retry logic

    async def analyze_single(trajectory):
        async with semaphore:
            # Analyze trajectory with teacher
            response = await client.chat(
                messages=[{"role": "user", "content": teacher_prompt(trajectory)}],
                temperature=0.7,
            )
            return parse_teacher_response(response)

    # Launch all analyses in parallel
    results = await asyncio.gather(*[analyze_single(t) for t in trajectories])
    return results
```

### Early Stopping / Convergence Detection
```python
# Based on: https://keras.io/api/callbacks/early_stopping/
# Adapted pattern for evolution loop

class ConvergenceDetector:
    """Detect when performance plateaus."""

    def __init__(self, patience: int = 5, min_delta: float = 0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = 0.0
        self.no_improvement_count = 0

    def check(self, current_value: float) -> bool:
        """Returns True if should stop (converged)."""
        if current_value > self.best_value + self.min_delta:
            self.best_value = current_value
            self.no_improvement_count = 0
            return False
        else:
            self.no_improvement_count += 1
            return self.no_improvement_count >= self.patience

# Usage in evolution loop
detector = ConvergenceDetector(patience=5, min_delta=0.01)
for iteration in range(max_iterations):
    metrics = run_iteration(iteration)

    if detector.check(metrics.overall_success_rate):
        print(f"Converged at iteration {iteration}")
        break
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual experiment tracking in CSV/spreadsheets | W&B/MLflow/TensorBoard | ~2018-2020 | Live dashboards, automatic versioning, reproducibility |
| Single teacher-student distillation | TrajTune: trajectory-aware multi-LLM feedback loops | 2024-2026 | 40% reduction in hallucinations, 30% better tool success rates |
| Full model fine-tuning for skill learning | Frozen model + evolving skill library (in-context) | 2025-2026 | No GPU training costs, faster iteration, interpretable skills |
| Fixed prompts | Meta-prompting and iterative prompt refinement | 2024-2025 | LLMs improve their own prompts based on execution feedback |
| Sequential API calls | Batch APIs and parallel processing | 2025-2026 | 50% cost reduction on batch workloads, higher throughput |

**Deprecated/outdated:**
- DeepSeek V3 (superseded by V3.2 in Dec 2025) - V3.2 adds thinking-mode tool-use integration
- wandb < 0.24.x (older versions) - Latest has improved table performance and API
- Manual retry logic for API calls - Use tenacity library (already in codebase)

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal batch size for teacher analysis**
   - What we know: DeepSeek V3.2 has 64K context, supports tool calling in thinking mode
   - What's unclear: Actual token counts for compressed trajectory representations; needs empirical testing
   - Recommendation: Start with batch_size=10, monitor API response times and errors, adjust dynamically

2. **Skill granularity constraints**
   - What we know: Skills must be abstract and transferable, not task-specific
   - What's unclear: How to automatically enforce granularity (too general = useless, too specific = overfitting)
   - Recommendation: Prompt engineering with examples, post-process validation, iterative refinement

3. **Pruning threshold for skill removal**
   - What we know: Usage tracking available (usage_count, last_used_iteration)
   - What's unclear: What constitutes "unhelpful" (usage_count=0 after how many iterations?)
   - Recommendation: Conservative threshold (5+ iterations unused) to avoid premature pruning; evaluate empirically

4. **DeepSeek V3.2 thinking mode for teacher**
   - What we know: V3.2 supports thinking mode with tool calling, but docs are sparse
   - What's unclear: Whether thinking mode improves trajectory analysis quality vs standard chat
   - Recommendation: Start with deepseek-chat (already working), evaluate thinking mode if analysis quality is insufficient

5. **Convergence criteria sensitivity**
   - What we know: Early stopping prevents infinite loops, patience parameter controls sensitivity
   - What's unclear: Whether plateau in aggregate success_rate means true convergence (per-subtask might still improve)
   - Recommendation: Monitor both aggregate and per-subtask metrics; require plateau in aggregate before stopping

## Sources

### Primary (HIGH confidence)
- W&B Python SDK Documentation: https://docs.wandb.ai/ref/python/
- W&B Track Experiments Tutorial: https://docs.wandb.ai/tutorials/experiments/
- W&B Logging Guide: https://docs.wandb.ai/guides/track/log/
- W&B Table Documentation: https://docs.wandb.ai/ref/python/data-types/table/
- DeepSeek V3.2 Release Notes: https://api-docs.deepseek.com/news/news251201
- wandb PyPI (v0.24.2): https://pypi.org/project/wandb/

### Secondary (MEDIUM confidence)
- TrajTune Paper (OpenReview): https://openreview.net/forum?id=vXrOyzt0Ea - Trajectory analysis framework
- Medium W&B Tutorial (Jan 2026): https://medium.com/@digitalconsumer777/weights-biases-wandb-tutorial-track-ml-experiments-like-a-pro-7ab749372986
- FAISS Semantic Search Guide (Hugging Face): https://huggingface.co/learn/llm-course/en/chapter5/6
- Iterative RL/SFT Research: https://www.emergentmind.com/topics/iterative-reinforcement-learning-sft

### Tertiary (LOW confidence)
- LLM Cost Optimization 2026 Guide: https://www.silicondata.com/blog/llm-cost-per-token - General batch processing patterns
- Prompt Engineering Guide 2026: https://www.lakera.ai/blog/prompt-engineering-guide - Trajectory analysis prompting
- Evolutionary Reinforcement Learning Survey: https://www.mdpi.com/2227-7390/13/5/833 - Convergence patterns

## Metadata

**Confidence breakdown:**
- Standard stack (wandb, openai, tenacity): HIGH - Official docs verified, versions confirmed
- Architecture patterns (evolution loop, W&B logging): HIGH - Official examples and existing codebase patterns
- Don't hand-roll (W&B vs custom, async batching): HIGH - Industry standard practices
- Pitfalls (context limits, task-specific skills, pruning): MEDIUM - Based on LLM analysis patterns and best practices, not ALFWorld-specific empirical data
- Code examples (W&B API, async patterns): HIGH - Verified from official docs and existing codebase
- Teacher analysis strategy: MEDIUM - Based on TrajTune research and general LLM reasoning patterns, needs domain-specific validation

**Research date:** 2026-02-11
**Valid until:** ~2026-03-15 (30 days - relatively stable domain, but LLM API capabilities evolve quickly)
