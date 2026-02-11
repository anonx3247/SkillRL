# Domain Pitfalls: Frozen-Model Skill Evolution on ALFWorld

**Project:** SkillRL Frozen-Model Ablation
**Domain:** LLM agent skill evolution, embodied AI, reinforcement learning ablations
**Researched:** 2026-02-11
**Confidence:** MEDIUM (based on training knowledge of LLM agents, ALFWorld, experiment design; no external verification possible)

## Critical Pitfalls

Mistakes that cause rewrites, invalid experiments, or unrecoverable data loss.

### Pitfall 1: API Rate Limiting Without Batching Strategy

**What goes wrong:** 134 tasks × 5 iterations × 50 steps avg = 33,500+ API calls. DeepSeek API rate limits (likely 100-500 RPM) will cause exponential backoff delays or failures. Naive retry logic leads to 10+ hour runtimes or incomplete iterations.

**Why it happens:** Optimistic assumption that "API calls just work." No rate limit handling until production run hits limits at iteration 3, corrupting results.

**Consequences:**
- Incomplete iterations (some tasks timeout, others succeed → biased metrics)
- Non-reproducible results (rate limit state varies between runs)
- Wasted compute time (hours of backoff delays)
- Cost overruns (failed calls still count against quota)

**Prevention:**
1. **Implement rate limiter from day 1:** Token bucket or leaky bucket algorithm matching provider limits
2. **Batch-aware scheduling:** Queue all tasks, process with controlled concurrency (e.g., 10 parallel tasks max)
3. **Exponential backoff with jitter:** Prevent thundering herd when rate limit resets
4. **Cost tracking per iteration:** Log tokens used, estimate cost before full run
5. **Checkpoint granularity:** Save after each task completes, not just after iteration

**Detection:**
- HTTP 429 errors in logs
- Iteration runtime suddenly 3-5x longer than expected
- Some tasks succeed, others fail with timeout
- API cost spikes without corresponding completion

**Phase mapping:** Phase 1 (agent loop foundation) must include rate limiting. Cannot defer to "optimization phase."

---

### Pitfall 2: Confounds in Frozen-Model Ablations

**What goes wrong:** Performance improvements attributed to skills are actually caused by:
- Changing prompt format between iterations (affects reasoning)
- Temperature drift (non-deterministic behavior masks skill effect)
- Context length variations (later iterations have more skills → different attention patterns)
- Teacher prompt evolution (iteratively refining teacher prompts → better skill distillation, not better skills)

**Why it happens:** Researcher tweaks "just the prompt wording" or "fixes a bug" in the teacher during iteration 2, invalidating comparison with iteration 1. The ablation is supposed to test "skills alone" but accidentally tests "skills + better prompts."

**Consequences:**
- **Paper-killing:** Results are invalid. If challenged at review, no way to re-run with 18 months of API costs and time.
- False conclusions about skill effectiveness
- Wasted compute resources (entire experiment must be re-run)

**Prevention:**
1. **Freeze all prompts before iteration 0:**
   - Agent system prompt
   - Tool call formatting
   - Teacher analysis prompt
   - Skill distillation prompt
   - Commit hash + SHA256 of prompt files in results metadata
2. **Lock hyperparameters:**
   - Temperature, top_p, max_tokens
   - Skill retrieval K
   - Max steps per task
   - Random seeds for any sampling
3. **Version all inputs:**
   - ALFWorld task definitions (seed, initial state)
   - Model version (DeepSeek V3.2 Reasoner → if provider updates, document)
4. **Ablation-specific testing:**
   - Before full run, do 10-task pilot with iteration 0 and iteration 1 prompts
   - Verify that baseline (no skills) is reproducible across runs
5. **Change log discipline:**
   - Any mid-experiment change (even "bug fixes") invalidates results
   - If must fix critical bug, restart from iteration 0 with new experiment ID

**Detection:**
- Success rate improvements that are "too smooth" (suggests prompt tuning)
- Baseline performance differs between early and late experiment runs
- Git history shows prompt changes during experiment window
- Unable to reproduce iteration 0 results when re-running

**Phase mapping:**
- Phase 1: Establish prompt freezing protocol and version tracking
- Phase 2: Pre-commit hooks that prevent prompt changes once experiment starts
- Phase 5: Ablation validation tests (baseline reproducibility)

---

### Pitfall 3: Skill Overfitting to Task-Specific Details

**What goes wrong:** Skills degrade into task-specific answers:
- BAD: "The apple is usually in cabinet 3 in room 2" (task-specific)
- GOOD: "Check typical storage locations for food items (cabinets, fridges)" (general)

Teacher fails to enforce generality. Skill library becomes a lookup table of task solutions, not transferable strategies. Performance appears to improve but skills don't transfer to unseen tasks.

**Why it happens:**
- Teacher prompt says "extract strategies" but doesn't define what "general" means
- Success bias: Teacher sees skill "apple in cabinet 3" correlates with success, adds it
- No cross-validation: Skills tested on same tasks they were distilled from
- Trajectory context leak: Teacher sees task description in trajectory, includes details in skill

**Consequences:**
- **Invalidates the research question:** The point is "do general skills help?" If skills are task-specific, the answer is meaningless.
- Poor generalization to new tasks
- Skill library explodes (one skill per task variant)
- Benchmark gaming (overfitting to test set)

**Prevention:**
1. **Teacher prompt constraints:**
   ```
   CRITICAL: Skills must be GENERAL PRINCIPLES, not task-specific details.

   REJECT skills that:
   - Name specific objects ("apple", "cabinet 3")
   - Describe specific locations ("room 2", "shelf by the door")
   - Contain numeric specifics ("heat for 30 seconds")

   ACCEPT skills that:
   - Describe strategic patterns ("verify object in hand before placing")
   - Reference abstract categories ("check typical storage for food items")
   - Capture failure modes ("if object not visible, check containers first")
   ```
2. **Skill quality checks:**
   - Regex filters: Flag skills containing "cabinet \d+", "room \d+", specific object names from task set
   - Length heuristic: Task-specific skills tend to be shorter (1 sentence vs 2-3 for general principles)
   - Template matching: Flag skills that are suspiciously similar to task descriptions
3. **Held-out validation:**
   - Reserve 20% of tasks for skill distillation only (never used for eval)
   - Use remaining 80% for both distillation and eval
   - Skills distilled from 20% should help on 80%
4. **Manual audits:**
   - Every 10 skills added, human reviews for task-specific leakage
   - Flag suspicious skills for removal

**Detection:**
- Skill descriptions contain object names from task definitions
- Skill library grows linearly with iterations (1 new skill per failed task)
- Success rate plateaus quickly then stops improving
- Skills don't improve performance on tasks they weren't distilled from

**Phase mapping:**
- Phase 2: Teacher prompt with explicit generality constraints
- Phase 3: Automated skill quality filters
- Phase 5: Held-out validation split and manual audit protocol

---

### Pitfall 4: State Persistence Corruption

**What goes wrong:** Skill library file corrupted during write (e.g., process killed mid-write). On resume:
- Old skills lost
- Duplicate skills with conflicting IDs
- Eval results from iteration N-1 mixed with iteration N
- No way to know corruption happened until results look wrong

**Why it happens:**
- Direct JSON write without atomic operations
- No write-ahead log or backup
- Process killed during skill evolution (user Ctrl+C, OOM, API timeout)
- Concurrent writes from parallel tasks (if poorly designed)

**Consequences:**
- Entire experiment invalidated (can't trust any iteration after corruption)
- Hours/days of compute wasted
- Debugging takes longer than re-running experiment
- Silent corruption (results look plausible but are wrong)

**Prevention:**
1. **Atomic writes:**
   ```python
   # Write to temp file, then atomic rename
   with open('skills.json.tmp', 'w') as f:
       json.dump(skills, f)
   os.replace('skills.json.tmp', 'skills.json')  # Atomic on POSIX
   ```
2. **Write-ahead log:**
   - Log all skill operations (add, update, remove) to append-only file
   - Rebuild state from log on resume
   - Detect inconsistencies by comparing log replay vs loaded state
3. **Backup snapshots:**
   - After each iteration, copy state to `state_iter_{N}.json`
   - Keep last 3 iterations, delete older
4. **Checksums and validation:**
   - Store SHA256 of skills.json in metadata
   - On load, verify checksum matches
   - Validate JSON schema (all required fields present)
5. **Single-writer guarantee:**
   - Only one process modifies skill library at a time
   - Use file locks if multi-process design
6. **Graceful shutdown:**
   - Catch SIGINT/SIGTERM
   - Flush all writes, save current state, exit cleanly

**Detection:**
- JSON parse errors on load
- Skill IDs not unique
- Iteration count mismatch (metadata says iter 5, skills show iter 3)
- Success rate suddenly drops or spikes
- Missing skills that should exist based on logs

**Phase mapping:**
- Phase 1: Implement atomic writes and basic validation
- Phase 2: Add write-ahead log and backup snapshots
- Phase 4: Graceful shutdown handling

---

### Pitfall 5: Step Count Metrics Without Failure Handling

**What goes wrong:** Task fails after 50 steps. Recorded as "50 steps, 0% success." But 50-step failure is NOT equivalent to:
- 5-step failure (immediate error)
- 20-step partial progress then failure
- 49-step near-success then failure

Averaging step counts across successes and failures produces meaningless numbers. "Average 30 steps" could be 20 successes at 10 steps + 10 failures at 50 steps, or 30 successes at 30 steps + 0 failures. Totally different skill quality.

**Why it happens:** Naive metric design: `avg_steps = sum(steps) / len(tasks)` without conditioning on success.

**Consequences:**
- Can't distinguish "fast failures" from "slow near-successes"
- Efficiency improvements masked (skill reduces steps from 30→15 but failures still at 50)
- Misleading graphs (step count appears to increase when success rate improves)

**Prevention:**
1. **Separate metrics:**
   ```python
   metrics = {
       'success_rate': successes / total,
       'avg_steps_success': sum(steps for success tasks) / successes,
       'avg_steps_failure': sum(steps for failed tasks) / failures,
       'step_distribution': histogram([5-10, 11-20, 21-30, 31-40, 41-50])
   }
   ```
2. **Efficiency metric:**
   - Success-weighted efficiency: Only count steps for successful tasks
   - Normalized efficiency: `(success_rate) * (max_steps - avg_steps_success) / max_steps`
   - Reward function style: +1 for success, -0.02 per step taken
3. **Failure analysis:**
   - Categorize failures: early (0-10 steps), mid (11-30), late (31-50)
   - Track "progress indicators" (e.g., did agent pick up target object before failing?)
4. **Per-task-type metrics:**
   - Pick tasks may need 5-10 steps
   - Clean tasks may need 20-30 steps
   - Averaging across types hides task-specific efficiency

**Detection:**
- Step count increases as success rate increases (should be inverse)
- Step count distribution is bimodal (successes cluster at 10-20, failures at 50)
- Can't answer "did skills make successful tasks more efficient?"

**Phase mapping:**
- Phase 3: Define comprehensive metrics before first eval
- Phase 5: Add per-task-type and failure category metrics

---

### Pitfall 6: ALFWorld Environment State Leakage

**What goes wrong:** Agent sees information it shouldn't have:
- Full object inventory at start (should explore to discover)
- Goal description too explicit ("put apple in fridge" vs "task: pick and place")
- Previous task outcomes visible in environment state
- Observation text contains action success hints ("You heat the potato. It is now hot." vs deterministic success)

**Why it happens:** ALFWorld has multiple task formats and observation levels. Unclear specification of what agent receives. Accidentally giving too much info makes task easier, inflating performance.

**Consequences:**
- Results not comparable to paper baseline (different observation format)
- Skills optimized for privileged information, won't transfer
- Easier tasks → inflated success rate → false conclusions

**Prevention:**
1. **Explicit observation policy:**
   - Agent receives: `observation` (text description of immediate surroundings)
   - Agent does NOT receive: full state dict, object locations, goal description beyond high-level type
2. **Match paper protocol:**
   - Check ALFWorld paper and SkillRL paper for exact observation format
   - Use `TextWorld` interface, not `GameState` interface (if ALFWorld offers both)
3. **Blind information hiding:**
   - Tool response format: Only return what agent would see ("You don't see that here" not "Object not in room 2, try room 3")
4. **Reset validation:**
   - Ensure each task starts fresh (no state from previous task)
   - Randomize task order to prevent curriculum effects

**Detection:**
- Success rate higher than paper baselines with fewer iterations
- Skills reference information that requires exploration ("apple always in room 2")
- Agent succeeds without exploration actions

**Phase mapping:**
- Phase 1: Define exact observation format, validate against papers
- Phase 2: Implement observation filtering in FastMCP tools

---

## Moderate Pitfalls

Mistakes that cause delays, technical debt, or experimental noise.

### Pitfall 7: Teacher Prompt Drift via In-Context Examples

**What goes wrong:** Teacher prompt includes example skills. Over iterations, new skills resemble examples too closely. Skill diversity decreases. Library converges to variations of initial examples.

**Why it happens:** Few-shot prompting anchors teacher to example style. Teacher interprets "generate skills like this" too literally.

**Prevention:**
- Use zero-shot teacher prompt with principle-based guidelines, not examples
- If examples necessary, rotate them between iterations
- Measure skill diversity: cosine similarity of embeddings, flag if avg similarity > 0.8

**Detection:**
- Skills become repetitive
- Most new skills are minor variations of existing ones
- Skill library grows but performance plateaus

**Phase mapping:** Phase 2 (skill distillation design)

---

### Pitfall 8: Skill Retrieval Bias Toward Recent Skills

**What goes wrong:** Semantic similarity retrieval favors recently added skills (more similar to recent failures) over older, more general skills. Skill evolution creates recency bias.

**Why it happens:** Embedding-based retrieval without temporal balancing. Skills added in iteration 5 are more similar to iteration 5 tasks than skills from iteration 1.

**Prevention:**
- Temporal diversity in retrieval: Force inclusion of skills from different iteration ranges
- Usage tracking: Boost underused skills in retrieval
- Hybrid retrieval: Top K/2 by similarity, random K/2 from rest of library

**Detection:**
- Old skills never retrieved (usage count = 0)
- Skill removal rate accelerates (old skills pruned as "unhelpful")
- Performance gains come from new skills only, never from old skills

**Phase mapping:** Phase 3 (skill retrieval implementation)

---

### Pitfall 9: Eval Task Ordering Effects

**What goes wrong:** Running tasks in fixed order creates warm-up effects. First few tasks in iteration have cold API (slower), later tasks benefit from any server-side caching or model warm-up.

**Why it happens:** Tasks run sequentially in same order every iteration.

**Prevention:**
- Randomize task order each iteration (with fixed seed per iteration for reproducibility)
- OR: Shuffle once at experiment start, then use same order all iterations (controls for task difficulty ordering)

**Detection:**
- Tasks at end of eval sequence have consistently higher success rate
- Step count decreases over course of eval run

**Phase mapping:** Phase 3 (evaluation loop design)

---

### Pitfall 10: Skill Pruning Based on Single-Iteration Performance

**What goes wrong:** Skill unused or unhelpful in iteration N gets pruned. Would have been helpful in iteration N+1 with different library composition.

**Why it happens:** Greedy pruning without understanding skill interaction effects.

**Prevention:**
- Multi-iteration usage tracking: Only prune if unused for 3+ consecutive iterations
- Prune conservatively: Remove bottom 10% by usage, not all unused skills
- Never prune general skills (only task-specific skills are prunable)

**Detection:**
- Skill library size oscillates (add 10, prune 8, add 12, prune 10)
- Performance degrades after aggressive pruning

**Phase mapping:** Phase 4 (skill evolution and pruning)

---

### Pitfall 11: Tool Call Parsing Ambiguity

**What goes wrong:** Agent output formatted incorrectly for FastMCP. Parsing fails. Counted as action failure, but actually a formatting issue, not reasoning failure.

**Why it happens:** DeepSeek model output varies slightly in JSON formatting, extra whitespace, field order.

**Prevention:**
- Robust parsing: Try multiple parsers (strict JSON, lenient JSON, regex fallback)
- Tool call schema validation with clear error messages
- Log unparseable outputs for debugging
- Few-shot examples in system prompt showing exact format

**Detection:**
- High rate of "invalid tool call" errors
- Agent output looks correct to human but fails parsing
- Errors clustered around specific tools

**Phase mapping:** Phase 1 (agent-tool integration)

---

### Pitfall 12: Cost Estimation Ignoring Teacher Calls

**What goes wrong:** Budget estimated as 134 tasks × 50 steps × N iterations. Ignores teacher analysis calls (long context windows analyzing trajectories). Teacher cost can be 2-5x agent cost.

**Why it happens:** Focus on agent loop, forgetting teacher is also API-intensive.

**Prevention:**
- Estimate teacher cost separately: 134 tasks × avg trajectory length (20 steps × 200 tokens/step = 4000 tokens input) × iterations
- Track tokens per component: agent_tokens, teacher_tokens, total_tokens
- Set budget alerts (pause if cost > $X)

**Detection:**
- Actual cost 3x initial estimate
- No breakdown of where tokens went

**Phase mapping:** Phase 2 (cost tracking implementation)

---

## Minor Pitfalls

Mistakes that cause annoyance or confusion but are easily fixed.

### Pitfall 13: Logging Too Much or Too Little

**What goes wrong:**
- Too much: 1GB+ logs per iteration, impossible to debug
- Too little: Can't diagnose why task failed

**Prevention:**
- Structured logging with levels: DEBUG (all tool calls), INFO (task outcomes), WARN (retries), ERROR (failures)
- Per-task log files + aggregate summary
- Log rotation and compression

**Phase mapping:** Phase 1

---

### Pitfall 14: Hardcoded File Paths

**What goes wrong:** Paths like `/Users/researcher/skills.json` break on different machines or docker containers.

**Prevention:**
- Environment variables for all paths
- Config file with path specifications
- Use relative paths from project root

**Phase mapping:** Phase 1

---

### Pitfall 15: No Experiment ID or Run Tracking

**What goes wrong:** Multiple experimental runs overwrite each other. Can't compare ablations. No way to reference "that run from last week."

**Prevention:**
- UUID per experiment run
- Directory structure: `results/{experiment_id}/iteration_{N}/`
- Metadata file: experiment config, start time, Git commit hash

**Phase mapping:** Phase 1

---

### Pitfall 16: Forgetting to Test Iteration 0 Baseline

**What goes wrong:** Rush to iteration 1 with skills. Never establish proper no-skill baseline. Can't measure skill contribution.

**Prevention:**
- Iteration 0 = no skills, agent with base prompt only
- Run full eval before any skill distillation
- This is the MemRL baseline to beat (21.4% in paper)

**Phase mapping:** Phase 5 (evaluation strategy)

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Agent loop foundation | API rate limiting (Pitfall 1) | Rate limiter + batching from day 1 |
| FastMCP integration | Tool call parsing (Pitfall 11) | Robust parsing + validation |
| Environment setup | State leakage (Pitfall 6) | Explicit observation policy |
| Skill distillation | Overfitting (Pitfall 3) | Teacher prompt constraints + filters |
| Skill retrieval | Recency bias (Pitfall 8) | Temporal diversity in retrieval |
| Skill evolution | Teacher prompt drift (Pitfall 7) | Zero-shot prompting, no examples |
| Skill pruning | Single-iteration decisions (Pitfall 10) | Multi-iteration usage tracking |
| State persistence | File corruption (Pitfall 4) | Atomic writes + backups |
| Evaluation design | Confounds (Pitfall 2) | Prompt freezing protocol |
| Evaluation metrics | Step count issues (Pitfall 5) | Separate success/failure metrics |
| Cost management | Teacher cost ignored (Pitfall 12) | Component-level tracking |
| Experiment tracking | No baseline (Pitfall 16) | Iteration 0 with no skills |

---

## Ablation-Specific Risks

### Risk 1: Results Comparable to MemRL, Not SkillRL Full

**Scenario:** Frozen model + skills achieves 25% (only 4pp above MemRL's 21.4%, far below SkillRL's 89.9%).

**Interpretation question:** Is this because:
1. Skills alone aren't enough (needs training) ← Valid scientific finding
2. Implementation bugs ← Invalidates experiment
3. DeepSeek V3.2 not capable enough (paper used different model) ← Confound
4. Skill quality poor (teacher prompt insufficient) ← Implementation issue

**Mitigation:**
- Qualitative skill analysis: Are skills actually general and strategic?
- Compare skill library size and diversity to paper's library
- Manual evaluation: Do skills make sense to human expert?
- Ablation within ablation: Test with hand-written gold skills (if those fail too, it's model capacity)

---

### Risk 2: Success Rate Improves but Step Count Increases

**Scenario:** Success rate: 20% → 40% → 60%, but avg steps: 15 → 25 → 35.

**Interpretation:** Skills help agent persist longer, but not work smarter. Inefficient success.

**Mitigation:**
- Track efficiency metrics (success rate × steps efficiency)
- Analyze successful trajectories: Are they different strategies or just more retries?
- May need to add efficiency reward to teacher prompt ("prefer concise skills")

---

### Risk 3: High Variance Between Iterations

**Scenario:** Success rate: 30% → 45% → 25% → 50% (not monotonic).

**Interpretation:** Skill evolution is chaotic, not converging.

**Mitigation:**
- More conservative pruning (don't remove skills too aggressively)
- Larger K for retrieval (provide more skill options)
- Teacher uncertainty filtering (only add high-confidence skills)
- May indicate DeepSeek's reasoning is less stable than expected

---

## Meta-Pitfall: Premature Optimization

**What goes wrong:** Spending 3 weeks building perfect skill pruning algorithm before running first eval. Project stalls in infrastructure.

**Prevention:**
1. **Ship iteration 0 baseline in week 1:** Proves end-to-end system works
2. **Naive implementations first:** Simple skill add/retrieve, no pruning
3. **Optimize only when blocking:** If iteration takes 8 hours, optimize then, not preemptively
4. **Measure before optimizing:** Profile to find actual bottleneck (might be API latency, not code)

**Phase mapping:** Philosophy for all phases. Bias toward working code over perfect code.

---

## Critical Path Items (Must Address Before First Full Run)

- [ ] Prompt freezing protocol (Pitfall 2)
- [ ] API rate limiting (Pitfall 1)
- [ ] Skill generality constraints in teacher prompt (Pitfall 3)
- [ ] Atomic state persistence (Pitfall 4)
- [ ] Observation format matching papers (Pitfall 6)
- [ ] Proper step count metrics (Pitfall 5)
- [ ] Iteration 0 baseline test (Pitfall 16)

## Can Defer Until Later

- Skill retrieval bias mitigation (Pitfall 8) — only matters after 20+ skills
- Teacher cost optimization (Pitfall 12) — track first, optimize later
- Advanced pruning strategies (Pitfall 10) — start with no pruning, add when library bloats

---

## Source Confidence Notes

**Confidence level: MEDIUM**

**What I'm confident about:**
- LLM API rate limiting patterns (standard practice)
- Ablation experimental design principles (PhD-level research methodology)
- State persistence and atomic operations (systems programming fundamentals)
- Metric design for RL-style evaluations (standard in embodied AI)

**What I'm less confident about:**
- DeepSeek V3.2 Reasoner specific rate limits (no access to current docs)
- ALFWorld's exact observation format options (remember general structure, not specifics)
- FastMCP tool call parsing edge cases (depends on implementation version)
- Current (2026) best practices for LLM agent loops (my training is Jan 2025)

**What I couldn't verify:**
- No access to Context7, WebSearch, or official docs during this research
- ALFWorld and DeepSeek-specific pitfalls based on training knowledge only
- Some pitfalls are inferred from general ML research patterns, not domain-specific sources

**Recommendation:** Before implementing prevention strategies for Pitfalls 6, 8, 11, consult:
- ALFWorld official docs for observation format specifications
- DeepSeek API documentation for current rate limits and best practices
- FastMCP documentation for tool call schema and error handling

All other pitfalls (1-5, 7, 9-10, 12-16) are high-confidence based on fundamental principles.
