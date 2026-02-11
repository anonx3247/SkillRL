# Project Research Summary

**Project:** SkillRL Frozen-Model Ablation
**Domain:** LLM Agent Skill Evolution on Embodied AI Benchmark (ALFWorld)
**Researched:** 2026-02-11
**Confidence:** MEDIUM (training knowledge + project context, no web verification)

## Executive Summary

This project implements a controlled ablation of the SkillRL paper to test a critical research question: **Can skill library evolution alone — without model fine-tuning — drive performance improvements in LLM agents?** The system uses a frozen DeepSeek V3.2 Reasoner model in dual roles (agent and teacher) to execute ALFWorld household tasks and iteratively evolve a hierarchical skill library based on failure analysis.

The recommended architecture centers on clean separation via FastMCP tool servers: the agent interacts with ALFWorld and retrieves skills exclusively through MCP tool calls, while a teacher system analyzes logged trajectories offline to distill new skills and propose library evolutions. The core challenge is enforcing skill generality — preventing the teacher from extracting task-specific solutions that would invalidate the research question. Success requires freezing all prompts before iteration 0 to eliminate confounds, implementing robust state persistence with atomic writes, and carefully managing API rate limits across 134 tasks × 50 steps × N iterations.

**Key risk:** Without rigorous constraints, the teacher will distill task-specific "skills" (e.g., "apple in cabinet 3") rather than transferable strategies (e.g., "check typical storage locations systematically"). This single failure mode invalidates the entire experiment. The mitigation is explicit teacher prompt constraints, automated validation filters, and held-out task validation.

## Key Findings

### Recommended Stack

The stack emphasizes simplicity and research reproducibility over production polish. Python 3.10+ provides modern async support for parallel task execution, reducing 134-task evaluation from 3.7 hours sequential to ~30 minutes with 10 concurrent workers.

**Core technologies:**
- **Python 3.10+**: Runtime — modern asyncio for parallel eval, ALFWorld compatibility, DeepSeek client support
- **openai package (1.0+)**: DeepSeek API client — DeepSeek is OpenAI-compatible, better than raw requests (built-in retry, streaming, error handling)
- **FastMCP (mcp package)**: Tool protocol — clean decorator-based tool servers for ALFWorld environment and skill library access
- **ALFWorld (0.3.3+)**: Benchmark environment — official 134-task test split, standard text-based interface
- **sentence-transformers (2.2+)**: Skill embeddings — offline semantic retrieval (all-MiniLM-L6-v2 model, 384-dim, fast)
- **faiss-cpu (1.7+)**: Vector search — fast K-nearest for skill retrieval, scales to hundreds of skills
- **JSON + filesystem**: State persistence — human-readable, git-trackable, no database overhead

**Alternatives rejected:**
- SQLite for skills (overkill, harder debugging)
- Vector DBs like Pinecone (infrastructure overhead)
- Langchain (heavy framework, not needed)
- MLflow/W&B (framework overkill for single-experiment ablation)

**Critical version note:** FastMCP is new (0.x), may have flux in API. ALFWorld 0.3.3 from training data may be outdated. Both need verification before implementation.

### Expected Features

The feature set divides cleanly into table stakes (system doesn't work without them), differentiators (elevate research quality), and anti-features (common mistakes that invalidate experiments).

**Must have (table stakes):**
- **ALFWorld task execution** — 134 tasks across 6 subtask types (pick, look, clean, heat, cool, pick2) with think-act-observe loop
- **Skill library storage** — hierarchical (12 general skills + 6 task-specific categories), semantic retrieval with K=6 nearest
- **Teacher distillation** — extract skills from failed trajectories with generality enforcement
- **Skill application** — agent injects retrieved skills into prompt, uses for decision-making
- **Full re-evaluation** — 134 tasks per iteration with deterministic initialization for fair comparison
- **Binary success metric** — clear task completion indicator
- **Step count tracking** — efficiency measure (but separate success vs failure steps)
- **State persistence** — checkpoint/resume capability with atomic writes
- **Skill pruning** — remove skills with low usage after 3+ iterations

**Should have (differentiators):**
- **Skill usage attribution** — track which skills influenced decisions per task
- **Performance curve visualization** — success rate and efficiency over iterations
- **Parallel task execution** — 10 concurrent workers to speed eval from hours to minutes
- **Skill evolution provenance** — track skill creation, modification, pruning history
- **Failure mode categorization** — classify failures (early/mid/late, action error, timeout)
- **Ablation study support** — compare no-skills baseline, only-general, only-task-specific
- **Reproducibility package** — Git commit hash, model version, library hash, random seeds

**Anti-features (avoid these):**
- **Task-specific skill validation** — defeats generality requirement
- **Real-time skill modification** — breaks reproducibility (only update between iterations)
- **Human-in-the-loop editing** — introduces subjective bias
- **Partial evaluation** — sampling biases results, always use full 134 tasks
- **Complex skill voting/ensembles** — obscures individual skill value
- **GUI/Dashboard** — scope creep, use CLI + file outputs

### Architecture Approach

The architecture enforces stateless agent behavior and offline teacher analysis through FastMCP tool servers as the sole interface between the model and external systems. This enables clean checkpointing and prevents "cheating" via direct state access.

**Major components:**

1. **Experiment Orchestrator** — top-level control loop managing iterations, spawning parallel agent tasks (10 workers), triggering teacher evolution, and persisting state after each iteration

2. **Agent Runner** — stateless executor for single tasks, calls DeepSeek API with (task description + retrieved skills + observation history), dispatches tool calls to MCP servers, logs trajectory

3. **ALFWorld Environment MCP Server** — exposes 11 action tools (go_to, take, put, open, close, clean, heat, cool, use, examine, inventory, task_completed), wraps ALFWorld gym, formats observations as natural language

4. **Skill Library MCP Server** — manages hierarchical storage (general + 6 task categories), retrieval via semantic similarity (K=6 nearest + all general skills), teacher CRUD operations (add/update/remove skills with generality validation)

5. **Teacher/Distiller System** — analyzes batched failed trajectories offline (no agent execution), calls DeepSeek API with failure analysis prompts, proposes skill operations (add/update/remove), enforces generality via prompt constraints + validation

6. **State Manager** — atomic file writes for skill library, trajectory logs, iteration checkpoints, and metrics (prevents corruption from mid-write failures)

**Key patterns:**
- MCP as only interface (agent never imports environment directly)
- Stateless agent, stateful orchestrator (enables parallel execution)
- Trajectory as first-class data (complete, immutable, logged immediately)
- Teacher operates offline (analyzes logs, never runs agent)
- Hierarchical skill retrieval guarantee (general skills always included, task-specific retrieved by similarity)

### Critical Pitfalls

Research identified several experiment-killing mistakes that require prevention from day 1:

1. **API Rate Limiting Without Batching** — 134 tasks × 50 steps × N iterations = 33,500+ calls. DeepSeek rate limits (likely 100-500 RPM) cause exponential backoff or incomplete iterations. **Prevention:** Token bucket rate limiter, 10 concurrent worker max, checkpoint after each task, track cost per iteration.

2. **Confounds in Frozen-Model Ablation** — Performance improvements attributed to skills when actually caused by prompt tweaks, temperature drift, or teacher prompt evolution. **Prevention:** Freeze ALL prompts before iteration 0 (commit hash + SHA256 in metadata), lock hyperparameters (temperature, top_p, K, max_steps, seeds), restart experiment from scratch if any change needed mid-run.

3. **Skill Overfitting to Task-Specific Details** — Teacher extracts "apple in cabinet 3" instead of "check typical food storage locations." **Prevention:** Explicit teacher prompt constraints (REJECT specific objects/locations/numerics, ACCEPT strategic patterns), regex validation filters, held-out task validation, manual audits every 10 skills.

4. **State Persistence Corruption** — Skill library corrupted during write (process killed, OOM, Ctrl+C). **Prevention:** Atomic writes (temp file + rename), write-ahead log, backup snapshots after each iteration, checksums + schema validation on load, graceful shutdown handlers.

5. **Step Count Metrics Without Failure Handling** — Averaging step counts across successes and failures produces meaningless numbers (50-step failure ≠ 5-step failure). **Prevention:** Separate metrics for success/failure steps, efficiency = success_rate × (max_steps - avg_success_steps) / max_steps, per-task-type analysis, failure categorization (early/mid/late).

6. **ALFWorld Environment State Leakage** — Agent sees privileged information (full inventory, explicit goal, previous task outcomes). **Prevention:** Explicit observation policy matching paper protocol (only immediate surroundings text, no state dict), blind information hiding in tool responses ("object not here" not "try room 3"), reset validation between tasks.

## Implications for Roadmap

Based on dependency analysis and integration complexity, the recommended phase structure prioritizes foundational infrastructure before evolution complexity.

### Phase 1: Environment Integration & Basic Agent Loop
**Rationale:** All downstream components need ALFWorld working and agent execution loop proven. The environment defines observation format (affects prompt engineering) and action validation logic. This is the integration complexity bottleneck — must work before anything else.

**Delivers:**
- ALFWorld Environment MCP Server with 11 action tools
- Basic agent runner with DeepSeek API integration (think-act-observe loop)
- Single-task execution with trajectory logging
- Rate limiting infrastructure (token bucket, 10 worker concurrency)
- Observation formatter (natural language responses)

**Addresses:** ALFWorld task execution (table stakes), API rate limiting (Critical Pitfall 1), tool call parsing (Moderate Pitfall 11)

**Avoids:** Environment state leakage (Critical Pitfall 6) via explicit observation policy

**Research flags:** Needs verification of ALFWorld observation format, DeepSeek API rate limits, FastMCP tool server setup patterns

---

### Phase 2: Skill Library Infrastructure
**Rationale:** Agent needs skills to run meaningfully, but retrieval can come later. Start with static skill CRUD to unblock agent development, add semantic retrieval in Phase 6 once evaluation loop works.

**Delivers:**
- Skill Library MCP Server (hierarchical storage: general + 6 categories)
- CRUD tools for teacher (add_skill, update_skill, remove_skill, list_skills)
- Skill format validation (enforce required fields)
- Manual skill seeding (12 initial general skills)
- NO retrieval yet (all general skills injected, Phase 6 adds TopK)

**Addresses:** Skill library storage (table stakes)

**Avoids:** Unbounded growth (seed with conservative general skills only)

**Research flags:** Standard pattern (JSON storage + Pydantic validation), no deep research needed

---

### Phase 3: Full Evaluation Loop & State Persistence
**Rationale:** Establishes baseline performance and proves end-to-end system works. Parallel execution reduces 134-task eval from hours to minutes. State persistence enables long-running experiments without losing progress.

**Delivers:**
- Experiment Orchestrator (multi-iteration loop)
- Parallel task execution (10 concurrent agent runners)
- State Manager (atomic writes, checkpoint/resume, trajectory storage)
- Iteration 0 baseline (no skills, MemRL comparison at 21.4%)
- Performance metrics (success rate, avg steps for successes/failures separately)

**Addresses:** Full re-evaluation (table stakes), parallel execution (differentiator), state persistence (table stakes), baseline test (Minor Pitfall 16)

**Avoids:** State corruption (Critical Pitfall 4) via atomic writes + backups, step count issues (Critical Pitfall 5) via separate success/failure metrics, eval ordering effects (Moderate Pitfall 9) via randomized task order

**Research flags:** Standard patterns (ThreadPoolExecutor, JSON serialization), no deep research

---

### Phase 4: Skill Retrieval
**Rationale:** Required before teacher evolution makes sense. Retrieval quality affects agent performance and skill usage tracking. Must validate retrieval before adding evolution complexity.

**Delivers:**
- Semantic similarity retrieval (sentence-transformers + faiss-cpu)
- retrieve_skills(task_description, k=6) MCP tool
- Hierarchical logic (all general + TopK task-specific by similarity)
- Embedding model integration (all-MiniLM-L6-v2)
- Retrieval quality tests (manual task description samples)

**Addresses:** Skill application (table stakes with retrieval), skill library (complete with semantic search)

**Avoids:** Retrieval bias (Moderate Pitfall 8) via temporal diversity (for now, just TopK; optimize later if old skills never retrieved)

**Research flags:** sentence-transformers documentation (model selection, API), faiss-cpu usage patterns (index types, exact vs approximate search)

---

### Phase 5: Teacher Distillation System
**Rationale:** Complex prompt engineering challenge. Need trajectory analysis working before auto-evolution. Can validate skill quality manually before adding evolution automation.

**Delivers:**
- Teacher system (offline trajectory analysis)
- distill_success(trajectory) → skill extraction
- distill_failure(trajectory) → counterfactual lesson
- Skill generality enforcement (prompt constraints + regex validation)
- Manual distillation workflow (orchestrator provides trajectories, teacher returns skills, orchestrator executes via MCP)

**Addresses:** Teacher distillation (table stakes), skill abstraction validation (differentiator)

**Avoids:** Skill overfitting (Critical Pitfall 3) via explicit prompt constraints + filters, teacher prompt drift (Moderate Pitfall 7) via zero-shot prompting without in-context examples

**Research flags:** HIGH — Teacher prompt engineering is critical and experimental. Needs iteration on generality constraints, likely multiple attempts to get right. Budget time for prompt refinement.

---

### Phase 6: Skill Evolution & Pruning
**Rationale:** Final integration. Requires all prior components working. Enables multi-iteration experiments and autonomous library growth.

**Delivers:**
- propose_evolutions(failures, skills) → operations (add/update/remove)
- Orchestrator integration (trigger evolution after eval if success_rate < 85%)
- Category-based failure analysis
- Diversity-aware trajectory sampling (avoid redundant failures)
- Skill pruning (unused for 3+ iterations, success_rate < 0.3 when used)
- Multi-iteration experiment runner (10 iterations target)

**Addresses:** Skill evolution (table stakes), skill pruning (table stakes), skill evolution provenance (differentiator)

**Avoids:** Single-iteration pruning (Moderate Pitfall 10) via multi-iteration usage tracking, premature optimization (Meta-Pitfall) by deferring pruning sophistication until library actually bloats

**Research flags:** Standard patterns once teacher working, no deep research

---

### Phase 7: Analysis & Reproducibility (Post-MVP)
**Rationale:** After core loop works, add publication-quality analysis and reproducibility features. These elevate research quality but aren't blocking.

**Delivers:**
- Performance curve visualization (success rate, efficiency over iterations)
- Skill usage attribution (track which skills influenced decisions)
- Failure mode categorization (early/mid/late, timeout, action error)
- Skill library diff tool (compare iterations)
- Reproducibility package (Git hash, model version, seeds, full config)
- Ablation configurations (no-skills, only-general, only-task-specific)

**Addresses:** Differentiators for publication quality

**Research flags:** Standard patterns (matplotlib, pandas analysis), no deep research

---

### Phase Ordering Rationale

**Why environment first (Phase 1)?**
- Defines observation format → affects all prompts
- Integration bottleneck (ALFWorld, FastMCP, DeepSeek)
- Can test standalone before agent complexity

**Why skills second (Phase 2)?**
- Agent needs something to run with
- Simple CRUD unblocks agent development
- Retrieval can wait (inject all general skills initially)

**Why evaluation third (Phase 3)?**
- Establishes baseline (proves system works)
- Parallel execution critical for iteration speed
- State persistence enables long experiments
- Iteration 0 baseline is research requirement

**Why retrieval fourth (Phase 4)?**
- Required before evolution makes sense
- Affects what skills agent sees
- Validates retrieval quality before teacher complexity

**Why teacher fifth (Phase 5)?**
- Most complex prompt engineering
- Needs working trajectories to analyze
- Generality enforcement is experiment-critical
- Manual validation before auto-evolution

**Why evolution last (Phase 6)?**
- Requires everything else working
- Final integration piece
- Enables full autonomous multi-iteration loop

**Dependency chain:** Environment → Agent → Evaluation → Skills → Retrieval → Teacher → Evolution

### Research Flags

**Needs deeper research during planning:**
- **Phase 1:** ALFWorld observation format (match paper protocol exactly), DeepSeek API specifics (rate limits, context length, tool call format), FastMCP tool server setup (package name, decorator API, async patterns)
- **Phase 4:** sentence-transformers model selection (quality vs speed tradeoffs), faiss-cpu index types (exact vs approximate for <1000 skills)
- **Phase 5:** Teacher prompt engineering (generality constraints, few-shot vs zero-shot, output format), skill validation heuristics (what regex patterns detect task-specific leakage)

**Standard patterns (skip research-phase):**
- **Phase 2:** JSON file storage, Pydantic models (well-documented)
- **Phase 3:** ThreadPoolExecutor, pandas metrics, JSON checkpoints (standard Python)
- **Phase 6:** Orchestrator integration (builds on Phase 5)
- **Phase 7:** matplotlib, analysis tools (standard patterns)

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM | Python/openai/ALFWorld HIGH (standard choices), FastMCP LOW (new protocol, can't verify 2026 status), sentence-transformers/faiss MEDIUM (good choices but versions unverified) |
| Features | MEDIUM | Table stakes derived from project requirements (HIGH), differentiators from domain knowledge (MEDIUM), anti-features from research experience (HIGH) |
| Architecture | HIGH | Component boundaries from SkillRL paper Algorithm 1 (HIGH), MCP patterns from tool server standards (HIGH), build order from dependency analysis (HIGH) |
| Pitfalls | MEDIUM | Critical pitfalls from research methodology fundamentals (HIGH), ALFWorld-specific from training knowledge (MEDIUM), DeepSeek-specific from inference (LOW — needs verification) |

**Overall confidence:** MEDIUM

Research is sufficient for roadmap creation and Phase 1-2 planning. Later phases (especially teacher prompt engineering) will need refinement during implementation.

### Gaps to Address

**Critical gaps needing validation before implementation:**

1. **FastMCP current state (2026)** — Training data from Jan 2025, package may have evolved. Need to verify: actual package name for pip install, decorator API (@mcp.tool() signature), async patterns, tool schema format.

2. **ALFWorld observation format** — Must match SkillRL paper protocol exactly. Need to verify: does ALFWorld offer multiple observation levels? Which level gives "natural language only, no privileged info"? How to configure observation format?

3. **DeepSeek V3.2 Reasoner specifics** — Need to verify: rate limits (RPM/TPM), context window size (affects skill library size), tool call format (OpenAI function calling compatible?), pricing (affects cost estimation).

4. **Skill generality validation** — What regex patterns effectively catch task-specific leakage without false positives? This needs experimentation during Phase 5.

**Handling strategy:**
- Gaps 1-3: Use `/gsd:research-phase` during Phase 1 planning to verify external dependencies
- Gap 4: Treat as experimental during Phase 5, iterate on validation heuristics based on actual skills generated

**Non-blocking gaps (defer to implementation):**
- Skill retrieval bias mitigation (only matters after 20+ skills, Phase 4 can use naive TopK)
- Optimal embedding model (all-MiniLM-L6-v2 is good enough, can optimize later)
- Teacher cost optimization (track first, optimize if needed)
- Advanced pruning strategies (start with simple thresholds, refine if library bloats)

## Sources

### Primary (HIGH confidence)
- SkillRL paper methodology (Algorithm 1, hierarchical skill library design, recursive evolution)
- PROJECT.md requirements (frozen-model ablation, FastMCP constraint, DeepSeek V3.2 Reasoner)
- ALFWorld benchmark structure (134 test tasks, 6 subtask types, max 50 steps)

### Secondary (MEDIUM confidence)
- LLM agent architecture patterns (training knowledge on tool servers, trajectory logging)
- Python async patterns (asyncio, ThreadPoolExecutor for parallel execution)
- Embodied AI experiment design (metrics, baselines, state persistence)
- sentence-transformers and faiss-cpu usage patterns (standard embedding + vector search)

### Tertiary (LOW confidence — needs verification)
- FastMCP package details (inferred from Model Context Protocol spec, actual package may differ)
- DeepSeek V3.2 Reasoner API specifics (assumed OpenAI-compatible based on provider docs)
- ALFWorld 0.3.3 version (from training data, may be outdated by 2026)

**Verification needed before implementation:**
- [ ] FastMCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
- [ ] ALFWorld current version and observation format: https://github.com/alfworld/alfworld
- [ ] DeepSeek API documentation: https://platform.deepseek.com/docs
- [ ] sentence-transformers model comparison: https://www.sbert.net/docs/pretrained_models.html

---

**Research completed:** 2026-02-11
**Ready for roadmap:** Yes
**Recommended next step:** Create requirements document with Phase 1-7 structure, flag Phase 1 for `/gsd:research-phase` to verify FastMCP/ALFWorld/DeepSeek specifics before detailed planning.
