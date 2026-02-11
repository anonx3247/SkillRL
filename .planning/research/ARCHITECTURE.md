# Architecture Patterns: Frozen-Model Skill Evolution System

**Domain:** LLM Agent with Skill Library Evolution on ALFWorld
**Researched:** 2026-02-11
**Confidence:** HIGH (based on paper methodology and established agent architecture patterns)

## Executive Summary

This system implements a frozen-model ablation of SkillRL, where a single DeepSeek V3.2 Reasoner model serves dual roles (agent and teacher) to test whether skill evolution alone — without weight updates — can drive performance improvements. The architecture centers on clean separation between five major subsystems: (1) agent execution runtime, (2) ALFWorld environment wrapper exposed via MCP, (3) skill library with semantic retrieval, (4) teacher distillation system, and (5) experiment orchestrator managing evaluation loops and persistence.

**Key architectural principle:** MCP tool servers act as the **only** interface between the model and external systems (environment, skill library). This enforces stateless agent behavior and enables clean checkpointing.

## Recommended Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Experiment Orchestrator                      │
│  - Iteration loop controller                                 │
│  - State persistence (checkpoint/resume)                     │
│  - Performance tracking (metrics, curves)                    │
└────────────┬──────────────────────────────┬─────────────────┘
             │                              │
             │ spawn & monitor              │ read/write state
             ▼                              ▼
┌────────────────────────┐      ┌──────────────────────────┐
│   Agent Runner         │      │   State Manager          │
│  - Task execution loop │      │  - Skill library store   │
│  - Model invocation    │      │  - Eval results store    │
│  - Tool dispatch       │      │  - Trajectory logs       │
│  - Trajectory logging  │      │  - Iteration checkpoints │
└───────┬────────────────┘      └──────────────────────────┘
        │
        │ MCP protocol
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│              FastMCP Tool Servers (2 servers)             │
├──────────────────────────────┬───────────────────────────┤
│  ALFWorld Environment Server │  Skill Library Server     │
│  - go_to_object(obj)         │  - retrieve_skills(task)  │
│  - take_object(obj)          │  - add_skill(skill_dict)  │
│  - put_object(obj, rec)      │  - update_skill(id, diff) │
│  - open_receptacle(rec)      │  - remove_skill(id)       │
│  - close_receptacle(rec)     │  - list_skills(category)  │
│  - clean_object(obj)         │                           │
│  - heat_object(obj)          │  Internal:                │
│  - cool_object(obj)          │  - Embedding model        │
│  - use_object(obj)           │  - Vector similarity      │
│  - examine_object(obj)       │  - Hierarchical indexing  │
│  - inventory()               │                           │
│  - task_completed(success)   │                           │
│                              │                           │
│  Internal:                   │                           │
│  - ALFWorld gym env          │                           │
│  - Observation formatting    │                           │
│  - Action validation         │                           │
└──────────────────────────────┴───────────────────────────┘
        ▲                              ▲
        │                              │
        │ direct access (no MCP)       │
        │                              │
┌───────┴──────────────────────────────┴───────────────────┐
│               Teacher/Distiller System                    │
│  - analyze_failures(trajectories, current_skills)         │
│  - distill_success(trajectory) -> skill                   │
│  - distill_failure(trajectory) -> lesson                  │
│  - propose_evolutions(failures, skills) -> operations     │
│                                                           │
│  Uses direct LLM calls (no MCP) + direct skill server    │
└───────────────────────────────────────────────────────────┘
```

## Component Boundaries

### 1. Experiment Orchestrator

**Responsibility:** Top-level control flow for multi-iteration experiments.

**Key operations:**
- Initialize system (load checkpoint or start fresh)
- Run full evaluation: spawn 134 agent tasks in parallel
- Collect trajectories and metrics
- Trigger teacher analysis on failures
- Update skill library via teacher recommendations
- Save state after each iteration
- Track performance curves over time

**Does NOT:**
- Directly invoke the model (delegates to agent runner)
- Parse environment observations (delegates to env wrapper)
- Manage skill retrieval logic (delegates to skill library server)

**State it owns:**
- Iteration counter
- Performance history (success rates, avg steps per iteration)
- Configuration (max steps, skill retrieval K, evolution threshold)

**Interfaces:**
- Spawns agent runner processes (one per task)
- Invokes teacher distillation system
- Reads/writes via state manager

---

### 2. Agent Runner

**Responsibility:** Execute a single ALFWorld task autonomously until completion or max steps.

**Key operations:**
- Initialize task: retrieve skills via MCP, get initial observation
- Execution loop:
  1. Call model with (task description, observation history, retrieved skills)
  2. Model returns tool call(s)
  3. Dispatch tool calls to MCP servers
  4. Collect observations
  5. Repeat until `task_completed` called or max steps hit
- Log full trajectory (thoughts, actions, observations) for later distillation
- Return final status (success/failure, step count, trajectory)

**Does NOT:**
- Know about ALFWorld internals (interacts only via MCP)
- Manage skill library updates (read-only skill access)
- Make teacher-role decisions (no distillation, no skill proposals)

**State it owns:**
- Current task context (task description, observation history)
- Step counter
- Trajectory log (append-only during execution)

**Interfaces:**
- Calls DeepSeek V3.2 Reasoner API (OpenAI-compatible endpoint)
- Dispatches tool calls to FastMCP servers
- Returns trajectory to orchestrator

**Design notes:**
- Stateless between tasks (no memory persists across runs)
- Can be run in parallel (one runner per task)
- Single model invocation per step (no multi-round internal reasoning)

---

### 3. ALFWorld Environment MCP Server

**Responsibility:** Expose ALFWorld environment as MCP tools, handling action execution and observation formatting.

**Exposed tools:**
```python
# Navigation & interaction
go_to_object(object_name: str) -> str
take_object(object_name: str) -> str
put_object(object_name: str, receptacle: str) -> str
open_receptacle(receptacle: str) -> str
close_receptacle(receptacle: str) -> str
examine_object(object_name: str) -> str

# Object manipulation
clean_object(object_name: str) -> str
heat_object(object_name: str) -> str
cool_object(object_name: str) -> str
use_object(object_name: str) -> str

# State queries
inventory() -> str

# Task control
task_completed(success: bool, reasoning: str) -> str
```

**Internal responsibilities:**
- Wrap ALFWorld gym environment
- Translate MCP tool calls to ALFWorld actions
- Format observations as natural language
- Validate actions (e.g., can't take object not in room)
- Track episode state (current location, inventory, step count)
- Reset environment between tasks

**Does NOT:**
- Make decisions (passive responder to tool calls)
- Track skill usage (agent's responsibility to log reasoning)
- Manage trajectories (returns observations only)

**State it owns:**
- Current ALFWorld environment instance
- Episode state (per-task, reset between tasks)

**Design notes:**
- One server instance shared by all parallel agent runners
- Thread-safe: each task gets isolated env instance
- Observations are descriptive strings (e.g., "You are in the kitchen. You see: apple, knife, countertop.")
- `task_completed` is the **only** way to end task (model must explicitly call it)

---

### 4. Skill Library MCP Server

**Responsibility:** Manage hierarchical skill library with semantic retrieval.

**Exposed tools (Agent role):**
```python
retrieve_skills(task_description: str, k: int = 6) -> list[dict]
```

**Exposed tools (Teacher role):**
```python
add_skill(skill: dict) -> str  # {name, principle, when_to_apply, category}
update_skill(skill_id: str, updates: dict) -> str
remove_skill(skill_id: str) -> str
list_skills(category: str = None) -> list[dict]
```

**Internal responsibilities:**
- Store skills in hierarchical structure (general vs task-specific)
- Embed task descriptions and skill texts for semantic search
- Retrieve: general skills (always) + TopK task-specific skills by similarity
- Validate skill format (enforce generality constraints)
- Persist library to disk

**Does NOT:**
- Decide when to evolve skills (orchestrator's responsibility)
- Analyze trajectories (teacher's responsibility)
- Track skill usage statistics (logged by agent, analyzed by orchestrator)

**State it owns:**
- Skill database (hierarchical: general + 6 task categories)
- Embedding model for semantic similarity
- Skill ID counter

**Data model:**
```python
Skill = {
    "id": str,           # unique identifier
    "name": str,         # e.g., "systematic_exploration"
    "principle": str,    # e.g., "When searching for objects..."
    "when_to_apply": str,# e.g., "At task start or when stuck"
    "category": str,     # "general" | "pick" | "look" | "clean" | "heat" | "cool" | "pick2"
    "created": datetime,
    "used_count": int,   # tracked by orchestrator
    "success_rate": float # tracked by orchestrator
}
```

**Design notes:**
- Teacher and agent access same server (different tool subsets)
- Retrieval is **deterministic** given (task, skills, K)
- Embedding model: lightweight (e.g., sentence-transformers/all-MiniLM-L6-v2)
- General skills (~12) always included, don't count toward K limit

---

### 5. Teacher/Distiller System

**Responsibility:** Analyze trajectories to distill skills and propose library evolutions.

**Key operations:**
- `distill_success(trajectory)`: Extract strategic patterns from successful run
- `distill_failure(trajectory)`: Identify failure point, synthesize counterfactual lesson
- `propose_evolutions(failed_trajectories, current_skills)`: Analyze failure patterns, return skill operations (add/update/remove)

**Prompt structure (critical):**
```
You are a teacher analyzing agent performance to improve a skill library.

CONSTRAINT: Skills must be GENERAL principles that transfer across tasks.
NEVER include task-specific details (e.g., "apple is in cabinet 3").
Focus on strategic patterns (e.g., "check inventory before searching").

[For distill_success:]
Analyze this successful trajectory and extract 1-3 reusable strategic patterns...

[For distill_failure:]
Identify: (1) failure point, (2) flawed reasoning, (3) what should have been done, (4) general principle...

[For propose_evolutions:]
Given these failure trajectories and current skills:
1. Identify failure patterns NOT addressed by current skills
2. Propose NEW skills to cover gaps (max 3 per category)
3. Suggest refinements to existing ineffective skills (with skill_id)
4. Propose removals of redundant/harmful skills

Return JSON: {"add": [...], "update": [...], "remove": [...]}
```

**Does NOT:**
- Run in MCP (direct API calls to DeepSeek)
- Manage its own state (operates on inputs provided by orchestrator)
- Directly modify skill library (returns operations, orchestrator executes via MCP)

**Interfaces:**
- Receives trajectories from orchestrator
- Calls DeepSeek V3.2 Reasoner API (same model as agent)
- Returns skill dictionaries or operation lists
- Orchestrator executes operations via skill library MCP server

**Design notes:**
- Same model as agent (DeepSeek V3.2 Reasoner) but different prompt/role
- Stateless function calls (no memory between invocations)
- Operates on batches (e.g., all failures for a category)
- Prompt engineering is **critical** to enforce generality

---

### 6. State Manager

**Responsibility:** Persistent storage and checkpoint/resume functionality.

**Storage structure:**
```
.state/
├── skills/
│   ├── general.json
│   ├── pick.json
│   ├── look.json
│   ├── clean.json
│   ├── heat.json
│   ├── cool.json
│   └── pick2.json
├── trajectories/
│   ├── iteration_001/
│   │   ├── task_000.json
│   │   ├── task_001.json
│   │   └── ...
│   └── iteration_002/
│       └── ...
├── results/
│   ├── iteration_001.json
│   └── iteration_002.json
├── metrics/
│   └── performance_curves.json
└── checkpoint.json  # current iteration, config
```

**Key operations:**
- `save_checkpoint(iteration, config)`: Save current state
- `load_checkpoint() -> (iteration, config, skills)`: Resume from disk
- `save_trajectory(iteration, task_id, trajectory)`: Log full trajectory
- `save_results(iteration, results)`: Store eval metrics
- `get_trajectories(iteration, filter) -> list[trajectory]`: Load for teacher analysis

**Does NOT:**
- Parse trajectory content (stores opaque JSON)
- Analyze performance trends (orchestrator does this)
- Enforce schema (components own their data format)

**Design notes:**
- All writes are atomic (temp file + rename)
- JSON for human readability and debugging
- Trajectories are write-once (no updates)
- Checkpoint enables stop/resume at iteration boundaries

---

## Data Flow

### Startup Flow

```
1. Orchestrator loads checkpoint (or initializes fresh)
2. Orchestrator starts MCP servers:
   - ALFWorld environment server (listens on port 5001)
   - Skill library server (listens on port 5002, loads skills from disk)
3. Orchestrator enters main loop
```

### Task Execution Flow (per task)

```
1. Orchestrator spawns agent runner with (task_id, task_description)
2. Agent runner connects to MCP servers
3. Agent calls skill_library.retrieve_skills(task_description, k=6)
   -> Returns: [general_skills (all)] + [top-6 task-specific by similarity]
4. Agent calls alfworld.get_observation() for initial state
5. Agent enters loop (max 50 steps):
   a. Build context: task + skills + observation_history
   b. Call DeepSeek API: model returns tool_calls
   c. For each tool_call:
      - If env action: dispatch to alfworld MCP server -> observation
      - If task_completed: break loop
   d. Append (action, observation) to trajectory
6. Agent returns trajectory to orchestrator
7. Orchestrator saves trajectory to disk via state manager
```

### Skill Evolution Flow (after full eval)

```
1. Orchestrator collects all failed trajectories for iteration
2. Group failures by task category (pick, look, clean, etc.)
3. For each category with success_rate < threshold (e.g., 85%):
   a. Load current skills for category
   b. Sample failed trajectories (diversity-aware)
   c. Call teacher.propose_evolutions(failures, current_skills)
   d. Teacher returns: {"add": [...], "update": [...], "remove": [...]}
   e. For each operation:
      - add: skill_library.add_skill(skill_dict)
      - update: skill_library.update_skill(id, updates)
      - remove: skill_library.remove_skill(id)
4. Skill library persists updated library to disk
5. Orchestrator proceeds to next iteration
```

### Checkpoint/Resume Flow

```
Save (after each iteration):
1. Orchestrator calls state_manager.save_checkpoint(iteration, config)
2. Skill library auto-saves on every modification
3. Trajectories and results already saved during execution

Resume:
1. Orchestrator calls state_manager.load_checkpoint()
2. Returns: (iteration_num, config, skill_state)
3. Orchestrator starts MCP servers with loaded skill state
4. Continue from iteration_num + 1
```

---

## Suggested Build Order

Based on dependency analysis and integration complexity:

### Phase 1: Foundation (Environment + Tools)
**Build first: ALFWorld Environment MCP Server**

**Why:**
- All downstream components need this to test
- Contains complex integration (ALFWorld gym, action validation)
- Defines observation format (affects prompt engineering)
- Can be tested standalone (manual tool calls)

**Deliverable:**
- FastMCP server exposing ALFWorld tools
- Observation formatter
- Action validator
- Test script: execute pre-scripted action sequences

**Dependency:** None

---

### Phase 2: Skill Storage (No Retrieval Yet)
**Build second: Skill Library MCP Server (basic CRUD)**

**Why:**
- Agent needs skills to run meaningfully
- Start with static skills (no retrieval) to unblock agent development
- Retrieval can be added later without breaking interface

**Deliverable:**
- Skill storage (hierarchical: general + 6 categories)
- MCP tools: add_skill, update_skill, remove_skill, list_skills
- Manual skill loading from JSON seed files
- NO retrieval yet (Phase 3)

**Dependency:** None (operates independently)

---

### Phase 3: Agent Execution Runtime
**Build third: Agent Runner + Orchestrator (single task)**

**Why:**
- Core execution loop drives everything else
- Can test with static skills (no retrieval)
- Establishes trajectory logging format
- Enables end-to-end testing

**Deliverable:**
- Agent runner: task execution loop with MCP dispatch
- Model integration (DeepSeek V3.2 Reasoner API)
- Trajectory logging
- Orchestrator: single-task runner (no multi-task eval yet)
- Test: run one ALFWorld task with pre-seeded skills

**Dependency:** Phases 1 & 2 (needs env + skills)

---

### Phase 4: State Persistence
**Build fourth: State Manager**

**Why:**
- Enables multi-task evaluation (need to save results)
- Checkpoint/resume unblocks long experiments
- Teacher needs trajectory access

**Deliverable:**
- File-based state storage (.state/ directory)
- save_checkpoint, load_checkpoint
- save_trajectory, get_trajectories
- save_results
- Test: run 10 tasks, save, resume, verify continuity

**Dependency:** Phase 3 (needs trajectory format)

---

### Phase 5: Full Evaluation Loop
**Build fifth: Multi-task orchestrator**

**Why:**
- Establishes baseline performance (iteration 0)
- Validates parallel execution
- Enables performance tracking

**Deliverable:**
- Orchestrator: spawn 134 tasks in parallel
- Aggregate results (success rate, avg steps, per-category metrics)
- Performance curve tracking
- Test: full eval, verify all tasks logged

**Dependency:** Phase 4 (needs state persistence)

---

### Phase 6: Skill Retrieval
**Build sixth: Semantic similarity retrieval in skill library**

**Why:**
- Required before skill evolution makes sense
- Can validate retrieval quality before teacher complexity
- Affects agent performance (need baseline with retrieval)

**Deliverable:**
- Embedding model integration (sentence-transformers)
- retrieve_skills(task_description, k) MCP tool
- TopK similarity with threshold
- General skills always included
- Test: retrieval quality on sample task descriptions

**Dependency:** Phase 2 (extends skill library)

---

### Phase 7: Teacher Distillation
**Build seventh: Teacher system (distill only, no evolution)**

**Why:**
- Complex prompt engineering
- Need trajectory analysis before evolution proposals
- Can validate skill quality before auto-evolution

**Deliverable:**
- distill_success(trajectory) -> skill
- distill_failure(trajectory) -> lesson
- Skill generality enforcement (prompt + validation)
- Test: manual distillation on sample trajectories

**Dependency:** Phase 5 (needs trajectories)

---

### Phase 8: Skill Evolution
**Build eighth: Teacher propose_evolutions + orchestrator integration**

**Why:**
- Final integration piece
- Requires all prior components
- Enables multi-iteration experiments

**Deliverable:**
- propose_evolutions(failures, skills) -> operations
- Orchestrator: trigger evolution after eval
- Category-based failure analysis
- Diversity-aware trajectory sampling
- Test: run 3 iterations, verify skill library growth

**Dependency:** Phases 6 & 7 (needs retrieval + distillation)

---

## Build Order Summary Table

| Phase | Component | Can Test Independently? | Blocks |
|-------|-----------|------------------------|--------|
| 1 | ALFWorld MCP Server | YES (manual tool calls) | Agent, orchestrator |
| 2 | Skill Library (CRUD) | YES (manual operations) | Agent, teacher |
| 3 | Agent Runner + Basic Orchestrator | NO (needs 1 & 2) | Multi-task eval, teacher |
| 4 | State Manager | NO (needs 3 for format) | Multi-task, resume |
| 5 | Multi-task Orchestrator | NO (needs 4) | Performance tracking |
| 6 | Skill Retrieval | NO (needs 2 & 5) | Evolution |
| 7 | Teacher Distillation | NO (needs 5) | Evolution |
| 8 | Skill Evolution | NO (needs 6 & 7) | Full system |

---

## Patterns to Follow

### Pattern 1: MCP as the Only Interface
**What:** Agent interacts with environment and skills **exclusively** via MCP tool calls. No direct imports, no shared state.

**Why:**
- Clean separation enables independent testing
- Agent can't "cheat" by accessing internals
- Tool calls are logged automatically (trajectory transparency)
- Easy to swap implementations (e.g., mock env for testing)

**Example:**
```python
# GOOD: Agent uses MCP
response = agent_model.call(
    messages=[...],
    tools=mcp_client.list_tools()  # ALFWorld + skill library
)
for tool_call in response.tool_calls:
    result = mcp_client.execute(tool_call)

# BAD: Agent imports environment directly
from alfworld import ALFWorldEnv
env = ALFWorldEnv()
obs = env.step(action)  # Bypasses MCP, breaks logging
```

---

### Pattern 2: Stateless Agent, Stateful Orchestrator
**What:** Agent runner maintains no state between tasks. Orchestrator owns all persistent state.

**Why:**
- Enables parallel execution (no shared memory)
- Simplifies agent logic (pure function: task -> trajectory)
- Checkpoint/resume happens at orchestrator level

**Example:**
```python
# Orchestrator (stateful)
for iteration in range(start_iteration, max_iterations):
    results = []
    for task in eval_tasks:
        trajectory = run_agent(task, skills)  # spawns fresh agent
        results.append(trajectory)
    state_manager.save_checkpoint(iteration, results)

# Agent (stateless)
def run_agent(task, skills):
    # No class state, no memory between calls
    trajectory = []
    for step in range(max_steps):
        action = model.call(task, skills, trajectory)
        obs = mcp.execute(action)
        trajectory.append((action, obs))
    return trajectory
```

---

### Pattern 3: Trajectory as First-Class Data
**What:** Trajectories are complete, self-contained records logged immediately and never modified.

**Why:**
- Enables offline analysis (teacher doesn't re-execute)
- Debugging (replay exact agent behavior)
- Reproducibility (audit model decisions)

**Trajectory format:**
```json
{
    "task_id": "pick_apple_001",
    "task_description": "Put a clean apple in the refrigerator",
    "category": "pick",
    "retrieved_skills": [
        {"id": "gen_001", "name": "systematic_exploration", ...},
        {"id": "pick_003", "name": "verify_object_properties", ...}
    ],
    "steps": [
        {
            "step": 1,
            "model_reasoning": "I need to locate the apple first...",
            "action": {"tool": "go_to_object", "args": {"object_name": "apple"}},
            "observation": "You are in the kitchen. You see: apple, knife, countertop."
        },
        ...
    ],
    "outcome": {
        "success": true,
        "total_steps": 12,
        "task_completed_reasoning": "Apple is now clean and in refrigerator"
    }
}
```

---

### Pattern 4: Teacher Operates Offline
**What:** Teacher analyzes logged trajectories, never runs the agent.

**Why:**
- Separation of concerns (agent = doer, teacher = analyzer)
- Teacher can process batches efficiently
- No risk of teacher "gaming" the environment

**Example:**
```python
# GOOD: Teacher analyzes logged data
failed_trajectories = state_manager.get_trajectories(
    iteration=5,
    filter=lambda t: t['outcome']['success'] == False
)
new_skills = teacher.distill_failures(failed_trajectories)

# BAD: Teacher runs agent
for task in tasks:
    trajectory = agent.run(task)  # Teacher shouldn't control agent
    skill = teacher.analyze(trajectory)
```

---

### Pattern 5: Hierarchical Skill Library with Retrieval Guarantee
**What:** General skills are **always** included, task-specific skills are retrieved by similarity, total context is bounded.

**Why:**
- Guarantees foundational knowledge (e.g., "use task_completed to end")
- Task-specific skills add targeted expertise
- Context limit prevents bloat (K=6 task-specific max)

**Retrieval logic:**
```python
def retrieve_skills(task_description: str, k: int = 6) -> list[dict]:
    # Step 1: Always include all general skills
    result = list(general_skills)  # ~12 skills

    # Step 2: Embed task description
    task_emb = embedding_model.encode(task_description)

    # Step 3: Score task-specific skills by similarity
    candidates = []
    for skill in task_specific_skills:
        skill_emb = embedding_model.encode(skill['principle'])
        sim = cosine_similarity(task_emb, skill_emb)
        if sim > threshold:  # e.g., 0.3
            candidates.append((sim, skill))

    # Step 4: TopK task-specific
    candidates.sort(reverse=True)
    result.extend([skill for _, skill in candidates[:k]])

    return result
```

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Agent Manages Skill Library
**What:** Agent runner directly modifies skills during execution.

**Why bad:**
- Breaks separation of concerns (agent = executor, teacher = analyzer)
- Online updates are noisy (single trajectory is not enough signal)
- Makes checkpointing complex (when to save?)

**Instead:** Agent is read-only consumer. Teacher analyzes batches offline, proposes updates, orchestrator executes.

---

### Anti-Pattern 2: Synchronous Multi-Task Evaluation
**What:** Run 134 tasks sequentially (one at a time).

**Why bad:**
- Each task takes 30-60 seconds, full eval takes 1-2 hours
- Wastes resources (API calls have latency, parallelism is free)

**Instead:** Spawn tasks in parallel (e.g., 10 concurrent runners). ALFWorld MCP server gives each task an isolated env instance.

---

### Anti-Pattern 3: Task-Specific Skills Sneak In
**What:** Skills like "apple is in cabinet 3" or "use microwave for heating".

**Why bad:**
- Defeats the purpose (testing generalization, not memorization)
- Doesn't transfer (each task instance has randomized object locations)
- Games the benchmark (success without learning strategy)

**Instead:** Enforce generality in teacher prompt + validation. Skills must be abstract principles (e.g., "check all receptacles systematically" not "check cabinet 3").

**Validation heuristic:**
```python
def validate_skill_generality(skill: dict) -> bool:
    # Red flags: specific object names, locations, or action sequences
    red_flags = [
        r'\bcabinet \d+\b',  # "cabinet 3"
        r'\bapple\b',         # specific object
        r'\bmicrowave\b',     # specific receptacle (unless principle is about microwaves)
        r'first.*then.*then', # overly specific sequence
    ]
    text = skill['principle'] + skill['when_to_apply']
    for pattern in red_flags:
        if re.search(pattern, text, re.IGNORECASE):
            return False
    return True
```

---

### Anti-Pattern 4: Unbounded Skill Growth
**What:** Keep adding skills without pruning, library grows to 500+ skills.

**Why bad:**
- Context bloat (retrieval returns too many, or individual skills are too long)
- Redundancy (many skills say the same thing differently)
- Noise (outdated skills conflict with better learned strategies)

**Instead:** Prune low-impact skills. Track usage + success rate per skill. Remove skills with:
- `used_count < 5 and iterations_since_creation > 3` (never useful)
- `success_rate < 0.3 when used` (actively harmful)
- Redundancy with higher-quality skill (teacher identifies during evolution)

---

### Anti-Pattern 5: Forgetting to Call task_completed
**What:** Agent hits max steps without calling `task_completed`, orchestrator doesn't know if task succeeded.

**Why bad:**
- Success is ambiguous (did agent think it succeeded?)
- Metrics are noisy (false negatives)

**Instead:**
- Make `task_completed` mandatory in agent prompt: "You MUST call task_completed when you believe the task is done"
- Orchestrator: if max steps reached without task_completed, log as failure with reason "timeout"
- Track "timeout rate" as separate metric (indicates agent isn't learning termination conditions)

---

## Scalability Considerations

### At 134 Tasks (MVP)

**Approach:**
- Parallel execution: 10 concurrent agent runners
- In-memory skill library (full library is ~100 skills, <100KB)
- File-based state (JSON, human-readable for debugging)
- Single machine, no distributed system

**Bottleneck:** API rate limits (DeepSeek)

---

### At 1000 Tasks (Future expansion)

**Approach:**
- Same architecture, increase parallelism to 50 concurrent runners
- Skill library still in-memory (even 500 skills is <500KB)
- Trajectory storage: compress old iterations (gzip JSON)
- Consider batch teacher analysis (process 100 failures at once)

**Bottleneck:** Trajectory disk I/O (mitigate with compression)

---

### At 10K Tasks (Research scale)

**Approach:**
- Distributed orchestrator (e.g., Ray for task scheduling)
- Skill library: move to vector DB (e.g., ChromaDB, FAISS) for faster retrieval
- Teacher analysis: batch + parallelize (analyze multiple categories concurrently)
- Trajectory storage: database (SQLite or Postgres) with indexed queries
- Streaming metrics (don't load all trajectories into memory)

**Bottleneck:** Teacher LLM throughput (mitigate with multiple API keys or local model)

---

## FastMCP Tool Server Design

### Server Architecture

**Two independent FastMCP servers:**

1. **ALFWorld Environment Server** (port 5001)
   - Exposes environment interaction tools
   - Manages ALFWorld gym instances
   - Thread-safe: one env instance per task

2. **Skill Library Server** (port 5002)
   - Exposes skill management tools
   - Manages hierarchical skill storage + retrieval
   - Shared by all agents (read-only for agent role, read-write for teacher role)

### Tool Design Principles

**Principle 1: Descriptive Schemas**
Each tool has rich descriptions and parameter schemas. The model learns from tool descriptions.

Example:
```python
@mcp.tool()
async def go_to_object(object_name: str) -> str:
    """
    Navigate to the specified object in the current environment.

    Use this when you need to interact with an object (take, examine, etc.)
    but are not currently at its location.

    Args:
        object_name: Name of the object to navigate to (e.g., "apple", "microwave")

    Returns:
        Observation describing your new location and visible objects.

    Failure modes:
        - Object not found in current room -> returns error, try examining other receptacles
        - Already at object -> returns success immediately
    """
    # Implementation...
```

**Principle 2: Natural Language Returns**
Tools return observations as descriptive text, not JSON. The agent is an LLM, not a parser.

```python
# GOOD
return "You are now at the apple. You see: apple (on countertop), knife, plate."

# BAD
return {"location": "apple", "visible": ["apple", "knife", "plate"]}
```

**Principle 3: Error Handling**
Tools never raise exceptions to the agent. All errors are observation strings.

```python
# GOOD
if object_name not in visible_objects:
    return f"Error: {object_name} is not in this room. Visible objects: {', '.join(visible_objects)}"

# BAD
raise ValueError(f"{object_name} not found")  # Agent can't handle exceptions
```

### Example Tool Implementations

**Environment tool (ALFWorld):**
```python
@mcp.tool()
async def take_object(object_name: str) -> str:
    """Take an object and add it to your inventory."""
    try:
        obs, reward, done, info = env.step(f"take {object_name}")
        return format_observation(obs)
    except Exception as e:
        return f"Error: Could not take {object_name}. Reason: {str(e)}"
```

**Skill library tool (Retrieval):**
```python
@mcp.tool()
async def retrieve_skills(task_description: str, k: int = 6) -> list[dict]:
    """
    Retrieve relevant skills for the given task.

    Returns general skills (always included) plus top-K task-specific skills
    ranked by semantic similarity to the task description.

    Args:
        task_description: Natural language description of the task
        k: Maximum number of task-specific skills to retrieve (default: 6)

    Returns:
        List of skill dictionaries with keys: id, name, principle, when_to_apply, category
    """
    # Always include general skills
    result = list(skill_library.get_general_skills())

    # Retrieve task-specific by similarity
    task_emb = embedding_model.encode(task_description)
    candidates = []
    for skill in skill_library.get_task_specific_skills():
        skill_emb = embedding_model.encode(skill['principle'])
        sim = cosine_similarity(task_emb, skill_emb)
        candidates.append((sim, skill))

    candidates.sort(reverse=True)
    result.extend([skill for _, skill in candidates[:k]])

    return result
```

**Skill library tool (Teacher role):**
```python
@mcp.tool()
async def add_skill(skill: dict) -> str:
    """
    Add a new skill to the library (teacher role only).

    Args:
        skill: Dictionary with keys:
            - name: Short identifier (e.g., "systematic_exploration")
            - principle: Description of the strategy
            - when_to_apply: Conditions for applicability
            - category: "general" or task type ("pick", "look", etc.)

    Returns:
        Success message with assigned skill ID, or error if validation fails.
    """
    # Validate generality
    if not validate_skill_generality(skill):
        return f"Error: Skill too specific. Principles must be general strategies, not task-specific details."

    # Assign ID and save
    skill_id = skill_library.add(skill)
    return f"Success: Skill '{skill['name']}' added with ID {skill_id}"
```

---

## Experiment Orchestrator Deep Dive

### Main Loop Structure

```python
class ExperimentOrchestrator:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.state_manager = StateManager(self.config['state_dir'])
        self.mcp_servers = self.start_mcp_servers()
        self.iteration = self.state_manager.load_checkpoint()['iteration']

    def run(self):
        """Main experiment loop."""
        for iteration in range(self.iteration, self.config['max_iterations']):
            print(f"=== Iteration {iteration} ===")

            # Phase 1: Full evaluation (134 tasks)
            results = self.run_full_evaluation()
            self.state_manager.save_results(iteration, results)

            # Phase 2: Analyze performance
            metrics = self.compute_metrics(results)
            self.state_manager.save_metrics(iteration, metrics)
            print(f"Success rate: {metrics['success_rate']:.2%}")
            print(f"Avg steps: {metrics['avg_steps']:.1f}")

            # Phase 3: Skill evolution (if needed)
            if self.should_evolve_skills(metrics):
                self.evolve_skills(results, iteration)

            # Phase 4: Checkpoint
            self.state_manager.save_checkpoint(iteration, self.config)

    def run_full_evaluation(self) -> list[dict]:
        """Run all 134 ALFWorld test tasks in parallel."""
        tasks = self.load_eval_tasks()

        # Parallel execution with progress bar
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(self.run_single_task, task)
                for task in tasks
            ]
            results = [f.result() for f in tqdm(futures, desc="Evaluating")]

        return results

    def run_single_task(self, task: dict) -> dict:
        """Run one task via agent runner."""
        agent = AgentRunner(
            model_endpoint=self.config['model_endpoint'],
            mcp_servers=self.mcp_servers,
            max_steps=self.config['max_steps']
        )
        trajectory = agent.run(task)
        self.state_manager.save_trajectory(self.iteration, task['id'], trajectory)
        return trajectory

    def should_evolve_skills(self, metrics: dict) -> bool:
        """Decide if skill evolution is needed."""
        # Evolve if any category is below threshold
        threshold = self.config['evolution_threshold']  # e.g., 0.85
        for category, success_rate in metrics['by_category'].items():
            if success_rate < threshold:
                return True
        return False

    def evolve_skills(self, results: list[dict], iteration: int):
        """Trigger teacher analysis and update skills."""
        # Group failures by category
        failures_by_category = defaultdict(list)
        for result in results:
            if not result['outcome']['success']:
                failures_by_category[result['category']].append(result)

        # Evolve each struggling category
        teacher = TeacherSystem(self.config['model_endpoint'])
        for category, failures in failures_by_category.items():
            # Sample diverse failures (avoid redundancy)
            sampled = self.sample_diverse_failures(failures, max_samples=20)

            # Get current skills
            current_skills = self.mcp_servers['skills'].list_skills(category)

            # Propose evolutions
            operations = teacher.propose_evolutions(sampled, current_skills)

            # Execute operations via MCP
            for op_type, op_data in operations.items():
                if op_type == 'add':
                    for skill in op_data:
                        self.mcp_servers['skills'].add_skill(skill)
                elif op_type == 'update':
                    for skill_id, updates in op_data.items():
                        self.mcp_servers['skills'].update_skill(skill_id, updates)
                elif op_type == 'remove':
                    for skill_id in op_data:
                        self.mcp_servers['skills'].remove_skill(skill_id)

            print(f"Evolved {category}: +{len(operations['add'])} -{len(operations['remove'])}")
```

### Configuration Schema

```json
{
    "model_endpoint": "https://api.deepseek.com/v1",
    "model_name": "deepseek-reasoner",
    "api_key": "${DEEPSEEK_API_KEY}",
    "max_iterations": 10,
    "max_steps_per_task": 50,
    "skill_retrieval_k": 6,
    "evolution_threshold": 0.85,
    "state_dir": ".state",
    "mcp_servers": {
        "alfworld": {"host": "localhost", "port": 5001},
        "skills": {"host": "localhost", "port": 5002}
    },
    "alfworld_config": {
        "data_path": "alfworld/data",
        "eval_split": "test"
    },
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "teacher_prompts": {
        "distill_success": "prompts/distill_success.txt",
        "distill_failure": "prompts/distill_failure.txt",
        "propose_evolutions": "prompts/propose_evolutions.txt"
    }
}
```

---

## Testing Strategy

### Phase 1: Component Tests (Unit-level)

**ALFWorld MCP Server:**
```bash
# Start server
python -m mcp_servers.alfworld --port 5001

# Manual test script
python tests/test_alfworld_manual.py
# - Executes pre-scripted action sequence
# - Verifies observations are natural language
# - Tests error handling (invalid actions)
```

**Skill Library MCP Server:**
```bash
# Start server with seed skills
python -m mcp_servers.skills --port 5002 --seed data/initial_skills.json

# Manual test script
python tests/test_skills_manual.py
# - Add/update/remove skills
# - Test retrieval (verify TopK + general always included)
# - Test embedding similarity
```

---

### Phase 2: Integration Tests

**Agent + Environment:**
```python
# tests/test_agent_basic.py
def test_single_task_execution():
    """Run one simple task (e.g., pick apple) with static skills."""
    agent = AgentRunner(model_endpoint, mcp_servers, max_steps=50)
    task = load_test_task("pick_apple_simple")
    trajectory = agent.run(task)

    assert len(trajectory['steps']) > 0
    assert 'outcome' in trajectory
    # Don't assert success (model might fail), just test execution
```

**Agent + Skills:**
```python
# tests/test_agent_skills.py
def test_skill_retrieval_during_execution():
    """Verify agent retrieves skills at task start."""
    agent = AgentRunner(model_endpoint, mcp_servers, max_steps=50)
    task = load_test_task("clean_apple_medium")
    trajectory = agent.run(task)

    assert 'retrieved_skills' in trajectory
    assert len(trajectory['retrieved_skills']) > 6  # general + task-specific
    assert any(s['category'] == 'general' for s in trajectory['retrieved_skills'])
```

---

### Phase 3: System Tests

**Full Evaluation (Small):**
```python
# tests/test_orchestrator_small.py
def test_mini_evaluation():
    """Run 10 tasks (mixed categories), verify state persistence."""
    config = load_config("configs/test_mini.json")
    config['eval_tasks'] = load_test_tasks(count=10)

    orchestrator = ExperimentOrchestrator(config)
    orchestrator.run()

    # Verify results saved
    results = state_manager.load_results(iteration=0)
    assert len(results) == 10

    # Verify trajectories saved
    trajectories = state_manager.get_trajectories(iteration=0)
    assert len(trajectories) == 10
```

**Skill Evolution (Simulated):**
```python
# tests/test_evolution_pipeline.py
def test_skill_evolution_cycle():
    """Simulate failure -> teacher analysis -> skill update."""
    # Load pre-recorded failed trajectories
    failures = load_test_trajectories("data/test_failures.json")
    current_skills = skill_library.list_skills(category="pick")

    teacher = TeacherSystem(model_endpoint)
    operations = teacher.propose_evolutions(failures, current_skills)

    # Verify structure
    assert 'add' in operations
    assert 'update' in operations
    assert 'remove' in operations

    # Execute operations
    for skill in operations['add']:
        assert validate_skill_generality(skill), f"Skill too specific: {skill}"
        skill_library.add_skill(skill)

    # Verify library updated
    updated_skills = skill_library.list_skills(category="pick")
    assert len(updated_skills) > len(current_skills)
```

---

## Sources

**Architecture patterns:**
- SkillRL paper (method section): Algorithm 1, hierarchical skill library design, recursive evolution mechanism
- PROJECT.md: Frozen-model ablation requirements, FastMCP tool interface constraints
- ALFWorld benchmark: 134 test tasks, 6 task categories, max 50 steps per task

**Confidence:** HIGH
- Component boundaries derived directly from paper methodology (agent, teacher, skill library, environment)
- Data flow matches paper Algorithm 1 (rollout → distill → retrieve → evolve)
- Build order based on dependency analysis (environment → skills → agent → orchestrator → teacher)
- FastMCP architecture follows standard tool server patterns (stateless tools, natural language I/O)
- No WebSearch needed: architecture is fully specified by paper + project requirements

**Key architectural decisions:**
1. MCP as only interface: Enforces clean separation, enables logging, prevents agent from accessing internals
2. Stateless agent: Enables parallel execution, simplifies orchestrator checkpointing
3. Offline teacher: Separates concerns (agent = executor, teacher = analyzer), enables batch processing
4. Hierarchical skill library: General skills always included, task-specific retrieved, bounded context
5. Two MCP servers: Environment and skills are independent subsystems with different lifecycles
6. Build order: Foundation (env + skills) → agent → persistence → evaluation → retrieval → teacher → evolution
