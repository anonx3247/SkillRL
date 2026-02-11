# Phase 2: Skill System & Evaluation - Research

**Researched:** 2026-02-11
**Domain:** Semantic retrieval (sentence-transformers + FAISS), parallel evaluation orchestration, skill library design, state persistence
**Confidence:** HIGH

## Summary

Phase 2 implements three integrated systems: (1) a flat skill library with semantic retrieval using sentence-transformers + FAISS for TopK skill injection into agent prompts, (2) parallel 134-task evaluation with asyncio semaphore-limited workers and tqdm progress tracking, and (3) atomic state persistence for skill library and iteration checkpoints enabling stop/resume. The established approach uses all-MiniLM-L6-v2 (384-dim embeddings) with IndexFlatIP for exact nearest neighbor search on small corpora (<1000 skills), asyncio.Semaphore for 10 concurrent workers with TaskGroup for structured concurrency, and atomic write patterns (temp + fsync + os.replace) extended from Phase 1's trajectory storage.

Research confirms all components are production-ready as of early 2026. sentence-transformers 3.4+ provides normalized embeddings via `normalize_embeddings=True`, making IndexFlatIP equivalent to cosine similarity through dot product. asyncio.TaskGroup (Python 3.11+) offers safer concurrency than gather() with automatic cancellation on exceptions. The skill library design follows RAG patterns: simple JSON storage with semantic retrieval, no vector database needed for this scale. Checkpoint patterns from distributed training frameworks (Ray Tune, PyTorch Lightning) inform iteration state persistence, though this project's simpler requirements allow custom atomic-write checkpointing.

**Primary recommendation:** Use sentence-transformers/all-MiniLM-L6-v2 with faiss-cpu IndexFlatIP (normalize embeddings), asyncio.Semaphore(10) with TaskGroup for concurrent evaluation, JSON skill library with atomic writes, and iteration checkpoint files tracking progress + skill library state for stop/resume capability.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| sentence-transformers | 3.4+ | Semantic embedding generation | Production-ready, 10k+ models on HuggingFace, all-MiniLM-L6-v2 (384-dim) is fast and accurate |
| faiss-cpu | 1.13+ | Vector similarity search | Meta's industry standard, IndexFlatIP for exact search on small corpora, no GPU needed for <10k vectors |
| tqdm | 4.66+ | Progress tracking for asyncio | Built-in asyncio support via tqdm.asyncio.gather, concurrent task visualization |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | latest | Vector operations for embeddings | Already dependency of sentence-transformers, used for faiss.normalize_L2() |
| asyncio (stdlib) | Python 3.11+ | Concurrent task execution | TaskGroup for structured concurrency (safer than gather) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| faiss-cpu | ChromaDB, Pinecone | Vector DBs overkill for <1000 skills. FAISS is lightweight, no server needed, exact search sufficient |
| IndexFlatIP | IndexHNSWFlat, IndexIVFFlat | ANN methods for scale (millions), but Phase 2 has ~100-200 skills max. Flat index is simpler and exact |
| all-MiniLM-L6-v2 | all-mpnet-base-v2 | mpnet is more accurate (768-dim) but 3x slower. MiniLM balances speed/quality for retrieval |
| asyncio.Semaphore | ThreadPoolExecutor | Threads for I/O-bound tasks, but async already used in Phase 1 agent loop. Stay consistent |

**Installation:**
```bash
pip install sentence-transformers>=3.4
pip install faiss-cpu>=1.13
pip install tqdm>=4.66
# numpy already installed as sentence-transformers dependency
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── skills/                    # Skill library system (NEW)
│   ├── models.py             # Skill dataclass (name, principle, when_to_apply)
│   ├── library.py            # Skill library: add/update/remove, persistence
│   ├── retrieval.py          # Semantic retrieval: encode + FAISS search
│   └── storage.py            # Atomic JSON write for skill library
├── evaluation/               # Full 134-task evaluation (NEW)
│   ├── orchestrator.py       # Parallel task execution with semaphore
│   ├── metrics.py            # Per-task and aggregate metrics
│   └── checkpoint.py         # Iteration state persistence (stop/resume)
├── agent/                    # Extend from Phase 1
│   ├── prompts.py            # Add skill injection to system prompt
│   └── loop.py               # run_task() unchanged (no modifications needed)
├── environment/              # Phase 1 - no changes
├── trajectory/               # Phase 1 - no changes
└── main.py                   # New CLI: run full evaluation
```

### Pattern 1: Semantic Skill Retrieval
**What:** Encode task descriptions and skills, use FAISS for TopK nearest neighbor search
**When to use:** Every task execution to inject relevant skills into agent prompt
**Example:**
```python
# Source: https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class SkillRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.skills = []

    def index_skills(self, skills: list[dict]):
        """Build FAISS index from skill library."""
        self.skills = skills

        # Encode skill descriptions (concatenate for richer context)
        skill_texts = [
            f"{s['name']}: {s['principle']}. {s['when_to_apply']}"
            for s in skills
        ]

        embeddings = self.model.encode(
            skill_texts,
            normalize_embeddings=True,  # Enable for cosine similarity via dot product
            convert_to_tensor=False      # Return numpy arrays for FAISS
        )

        # Create FAISS index (IndexFlatIP for inner product / cosine similarity)
        dimension = embeddings.shape[1]  # 384 for all-MiniLM-L6-v2
        self.index = faiss.IndexFlatIP(dimension)

        # Normalize for cosine similarity (IndexFlatIP uses dot product)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve TopK most relevant skills for query."""
        if self.index is None:
            return []

        # Encode query
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_tensor=False
        )

        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        # Return skills with scores
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.skills):
                results.append({
                    **self.skills[idx],
                    "relevance_score": float(score)
                })

        return results
```

**Key points:**
- `normalize_embeddings=True` in encode() enables cosine similarity via dot product
- `faiss.normalize_L2()` normalizes numpy arrays for IndexFlatIP
- IndexFlatIP returns exact results (no approximation) for small corpora
- Concatenate skill fields for richer semantic representation

### Pattern 2: Parallel Task Execution with Semaphore
**What:** Limit concurrent tasks using asyncio.Semaphore, track progress with tqdm
**When to use:** Full 134-task evaluation with 10 concurrent workers
**Example:**
```python
# Source: https://rednafi.com/python/limit-concurrency-with-semaphore/
import asyncio
from tqdm.asyncio import tqdm_asyncio

async def run_full_evaluation(
    env_manager,
    client,
    tools_spec,
    max_concurrent: int = 10,
    max_steps: int = 50
):
    """Run all 134 tasks with limited concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)

    # Load all tasks from ALFWorld
    tasks = []
    for task_idx in range(134):
        obs, info = env_manager.reset()
        tasks.append({
            "task_id": env_manager.get_task_id(),
            "task_type": env_manager.get_task_type(),
            "description": obs,
        })

    async def bounded_task(task_info):
        """Run single task with semaphore limiting."""
        async with semaphore:
            # Reset to specific task (will need task index tracking)
            trajectory = await run_task(
                task_description=task_info["description"],
                task_id=task_info["task_id"],
                task_type=task_info["task_type"],
                env_manager=env_manager,
                tools_spec=tools_spec,
                client=client,
                max_steps=max_steps,
            )
            return trajectory

    # Use tqdm_asyncio.gather for progress bar
    trajectories = await tqdm_asyncio.gather(
        *[bounded_task(t) for t in tasks],
        desc="Evaluating tasks",
        total=len(tasks)
    )

    return trajectories
```

**Modern alternative using TaskGroup (Python 3.11+):**
```python
# Source: https://docs.python.org/3/library/asyncio-task.html
async def run_full_evaluation_taskgroup(tasks, max_concurrent=10):
    """Run tasks with TaskGroup (safer exception handling)."""
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async with asyncio.TaskGroup() as tg:
        async def bounded_task(task_info):
            async with semaphore:
                return await run_task(...)

        # Create all tasks (execution managed by TaskGroup)
        task_handles = [
            tg.create_task(bounded_task(t))
            for t in tasks
        ]

    # All tasks complete (or group raises on first exception)
    results = [t.result() for t in task_handles]
    return results
```

**Key points:**
- Semaphore limits concurrent execution (prevents overwhelming API or environment)
- TaskGroup automatically cancels remaining tasks on first exception (safer)
- tqdm.asyncio.gather provides progress bar without intruding into task logic
- Semaphore value of 10 balances throughput and API courtesy

### Pattern 3: Skill Library Persistence with Atomic Writes
**What:** Extend Phase 1's atomic write pattern to skill library JSON
**When to use:** Saving skill library after teacher updates
**Example:**
```python
# Source: Phase 1 trajectory/storage.py pattern + https://code.activestate.com/recipes/579097-safely-and-atomically-write-to-a-file/
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class Skill:
    name: str
    principle: str
    when_to_apply: str
    created_iteration: int = 0
    last_used_iteration: int = 0
    usage_count: int = 0

class SkillLibrary:
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.skills: dict[str, Skill] = {}

    def load(self):
        """Load skills from JSON file."""
        if not self.storage_path.exists():
            self.skills = {}
            return

        with open(self.storage_path, 'r') as f:
            data = json.load(f)

        self.skills = {
            name: Skill(**skill_dict)
            for name, skill_dict in data.items()
        }

    def save(self):
        """Save skills to JSON with atomic write."""
        # Serialize to dict
        data = {
            name: asdict(skill)
            for name, skill in self.skills.items()
        }

        # Atomic write pattern: temp + fsync + replace
        temp_path = self.storage_path.with_suffix('.tmp')

        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # Atomic replace
        os.replace(temp_path, self.storage_path)

    def add_skill(self, skill: Skill):
        """Add or update skill in library."""
        self.skills[skill.name] = skill
        self.save()

    def remove_skill(self, name: str):
        """Remove skill from library."""
        if name in self.skills:
            del self.skills[name]
            self.save()
```

**Key points:**
- Same atomic write pattern as trajectory storage (proven in Phase 1)
- JSON format (not JSONL) since entire library fits in memory
- Dict keyed by skill name for O(1) lookup and deduplication
- Include metadata (usage_count, created_iteration) for future pruning

### Pattern 4: Iteration Checkpoint for Stop/Resume
**What:** Persist iteration state to resume evaluation after interruption
**When to use:** After each iteration or periodically during long evaluation runs
**Example:**
```python
# Source: Ray Tune patterns + custom adaptation for this project
from dataclasses import dataclass
import json

@dataclass
class IterationCheckpoint:
    iteration: int
    tasks_completed: int
    total_tasks: int
    skill_library_snapshot: str  # Path to skill library JSON at this iteration
    trajectories_file: str       # Path to trajectories JSONL
    aggregate_metrics: dict      # Success rate, avg steps, per-subtask metrics
    timestamp: float

class CheckpointManager:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, checkpoint: IterationCheckpoint):
        """Save iteration checkpoint with atomic write."""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_iter_{checkpoint.iteration}.json"
        latest_link = self.checkpoint_dir / "checkpoint_latest.json"

        # Write checkpoint
        temp_path = checkpoint_file.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(asdict(checkpoint), f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, checkpoint_file)

        # Update latest symlink (for easy resume)
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(checkpoint_file.name)

    def load_latest(self) -> IterationCheckpoint | None:
        """Load most recent checkpoint."""
        latest_link = self.checkpoint_dir / "checkpoint_latest.json"
        if not latest_link.exists():
            return None

        with open(latest_link, 'r') as f:
            data = json.load(f)

        return IterationCheckpoint(**data)
```

**Key points:**
- Checkpoint after each iteration (not per-task, too granular)
- Include snapshot paths (not full data) to keep checkpoints small
- Symlink "latest" checkpoint for easy resume
- Atomic writes prevent checkpoint corruption

### Pattern 5: Injecting Skills into Agent Prompt
**What:** Modify system prompt to include retrieved skills before task description
**When to use:** Every task execution when skills are available
**Example:**
```python
# Source: RAG prompt patterns + Phase 1 prompts.py
def build_agent_prompt_with_skills(retrieved_skills: list[dict]) -> str:
    """Build system prompt with injected skills."""
    base_prompt = """You are an autonomous agent in an ALFWorld household simulation.

Available Tools:
[... tool list from Phase 1 ...]

Instructions:
[... instructions from Phase 1 ...]
"""

    if not retrieved_skills:
        return base_prompt

    # Inject skills section
    skills_section = "\n\nRelevant Skills (learned from past experience):\n"
    for i, skill in enumerate(retrieved_skills, 1):
        skills_section += f"\n{i}. {skill['name']}\n"
        skills_section += f"   Principle: {skill['principle']}\n"
        skills_section += f"   When to apply: {skill['when_to_apply']}\n"

    skills_section += "\nConsider these skills when planning your approach.\n"

    # Insert skills after tool list, before instructions
    return base_prompt.replace(
        "Instructions:",
        skills_section + "\nInstructions:"
    )
```

**Key points:**
- Skills inserted between tools and instructions (natural reading order)
- Include relevance_score in prompt (optional, for debugging)
- Keep skills concise (no full trajectories, just principles)
- "Consider" phrasing (not "must use") allows agent flexibility

### Anti-Patterns to Avoid
- **Using vector databases (ChromaDB, Pinecone) for small corpora:** Overkill for <1000 skills. FAISS in-memory is faster and simpler
- **Not normalizing embeddings before IndexFlatIP:** Results in dot product, not cosine similarity. Always normalize with `faiss.normalize_L2()`
- **Using asyncio.gather() without return_exceptions:** First exception kills all tasks. Use TaskGroup or gather(return_exceptions=True)
- **Checkpointing after every task:** 134 checkpoints per iteration is excessive. Checkpoint per iteration only
- **Loading entire skill library into every prompt:** Retrieve TopK (3-5) most relevant skills, not all skills
- **Forgetting to reset environment between tasks:** ALFWorld requires reset() to load new task; reusing same environment instance

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Semantic similarity search | Custom cosine similarity loops | sentence-transformers + FAISS IndexFlatIP | Optimized C++ implementation, 100x faster than Python loops |
| Progress tracking for async tasks | Manual print statements | tqdm.asyncio.gather | Clean progress bars, no intrusion into business logic, handles updates correctly |
| Embedding model selection | Train custom embeddings | Pre-trained all-MiniLM-L6-v2 | Trained on 1B+ pairs, good out-of-box performance, 22MB model |
| Concurrent task limiting | Manual task queues | asyncio.Semaphore | Prevents deadlocks, integrates with async context managers |
| Vector normalization | Manual L2 norm computation | faiss.normalize_L2() | SIMD-optimized, handles batch operations efficiently |
| Iteration checkpointing | Custom state serialization | Extend Phase 1 atomic write pattern | Already proven in trajectory storage, crash-resilient |

**Key insight:** Semantic retrieval and parallel evaluation have mature Python ecosystems as of 2026. sentence-transformers (10k+ models), FAISS (Meta's standard), and asyncio patterns are production-ready. Custom solutions introduce performance penalties (Python loops vs SIMD) and bugs (forgetting edge cases in normalization, semaphore deadlocks).

## Common Pitfalls

### Pitfall 1: IndexFlatIP Without Normalization
**What goes wrong:** FAISS returns dot product scores instead of cosine similarity, ranking skills incorrectly
**Why it happens:** IndexFlatIP computes inner product. Cosine similarity requires normalized vectors (cos θ = dot(u,v) / (||u|| * ||v||))
**How to avoid:**
- Set `normalize_embeddings=True` in SentenceTransformer.encode()
- Call `faiss.normalize_L2(embeddings)` before adding to index
- Normalize query embeddings before search
**Warning signs:**
- Longer skill descriptions always rank higher (not normalized by length)
- Scores outside [-1, 1] range (cosine similarity is bounded)
- Retrieval favors verbose skills over semantically relevant ones

### Pitfall 2: Semaphore Deadlock with TaskGroup
**What goes wrong:** Tasks hang indefinitely when combining Semaphore with TaskGroup exception handling
**Why it happens:** TaskGroup cancels remaining tasks on first exception, but semaphore may still be held by cancelled task, blocking others
**How to avoid:**
- Always use `async with semaphore:` to ensure release on cancellation
- Handle exceptions inside bounded_task(), not outside
- Use gather(return_exceptions=True) if continuing on errors is desired
**Warning signs:**
- Some tasks complete but others hang forever
- Semaphore count never reaches capacity after exception
- No clear error message (silent hang)

### Pitfall 3: ALFWorld Task Indexing vs Reset
**What goes wrong:** Calling reset() multiple times to reach task N doesn't guarantee task N loads
**Why it happens:** ALFWorld's reset() advances to next task in sequence, but sequence may not be deterministic across runs
**How to avoid:**
- Load environment once, iterate through all 134 tasks sequentially with reset()
- Collect all task descriptions in first pass, then evaluate
- Don't try to "jump" to specific task index by calling reset() N times
**Warning signs:**
- Tasks appear in different order across evaluation runs
- Same task appears multiple times in single run
- Total tasks ≠ 134

### Pitfall 4: Skill Retrieval at Wrong Granularity
**What goes wrong:** Retrieving skills once per evaluation instead of once per task, losing task-specific relevance
**Why it happens:** Confusion between "skill library" (global) and "retrieved skills" (task-specific)
**How to avoid:**
- Retrieve skills for each task based on task description
- Different tasks get different TopK skills (semantic matching)
- Don't inject all skills into all prompts
**Warning signs:**
- Same skills appear in all task prompts
- Agent receives skills irrelevant to current task
- No performance difference between tasks of different types

### Pitfall 5: Progress Bar Blocking in asyncio
**What goes wrong:** Using synchronous tqdm with asyncio causes blocked event loop
**Why it happens:** Regular tqdm.tqdm() doesn't know about asyncio, updates can block
**How to avoid:**
- Use tqdm.asyncio.tqdm_asyncio.gather() instead of asyncio.gather()
- For manual updates, use tqdm.asyncio.tqdm(asyncio=True)
- Never call synchronous operations inside async functions
**Warning signs:**
- Progress bar doesn't update until tasks complete
- Event loop latency spikes during updates
- Tasks appear to run serially instead of concurrently

### Pitfall 6: Not Handling Empty Skill Library
**What goes wrong:** Code crashes when retrieving from empty library (iteration 0 baseline)
**Why it happens:** Iteration 0 has no skills, but retrieval code assumes skills exist
**How to avoid:**
- Check if skill library is empty before retrieval
- Return empty list for TopK retrieval on empty library
- Agent prompt gracefully handles absence of skills section
**Warning signs:**
- Iteration 0 fails but iteration 1+ succeeds
- FAISS index.search() raises on empty index
- Prompt has "Relevant Skills:" header but no skills listed

## Code Examples

Verified patterns from official sources:

### Complete Skill Retrieval System
```python
# Source: https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dataclasses import dataclass, asdict
import json
from pathlib import Path

@dataclass
class Skill:
    name: str
    principle: str
    when_to_apply: str

class SkillLibraryWithRetrieval:
    def __init__(
        self,
        storage_path: Path,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.storage_path = storage_path
        self.model = SentenceTransformer(model_name)
        self.skills: list[Skill] = []
        self.index: faiss.IndexFlatIP | None = None

    def load(self):
        """Load skills from JSON file."""
        if not self.storage_path.exists():
            self.skills = []
            return

        with open(self.storage_path, 'r') as f:
            data = json.load(f)

        self.skills = [Skill(**skill) for skill in data]

        # Build FAISS index
        if self.skills:
            self._build_index()

    def _build_index(self):
        """Build FAISS index from current skills."""
        # Concatenate skill fields for richer semantic representation
        skill_texts = [
            f"{s.name}: {s.principle}. {s.when_to_apply}"
            for s in self.skills
        ]

        # Encode with normalization
        embeddings = self.model.encode(
            skill_texts,
            normalize_embeddings=True,
            convert_to_tensor=False,
            show_progress_bar=False
        )

        # Create and populate index
        dimension = embeddings.shape[1]  # 384 for all-MiniLM-L6-v2
        self.index = faiss.IndexFlatIP(dimension)

        # Add normalized embeddings
        faiss.normalize_L2(embeddings)  # Ensure normalized for cosine similarity
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 3) -> list[Skill]:
        """Retrieve TopK most relevant skills."""
        if not self.skills or self.index is None:
            return []

        # Limit top_k to available skills
        top_k = min(top_k, len(self.skills))

        # Encode query with normalization
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_tensor=False,
            show_progress_bar=False
        )

        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        # Return skills (scores available for debugging if needed)
        return [self.skills[idx] for idx in indices[0] if idx < len(self.skills)]
```

### Parallel Evaluation with Progress Tracking
```python
# Source: https://www.dataleadsfuture.com/using-tqdm-with-asyncio-in-python/
import asyncio
from tqdm.asyncio import tqdm_asyncio
from typing import Any

async def evaluate_all_tasks(
    client,
    tools_spec,
    skill_library,
    max_concurrent: int = 10,
    max_steps: int = 50
) -> list[Trajectory]:
    """Evaluate all 134 ALFWorld tasks with parallel execution."""

    # Initialize environment (module-level singleton from Phase 1)
    from src.environment.env_manager import EnvManager
    env_manager = EnvManager()
    env_manager.load()

    # Collect all task descriptions
    tasks = []
    for task_idx in range(134):
        obs, info = env_manager.reset()
        tasks.append({
            "index": task_idx,
            "task_id": env_manager.get_task_id(),
            "task_type": env_manager.get_task_type(),
            "description": obs,
        })

    # Semaphore for concurrency limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_single_task(task_info: dict) -> Trajectory:
        """Run single task with semaphore limiting."""
        async with semaphore:
            # Create fresh environment for this task
            env = EnvManager()
            env.load()

            # Reset to correct task (need to iterate from 0)
            for i in range(task_info["index"] + 1):
                obs, info = env.reset()

            # Retrieve relevant skills
            retrieved_skills = skill_library.retrieve(
                task_info["description"],
                top_k=3
            )

            # Run task (with skills injected in prompt)
            trajectory = await run_task(
                task_description=task_info["description"],
                task_id=task_info["task_id"],
                task_type=task_info["task_type"],
                env_manager=env,
                tools_spec=tools_spec,
                client=client,
                max_steps=max_steps,
                retrieved_skills=retrieved_skills  # Pass to run_task
            )

            return trajectory

    # Run all tasks with progress bar
    trajectories = await tqdm_asyncio.gather(
        *[run_single_task(task) for task in tasks],
        desc="Evaluating 134 tasks",
        unit="task"
    )

    return trajectories
```

### Metrics Aggregation
```python
# Source: Custom pattern for this project's requirements
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class AggregateMetrics:
    iteration: int
    overall_success_rate: float
    per_subtask_success_rate: dict[str, float]  # task_type -> success_rate
    avg_steps_success: float
    avg_steps_failure: float
    total_tasks: int
    successful_tasks: int

def compute_metrics(trajectories: list[Trajectory], iteration: int) -> AggregateMetrics:
    """Compute aggregate metrics from trajectories."""
    total = len(trajectories)
    successful = sum(1 for t in trajectories if t.success)

    # Per-subtask success rates
    subtask_counts = defaultdict(lambda: {"total": 0, "success": 0})
    for t in trajectories:
        subtask_counts[t.task_type]["total"] += 1
        if t.success:
            subtask_counts[t.task_type]["success"] += 1

    per_subtask_rate = {
        task_type: counts["success"] / counts["total"]
        for task_type, counts in subtask_counts.items()
    }

    # Average steps (separate for success/failure)
    success_steps = [t.total_steps for t in trajectories if t.success]
    failure_steps = [t.total_steps for t in trajectories if not t.success]

    avg_steps_success = sum(success_steps) / len(success_steps) if success_steps else 0
    avg_steps_failure = sum(failure_steps) / len(failure_steps) if failure_steps else 0

    return AggregateMetrics(
        iteration=iteration,
        overall_success_rate=successful / total if total > 0 else 0,
        per_subtask_success_rate=per_subtask_rate,
        avg_steps_success=avg_steps_success,
        avg_steps_failure=avg_steps_failure,
        total_tasks=total,
        successful_tasks=successful,
    )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Custom embedding models | Pre-trained sentence-transformers | 2019-2020 | all-MiniLM-L6-v2 matches custom models trained on domain data for most tasks |
| Annoy, hnswlib for ANN | FAISS (Facebook) | 2017-2020 | Industry standard, better optimized, supports both exact and ANN |
| asyncio.gather() for concurrency | asyncio.TaskGroup() | Python 3.11 (2022) | Structured concurrency, automatic cancellation on exceptions |
| Vector databases (Pinecone, Weaviate) | In-memory FAISS for small corpora | 2023-2024 | No server overhead, faster for <10k vectors, simpler deployment |
| Manual progress tracking | tqdm.asyncio.gather | tqdm 4.50+ (2020) | Clean async progress bars without event loop blocking |
| Pydantic for all data | Dataclass for internal (Phase 1 decision) | 2024-2025 | 10-100x faster, JSON-serializable without overhead |

**Deprecated/outdated:**
- **sentence-transformers <3.0:** Major API changes in 3.0+ (2024), use latest version
- **asyncio.gather() without exception handling:** TaskGroup is safer for production code
- **faiss.IndexIVFFlat for small datasets:** Overhead of clustering not worth it for <10k vectors
- **Loading full skill trajectories into prompts:** RAG pattern: retrieve summaries (principles), not full context

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal TopK for Skill Retrieval**
   - What we know: RAG systems typically use 3-5 retrieved documents
   - What's unclear: Optimal K for skill library (shorter than documents), impact on success rate
   - Recommendation: Start with K=3, experiment with K=1,3,5 in Phase 3. Track "skills used" in trajectories to measure relevance

2. **Skill Description Format for Retrieval**
   - What we know: Concatenating skill fields improves semantic matching
   - What's unclear: Best format: "name: principle. when_to_apply" vs "principle when_to_apply" vs separate embeddings
   - Recommendation: Use concatenated format with name (helps distinguish similar principles). Consider weighted combination in Phase 3 if needed

3. **Environment Isolation for Parallel Execution**
   - What we know: ALFWorld uses global state (TextWorld engine), unclear if thread-safe
   - What's unclear: Can single EnvManager instance serve 10 concurrent tasks, or need 10 instances?
   - Recommendation: Test with single shared instance first (simpler). If state corruption occurs, create env_manager per worker. Phase 1 used single instance for sequential tasks.

4. **Task Reset Determinism**
   - What we know: env.reset() advances to next task in sequence
   - What's unclear: Is sequence order deterministic across runs? Does it depend on random seed?
   - Recommendation: Collect all 134 task descriptions in first pass (single sequential reset loop), then run evaluation. Don't assume reset order is repeatable.

5. **Checkpoint Granularity Trade-offs**
   - What we know: Per-iteration checkpoints are sufficient for stop/resume
   - What's unclear: Should we also checkpoint every N tasks during evaluation (e.g., every 25 tasks)?
   - Recommendation: Start with per-iteration only (simpler). Add intra-iteration checkpoints if evaluation takes >1 hour and interruptions are common. Not critical for Phase 2.

## Sources

### Primary (HIGH confidence)
- [sentence-transformers Semantic Search Docs](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html) - Recommended approach, model selection, code examples
- [sentence-transformers API Reference](https://sbert.net/docs/package_reference/util.html) - normalize_embeddings parameter, similarity methods
- [FAISS GitHub Wiki - Indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) - IndexFlatIP vs IndexFlatL2, normalization requirements
- [Python asyncio Documentation - Tasks](https://docs.python.org/3/library/asyncio-task.html) - TaskGroup, gather(), Semaphore patterns
- [tqdm.asyncio Documentation](https://tqdm.github.io/docs/asyncio/) - tqdm_asyncio.gather usage
- [Hugging Face - all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - Model card, specs, performance

### Secondary (MEDIUM confidence)
- [Limit Concurrency with Semaphore (rednafi.com)](https://rednafi.com/python/limit-concurrency-with-semaphore/) - Verified pattern for asyncio.Semaphore
- [Using tqdm with Asyncio (Towards Data Science)](https://towardsdatascience.com/using-tqdm-with-asyncio-in-python-5c0f6e747d55/) - Progress tracking patterns
- [FAISS Cosine Similarity Guide (MyScale)](https://www.myscale.com/blog/faiss-cosine-similarity-enhances-search-efficiency/) - IndexFlatIP with normalization
- [Better File Writing in Python (Medium)](https://sahmanish20.medium.com/better-file-writing-in-python-embrace-atomic-updates-593843bfab4f) - Atomic write patterns
- [ALFWorld Paper (OpenReview)](https://openreview.net/pdf?id=0IOX0YcCdTn) - 134 tasks, 6 subtask types, evaluation protocol

### Tertiary (LOW confidence)
- WebSearch: "sentence-transformers all-MiniLM-L6-v2 latest version" - 3.4+ confirmed from PyPI, January 2026 release
- WebSearch: "FAISS IndexFlatIP performance vs IndexFlatL2" - IndexFlatIP slightly faster, needs verification with benchmark
- WebSearch: "python checkpoint resume patterns 2026" - Multiple frameworks (Ray, PyTorch Lightning), general patterns extracted

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - sentence-transformers, FAISS, tqdm all have official docs verified
- Architecture: HIGH - Patterns verified from official docs (sentence-transformers, asyncio), established RAG principles
- Pitfalls: MEDIUM to HIGH - Normalization pitfall from FAISS GitHub issues (HIGH), semaphore deadlock from community experience (MEDIUM), task reset from ALFWorld experimentation needed (MEDIUM)

**Research date:** 2026-02-11
**Valid until:** ~2026-03-15 (30 days for stable domain—sentence-transformers and FAISS are mature libraries, asyncio patterns stable since Python 3.11)

**Validation needed:**
- ALFWorld task reset determinism (test with repeated runs)
- Environment thread-safety for concurrent execution (test with parallel workers)
- Optimal TopK for skill retrieval (experiment in Phase 2 or 3)
- Checkpoint frequency trade-offs (monitor evaluation runtime)
