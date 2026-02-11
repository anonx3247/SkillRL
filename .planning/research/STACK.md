# Technology Stack

**Project:** SkillRL Frozen-Model Ablation
**Researched:** 2026-02-11
**Overall Confidence:** MEDIUM (web verification tools unavailable, relying on training data + project constraints)

## Recommended Stack

### Core Environment

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| Python | 3.10+ | Runtime | ALFWorld requires 3.8+, FastMCP works best with 3.10+, asyncio improvements | HIGH |
| ALFWorld | 0.3.3+ | Benchmark environment | Official benchmark from paper, standard text-based household tasks | MEDIUM |
| FastMCP | Latest (0.x) | Tool protocol server | Official Python MCP SDK, clean tool interface for agent-environment interaction | LOW |

**Rationale for Python 3.10+:**
- ALFWorld has loose requirements (3.8+) but uses type hints extensively
- FastMCP leverages modern asyncio (structured concurrency)
- DeepSeek API client (openai package) works best with 3.10+
- No reason to target older versions for greenfield project

**ALFWorld Installation:**
```bash
pip install alfworld
# or from source for latest:
pip install git+https://github.com/alfworld/alfworld.git
```

**Note on ALFWorld:**
- Depends on TextWorld engine (>=1.5.3)
- Includes pretrained models (unused in frozen setup)
- Provides 134 test tasks standard split
- May require downloading game files on first run

### API Client

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| openai | 1.0+ | DeepSeek API client | DeepSeek is OpenAI-compatible, official client handles retries/streaming/errors | HIGH |

**Rationale for openai package:**
- DeepSeek V3.2 API explicitly OpenAI-compatible
- Better than requests: built-in retry logic, streaming, error handling, type hints
- Simpler than litellm: no need for multi-provider abstraction
- Native async support for concurrent task execution

**Configuration:**
```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-...",  # DeepSeek API key
    base_url="https://api.deepseek.com/v1"
)
```

**Alternative NOT recommended:**
- `requests` library: Too low-level, manual retry logic, no streaming support
- `litellm`: Overkill for single provider, adds complexity
- `anthropic`: Wrong API format

### MCP Tool Server

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| mcp | 0.x (Python SDK) | FastMCP tool definitions | Official MCP Python SDK for tool servers | LOW |
| pydantic | 2.0+ | Tool schema validation | MCP uses Pydantic models for tool parameters | HIGH |

**Rationale for MCP Python SDK:**
- Official implementation of Model Context Protocol
- Clean decorator-based tool definition (`@mcp.tool()`)
- Automatic JSON schema generation from type hints
- Built-in async support for tool execution

**Tool Server Structure:**
```python
# alfworld_server.py - Environment tools
@mcp.tool()
async def go_to(target: str) -> str:
    """Navigate to target object/location"""

# skill_server.py - Skill management tools
@mcp.tool()
async def add_skill(category: str, skill: dict) -> str:
    """Add skill to library"""
```

**Alternative NOT recommended:**
- Custom JSON-RPC server: Reinventing the wheel, no standard compliance
- Function calling via system prompt: Fragile, no schema validation
- Langchain tools: Heavy dependency, overkill for this use case

### Skill Library Storage

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| JSON files | stdlib | Skill persistence | Simple, human-readable, git-friendly, no DB overhead | HIGH |
| sentence-transformers | 2.2+ | Semantic retrieval | Embed skills for similarity-based retrieval (K=6 nearest) | MEDIUM |
| faiss-cpu | 1.7+ | Vector search | Fast ANN search for skill retrieval, scales to 100s of skills | MEDIUM |

**Rationale for JSON + embeddings:**
- **JSON files** for persistence:
  - Human-readable for debugging skill content
  - Git-trackable for evolution history
  - No database setup/maintenance
  - Fast enough for 100-skill library
  - Structure: `skills/general.json`, `skills/pick.json`, etc.

- **sentence-transformers** for embeddings:
  - `all-MiniLM-L6-v2` model: fast, 384-dim, good quality
  - Offline inference (no API calls)
  - Consistent with paper's semantic retrieval approach

- **faiss-cpu** for search:
  - Exact search fast enough for small library (<1000 skills)
  - Scales if library grows unexpectedly
  - Simple API: `index.add()`, `index.search()`

**Storage Structure:**
```
skills/
  general.json          # 12 initial general skills
  pick.json            # Task-specific for Pick subtask
  look.json            # Task-specific for Look subtask
  clean.json           # ...
  heat.json
  cool.json
  pick2.json
  embeddings.npy       # Precomputed embeddings
  index.faiss          # FAISS index
```

**Alternative NOT recommended:**
- SQLite: Overkill for 100 skills, harder to inspect/debug
- Vector DB (Qdrant, Weaviate): Infrastructure overhead, unnecessary complexity
- Pickle files: Not human-readable, not git-friendly
- Raw regex matching: No semantic understanding, fragile

### Experiment State Management

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| JSON + filesystem | stdlib | Iteration state | Simple checkpointing, stop/resume, no framework needed | HIGH |
| pandas | 2.0+ | Metrics tracking | Tabular results, easy plotting, standard analysis format | HIGH |

**Rationale for JSON checkpointing:**
- **State to persist:**
  - Current iteration number
  - Skill library version hash
  - Per-task results (success, steps, skills_used)
  - Evolution history (skills added/removed each iteration)

- **Why not a framework:**
  - MLflow/W&B: Overkill for single-experiment ablation
  - Sacred/Hydra: Configuration management, not needed here
  - Custom is 50 lines, more explicit

**Checkpoint Structure:**
```
state/
  iteration_0/
    results.jsonl      # One JSON per task (134 lines)
    skills/            # Skill library snapshot
    metrics.json       # Aggregated: success_rate, avg_steps
  iteration_1/
    ...
  latest -> iteration_3  # Symlink for resume
```

**Pandas for analysis:**
- Load all `results.jsonl` → DataFrame
- Groupby iteration → success rate, avg steps
- Easy plotting: `df.plot(x='iteration', y='success_rate')`

**Alternative NOT recommended:**
- HDF5/Parquet: Binary formats, harder debugging
- Database: Infrastructure for 134 rows per iteration
- In-memory only: Loses work on crash
- Pickle: Fragile across Python versions

### Supporting Libraries

| Library | Version | Purpose | When to Use | Confidence |
|---------|---------|---------|-------------|------------|
| asyncio | stdlib | Concurrent execution | Run multiple ALFWorld tasks in parallel (speed up eval) | HIGH |
| aiofiles | 23.0+ | Async file I/O | Non-blocking writes during concurrent task execution | MEDIUM |
| numpy | 1.24+ | Array operations | Embeddings, FAISS indexing, metrics computation | HIGH |
| tqdm | 4.65+ | Progress bars | Iteration progress, task execution tracking | HIGH |
| pyyaml | 6.0+ | Config files | ALFWorld config, experiment hyperparameters | HIGH |
| pytest | 7.3+ | Testing | Unit tests for skill retrieval, tool execution | HIGH |
| python-dotenv | 1.0+ | Env vars | DeepSeek API key management | HIGH |

**Async execution rationale:**
- ALFWorld tasks are independent → parallelize evaluation
- 134 tasks × 50 steps × 2s/step = 3.7 hours sequential
- With 10 concurrent tasks → ~30 minutes
- DeepSeek API allows concurrent requests

**Testing focus:**
- Skill retrieval logic (K-nearest, category filtering)
- Tool execution (mocked ALFWorld responses)
- Checkpoint save/load (state integrity)
- NOT end-to-end (too slow for CI)

## Alternatives Considered

| Category | Recommended | Alternative | Why Not | Confidence |
|----------|-------------|-------------|---------|------------|
| API Client | openai | requests | Manual retry/streaming, more code | HIGH |
| API Client | openai | litellm | Multi-provider overkill | HIGH |
| Tool Protocol | FastMCP | Langchain | Heavy deps, agent framework not needed | MEDIUM |
| Tool Protocol | FastMCP | Custom JSON-RPC | Reinventing standard | HIGH |
| Skill Storage | JSON + FAISS | SQLite | Overkill, harder debugging | HIGH |
| Skill Storage | JSON + FAISS | Pinecone/Qdrant | Infrastructure overhead | HIGH |
| Embeddings | sentence-transformers | OpenAI embeddings | API cost, latency, offline preferred | HIGH |
| State Mgmt | JSON checkpoints | MLflow | Framework overkill for single experiment | HIGH |
| State Mgmt | JSON checkpoints | Pickle | Not human-readable, fragile | HIGH |

## Installation

### Core Dependencies

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Core packages
pip install openai>=1.0  # DeepSeek API client
pip install mcp  # FastMCP Python SDK (check actual package name)
pip install alfworld  # ALFWorld benchmark

# Skill library
pip install sentence-transformers>=2.2
pip install faiss-cpu>=1.7

# State management & analysis
pip install pandas>=2.0
pip install numpy>=1.24

# Utilities
pip install aiofiles>=23.0
pip install tqdm>=4.65
pip install pyyaml>=6.0
pip install python-dotenv>=1.0

# Development
pip install pytest>=7.3
pip install pytest-asyncio>=0.21
pip install black>=23.0  # Formatting
pip install ruff>=0.0.270  # Linting
```

### Environment Setup

```bash
# .env file
DEEPSEEK_API_KEY=sk-...

# Download ALFWorld data (first run)
python -c "import alfworld; alfworld.agents.environment.AlfredTWEnv()"
```

### Verification

```bash
# Test ALFWorld installation
python -c "import alfworld.agents.environment as ae; print('ALFWorld OK')"

# Test OpenAI client
python -c "from openai import OpenAI; print('OpenAI client OK')"

# Test sentence-transformers
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('all-MiniLM-L6-v2'); print('Embeddings OK')"

# Test FAISS
python -c "import faiss; print('FAISS OK')"
```

## Architecture Notes

### Project Structure

```
skillrl/
  agent/
    executor.py          # Agent execution loop
    tools.py            # ALFWorld tool wrappers
  teacher/
    distiller.py        # Trajectory → skill extraction
    evolver.py          # Skill library evolution logic
    tools.py            # Skill management MCP tools
  skills/
    library.py          # Skill storage & retrieval
    embedder.py         # Embedding & search
    data/               # JSON files
  evaluation/
    runner.py           # Run 134 tasks, save results
    metrics.py          # Compute success rate, avg steps
  state/
    checkpoint.py       # Save/load iteration state
  config.yaml           # Hyperparameters (K=6, max_steps=50)
  main.py              # Orchestrate iterations
```

### Key Interfaces

**Agent → Environment:**
```python
# Via MCP tools
await go_to("cabinet 1")
await take("apple")
await task_completed()
```

**Agent → Skills:**
```python
# Skills injected into system prompt
skills = library.retrieve(task_type="pick", current_obs=obs, k=6)
prompt = f"{system}\n\nRelevant Skills:\n{skills}\n\n{task}"
```

**Teacher → Skill Library:**
```python
# Via MCP tools
await add_skill(category="general", skill={
    "name": "verify_inventory_before_place",
    "content": "Always check you're holding the target object...",
    "success_rate": 0.0,  # Will track
    "usage_count": 0
})
```

## Version Pinning Strategy

**Pin major versions, allow minor/patch updates:**
```
# requirements.txt
openai>=1.0,<2.0
pandas>=2.0,<3.0
sentence-transformers>=2.2,<3.0
```

**Rationale:**
- Major versions: breaking API changes
- Minor versions: new features, safe to adopt
- Patch versions: bug fixes, always safe

**Exceptions (pin exact versions):**
- alfworld==0.3.3  # Benchmark consistency
- faiss-cpu==1.7.4  # Index compatibility

## Critical Dependencies Audit

| Package | Trust Level | Vulnerability Surface | Notes |
|---------|-------------|----------------------|-------|
| openai | HIGH | Network requests | Official SDK, wide usage |
| alfworld | MEDIUM | Game engine, file I/O | Research code, less scrutiny |
| sentence-transformers | HIGH | Model loading | Hugging Face ecosystem |
| faiss-cpu | HIGH | Native code (C++) | Facebook Research, battle-tested |
| mcp | LOW | New protocol | Check for maturity, may have bugs |

**Security considerations:**
- API key in environment variable (never commit)
- ALFWorld downloads game files (check checksums if paranoid)
- sentence-transformers downloads models (use local cache)
- No user input → low injection risk

## Performance Expectations

| Operation | Expected Time | Notes |
|-----------|---------------|-------|
| Task execution | 20-100s | 10-50 steps × 2s/step (API latency) |
| Full evaluation (134 tasks) | 30-60 min | With 10 concurrent workers |
| Skill retrieval | <10ms | FAISS + 100 skills |
| Embedding 1 skill | ~50ms | sentence-transformers CPU |
| Checkpoint save | <1s | JSON serialization |

**Bottlenecks:**
1. **DeepSeek API latency** (2-5s per call): Parallelize tasks
2. **ALFWorld step execution** (<100ms): Not a bottleneck
3. **Skill embedding** (50ms): Embed once, cache forever

## Common Pitfalls

### FastMCP Integration
**Issue:** MCP Python SDK may still be in flux (0.x versions)
**Mitigation:** Check GitHub for latest examples, be ready to update
**Confidence:** LOW (tool availability)

### ALFWorld Setup
**Issue:** First run downloads game files, may fail silently
**Mitigation:** Run initialization script before main loop
**Detection:** Check for `alfworld_data/` directory

### Async ALFWorld
**Issue:** ALFWorld may not be async-safe (global state)
**Mitigation:** Create separate environment instances per worker
**Detection:** Race conditions, inconsistent results

### Skill Embedding Cache
**Issue:** Forgetting to update embeddings when skills change
**Mitigation:** Hash skill content, rebuild index on mismatch
**Detection:** Retrieved skills don't match task relevance

### DeepSeek Rate Limits
**Issue:** API may rate limit concurrent requests
**Mitigation:** Start with 5 workers, increase if stable
**Detection:** 429 errors, throttling warnings

## Confidence Assessment

| Component | Confidence | Rationale |
|-----------|-----------|-----------|
| Python 3.10+ | HIGH | Standard choice, well-established |
| openai client | HIGH | DeepSeek explicitly OpenAI-compatible |
| ALFWorld 0.3.3 | MEDIUM | Version from training data, may be outdated |
| FastMCP/mcp | LOW | Cannot verify current state, new protocol |
| sentence-transformers | HIGH | Standard for embeddings, stable API |
| faiss-cpu | MEDIUM | Good choice but version unverified |
| JSON storage | HIGH | Simple, proven approach |
| pandas | HIGH | Standard for tabular data |

## Sources

**Note:** Web verification tools were unavailable during research. Recommendations based on:
- Training data (January 2025 cutoff)
- Project constraints (PROJECT.md)
- DeepSeek API documentation (known OpenAI-compatible)
- ALFWorld paper (standard benchmark)
- MCP protocol specification (from Anthropic)

**Critical verification needed:**
1. FastMCP Python SDK actual package name and installation
2. ALFWorld current version (0.3.3 may be outdated)
3. Current best practices for MCP tool server setup (2026)
4. DeepSeek V3.2 Reasoner specific API details

**Recommended validation steps:**
- [ ] Check https://github.com/modelcontextprotocol/python-sdk for MCP installation
- [ ] Check https://github.com/alfworld/alfworld for current version
- [ ] Check https://platform.deepseek.com/docs for API details
- [ ] Test all package installations before roadmap finalization
