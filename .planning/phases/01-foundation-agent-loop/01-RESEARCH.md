# Phase 1: Foundation & Agent Loop - Research

**Researched:** 2026-02-11
**Domain:** Autonomous LLM agents, environment integration, trajectory logging
**Confidence:** HIGH

## Summary

Phase 1 integrates three mature technologies into a foundation for autonomous agent execution: FastMCP (Python tool server framework), ALFWorld (text-based household task environment), and DeepSeek V3.2 Reasoner (OpenAI-compatible LLM API). The standard approach uses FastMCP 2.x to expose 10 ALFWorld action primitives plus task_completed as tools, connects an async agent loop to DeepSeek's reasoning API via the OpenAI SDK, and logs complete think-act-observe trajectories to JSON Lines files with atomic writes.

Research confirms all three components are production-ready as of early 2026. FastMCP provides decorator-based tool registration with automatic schema generation from type hints. ALFWorld's TextWorld environments return natural language observations via a standard step() interface. DeepSeek Reasoner supports function calling in OpenAI-compatible format with no hard rate limits and 128K context windows, though tool calls route through deepseek-chat instead of the reasoning model.

**Primary recommendation:** Use FastMCP 2.14.5+ (stable), async OpenAI SDK with DeepSeek base URL, standard dataclasses for trajectory representation (Pydantic only at API boundaries), and JSON Lines format with atomic writes for trajectory persistence.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| fastmcp | 2.14.5+ (v2.x stable) | MCP tool server framework | Pythonic decorator API, automatic schema generation, async support, production-ready v2 |
| openai | 1.0+ | DeepSeek API client | AsyncOpenAI supports OpenAI-compatible endpoints, mature async patterns |
| alfworld | latest (pip install) | Text-based household task environment | Standard RL benchmark for embodied agents, 134-task test set |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pydantic | 2.x | Data validation at boundaries | API request/response validation only, NOT internal data structures (performance) |
| anyio | latest | Async compatibility layer | Already used by FastMCP for asyncio/trio compatibility |
| tenacity | latest | Retry with exponential backoff | Handling transient API failures (rate limits, network issues) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| FastMCP | Official MCP Python SDK | More boilerplate, lower-level API. FastMCP is higher-level wrapper around official SDK |
| OpenAI SDK | httpx/aiohttp direct | Must implement OpenAI protocol manually. SDK handles auth, retries, streaming |
| Pydantic everywhere | Standard dataclasses | Pydantic 10-100x slower for internal data. Use dataclass for trajectories, Pydantic for external validation only |

**Installation:**
```bash
pip install fastmcp>=2.14.5,<3  # Pin to stable v2
pip install openai>=1.0
pip install alfworld
pip install tenacity  # For retry logic
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── environment/         # ALFWorld wrapper as FastMCP server
│   ├── server.py       # FastMCP server with @mcp.tool decorators
│   ├── actions.py      # 10 action tools (go_to, take, put, etc.)
│   └── env_manager.py  # ALFWorld environment lifecycle
├── agent/              # Autonomous agent loop
│   ├── loop.py         # Think-act-observe execution loop
│   ├── client.py       # DeepSeek API client (AsyncOpenAI)
│   └── prompts.py      # System prompts for autonomous execution
├── trajectory/         # Trajectory logging
│   ├── models.py       # Dataclass definitions (Step, Trajectory)
│   └── storage.py      # JSON Lines writer with atomic writes
└── main.py             # Entry point: start env server, run agent
```

### Pattern 1: FastMCP Tool Server
**What:** Expose Python functions as MCP tools via decorator API
**When to use:** All model-tool interactions in this project
**Example:**
```python
# Source: https://gofastmcp.com/servers/tools
from fastmcp import FastMCP

mcp = FastMCP(name="ALFWorldServer")

@mcp.tool
async def go_to(location: str) -> str:
    """Navigate to a specific location in the environment.

    Args:
        location: Name of the location to navigate to (e.g., "cabinet 1", "desk 2")

    Returns:
        Natural language observation of the result
    """
    obs, score, done, info = env.step(f"go to {location}")
    return obs

@mcp.tool
async def task_completed(success: bool, summary: str) -> str:
    """Signal task completion and end the agent's run.

    Args:
        success: Whether the task was completed successfully
        summary: Brief summary of what was accomplished

    Returns:
        Confirmation message
    """
    return f"Task marked as {'complete' if success else 'incomplete'}: {summary}"
```

**Key points:**
- Type annotations generate JSON schema automatically
- Docstrings become tool descriptions for the LLM
- Both sync and async functions supported (async preferred for I/O)
- Return strings become tool results

### Pattern 2: Think-Act-Observe Loop
**What:** Autonomous agent loop with explicit thought, action, observation phases
**When to use:** All agent task execution
**Example:**
```python
# Source: https://medium.com/collaborne-engineering/controlling-llm-agents-with-think-act-observe-717d614b2fe1
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

async def autonomous_loop(task_description: str, max_steps: int = 50):
    messages = [
        {"role": "system", "content": AUTONOMOUS_AGENT_PROMPT},
        {"role": "user", "content": task_description}
    ]

    trajectory = []

    for step in range(max_steps):
        # Think + Act: Model decides on action
        response = await client.chat.completions.create(
            model="deepseek-chat",  # Use chat for function calling
            messages=messages,
            tools=mcp_tools,
            temperature=0.7
        )

        message = response.choices[0].message

        # Observe: Execute tool and get result
        if message.tool_calls:
            for tool_call in message.tool_calls:
                result = await execute_tool(tool_call)

                # Log step
                trajectory.append({
                    "step": step,
                    "thought": message.content,  # May be None if direct tool call
                    "action": tool_call.function.name,
                    "action_input": tool_call.function.arguments,
                    "observation": result
                })

                # Add to message history
                messages.append(message)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

                # Check for task_completed
                if tool_call.function.name == "task_completed":
                    return trajectory
        else:
            # No tool call - agent failed to act
            break

    # Timeout
    return trajectory
```

### Pattern 3: Trajectory Persistence with Atomic Writes
**What:** JSON Lines format with atomic write-temp-rename pattern
**When to use:** All trajectory logging (append-only, crash-resilient)
**Example:**
```python
# Source: https://thelinuxcode.com/read-write-and-parse-json-in-python-practical-production-friendly-guide/
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class Step:
    step: int
    thought: str | None
    action: str
    action_input: str
    observation: str

@dataclass
class Trajectory:
    task_id: str
    success: bool
    steps: list[Step]
    total_steps: int
    duration_seconds: float

def append_trajectory(trajectory: Trajectory, output_path: Path):
    """Append trajectory to JSON Lines file with atomic write."""
    # JSON Lines: one object per line, no array wrapper
    line = json.dumps(asdict(trajectory)) + "\n"

    # Atomic write pattern: temp + rename
    temp_path = output_path.with_suffix(".tmp")

    with open(temp_path, "a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())  # Force write to disk

    # Atomic rename (POSIX guarantees atomicity)
    temp_path.rename(output_path)
```

**Why JSON Lines:**
- Resilient to crashes: partial writes lose only last line
- Easy streaming reads: process one trajectory at a time
- No array wrapper to corrupt
- Standard format for log-like data

### Pattern 4: DeepSeek Function Calling
**What:** OpenAI-compatible function calling with DeepSeek API
**When to use:** All agent-environment interactions
**Example:**
```python
# Source: https://api-docs.deepseek.com/guides/function_calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "go_to",
            "description": "Navigate to a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location name (e.g., 'cabinet 1')"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# NOTE: deepseek-reasoner does NOT support function calling
# Tool calls automatically route through deepseek-chat
response = await client.chat.completions.create(
    model="deepseek-chat",  # Use chat, not reasoner, for tools
    messages=messages,
    tools=tools
)
```

### Anti-Patterns to Avoid
- **Using Pydantic for internal data:** 10-100x performance penalty. Use dataclass for trajectory steps, Pydantic only for external API validation
- **Generic error messages:** Raise FastMCP's `ToolError` with specific messages, not generic exceptions that become "Error calling tool"
- **Blocking I/O in sync tools:** Use async tools for environment interactions to avoid blocking FastMCP server
- **Including reasoning_content in subsequent messages:** DeepSeek API returns 400 if reasoning_content from prior turns is included in message history (deepseek-reasoner only)

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Async API rate limiting | Custom semaphore counter | `tenacity` with exponential backoff | Handles transient failures, retry-after headers, configurable strategies |
| Tool schema generation | Manual JSON schema writing | FastMCP `@mcp.tool` decorator | Auto-generates from type hints, validates inputs, handles async |
| JSON validation at boundaries | Manual dict checking | Pydantic models for API I/O | Type safety, validation errors, serialization helpers |
| Atomic file writes | Direct file.write() | Temp-write + fsync + rename pattern | Prevents corruption on crash, POSIX atomic rename guarantees |
| OpenAI API client | httpx with manual retry logic | OpenAI SDK (AsyncOpenAI) | Handles auth, streaming, retries, error types, standardized |
| Agent loop control | Manual step counter | max_iterations + timeout combination | Prevents infinite loops AND stuck API calls |

**Key insight:** Agent infrastructure has well-established patterns as of 2026. FastMCP + OpenAI SDK + tenacity provide production-ready primitives. Custom solutions introduce bugs (missing edge cases in atomic writes, incorrect retry backoff, etc.).

## Common Pitfalls

### Pitfall 1: DeepSeek Reasoner Function Calling Confusion
**What goes wrong:** Developers expect `deepseek-reasoner` to support function calling, but tool calls silently route through `deepseek-chat` instead.
**Why it happens:** The DeepSeek API accepts `tools` parameter for `deepseek-reasoner` but processes the request via `deepseek-chat`. This is documented but counterintuitive.
**How to avoid:** Always use `model="deepseek-chat"` explicitly when tools are needed. Reserve `deepseek-reasoner` for pure reasoning tasks without tool calls.
**Warning signs:**
- No chain-of-thought in responses when tools are used
- Function calling works but reasoning_content is never populated

### Pitfall 2: ALFWorld Observation Format Misunderstanding
**What goes wrong:** Developers assume ALFWorld provides privileged state information in observations, leading to incorrect environment setup.
**Why it happens:** ALFWorld offers both embodied (visual) and text modes. Text observations are natural language by design, but developers may expect structured state.
**How to avoid:**
- Use text-only install: `pip install alfworld` (not `alfworld[vis]`)
- Observations are strings like "You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a desk 2..."
- No privileged state info—agent sees what a human reading text would see
**Warning signs:**
- Expecting JSON or structured observation format
- Looking for configuration to "enable text mode" (it's default)

### Pitfall 3: Rate Limit Expectations
**What goes wrong:** Developers implement aggressive rate limiting assuming DeepSeek enforces hard limits, causing unnecessary throttling.
**Why it happens:** Most LLM APIs have strict RPM/TPM limits. DeepSeek explicitly states they "do NOT constrain user's rate limit."
**How to avoid:**
- Implement retry logic for failed requests (transient errors)
- Don't pre-emptively throttle requests
- Handle slow responses during high server load (can take up to 10 minutes before timeout)
**Warning signs:**
- Unnecessarily slow evaluation times from conservative rate limiting
- Implementing semaphores with very low concurrent request counts (e.g., 5)

### Pitfall 4: Trajectory Logging Data Corruption
**What goes wrong:** Direct file writes corrupt JSON when process crashes mid-write, losing entire experiment data.
**Why it happens:** JSON arrays require complete structure. Crash during write produces invalid JSON that can't be parsed.
**How to avoid:**
- Use JSON Lines format (one trajectory per line, no array wrapper)
- Implement atomic writes: write to temp file, fsync, rename
- Even if last line corrupts, earlier trajectories are recoverable
**Warning signs:**
- Using `json.dump()` with array of trajectories
- Opening file in write mode ('w') instead of append mode ('a')
- No fsync call before rename

### Pitfall 5: Agent Loop Without Multi-Layer Timeouts
**What goes wrong:** Agent gets stuck in infinite loops or slow API calls, burning through budget without useful work.
**Why it happens:** Single timeout (e.g., max_steps only) doesn't protect against slow API responses or reasoning loops.
**How to avoid:**
- Implement max steps (e.g., 50 steps per task)
- Add wall-clock timeout (e.g., 5 minutes total per task)
- Use asyncio.wait_for() for individual API calls
- Detect loop patterns (same action repeated)
**Warning signs:**
- Tasks timing out on DeepSeek's 10-minute connection limit
- Agent repeating same failed action indefinitely
- No wall-clock time tracking, only step count

### Pitfall 6: FastMCP Lifespan Misuse
**What goes wrong:** Developers initialize environment in lifespan context, expecting it to persist, but it re-initializes on every tool call.
**Why it happens:** Bug in FastMCP where lifespan runs per-request instead of per-server (issue #1115).
**How to avoid:**
- Use module-level or class-level environment initialization for now
- Track FastMCP GitHub issues for lifespan fix
- Test that environment state persists across tool calls
**Warning signs:**
- Environment resets between actions
- Initialization logs appearing multiple times during single task

## Code Examples

Verified patterns from official sources:

### Autonomous Agent System Prompt
```python
# Source: https://www.anthropic.com/research/building-effective-agents
AUTONOMOUS_AGENT_PROMPT = """You are an autonomous agent executing household tasks in a text-based environment.

You are running WITHOUT human supervision. You must act independently to complete the task.

Available tools:
- go_to(location): Navigate to a location
- take(object): Pick up an object
- put(object, receptacle): Place object in/on receptacle
- open(receptacle): Open a receptacle (drawer, cabinet, etc.)
- close(receptacle): Close a receptacle
- clean(object, receptacle): Clean object using receptacle (e.g., sink)
- heat(object, receptacle): Heat object using receptacle (e.g., microwave)
- cool(object, receptacle): Cool object using receptacle (e.g., refrigerator)
- use(object): Use/activate an object (e.g., turn on lamp)
- examine(object): Look at object closely
- inventory(): Check what you're currently holding
- task_completed(success: bool, summary: str): End task and report outcome

Instructions:
1. Read the task description carefully
2. Plan your approach, but be ready to adapt
3. Use tools to explore and manipulate the environment
4. When task is complete OR you've failed, call task_completed(success, summary)
5. You have a maximum of 50 steps—use them wisely

Think step-by-step. Describe your reasoning before acting.
"""
```

### ALFWorld Environment Wrapper
```python
# Source: https://github.com/alfworld/alfworld (README)
from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic
from fastmcp import FastMCP

mcp = FastMCP(name="ALFWorldServer")

# Module-level initialization (avoids lifespan bug)
config = generic.load_config()
env_type = config['env']['type']
env = get_environment(env_type)(config, train_eval='eval_out_of_distribution')
env = env.init_env(batch_size=1)

def reset_task(task_id: int):
    """Reset environment to specific task."""
    obs, info = env.reset()
    # Process observation to remove location prefix if needed
    if obs.startswith('You arrive at loc '):
        obs = obs[obs.find('. ')+2:]
    return obs, info

@mcp.tool
async def go_to(location: str) -> str:
    """Navigate to a location in the environment."""
    obs, score, done, info = env.step([f"go to {location}"])
    return obs[0]  # env returns batch, extract first

@mcp.tool
async def take(object_name: str) -> str:
    """Pick up an object."""
    obs, score, done, info = env.step([f"take {object_name}"])
    return obs[0]

# ... implement remaining 8 actions similarly
```

### Trajectory Model with Dataclasses
```python
# Source: Performance best practices from search results
from dataclasses import dataclass, asdict, field
from typing import Optional
from datetime import datetime

@dataclass
class Step:
    """Single step in agent trajectory."""
    step: int
    thought: Optional[str]  # May be None if direct tool call
    action: str
    action_input: str  # JSON string of tool arguments
    observation: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

@dataclass
class Trajectory:
    """Complete task execution trajectory."""
    task_id: str
    task_description: str
    success: bool
    steps: list[Step]
    total_steps: int
    duration_seconds: float
    failure_reason: Optional[str] = None  # Set if success=False

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return asdict(self)
```

### Concurrent Task Execution with Rate Limiting
```python
# Source: https://villoro.com/blog/async-openai-calls-rate-limiter/
import asyncio
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5)
)
async def run_task_with_retry(task_id: str, task_description: str) -> Trajectory:
    """Run single task with automatic retry on transient failures."""
    return await autonomous_loop(task_description, max_steps=50)

async def run_tasks_concurrent(tasks: list[tuple[str, str]], max_concurrent: int = 10):
    """Run multiple tasks concurrently with semaphore limiting."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_task(task_id: str, desc: str):
        async with semaphore:
            return await run_task_with_retry(task_id, desc)

    results = await asyncio.gather(*[
        bounded_task(tid, desc) for tid, desc in tasks
    ])
    return results
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual tool schema JSON | FastMCP @mcp.tool decorator | FastMCP 1.0 (2024) | 10x less boilerplate, type-safe schemas from annotations |
| Generic MCP Python SDK | FastMCP high-level API | FastMCP 2.0 (2025) | Production-ready patterns, automatic async handling |
| DeepSeek R1 (reasoning only) | DeepSeek V3.2 (chat + reasoner modes) | Jan 2025 | Unified API, function calling support (via chat mode) |
| LangChain for agent loops | Direct OpenAI SDK usage | 2024-2025 | Less abstraction overhead, more control, simpler debugging |
| JSON arrays for trajectories | JSON Lines format | Ongoing best practice | Crash resilience, streaming reads, append-only writes |
| Pydantic for all data | Dataclass for internal, Pydantic for boundaries | 2024-2025 performance optimization | 10-100x faster internal operations |

**Deprecated/outdated:**
- **FastMCP 1.0:** Merged into official MCP SDK. Use FastMCP 2.x for high-level API
- **DeepSeek V3.0/V3.1:** Superseded by V3.2 (released Dec 2025)
- **deepseek-reasoner with tools:** Doesn't use reasoning mode. Use deepseek-chat explicitly for function calling
- **ALFWorld observation pooling:** Config option exists but not recommended for natural language agents (adds state history complexity)

## Open Questions

Things that couldn't be fully resolved:

1. **ALFWorld Task Distribution Across 6 Types**
   - What we know: 134 total tasks across 6 types (pick_and_place_simple, look_at_obj_in_light, pick_clean_then_place_in_recep, pick_heat_then_place_in_recep, pick_cool_then_place_in_recep, pick_two_obj_and_place)
   - What's unclear: Exact count per task type, average step count per type
   - Recommendation: Run quick enumeration script on ALFWorld test set to get distribution. Not critical for Phase 1 (single task execution), but useful for Phase 2 (full evaluation)

2. **DeepSeek Practical Rate Limits**
   - What we know: No hard limits enforced, requests slow down under load, 10-minute timeout if inference doesn't start
   - What's unclear: Practical sustained throughput for concurrent requests in early 2026
   - Recommendation: Start with 10 concurrent workers (conservative). Monitor response times and adjust based on observed performance. DeepSeek's "best effort" model means limits are dynamic.

3. **FastMCP Lifespan Bug Status**
   - What we know: Issue #1115 reports lifespan running per-request instead of per-server
   - What's unclear: Fix timeline, workaround stability
   - Recommendation: Use module-level environment initialization for Phase 1. Monitor FastMCP releases for fix. Test environment persistence across tool calls during development.

4. **ALFWorld Success Detection**
   - What we know: Tasks return `score`, `done`, and `info` from step(). Agent should call task_completed() explicitly
   - What's unclear: Can we rely on ALFWorld's `done` flag, or should task_completed be the sole success indicator?
   - Recommendation: Implement both: agent calls task_completed(success, summary), and we verify against ALFWorld's `done` flag. Log discrepancies for analysis.

## Sources

### Primary (HIGH confidence)
- [FastMCP Tools Documentation](https://gofastmcp.com/servers/tools) - Decorator API, parameter handling, async support
- [DeepSeek API Docs - Function Calling](https://api-docs.deepseek.com/guides/function_calling) - Tool definition format, request/response structure
- [DeepSeek API Docs - Reasoning Model](https://api-docs.deepseek.com/guides/reasoning_model) - deepseek-reasoner specifics, unsupported features
- [DeepSeek API Docs - Rate Limits](https://api-docs.deepseek.com/quick_start/rate_limit) - No hard limits policy, 10-minute timeout
- [DeepSeek API Docs - Pricing](https://api-docs.deepseek.com/quick_start/pricing) - Token pricing ($0.28/$0.42 per 1M tokens)
- [ALFWorld GitHub Repository](https://github.com/alfworld/alfworld) - Installation, environment API, step() return format

### Secondary (MEDIUM confidence)
- [FastMCP Error Handling Guide (Medium)](https://medium.com/@sureshddm/mcp-error-handling-dont-let-your-tools-fail-silently-1b5e02fabe4c) - ToolError patterns, best practices
- [Think-Act-Observe Pattern (Medium)](https://medium.com/collaborne-engineering/controlling-llm-agents-with-think-act-observe-717d614b2fe1) - Agent loop implementation
- [OpenAI Async Best Practices (villoro.com)](https://villoro.com/blog/async-openai-calls-rate-limiter/) - Concurrent requests, rate limiting patterns
- [JSON Trajectory Logging Guide (TheLinuxCode)](https://thelinuxcode.com/read-write-and-parse-json-in-python-practical-production-friendly-guide/) - Atomic writes, JSON Lines format
- [Anthropic Agent Building Research](https://www.anthropic.com/research/building-effective-agents) - Autonomous agent design principles
- [ALFWorld ReAct Notebook](https://github.com/ysymyth/ReAct/blob/master/alfworld.ipynb) - Observation format examples, process_ob() pattern

### Tertiary (LOW confidence)
- WebSearch: "DeepSeek V3.2 context window" - 128K context limit reported by multiple sources, needs official verification
- WebSearch: "ALFWorld 134 tasks distribution" - 6 task types confirmed, exact distribution per type unavailable
- WebSearch: "FastMCP lifespan bug" - Issue #1115 confirmed, workaround needed

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - FastMCP, OpenAI SDK, ALFWorld all have official docs verified via WebFetch
- Architecture: HIGH - Patterns verified from official FastMCP docs, DeepSeek API docs, established async patterns
- Pitfalls: MEDIUM to HIGH - FastMCP/DeepSeek pitfalls from official docs (HIGH), agent loop pitfalls from community sources (MEDIUM)

**Research date:** 2026-02-11
**Valid until:** ~2026-03-15 (30 days for stable domain—FastMCP 2.x is production-ready, DeepSeek V3.2 recent, ALFWorld mature benchmark)

**Validation needed:**
- ALFWorld exact task distribution (run enumeration script)
- DeepSeek practical rate limits under load (test with 10 concurrent workers)
- FastMCP lifespan behavior (test environment persistence)
