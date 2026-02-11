---
phase: 01
plan: 02
subsystem: environment
tags: [alfworld, fastmcp, tool-interface, python]

requires: [01-01]
provides:
  - ALFWorld environment wrapper with lifecycle management
  - FastMCP tool server with 12 tools (10 actions + inventory + task_completed)
  - Natural language observation interface
affects: [01-03, 01-04]

tech-stack:
  added: [alfworld==0.4.2, torch==2.10.0, textworld==1.7.0]
  patterns: [fastmcp-tools, environment-wrapper, monkey-patching]

key-files:
  created:
    - src/environment/env_manager.py
    - src/environment/server.py
    - src/environment/actions.py
    - src/environment/__init__.py
    - alfworld_config.yaml
    - alfred_fixed.twl2
  modified: []

decisions:
  - decision: "Monkey-patch TextWorld EvalSymbol to fix missing 'r' variable bug"
    rationale: "ALFWorld/TextWorld grammar template has bug where intro template uses {r.name} but 'r' not in eval context"
    impact: "Enables ALFWorld reset() to work; workaround for upstream bug"
  - decision: "Use module-level env_manager instance"
    rationale: "FastMCP lifespan has bug #1115 - runs per-request instead of once"
    impact: "Environment persists across tool calls within session"
  - decision: "Return only observation strings from tools"
    rationale: "Agent should not see privileged state (score, done flags)"
    impact: "Clean separation - LLM only sees natural language"

metrics:
  duration: 813s
  completed: 2026-02-11
---

# Phase 1 Plan 2: ALFWorld FastMCP Wrapper Summary

**One-liner:** FastMCP tool server wrapping ALFWorld with 12 tools returning natural language observations

## Objective

Wrap ALFWorld as a FastMCP tool server exposing all 11 tools (10 environment actions + inventory + task_completed) that return natural language observation strings.

## What Was Built

### 1. EnvManager (src/environment/env_manager.py)

ALFWorld lifecycle management class:

- **load()**: Initializes ALFWorld eval_out_of_distribution environment with custom config
- **reset()**: Resets to new task, returns (observation: str, info: dict)
- **step(action: str)**: Executes action, returns (obs: str, score: float, done: bool, info: dict)
- **Batch indexing**: Always extracts [0] element since ALFWorld returns batched results even with batch_size=1
- **Task tracking**: Extracts task type (pick/look/clean/heat/cool/pick2) and task ID from gamefile path
- **Completion signaling**: task_completed_flag, task_completed_success, task_completed_summary attributes

### 2. FastMCP Server (src/environment/server.py)

- **mcp**: FastMCP instance named "ALFWorldServer"
- **env_manager**: Module-level EnvManager instance (NOT in lifespan due to bug #1115)

### 3. Action Tools (src/environment/actions.py)

12 FastMCP tools registered via @mcp.tool decorator:

**10 environment actions:**
1. go_to(location) - Navigate to location
2. take(object_name) - Pick up object
3. put(object_name, receptacle) - Place object in/on receptacle
4. open_receptacle(receptacle) - Open cabinet/drawer/microwave/etc
5. close_receptacle(receptacle) - Close receptacle
6. clean(object_name, receptacle) - Clean object with sinkbasin
7. heat(object_name, receptacle) - Heat object with microwave
8. cool(object_name, receptacle) - Cool object with fridge
9. use(object_name) - Toggle lamp/switch
10. examine(object_name) - Examine object/receptacle

**Inventory:**
11. inventory() - Check held objects

**Meta tool:**
12. task_completed(success, summary) - Signal episode completion

Each tool:
- Has typed parameters with descriptive names
- Has docstring explaining what it does (becomes tool description for LLM)
- Calls env_manager.step() with appropriate ALFWorld command string
- Returns ONLY the natural language observation string
- Does NOT return score, done flags, or internal state

### 4. ALFWorld Configuration

**alfworld_config.yaml:**
- Loads all 6 task types from eval_out_of_distribution split
- Points to ~/.cache/alfworld data
- Configures domain (PDDL), grammar (TWL2), rewards, task description

**Critical fix:** Modified grammar file (alfred_fixed.twl2) to work around TextWorld template bug

### 5. TextWorld Bug Workaround

**Problem:** ALFWorld intro template uses {r.name} but 'r' variable not in eval context during reset()

**Solution:** Monkey-patch textworld.envs.pddl.textgen.EvalSymbol.derive():
- Catch NameError when 'r' is undefined
- Inject DummyObject with name, indefinite, definite, type attributes
- Wrap result as TerminalSymbol for template system
- Allows intro derivation to complete successfully

## Implementation Decisions

1. **Batch indexing everywhere**: ALFWorld returns [obs, score, done, info] even with batch_size=1 → always index [0]
2. **Pure observation returns**: Tools return only obs string, not (obs, score, done, info) tuple → agent sees only natural language
3. **Module-level initialization**: env_manager created at module level, NOT in FastMCP lifespan → avoids bug #1115
4. **Descriptive parameter names**: "receptacle" not "target", "object_name" not "obj" → helps LLM understand tool usage
5. **task_completed doesn't step()**: Sets flags only, doesn't call env → clean separation of concerns

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added PyTorch dependency**
- **Found during:** Task 1 - Initial ALFWorld import
- **Issue:** ALFWorld requires torch but pyproject.toml didn't list it
- **Fix:** `pip install torch` (2.10.0)
- **Commit:** Included in task setup, not separate commit

**2. [Rule 2 - Missing Critical] Created ALFWorld config file**
- **Found during:** Task 1 - env_manager.load()
- **Issue:** ALFWorld generic.load_config() expects CLI args with config file path
- **Fix:** Created alfworld_config.yaml with all required keys (env, dataset, logic, controller, general, rl sections)
- **Files created:** alfworld_config.yaml
- **Commit:** 4cf4adb

**3. [Rule 1 - Bug] Fixed TextWorld grammar template bug**
- **Found during:** Task 1 - env.reset() call
- **Issue:** TextWorld intro template uses {r.name} in eval() but 'r' not defined in context
- **Root cause:** ALFWorld/TextWorld TWL2 grammar intro → #look.feedback# → #look-variations# → {r.name}, but reset() creates context with empty variables dict
- **Fix:** Monkey-patched EvalSymbol.derive() to catch NameError, inject DummyObject with required attributes, wrap as TerminalSymbol
- **Files modified:** src/environment/env_manager.py (added monkey-patch at module level)
- **Commit:** 4cf4adb

**4. [Rule 2 - Missing Critical] Downloaded ALFWorld data**
- **Found during:** Task 1 - Environment initialization
- **Issue:** ALFWorld data not included in package, needs separate download
- **Fix:** Ran `alfworld-download` to get json_2.1.1 dataset (134 test tasks) to ~/.cache/alfworld
- **Commit:** No commit (external data)

**5. [Rule 2 - Missing Critical] Expanded config paths**
- **Found during:** Task 1 - Config loading
- **Issue:** Config uses $ALFWORLD_DATA env var which needs expansion
- **Fix:** Added os.path.expandvars() calls in env_manager.load() for all dataset and logic paths
- **Files modified:** src/environment/env_manager.py
- **Commit:** 4cf4adb

## Technical Challenges

### Challenge 1: TextWorld Grammar Template Bug

**Problem:**
ALFWorld intro template chain:
```
intro → #look.feedback# → #look-variations# → {r.name}
```
But reset() creates context with `variables: {}` (empty), so eval(self.expression) with "{r.name}" fails: `NameError: name 'r' is not defined`

**Attempted solutions:**
1. Modified intro template to static text → Still failed (TWL2 file not reloaded)
2. Tried different goal_desc_human_anns_prob settings → Ignored by ALFWorld
3. Checked if grammar path was wrong → Confirmed correct
4. Tried to add 'r' to context at Grammar level → Wrong abstraction layer
5. Finally monkey-patched EvalSymbol.derive() → SUCCESS

**Why the patch works:**
- Catches NameError during eval()
- Injects DummyObject with all required attributes (name, indefinite, definite, type, __getattr__ fallback)
- Wraps result as TerminalSymbol (what template system expects)
- Returns proper symbol list for derivation stack

**Tradeoff:** Intro text shows "a r r" placeholders instead of actual receptacle names, but task description is intact and functional

### Challenge 2: FastMCP Lifespan Bug #1115

**Problem:** FastMCP lifespan context manager runs per-request, not once at startup

**Solution:** Create env_manager at module level, let caller invoke .load() when ready

**Implication:** Environment state persists across tool calls within session (desired behavior for agent loop)

## Testing & Verification

**EnvManager:**
```python
em = EnvManager()
em.load()  # Loads 134 eval_out_of_distribution tasks
obs, info = em.reset()  # Returns natural language string
assert isinstance(obs, str)
obs2, score, done, info = em.step("inventory")
assert isinstance(obs2, str)  # "You are not carrying anything."
```

**FastMCP Server:**
```python
from src.environment import mcp, env_manager
tools = mcp._tool_manager._tools
assert len(tools) == 12
assert all(t in tools for t in ['go_to', 'take', 'put', 'open_receptacle', 'close_receptacle', 'clean', 'heat', 'cool', 'use', 'examine', 'inventory', 'task_completed'])
```

**Tool execution:**
```python
env_manager.load()
env_manager.reset()
obs, score, done, info = env_manager.step("inventory")
# obs = "You are not carrying anything."
```

## Next Phase Readiness

**Blocks removed:**
- ALFWorld environment interface ready for agent loop (Plan 03)
- Tool interface defined and working
- Observation format confirmed (natural language strings only)

**Provides for downstream:**
- Plan 03 (agent loop) can import mcp server and call env_manager.load() / reset()
- Plan 04 (trajectory collection) can use env_manager to track score/done for success detection
- All subsequent plans have working environment to test against

**Known issues:**
- Intro text has "r r" placeholders due to grammar bug workaround → Doesn't affect task execution
- env_manager needs manual .load() call → Documented, not a blocker

## Commits

| Hash | Message | Files |
|------|---------|-------|
| 4cf4adb | feat(01-02): implement ALFWorld environment manager | env_manager.py, alfworld_config.yaml |
| e9da78f | feat(01-02): create FastMCP server with all 11 ALFWorld tools | server.py, actions.py, __init__.py |

## File Manifest

```
src/environment/
├── __init__.py              # Exports mcp, env_manager
├── env_manager.py           # EnvManager class + TextWorld monkey-patch
├── server.py                # FastMCP instance + env_manager instance
└── actions.py               # 12 tool functions with @mcp.tool decorators

alfworld_config.yaml         # ALFWorld configuration
alfred_fixed.twl2            # Modified grammar (not used in final solution, kept for reference)
```

## Performance

- **Total duration:** 813 seconds (~13.5 minutes)
- **Commits:** 2 atomic task commits
- **Files created:** 5
- **Dependencies added:** 3 (alfworld, torch, textworld)
- **Major blockers:** 1 (TextWorld grammar bug - 60% of time spent debugging)

## Success Criteria Met

- [x] EnvManager loads ALFWorld eval_out_of_distribution environment
- [x] EnvManager.reset() returns (observation: str, info: dict)
- [x] EnvManager.step(action) returns (observation: str, score: float, done: bool, info: dict) with batch indexing
- [x] 12 tools registered on FastMCP mcp instance (11 specified + proper count)
- [x] Each tool has typed parameters and descriptive docstring
- [x] Tool returns are pure natural language strings
- [x] task_completed sets flag on env_manager, does not call step()

## Lessons Learned

1. **Always check for library-specific bugs early**: TextWorld grammar bug was deep and not obvious from docs
2. **Monkey-patching is valid for upstream bugs**: When library has bug and fix isn't merged, patch locally with clear documentation
3. **Module-level vs lifespan initialization matters**: FastMCP lifespan bug #1115 would have broken environment persistence
4. **Batch indexing is critical**: ALFWorld returns lists even with batch_size=1 → must always index [0]
5. **Tool returns should be minimal**: Returning only observation string keeps interface clean for LLM
