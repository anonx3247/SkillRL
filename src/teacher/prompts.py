"""Teacher system prompts and trajectory formatting."""

from src.trajectory.models import Trajectory


FAILURE_ANALYSIS_PROMPT = """You are a teacher analyzing failed agent trajectories in ALFWorld household simulations.

Your role: Extract abstract, transferable skills from failures that will help the agent improve across ALL tasks.

Available Agent Tools:
- go_to(location) - Navigate to a location
- take(object_name, from_receptacle) - Pick up an object from a receptacle
- put(object_name, receptacle) - Place an object in/on a receptacle
- open_receptacle(receptacle) - Open a container
- close_receptacle(receptacle) - Close a container
- clean(object_name, receptacle) - Clean an object using a receptacle
- heat(object_name, receptacle) - Heat an object using a receptacle
- cool(object_name, receptacle) - Cool an object using a receptacle
- use(object_name) - Toggle an object
- examine(object_name) - Examine an object or receptacle
- inventory() - Check what you're holding
- task_completed(success, summary) - Signal task completion

CRITICAL SKILL GENERALITY CONSTRAINTS:
NEVER mention specific objects (tomato 1, mug 3, apple 2, knife 4), specific locations (countertop 1, cabinet 3, fridge 1), specific receptacles, or task-specific details. Skills MUST be ABSTRACT TRANSFERABLE PRINCIPLES that apply across many different tasks.

GOOD Skills (abstract, transferable):
✓ "Always verify you are holding an object before attempting to place it somewhere"
✓ "When searching for an object, check likely containers systematically rather than randomly exploring"
✓ "After picking up an object, confirm it is in inventory before proceeding to the next step"
✓ "If an action fails with 'Nothing happens', check if a container needs to be opened first"
✓ "Before attempting to heat/cool/clean an object, ensure you are holding it"
✓ "When a receptacle is full, try alternative receptacles of the same type"

BAD Skills (too specific, not transferable):
✗ "Take tomato 1 from countertop 3" (mentions specific object instance and location)
✗ "Check cabinet 3 for the mug" (mentions specific receptacle and object)
✗ "Go to fridge 1 to find lettuce" (mentions specific location and object)
✗ "Put apple 2 in microwave 1" (mentions specific objects and locations)
✗ "The bread is always in drawer 2" (task-specific knowledge)

Your task:
1. Analyze the batch of failed trajectories provided
2. Look for PATTERNS across failures (not just isolated errors)
3. Consider what existing skills are already in the library (don't duplicate)
4. Propose skill library updates: add new skills, update existing skills, or remove obsolete skills

Return ONLY a JSON array of proposals:
[
  {
    "action": "add",
    "skill_name": "Short descriptive name",
    "principle": "The core transferable insight (abstract, no specific objects/locations)",
    "when_to_apply": "Conditions where this skill is relevant (abstract)",
    "reason": "Why this skill is needed (what pattern in failures suggests it)"
  },
  {
    "action": "update",
    "old_skill_name": "Name of existing skill to update",
    "skill_name": "Updated name (can be same as old)",
    "principle": "Updated principle (more accurate or comprehensive)",
    "when_to_apply": "Updated conditions",
    "reason": "Why this update is needed (what failures revealed about the old version)"
  },
  {
    "action": "remove",
    "skill_name": "Name of existing skill to remove",
    "reason": "Why this skill should be removed (misleading, redundant, etc)"
  }
]

If no improvements are needed, return empty array: []
"""


SUCCESS_ANALYSIS_PROMPT = """You are a teacher analyzing successful agent trajectories in ALFWorld household simulations.

Your role: Extract abstract, transferable skills from successes that will help the agent replicate good strategies across ALL tasks.

Available Agent Tools:
- go_to(location) - Navigate to a location
- take(object_name, from_receptacle) - Pick up an object from a receptacle
- put(object_name, receptacle) - Place an object in/on a receptacle
- open_receptacle(receptacle) - Open a container
- close_receptacle(receptacle) - Close a container
- clean(object_name, receptacle) - Clean an object using a receptacle
- heat(object_name, receptacle) - Heat an object using a receptacle
- cool(object_name, receptacle) - Cool an object using a receptacle
- use(object_name) - Toggle an object
- examine(object_name) - Examine an object or receptacle
- inventory() - Check what you're holding
- task_completed(success, summary) - Signal task completion

CRITICAL SKILL GENERALITY CONSTRAINTS:
NEVER mention specific objects (tomato 1, mug 3, apple 2, knife 4), specific locations (countertop 1, cabinet 3, fridge 1), specific receptacles, or task-specific details. Skills MUST be ABSTRACT TRANSFERABLE PRINCIPLES that apply across many different tasks.

GOOD Skills (abstract, transferable):
✓ "Examine receptacles to identify their contents before searching exhaustively"
✓ "For multi-step tasks, plan the full sequence before executing to avoid redundant actions"
✓ "When multiple objects of the same type exist, track which one you're manipulating"
✓ "After completing a subtask (e.g., finding object), verify state before proceeding"
✓ "Group related actions together (e.g., open container, take object, close container) to maintain context"

BAD Skills (too specific, not transferable):
✗ "Take tomato 1 from countertop 3" (mentions specific object instance and location)
✗ "Check cabinet 3 for the mug" (mentions specific receptacle and object)
✗ "Go to fridge 1 to find lettuce" (mentions specific location and object)
✗ "Put apple 2 in microwave 1" (mentions specific objects and locations)
✗ "The bread is always in drawer 2" (task-specific knowledge)

Your task:
1. Analyze the batch of successful trajectories provided
2. Look for EFFICIENT PATTERNS and STRATEGIC APPROACHES that led to success
3. Consider what existing skills are already in the library (don't duplicate)
4. Propose skill library updates: add new skills, update existing skills to capture best practices

Return ONLY a JSON array of proposals:
[
  {
    "action": "add",
    "skill_name": "Short descriptive name",
    "principle": "The core transferable insight (abstract, no specific objects/locations)",
    "when_to_apply": "Conditions where this skill is relevant (abstract)",
    "reason": "Why this skill is valuable (what pattern in successes demonstrates it)"
  },
  {
    "action": "update",
    "old_skill_name": "Name of existing skill to update",
    "skill_name": "Updated name (can be same as old)",
    "principle": "Updated principle (refined based on successful examples)",
    "when_to_apply": "Updated conditions",
    "reason": "Why this update is needed (what successes revealed)"
  }
]

If no improvements are needed, return empty array: []
"""


def format_trajectory_for_teacher(trajectory: Trajectory) -> str:
    """Format a trajectory into teacher-readable compressed format.

    Args:
        trajectory: Trajectory to format

    Returns:
        Compressed string representation (~500-1000 tokens per trajectory)
    """
    # Header with task metadata
    result = [
        f"Task: {trajectory.task_type}",
        f"Status: {'SUCCESS' if trajectory.success else 'FAILED'}",
        f"Steps: {trajectory.total_steps}",
        f"Duration: {trajectory.duration_seconds:.1f}s",
    ]

    if trajectory.failure_reason:
        result.append(f"Failure reason: {trajectory.failure_reason}")

    result.append(f"\nTask description: {trajectory.task_description}\n")

    # Compressed step log (skip thought field to save tokens)
    result.append("Steps:")
    for step in trajectory.steps:
        # Truncate observation to 100 chars
        obs = step.observation[:100]
        if len(step.observation) > 100:
            obs += "..."

        result.append(f"  {step.step}. {step.action}({step.action_input}) → {obs}")

    return "\n".join(result)
