"""Trajectory data models for capturing agent interactions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict


@dataclass
class Step:
    """A single step in an agent trajectory.

    Captures the thought-action-observation cycle at each step.
    """
    step: int  # 0-indexed step number
    thought: str | None  # Model's reasoning (None if direct tool call)
    action: str  # Tool name called
    action_input: str  # JSON string of tool arguments
    observation: str  # Tool result / environment response
    timestamp: float = field(default_factory=time.time)  # Epoch seconds


@dataclass
class Trajectory:
    """Complete trajectory of an agent attempting a task.

    Includes all steps taken, task metadata, and outcome.
    """
    task_id: str  # ALFWorld task identifier
    task_description: str  # Natural language task prompt
    task_type: str  # One of: pick, look, clean, heat, cool, pick2
    success: bool  # Whether task was completed successfully
    steps: list[Step]  # All steps taken
    total_steps: int  # Length of steps
    duration_seconds: float  # Wall clock time
    failure_reason: str | None  # Set if success=False: "timeout", "no_tool_call", "agent_declared_failure"
    env_done: bool  # ALFWorld's own done flag (for discrepancy detection)

    def to_dict(self) -> dict:
        """Convert trajectory to dictionary for serialization."""
        return asdict(self)
