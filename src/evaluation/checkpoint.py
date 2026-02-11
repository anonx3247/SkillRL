"""Checkpoint management for evaluation iterations."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class IterationCheckpoint:
    """Checkpoint data for an evaluation iteration.

    Attributes:
        iteration: Iteration number
        total_tasks: Total number of tasks evaluated
        successful_tasks: Number of successful tasks
        overall_success_rate: Overall success rate
        per_subtask_success_rate: Success rate per task type
        avg_steps_success: Average steps for successful tasks
        avg_steps_failure: Average steps for failed tasks
        skill_library_path: Path to skill library used
        trajectories_path: Path to trajectories file
        timestamp: Unix timestamp when checkpoint was created
    """
    iteration: int
    total_tasks: int
    successful_tasks: int
    overall_success_rate: float
    per_subtask_success_rate: dict[str, float]
    avg_steps_success: float
    avg_steps_failure: float
    skill_library_path: str
    trajectories_path: str
    timestamp: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> IterationCheckpoint:
        """Create checkpoint from dictionary."""
        return cls(**data)


class CheckpointManager:
    """Manages iteration checkpoint save/load with atomic writes.

    Uses temp file + os.replace for atomic writes, symlink for latest.
    """

    def __init__(self, checkpoint_dir: Path):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, checkpoint: IterationCheckpoint) -> None:
        """Save checkpoint atomically with symlink update.

        Args:
            checkpoint: Checkpoint to save
        """
        # Save to iteration-specific file
        checkpoint_path = self.checkpoint_dir / f"iteration_{checkpoint.iteration:03d}.json"

        # Atomic write using temp + replace
        temp_path = checkpoint_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, checkpoint_path)

        # Update symlink to latest
        symlink_path = self.checkpoint_dir / "checkpoint_latest.json"
        temp_symlink = self.checkpoint_dir / "checkpoint_latest.tmp"

        # Create/update symlink atomically
        if temp_symlink.exists():
            temp_symlink.unlink()
        temp_symlink.symlink_to(checkpoint_path.name)
        os.replace(temp_symlink, symlink_path)

    def load_latest(self) -> IterationCheckpoint | None:
        """Load most recent checkpoint via symlink.

        Returns:
            Latest checkpoint if exists, else None
        """
        symlink_path = self.checkpoint_dir / "checkpoint_latest.json"
        if not symlink_path.exists():
            return None

        with open(symlink_path, "r") as f:
            data = json.load(f)

        return IterationCheckpoint.from_dict(data)

    def load_iteration(self, iteration: int) -> IterationCheckpoint | None:
        """Load checkpoint for specific iteration.

        Args:
            iteration: Iteration number to load

        Returns:
            Checkpoint if exists, else None
        """
        checkpoint_path = self.checkpoint_dir / f"iteration_{iteration:03d}.json"
        if not checkpoint_path.exists():
            return None

        with open(checkpoint_path, "r") as f:
            data = json.load(f)

        return IterationCheckpoint.from_dict(data)
