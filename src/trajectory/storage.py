"""JSONL storage for trajectories with crash-resilient atomic writes."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .models import Trajectory, Step


def append_trajectory(trajectory: Trajectory, output_path: Path) -> None:
    """Append a trajectory to a JSONL file with atomic write.

    Uses atomic write pattern: write to temp file, flush, fsync, then replace.
    This ensures crash resilience - either the write completes or it doesn't.

    Args:
        trajectory: Trajectory to append
        output_path: Path to JSONL file
    """
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing content if file exists
    existing_lines = []
    if output_path.exists():
        with open(output_path, 'r') as f:
            existing_lines = f.readlines()

    # Serialize trajectory to JSON
    traj_dict = trajectory.to_dict()
    json_line = json.dumps(traj_dict) + '\n'

    # Atomic write: temp file + fsync + replace
    temp_path = output_path.with_suffix('.tmp')
    with open(temp_path, 'w') as f:
        # Write existing content + new line
        for line in existing_lines:
            f.write(line)
        f.write(json_line)
        f.flush()
        os.fsync(f.fileno())

    # Atomic replace (os.replace is atomic on all platforms)
    os.replace(temp_path, output_path)


def load_trajectories(input_path: Path) -> list[Trajectory]:
    """Load trajectories from a JSONL file.

    Args:
        input_path: Path to JSONL file

    Returns:
        List of Trajectory objects (empty list if file doesn't exist)
    """
    if not input_path.exists():
        return []

    trajectories = []
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip blank lines
                continue

            # Deserialize JSON to dict
            traj_dict = json.loads(line)

            # Reconstruct Step objects from dicts
            steps_data = traj_dict.pop('steps')
            steps = [Step(**step_dict) for step_dict in steps_data]

            # Reconstruct Trajectory with steps
            trajectory = Trajectory(**traj_dict, steps=steps)
            trajectories.append(trajectory)

    return trajectories
