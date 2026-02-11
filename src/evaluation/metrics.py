"""Metrics computation for evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

from src.trajectory.models import Trajectory


@dataclass
class TaskMetrics:
    """Metrics for a single task evaluation.

    Attributes:
        task_id: Task identifier
        task_type: Task type (pick, look, clean, heat, cool, pick2)
        success: Whether task completed successfully
        step_count: Number of steps taken
        skills_retrieved: Number of skills retrieved for this task
    """
    task_id: str
    task_type: str
    success: bool
    step_count: int
    skills_retrieved: int


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all tasks in an iteration.

    Attributes:
        iteration: Iteration number
        overall_success_rate: Success rate across all tasks
        per_subtask_success_rate: Success rate per task type
        avg_steps_success: Average steps for successful tasks
        avg_steps_failure: Average steps for failed tasks
        total_tasks: Total number of tasks evaluated
        successful_tasks: Number of successful tasks
    """
    iteration: int
    overall_success_rate: float
    per_subtask_success_rate: dict[str, float]
    avg_steps_success: float
    avg_steps_failure: float
    total_tasks: int
    successful_tasks: int


def compute_metrics(
    trajectories: list[Trajectory],
    iteration: int,
    skills_per_task: dict[str, int] | None = None,
) -> AggregateMetrics:
    """Compute aggregate metrics from trajectories.

    Args:
        trajectories: List of trajectories from evaluation
        iteration: Iteration number
        skills_per_task: Optional dict mapping task_id to number of skills retrieved

    Returns:
        AggregateMetrics with computed statistics
    """
    if not trajectories:
        return AggregateMetrics(
            iteration=iteration,
            overall_success_rate=0.0,
            per_subtask_success_rate={},
            avg_steps_success=0.0,
            avg_steps_failure=0.0,
            total_tasks=0,
            successful_tasks=0,
        )

    skills_per_task = skills_per_task or {}

    # Compute per-task metrics
    task_metrics: list[TaskMetrics] = []
    for traj in trajectories:
        task_metrics.append(TaskMetrics(
            task_id=traj.task_id,
            task_type=traj.task_type,
            success=traj.success,
            step_count=traj.total_steps,
            skills_retrieved=skills_per_task.get(traj.task_id, 0),
        ))

    # Aggregate statistics
    total_tasks = len(task_metrics)
    successful_tasks = sum(1 for tm in task_metrics if tm.success)
    overall_success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0

    # Per-subtask success rate
    subtask_counts: dict[str, int] = defaultdict(int)
    subtask_successes: dict[str, int] = defaultdict(int)
    for tm in task_metrics:
        subtask_counts[tm.task_type] += 1
        if tm.success:
            subtask_successes[tm.task_type] += 1

    per_subtask_success_rate = {
        task_type: subtask_successes[task_type] / subtask_counts[task_type]
        for task_type in subtask_counts
    }

    # Average steps
    success_steps = [tm.step_count for tm in task_metrics if tm.success]
    failure_steps = [tm.step_count for tm in task_metrics if not tm.success]

    avg_steps_success = sum(success_steps) / len(success_steps) if success_steps else 0.0
    avg_steps_failure = sum(failure_steps) / len(failure_steps) if failure_steps else 0.0

    return AggregateMetrics(
        iteration=iteration,
        overall_success_rate=overall_success_rate,
        per_subtask_success_rate=per_subtask_success_rate,
        avg_steps_success=avg_steps_success,
        avg_steps_failure=avg_steps_failure,
        total_tasks=total_tasks,
        successful_tasks=successful_tasks,
    )
