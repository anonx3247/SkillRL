"""W&B experiment tracker for skill evolution."""

from __future__ import annotations

import wandb

from src.evaluation.metrics import AggregateMetrics


class ExperimentTracker:
    """W&B experiment tracker for skill evolution.

    Handles all Weights & Biases logging for the SkillRL evolution loop:
    - Core metrics (success rate, avg steps, skill count) per iteration
    - Per-subtask success rate breakdowns
    - Skill library state (names table) per iteration
    - Teacher decision proposals per iteration
    - Final run summary statistics

    Usage:
        tracker = ExperimentTracker(project="skillrl-evolution")
        tracker.start(config={"model": "deepseek-v3.2", "num_iterations": 10})
        try:
            for iteration in range(10):
                # ... run evaluation ...
                tracker.log_iteration(iteration, metrics, skill_count, skill_names)
                tracker.log_teacher_decisions(iteration, proposals)
            tracker.log_summary(best_success_rate, best_iteration, final_skill_count, total_iterations)
        finally:
            tracker.finish()
    """

    def __init__(self, project: str = "skillrl-evolution", config: dict | None = None):
        """Initialize tracker.

        Args:
            project: W&B project name
            config: Optional configuration dict to log with the run
        """
        self.project = project
        self.config = config
        self.run = None

    def start(self, config: dict | None = None):
        """Start a W&B run.

        Args:
            config: Optional configuration dict (overrides init config if provided)
        """
        run_config = config if config is not None else self.config
        self.run = wandb.init(project=self.project, config=run_config)

    def finish(self):
        """Finish the current W&B run."""
        if self.run is not None:
            wandb.finish()
            self.run = None

    def log_iteration(
        self,
        iteration: int,
        metrics: AggregateMetrics,
        skill_count: int,
        skill_names: list[str],
    ):
        """Log metrics and skill library state for an iteration.

        Args:
            iteration: Iteration number (x-axis for plots)
            metrics: Aggregate metrics from evaluation
            skill_count: Number of skills in library
            skill_names: List of skill names in library
        """
        # Log core metrics
        wandb.log(
            {
                "iteration": iteration,
                "success_rate": metrics.overall_success_rate,
                "avg_steps_success": metrics.avg_steps_success,
                "avg_steps_failure": metrics.avg_steps_failure,
                "skill_count": skill_count,
                "successful_tasks": metrics.successful_tasks,
                "total_tasks": metrics.total_tasks,
            },
            step=iteration,
        )

        # Log per-subtask success rates (in separate namespace)
        subtask_metrics = {
            f"subtask/{task_type}": rate
            for task_type, rate in metrics.per_subtask_success_rate.items()
        }
        if subtask_metrics:
            wandb.log(subtask_metrics, step=iteration)

        # Log skill names as W&B Table
        table = wandb.Table(columns=["skill_name"], data=[[name] for name in skill_names])
        wandb.log({f"skills/library_iter_{iteration}": table}, step=iteration)

    def log_teacher_decisions(self, iteration: int, proposals: list):
        """Log teacher decision proposals as a W&B table.

        Args:
            iteration: Iteration number
            proposals: List of SkillProposal objects (duck-typed, must have .action, .skill_name, .reason)
        """
        # Create table with columns for action, skill_name, reason
        table = wandb.Table(columns=["action", "skill_name", "reason"])

        # Add rows (one per proposal)
        for proposal in proposals:
            table.add_data(proposal.action, proposal.skill_name, proposal.reason)

        # Log the table (even if empty, shows no changes happened)
        wandb.log({f"teacher/decisions_iter_{iteration}": table}, step=iteration)

    def log_summary(
        self,
        best_success_rate: float,
        best_iteration: int,
        final_skill_count: int,
        total_iterations: int,
    ):
        """Log final summary statistics for the run.

        These appear in the W&B run summary and are used for run comparison.

        Args:
            best_success_rate: Best success rate achieved during evolution
            best_iteration: Iteration where best success rate occurred
            final_skill_count: Number of skills in final library
            total_iterations: Total number of iterations completed
        """
        wandb.run.summary["best_success_rate"] = best_success_rate
        wandb.run.summary["best_iteration"] = best_iteration
        wandb.run.summary["final_skill_count"] = final_skill_count
        wandb.run.summary["total_iterations"] = total_iterations
