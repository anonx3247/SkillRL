"""Evolution loop orchestrating evaluate-analyze-evolve cycle."""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.orchestrator import EvaluationOrchestrator
from src.skills.library import SkillLibrary
from src.skills.models import Skill
from src.teacher.analyzer import TeacherAnalyzer
from src.tracking.wandb_logger import ExperimentTracker
from src.trajectory.models import Trajectory, Step
from .convergence import ConvergenceDetector


class EvolutionLoop:
    """Orchestrates full skill evolution loop.

    Iterates: evaluate -> analyze -> update library -> repeat
    Stops early when success rate plateaus (convergence detection).
    """

    def __init__(
        self,
        skill_library_path: Path | str,
        output_dir: Path | str,
        max_iterations: int = 20,
        patience: int = 5,
        min_delta: float = 0.01,
        max_concurrent: int = 32,
        max_steps: int = 50,
        top_k_skills: int = 3,
        wandb_project: str = "skillrl-evolution",
    ):
        """Initialize evolution loop.

        Args:
            skill_library_path: Path to skill library JSON
            output_dir: Output directory for experiments
            max_iterations: Maximum evolution iterations
            patience: Iterations without improvement before early stopping
            min_delta: Minimum improvement to count as progress
            max_concurrent: Maximum concurrent workers for evaluation
            max_steps: Maximum steps per task
            top_k_skills: Number of skills to retrieve per task
            wandb_project: W&B project name
        """
        self.skill_library_path = Path(skill_library_path)
        self.output_dir = Path(output_dir)
        self.max_iterations = max_iterations
        self.max_concurrent = max_concurrent
        self.max_steps = max_steps
        self.top_k_skills = top_k_skills
        self.wandb_project = wandb_project

        # Create convergence detector
        self.convergence = ConvergenceDetector(patience=patience, min_delta=min_delta)

        # Components initialized in run() (not __init__)
        # This allows for clean separation between config and runtime

    async def run(self):
        """Run full evolution loop until convergence or max iterations."""
        print("\n" + "=" * 80)
        print("SKILL EVOLUTION LOOP")
        print("=" * 80)
        print(f"Skill Library: {self.skill_library_path}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Max Iterations: {self.max_iterations}")
        print(f"Convergence: patience={self.convergence.patience}, min_delta={self.convergence.min_delta}")
        print("=" * 80 + "\n")

        # Initialize components
        skill_library = SkillLibrary(self.skill_library_path)
        skill_library.load()  # Empty for iteration 0 if file doesn't exist
        print(f"Loaded skill library: {len(skill_library)} skills\n")

        # Create W&B tracker
        tracker = ExperimentTracker(project=self.wandb_project)
        config = {
            "max_iterations": self.max_iterations,
            "patience": self.convergence.patience,
            "min_delta": self.convergence.min_delta,
            "max_concurrent": self.max_concurrent,
            "max_steps": self.max_steps,
            "top_k_skills": self.top_k_skills,
            "skill_library_path": str(self.skill_library_path),
            "output_dir": str(self.output_dir),
        }
        tracker.start(config=config)

        # Create teacher and orchestrator
        analyzer = TeacherAnalyzer()
        orchestrator = EvaluationOrchestrator(
            skill_library_path=self.skill_library_path,
            output_dir=self.output_dir,
            max_concurrent=self.max_concurrent,
            max_steps=self.max_steps,
            top_k_skills=self.top_k_skills,
        )

        # Track best performance
        best_success_rate = 0.0
        best_iteration = 0

        try:
            # Main evolution loop
            for iteration in range(self.max_iterations):
                print(f"\n{'#' * 80}")
                print(f"# ITERATION {iteration}")
                print(f"{'#' * 80}\n")

                # Step 1: Run evaluation
                print(f"Running evaluation with {len(skill_library)} skills...")
                metrics = await orchestrator.run_iteration(iteration)

                # Reload library to pick up usage tracking data from evaluation
                skill_library.load()

                # Step 2: Log to W&B
                skill_names = [s.name for s in skill_library.get_all_skills()]
                tracker.log_iteration(iteration, metrics, len(skill_library), skill_names)

                # Step 3: Track best
                if metrics.overall_success_rate > best_success_rate:
                    best_success_rate = metrics.overall_success_rate
                    best_iteration = iteration
                    print(f"\n🎯 New best success rate: {best_success_rate:.2%} (iteration {best_iteration})")

                # Step 4: Check convergence (skip iteration 0)
                if iteration > 0 and self.convergence.check(metrics.overall_success_rate):
                    print(f"\n✓ Converged after {iteration + 1} iterations (no improvement for {self.convergence.patience} iterations)")
                    print(f"Best success rate: {best_success_rate:.2%} at iteration {best_iteration}")
                    break

                # Step 5: Teacher analysis and skill update (not on last iteration)
                if iteration < self.max_iterations - 1:
                    print(f"\nTeacher analyzing trajectories from iteration {iteration}...")

                    # Load trajectories
                    trajectories = self._load_iteration_trajectories(iteration)
                    print(f"Loaded {len(trajectories)} trajectories")

                    # Teacher analysis
                    proposals = await analyzer.analyze_and_propose(
                        trajectories,
                        skill_library.get_all_skills()
                    )
                    print(f"Teacher proposed {len(proposals)} updates")

                    # Apply proposals
                    if proposals:
                        self._apply_proposals(skill_library, proposals, iteration + 1)

                        # Log teacher decisions
                        tracker.log_teacher_decisions(iteration, proposals)
                    else:
                        print("No skill updates proposed")

            # Final summary
            print(f"\n{'=' * 80}")
            print("EVOLUTION COMPLETE")
            print(f"{'=' * 80}")
            print(f"Total iterations: {iteration + 1}")
            print(f"Best success rate: {best_success_rate:.2%} at iteration {best_iteration}")
            print(f"Final skill count: {len(skill_library)}")
            print(f"{'=' * 80}\n")

            # Log summary to W&B
            tracker.log_summary(
                best_success_rate=best_success_rate,
                best_iteration=best_iteration,
                final_skill_count=len(skill_library),
                total_iterations=iteration + 1,
            )

        finally:
            # Always finish W&B run
            tracker.finish()

    def _apply_proposals(self, skill_library: SkillLibrary, proposals: list, iteration: int):
        """Apply teacher proposals to skill library.

        Args:
            skill_library: Skill library to update
            proposals: List of SkillProposal objects
            iteration: Current iteration (for created_iteration tracking)
        """
        add_count = 0
        update_count = 0
        remove_count = 0

        for proposal in proposals:
            try:
                if proposal.action == "add":
                    # Create new skill
                    skill = Skill(
                        name=proposal.skill_name,
                        principle=proposal.principle,
                        when_to_apply=proposal.when_to_apply,
                        created_iteration=iteration,
                    )
                    skill_library.add_skill(skill)
                    add_count += 1
                    print(f"  ✓ Added skill: {proposal.skill_name}")

                elif proposal.action == "update":
                    # Update existing skill
                    skill_name = proposal.old_skill_name or proposal.skill_name
                    skill_library.update_skill(
                        skill_name,
                        principle=proposal.principle,
                        when_to_apply=proposal.when_to_apply,
                    )
                    update_count += 1
                    print(f"  ✓ Updated skill: {skill_name}")

                elif proposal.action == "remove":
                    # Remove skill
                    skill_library.remove_skill(proposal.skill_name)
                    remove_count += 1
                    print(f"  ✓ Removed skill: {proposal.skill_name}")

            except KeyError as e:
                print(f"  ⚠ Warning: {e} (skipping)")
                continue

        print(f"\nApplied {len(proposals)} proposals: {add_count} adds, {update_count} updates, {remove_count} removes")

        # Save updated library
        skill_library.save()
        print(f"Skill library saved to {self.skill_library_path}")

    def _load_iteration_trajectories(self, iteration: int) -> list[Trajectory]:
        """Load trajectories from an iteration's JSONL file.

        Args:
            iteration: Iteration number

        Returns:
            List of Trajectory objects
        """
        trajectories_path = self.output_dir / f"iteration_{iteration:03d}_trajectories.jsonl"

        trajectories = []
        with open(trajectories_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                # Parse JSON
                data = json.loads(line)

                # Reconstruct Step objects
                steps = [Step(**step_data) for step_data in data["steps"]]

                # Create Trajectory with reconstructed steps
                trajectory = Trajectory(
                    task_id=data["task_id"],
                    task_description=data["task_description"],
                    task_type=data["task_type"],
                    success=data["success"],
                    steps=steps,
                    total_steps=data["total_steps"],
                    duration_seconds=data["duration_seconds"],
                    failure_reason=data["failure_reason"],
                    env_done=data["env_done"],
                )
                trajectories.append(trajectory)

        return trajectories
