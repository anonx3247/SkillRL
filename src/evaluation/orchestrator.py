"""Parallel evaluation orchestrator for 134-task ALFWorld evaluation."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from src.agent import DeepSeekClient, build_tools_spec, run_task
from src.environment.env_manager import EnvManager
from src.skills.library import SkillLibrary
from src.skills.retrieval import SkillRetriever
from src.trajectory.models import Trajectory
from src.trajectory.storage import append_trajectory
from .checkpoint import CheckpointManager, IterationCheckpoint
from .metrics import compute_metrics, AggregateMetrics


class EvaluationOrchestrator:
    """Orchestrates parallel evaluation across all 134 ALFWorld tasks.

    Manages concurrent task execution, skill retrieval, metrics computation,
    and checkpoint persistence.
    """

    def __init__(
        self,
        skill_library_path: Path | str,
        output_dir: Path | str,
        max_concurrent: int = 10,
        max_steps: int = 50,
        top_k_skills: int = 3,
        wall_clock_timeout: float = 300.0,
    ):
        """Initialize evaluation orchestrator.

        Args:
            skill_library_path: Path to skill library JSON file
            output_dir: Directory for output (trajectories, checkpoints)
            max_concurrent: Maximum concurrent workers
            max_steps: Maximum steps per task
            top_k_skills: Number of skills to retrieve per task
            wall_clock_timeout: Timeout per task in seconds
        """
        self.skill_library_path = Path(skill_library_path)
        self.output_dir = Path(output_dir)
        self.max_concurrent = max_concurrent
        self.max_steps = max_steps
        self.top_k_skills = top_k_skills
        self.wall_clock_timeout = wall_clock_timeout

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_manager = CheckpointManager(self.output_dir / "checkpoints")

    async def run_iteration(self, iteration: int) -> AggregateMetrics:
        """Run full 134-task evaluation for one iteration.

        Args:
            iteration: Iteration number

        Returns:
            Aggregate metrics for this iteration
        """
        print(f"\n{'='*60}")
        print(f"Starting Iteration {iteration}")
        print(f"{'='*60}")

        start_time = time.time()

        # Step 1: Collect all 134 tasks (discovery pass — sequential)
        print("\nStep 1: Discovering all tasks...")
        tasks = await self._discover_all_tasks()
        print(f"Found {len(tasks)} tasks")

        # Step 2: Load skill library + retriever
        print(f"\nStep 2: Loading skill library from {self.skill_library_path}...")
        skill_library = SkillLibrary(self.skill_library_path)
        skill_library.load()
        all_skills = skill_library.get_all_skills()
        print(f"Loaded {len(all_skills)} skills")

        retriever = SkillRetriever()
        retriever.index_skills(all_skills)
        print("Skill index built")

        # Step 3: Run tasks in parallel (asyncio.Semaphore)
        print(f"\nStep 3: Running {len(tasks)} tasks with {self.max_concurrent} concurrent workers...")
        trajectories, skills_per_task = await self._run_tasks_parallel(
            tasks, retriever, iteration
        )
        print(f"Completed {len(trajectories)} tasks")

        # Step 4: Compute metrics
        print("\nStep 4: Computing metrics...")
        metrics = compute_metrics(trajectories, iteration, skills_per_task)

        # Step 5: Save results
        print("\nStep 5: Saving results...")
        trajectories_path = self.output_dir / f"iteration_{iteration:03d}_trajectories.jsonl"
        for trajectory in trajectories:
            append_trajectory(trajectory, trajectories_path)
        print(f"Trajectories saved to {trajectories_path}")

        # Save checkpoint
        checkpoint = IterationCheckpoint(
            iteration=iteration,
            total_tasks=metrics.total_tasks,
            successful_tasks=metrics.successful_tasks,
            overall_success_rate=metrics.overall_success_rate,
            per_subtask_success_rate=metrics.per_subtask_success_rate,
            avg_steps_success=metrics.avg_steps_success,
            avg_steps_failure=metrics.avg_steps_failure,
            skill_library_path=str(self.skill_library_path),
            trajectories_path=str(trajectories_path),
            timestamp=time.time(),
        )
        self.checkpoint_manager.save(checkpoint)
        print(f"Checkpoint saved")

        # Step 6: Print summary
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Iteration {iteration} Complete")
        print(f"{'='*60}")
        print(f"Overall Success Rate: {metrics.overall_success_rate:.2%}")
        print(f"Successful Tasks: {metrics.successful_tasks}/{metrics.total_tasks}")
        print(f"Avg Steps (Success): {metrics.avg_steps_success:.1f}")
        print(f"Avg Steps (Failure): {metrics.avg_steps_failure:.1f}")
        print(f"\nPer-Subtask Success Rates:")
        for task_type in sorted(metrics.per_subtask_success_rate.keys()):
            rate = metrics.per_subtask_success_rate[task_type]
            print(f"  {task_type:8s}: {rate:.2%}")
        print(f"\nTotal Time: {elapsed:.1f}s")
        print(f"{'='*60}\n")

        return metrics

    async def _discover_all_tasks(self) -> list[tuple[str, str, str]]:
        """Discover all 134 tasks via sequential reset.

        Returns:
            List of (task_id, task_description, task_type) tuples
        """
        env_manager = EnvManager()
        env_manager.load()

        tasks = []
        # ALFWorld eval_out_of_distribution has 134 tasks
        for _ in range(134):
            observation, info = env_manager.reset()
            task_id = env_manager.get_task_id()
            task_type = env_manager.get_task_type()
            tasks.append((task_id, observation, task_type))

        return tasks

    async def _run_tasks_parallel(
        self,
        tasks: list[tuple[str, str, str]],
        retriever: SkillRetriever,
        iteration: int,
    ) -> tuple[list[Trajectory], dict[str, int]]:
        """Run tasks in parallel with semaphore-limited concurrency.

        Args:
            tasks: List of (task_id, task_description, task_type) tuples
            retriever: SkillRetriever for semantic search
            iteration: Current iteration number

        Returns:
            Tuple of (trajectories, skills_per_task mapping)
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        client = DeepSeekClient()
        tools_spec = build_tools_spec()

        # Track progress
        completed = 0
        total = len(tasks)

        async def run_single_task(task_idx: int, task_id: str, task_description: str, task_type: str):
            nonlocal completed

            async with semaphore:
                # CRITICAL: Each worker creates its own EnvManager
                env_manager = EnvManager()
                env_manager.load()

                # Reset to this specific task
                for _ in range(task_idx + 1):
                    observation, info = env_manager.reset()

                # Retrieve skills for this task
                retrieved_skills = retriever.retrieve(task_description, self.top_k_skills)

                # Run task
                trajectory = await run_task(
                    task_description=task_description,
                    task_id=task_id,
                    task_type=task_type,
                    env_manager=env_manager,
                    tools_spec=tools_spec,
                    client=client,
                    max_steps=self.max_steps,
                    wall_clock_timeout=self.wall_clock_timeout,
                    retrieved_skills=retrieved_skills,
                )

                # Update progress
                completed += 1
                print(f"  [{completed}/{total}] {task_type:8s} | {task_id:40s} | {'SUCCESS' if trajectory.success else 'FAILED':7s} | {trajectory.total_steps:2d} steps")

                return trajectory, len(retrieved_skills)

        # Launch all tasks concurrently
        task_futures = [
            run_single_task(idx, task_id, task_description, task_type)
            for idx, (task_id, task_description, task_type) in enumerate(tasks)
        ]

        results = await asyncio.gather(*task_futures)

        # Separate trajectories and skills_per_task
        trajectories = [traj for traj, _ in results]
        skills_per_task = {
            traj.task_id: skill_count
            for (traj, skill_count) in results
        }

        return trajectories, skills_per_task
