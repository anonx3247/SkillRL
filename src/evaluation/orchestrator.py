"""Parallel evaluation orchestrator for 134-task ALFWorld evaluation."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import wandb

from src.agent import DeepSeekClient, build_tools_spec, run_task
from src.environment.env_manager import EnvManager
from src.skills.library import SkillLibrary
from src.skills.retrieval import SkillRetriever
from src.trajectory.models import Trajectory
from src.trajectory.storage import append_trajectory, load_trajectories
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
        max_concurrent: int = 32,
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

        Resumes from existing trajectory file if present — only runs tasks
        that don't already have results.

        Args:
            iteration: Iteration number

        Returns:
            Aggregate metrics for this iteration
        """
        print(f"\n{'='*60}")
        print(f"Starting Iteration {iteration}")
        print(f"{'='*60}")

        start_time = time.time()
        trajectories_path = self.output_dir / f"iteration_{iteration:03d}_trajectories.jsonl"

        # Step 1: Load existing trajectories for resume
        existing_trajectories = load_trajectories(trajectories_path)
        completed_task_ids = {t.task_id for t in existing_trajectories}
        if existing_trajectories:
            print(f"\nResuming: found {len(existing_trajectories)} existing trajectories, skipping them")

        # Step 2: Get game files + config (fast — no resets, just scans file paths)
        print("\nStep 1: Loading game files...")
        game_files, env_config = self._load_game_files()
        remaining = len(game_files) - len(completed_task_ids & set(game_files))
        print(f"Found {len(game_files)} tasks ({remaining} to run)")

        if remaining == 0:
            print("All tasks already completed for this iteration")
            all_trajectories = existing_trajectories
        else:
            # Step 3: Load skill library + retriever
            print(f"\nStep 2: Loading skill library from {self.skill_library_path}...")
            skill_library = SkillLibrary(self.skill_library_path)
            skill_library.load()
            all_skills = skill_library.get_all_skills()
            print(f"Loaded {len(all_skills)} skills")

            retriever = SkillRetriever()
            retriever.index_skills(all_skills)
            print("Skill index built")

            # Step 4: Run remaining tasks in parallel
            print(f"\nStep 3: Running {remaining} tasks with {self.max_concurrent} workers...")
            new_trajectories, skills_per_task = await self._run_tasks_parallel(
                game_files, env_config, retriever, iteration, completed_task_ids
            )
            print(f"Completed {len(new_trajectories)} new tasks")

            # Persist usage tracking data
            skill_library.save()

            # Append only new trajectories to file
            for trajectory in new_trajectories:
                append_trajectory(trajectory, trajectories_path)

            all_trajectories = existing_trajectories + new_trajectories

        # Compute metrics from ALL trajectories
        print("\nStep 4: Computing metrics...")
        skills_per_task_all = {}
        metrics = compute_metrics(all_trajectories, iteration, skills_per_task_all)

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

        # Print summary
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

    def _load_game_files(self) -> tuple[list[str], dict]:
        """Load game file list and config without resetting (fast, ~0.2s).

        Returns:
            Tuple of (game_files list, resolved config dict)
        """
        env_manager = EnvManager()
        env_manager.load()
        return env_manager.game_files, env_manager.config

    async def _run_tasks_parallel(
        self,
        game_files: list[str],
        env_config: dict,
        retriever: SkillRetriever,
        iteration: int,
        completed_task_ids: set[str] | None = None,
    ) -> tuple[list[Trajectory], dict[str, int]]:
        """Run tasks in parallel using per-worker lightweight environments.

        Each worker gets an env created from only its game files (instant).
        Workers call reset() themselves to get the task observation — no
        separate discovery pass needed. Completed tasks are skipped.

        Args:
            game_files: All game file paths
            env_config: Resolved ALFWorld config dict
            retriever: SkillRetriever for semantic search
            iteration: Current iteration number
            completed_task_ids: Task IDs to skip (already have trajectories)

        Returns:
            Tuple of (new trajectories, skills_per_task mapping)
        """
        completed_task_ids = completed_task_ids or set()
        client = DeepSeekClient()
        tools_spec = build_tools_spec()

        # Filter to only uncompleted game files
        pending_gamefiles = [gf for gf in game_files if gf not in completed_task_ids]
        total_to_run = len(pending_gamefiles)

        if total_to_run == 0:
            return [], {}

        num_workers = min(self.max_concurrent, total_to_run)
        completed = 0
        successes = 0
        new_results: list[tuple[Trajectory, int]] = []
        results_lock = asyncio.Lock()

        # Divide ONLY pending game files into chunks (no wasted workers on completed tasks)
        chunks: list[list[str]] = [[] for _ in range(num_workers)]
        for i, gf in enumerate(pending_gamefiles):
            chunks[i % num_workers].append(gf)

        print(f"  Launching {num_workers} workers with {len(chunks[0])}-{len(chunks[-1])} tasks each")

        async def worker(worker_id: int, worker_gamefiles: list[str]):
            nonlocal completed, successes

            # Create lightweight env from ONLY this worker's game files (instant)
            env_manager = EnvManager.from_gamefiles(worker_gamefiles, env_config)

            for gamefile in worker_gamefiles:
                # Reset gives us the observation + positions the env
                observation, info = env_manager.reset()
                task_id = env_manager.get_task_id()
                task_type = env_manager.get_task_type()

                # Retrieve skills for this task
                retrieved_skills = retriever.retrieve(
                    observation, self.top_k_skills, current_iteration=iteration
                )

                # Run task
                trajectory = await run_task(
                    task_description=observation,
                    task_id=task_id,
                    task_type=task_type,
                    env_manager=env_manager,
                    tools_spec=tools_spec,
                    client=client,
                    max_steps=self.max_steps,
                    wall_clock_timeout=self.wall_clock_timeout,
                    retrieved_skills=retrieved_skills,
                )

                async with results_lock:
                    new_results.append((trajectory, len(retrieved_skills)))
                    completed += 1
                    if trajectory.success:
                        successes += 1
                    running_rate = successes / completed
                    status = "SUCCESS" if trajectory.success else "FAILED"
                    print(
                        f"  [{completed}/{total_to_run}] {task_type:8s} | "
                        f"{status:7s} | {trajectory.total_steps:2d} steps | "
                        f"running {running_rate:.0%}"
                    )

                    # Per-task W&B progress
                    if wandb.run is not None:
                        wandb.log({
                            "eval/completed": completed,
                            "eval/running_success_rate": running_rate,
                            "eval/total_to_run": total_to_run,
                        })

        # Launch all workers concurrently
        await asyncio.gather(
            *(worker(i, chunk) for i, chunk in enumerate(chunks))
        )

        trajectories = [traj for traj, _ in new_results]
        skills_per_task = {
            traj.task_id: skill_count
            for traj, skill_count in new_results
        }

        return trajectories, skills_per_task
