"""Main entry point for running the agent on ALFWorld tasks."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from src.agent import DeepSeekClient, build_tools_spec, run_task
from src.environment.env_manager import EnvManager
from src.trajectory.storage import append_trajectory


async def main():
    """Main async execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run agent on ALFWorld tasks")
    parser.add_argument(
        "--task-index",
        type=int,
        default=0,
        help="Task index to run (default: 0)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum steps per task (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/trajectories",
        help="Output directory for trajectories (default: data/trajectories)",
    )
    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("Error: DEEPSEEK_API_KEY environment variable not set", file=sys.stderr)
        print("\nTo get an API key:", file=sys.stderr)
        print("1. Visit https://platform.deepseek.com/api_keys", file=sys.stderr)
        print("2. Create or copy your API key", file=sys.stderr)
        print("3. Set it in your environment: export DEEPSEEK_API_KEY=your-key-here", file=sys.stderr)
        sys.exit(1)

    # Load environment
    print("Loading ALFWorld environment...")
    env_manager = EnvManager()
    env_manager.load()

    # Reset to desired task
    print(f"Resetting to task index {args.task_index}...")
    for i in range(args.task_index + 1):
        observation, info = env_manager.reset()

    task_description = observation
    task_id = env_manager.get_task_id()
    task_type = env_manager.get_task_type()

    print(f"\nTask ID: {task_id}")
    print(f"Task Type: {task_type}")
    print(f"Task Description:\n{task_description}\n")

    # Create client and tools
    print("Initializing DeepSeek client...")
    client = DeepSeekClient()
    tools_spec = build_tools_spec()

    # Run task
    print(f"Running agent (max {args.max_steps} steps)...\n")
    try:
        trajectory = await run_task(
            task_description=task_description,
            task_id=task_id,
            task_type=task_type,
            env_manager=env_manager,
            tools_spec=tools_spec,
            client=client,
            max_steps=args.max_steps,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)

    # Print summary
    print("\n" + "=" * 60)
    print("TRAJECTORY SUMMARY")
    print("=" * 60)
    print(f"Task ID: {trajectory.task_id}")
    print(f"Task Type: {trajectory.task_type}")
    print(f"Success: {trajectory.success}")
    print(f"Total Steps: {trajectory.total_steps}")
    print(f"Duration: {trajectory.duration_seconds:.2f} seconds")
    if trajectory.failure_reason:
        print(f"Failure Reason: {trajectory.failure_reason}")

    # Check for discrepancy between agent and env
    if env_manager.done != trajectory.success:
        print("\n⚠️  WARNING: Discrepancy detected!")
        print(f"   Agent declared success={trajectory.success}")
        print(f"   Environment done={env_manager.done}")

    # Save trajectory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trajectory_file = output_path / "trajectories.jsonl"

    print(f"\nSaving trajectory to {trajectory_file}...")
    append_trajectory(trajectory, trajectory_file)

    print("\n✓ Complete!")


if __name__ == "__main__":
    asyncio.run(main())
