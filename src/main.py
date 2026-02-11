"""Main entry point for running the agent on ALFWorld tasks."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from src.agent import DeepSeekClient, build_tools_spec, format_step, run_task
from src.environment.env_manager import EnvManager
from src.evaluation.orchestrator import EvaluationOrchestrator
from src.evolution.loop import EvolutionLoop
from src.trajectory.storage import append_trajectory


async def run_single_task(args):
    """Run agent on a single task (original functionality)."""
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

    # Step display callback
    agent_idx = args.agent_index

    def on_step(step):
        print(format_step(step, agent_index=agent_idx, max_steps=args.max_steps))
        print()

    # Run task
    prefix = f"[Agent {agent_idx}]"
    print(f"{prefix} Running (max {args.max_steps} steps)...\n")
    try:
        trajectory = await run_task(
            task_description=task_description,
            task_id=task_id,
            task_type=task_type,
            env_manager=env_manager,
            tools_spec=tools_spec,
            client=client,
            max_steps=args.max_steps,
            on_step=on_step,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)

    # Print summary
    print("=" * 60)
    print(f"{prefix} SUMMARY")
    print("=" * 60)
    print(f"{prefix} Task:    {trajectory.task_type} | {trajectory.task_id}")
    print(f"{prefix} Result:  {'SUCCESS' if trajectory.success else 'FAILED'}")
    print(f"{prefix} Steps:   {trajectory.total_steps}/{args.max_steps}")
    print(f"{prefix} Time:    {trajectory.duration_seconds:.1f}s")
    if trajectory.failure_reason:
        print(f"{prefix} Reason:  {trajectory.failure_reason}")

    # Check for discrepancy between agent and env
    if env_manager.done != trajectory.success:
        print(f"\n{prefix} WARNING: Agent declared success={trajectory.success}, but env done={env_manager.done}")

    # Save trajectory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trajectory_file = output_path / "trajectories.jsonl"

    print(f"\nSaving trajectory to {trajectory_file}...")
    append_trajectory(trajectory, trajectory_file)

    print("\n✓ Complete!")


async def run_evaluation(args):
    """Run full 134-task evaluation with skill retrieval."""
    # Check for API key
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("Error: DEEPSEEK_API_KEY environment variable not set", file=sys.stderr)
        print("\nTo get an API key:", file=sys.stderr)
        print("1. Visit https://platform.deepseek.com/api_keys", file=sys.stderr)
        print("2. Create or copy your API key", file=sys.stderr)
        print("3. Set it in your environment: export DEEPSEEK_API_KEY=your-key-here", file=sys.stderr)
        sys.exit(1)

    # Create orchestrator
    orchestrator = EvaluationOrchestrator(
        skill_library_path=args.skill_library,
        output_dir=args.output_dir,
        max_concurrent=args.max_concurrent,
        max_steps=args.max_steps,
        top_k_skills=args.top_k,
    )

    # Run iteration
    try:
        metrics = await orchestrator.run_iteration(args.iteration)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)

    # Final summary
    print("\nEvaluation complete!")
    print(f"Results saved to: {args.output_dir}")


async def run_evolution_loop(args):
    """Run full skill evolution loop."""
    # Check for API key
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("Error: DEEPSEEK_API_KEY environment variable not set", file=sys.stderr)
        print("\nTo get an API key:", file=sys.stderr)
        print("1. Visit https://platform.deepseek.com/api_keys", file=sys.stderr)
        print("2. Create or copy your API key", file=sys.stderr)
        print("3. Set it in your environment: export DEEPSEEK_API_KEY=your-key-here", file=sys.stderr)
        sys.exit(1)

    # Check for WANDB_API_KEY
    if not os.environ.get("WANDB_API_KEY"):
        print("Warning: WANDB_API_KEY not set. W&B logging may require login.", file=sys.stderr)
        print("Set it via: export WANDB_API_KEY=your-key-here", file=sys.stderr)

    loop = EvolutionLoop(
        skill_library_path=args.skill_library,
        output_dir=args.output_dir,
        max_iterations=args.max_iterations,
        patience=args.patience,
        min_delta=args.min_delta,
        max_concurrent=args.max_concurrent,
        max_steps=args.max_steps,
        top_k_skills=args.top_k,
        wandb_project=args.wandb_project,
    )

    try:
        await loop.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)


async def main():
    """Main async execution function with subcommands."""
    parser = argparse.ArgumentParser(
        description="SkillRL: Frozen LLM + evolving skill library on ALFWorld"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # 'run' subcommand - single task execution
    run_parser = subparsers.add_parser(
        "run",
        help="Run agent on a single ALFWorld task"
    )
    run_parser.add_argument(
        "--task-index",
        type=int,
        default=0,
        help="Task index to run (default: 0)",
    )
    run_parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum steps per task (default: 50)",
    )
    run_parser.add_argument(
        "--output-dir",
        type=str,
        default="data/trajectories",
        help="Output directory for trajectories (default: data/trajectories)",
    )
    run_parser.add_argument(
        "--agent-index",
        type=int,
        default=0,
        help="Agent index prefix for display (default: 0)",
    )

    # 'evaluate' subcommand - full 134-task evaluation
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Run full 134-task evaluation with skill retrieval"
    )
    eval_parser.add_argument(
        "--iteration",
        type=int,
        default=0,
        help="Iteration number (default: 0)",
    )
    eval_parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent workers (default: 10)",
    )
    eval_parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum steps per task (default: 50)",
    )
    eval_parser.add_argument(
        "--output-dir",
        type=str,
        default="data/experiments",
        help="Output directory for results (default: data/experiments)",
    )
    eval_parser.add_argument(
        "--skill-library",
        type=str,
        default="data/skills/skills.json",
        help="Path to skill library JSON (default: data/skills/skills.json)",
    )
    eval_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of skills to retrieve per task (default: 3)",
    )

    # 'evolve' subcommand - full skill evolution loop
    evolve_parser = subparsers.add_parser(
        "evolve",
        help="Run full skill evolution loop (evaluate -> teach -> update -> repeat)"
    )
    evolve_parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum evolution iterations (default: 20)",
    )
    evolve_parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Iterations without improvement before early stopping (default: 5)",
    )
    evolve_parser.add_argument(
        "--min-delta",
        type=float,
        default=0.01,
        help="Minimum improvement to count as progress (default: 0.01)",
    )
    evolve_parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent workers (default: 10)",
    )
    evolve_parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum steps per task (default: 50)",
    )
    evolve_parser.add_argument(
        "--output-dir",
        type=str,
        default="data/experiments",
        help="Output directory (default: data/experiments)",
    )
    evolve_parser.add_argument(
        "--skill-library",
        type=str,
        default="data/skills/skills.json",
        help="Path to skill library JSON (default: data/skills/skills.json)",
    )
    evolve_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of skills to retrieve per task (default: 3)",
    )
    evolve_parser.add_argument(
        "--wandb-project",
        type=str,
        default="skillrl-evolution",
        help="W&B project name (default: skillrl-evolution)",
    )

    args = parser.parse_args()

    # Handle no command (backward compatibility - default to run with task-index 0)
    if args.command is None:
        print("No command specified. Use 'run', 'evaluate', or 'evolve'.")
        print("\nExamples:")
        print("  python -m src.main run --task-index 0")
        print("  python -m src.main evaluate --iteration 0")
        print("  python -m src.main evolve --max-iterations 20")
        parser.print_help()
        sys.exit(1)

    # Dispatch to appropriate handler
    if args.command == "run":
        await run_single_task(args)
    elif args.command == "evaluate":
        await run_evaluation(args)
    elif args.command == "evolve":
        await run_evolution_loop(args)


if __name__ == "__main__":
    asyncio.run(main())
