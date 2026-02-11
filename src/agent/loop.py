"""Agent execution loop: think-act-observe cycle."""

import asyncio
import json
import time
from typing import Any

from src.agent.client import DeepSeekClient
from src.agent.prompts import AUTONOMOUS_AGENT_PROMPT
from src.trajectory.models import Step, Trajectory


def build_tools_spec() -> list[dict[str, Any]]:
    """Build OpenAI function-calling schema for ALFWorld tools.

    Returns:
        List of tool definition dicts
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "go_to",
                "description": "Navigate to a specific location in the environment",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to go to (e.g., 'countertop 1', 'cabinet 3')",
                        }
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "take",
                "description": "Pick up an object from the current location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_name": {
                            "type": "string",
                            "description": "The name of the object to take",
                        }
                    },
                    "required": ["object_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "put",
                "description": "Place an object in or on a receptacle",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_name": {
                            "type": "string",
                            "description": "The object to place",
                        },
                        "receptacle": {
                            "type": "string",
                            "description": "The receptacle to place the object in/on",
                        },
                    },
                    "required": ["object_name", "receptacle"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "open_receptacle",
                "description": "Open a container or receptacle",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "receptacle": {
                            "type": "string",
                            "description": "The receptacle to open",
                        }
                    },
                    "required": ["receptacle"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "close_receptacle",
                "description": "Close a container or receptacle",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "receptacle": {
                            "type": "string",
                            "description": "The receptacle to close",
                        }
                    },
                    "required": ["receptacle"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "clean",
                "description": "Clean an object using a receptacle (e.g., sinkbasin)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_name": {
                            "type": "string",
                            "description": "The object to clean",
                        },
                        "receptacle": {
                            "type": "string",
                            "description": "The receptacle to use for cleaning (typically sinkbasin)",
                        },
                    },
                    "required": ["object_name", "receptacle"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "heat",
                "description": "Heat an object using a receptacle (e.g., microwave)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_name": {
                            "type": "string",
                            "description": "The object to heat",
                        },
                        "receptacle": {
                            "type": "string",
                            "description": "The receptacle to use for heating (typically microwave)",
                        },
                    },
                    "required": ["object_name", "receptacle"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cool",
                "description": "Cool an object using a receptacle (e.g., fridge)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_name": {
                            "type": "string",
                            "description": "The object to cool",
                        },
                        "receptacle": {
                            "type": "string",
                            "description": "The receptacle to use for cooling (typically fridge)",
                        },
                    },
                    "required": ["object_name", "receptacle"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "use",
                "description": "Use or toggle an object (e.g., turn on a lamp)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_name": {
                            "type": "string",
                            "description": "The object to use",
                        }
                    },
                    "required": ["object_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "examine",
                "description": "Look closely at an object to get more information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_name": {
                            "type": "string",
                            "description": "The object to examine",
                        }
                    },
                    "required": ["object_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inventory",
                "description": "Check what objects you are currently holding",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "task_completed",
                "description": "Signal that the task is complete (success or failure)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "success": {
                            "type": "boolean",
                            "description": "Whether the task was completed successfully",
                        },
                        "summary": {
                            "type": "string",
                            "description": "Brief summary of the outcome",
                        },
                    },
                    "required": ["success", "summary"],
                },
            },
        },
    ]


def _map_tool_to_command(tool_name: str, arguments: dict[str, Any]) -> str:
    """Map tool call to ALFWorld command string.

    Args:
        tool_name: Name of the tool being called
        arguments: Tool arguments

    Returns:
        ALFWorld command string
    """
    if tool_name == "go_to":
        return f"go to {arguments['location']}"
    elif tool_name == "take":
        return f"take {arguments['object_name']}"
    elif tool_name == "put":
        return f"put {arguments['object_name']} in/on {arguments['receptacle']}"
    elif tool_name == "open_receptacle":
        return f"open {arguments['receptacle']}"
    elif tool_name == "close_receptacle":
        return f"close {arguments['receptacle']}"
    elif tool_name == "clean":
        return f"clean {arguments['object_name']} with {arguments['receptacle']}"
    elif tool_name == "heat":
        return f"heat {arguments['object_name']} with {arguments['receptacle']}"
    elif tool_name == "cool":
        return f"cool {arguments['object_name']} with {arguments['receptacle']}"
    elif tool_name == "use":
        return f"use {arguments['object_name']}"
    elif tool_name == "examine":
        return f"examine {arguments['object_name']}"
    elif tool_name == "inventory":
        return "inventory"
    else:
        raise ValueError(f"Unknown tool: {tool_name}")


async def run_task(
    task_description: str,
    task_id: str,
    task_type: str,
    env_manager: Any,
    tools_spec: list[dict[str, Any]],
    client: DeepSeekClient,
    max_steps: int = 50,
    wall_clock_timeout: float = 300.0,
) -> Trajectory:
    """Run agent on a single task with think-act-observe loop.

    Args:
        task_description: Natural language task description
        task_id: Unique task identifier
        task_type: Task type (pick, look, clean, heat, cool, pick2)
        env_manager: EnvManager instance
        tools_spec: OpenAI function-calling schema
        client: DeepSeek API client
        max_steps: Maximum number of steps (default: 50)
        wall_clock_timeout: Wall clock timeout in seconds (default: 300)

    Returns:
        Trajectory with all steps and outcome
    """
    start_time = time.time()

    # Initialize conversation with system prompt and task
    messages = [
        {"role": "system", "content": AUTONOMOUS_AGENT_PROMPT},
        {"role": "user", "content": f"Complete this task:\n\n{task_description}"},
    ]

    steps: list[Step] = []
    task_complete = False
    success = False
    failure_reason: str | None = None
    agent_summary = ""

    try:
        async def execute_loop():
            nonlocal task_complete, success, failure_reason, agent_summary

            for step_num in range(max_steps):
                # Get model response
                response = await client.chat(messages, tools=tools_spec)
                response_message = response.choices[0].message

                # Add assistant message to conversation
                messages.append(response_message.model_dump())

                # Check if model made tool calls
                if not response_message.tool_calls:
                    # No tool call - agent failed to act
                    step = Step(
                        step=step_num,
                        thought=response_message.content,
                        action="no_action",
                        action_input="{}",
                        observation="Error: No tool was called. You must call a tool to interact with the environment.",
                    )
                    steps.append(step)
                    failure_reason = "no_tool_call"
                    break

                # Process each tool call
                for tool_call in response_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    # Check for task completion signal
                    if tool_name == "task_completed":
                        success = tool_args.get("success", False)
                        agent_summary = tool_args.get("summary", "")
                        step = Step(
                            step=step_num,
                            thought=response_message.content or "Task complete",
                            action="task_completed",
                            action_input=tool_call.function.arguments,
                            observation=f"Task marked as {'successful' if success else 'failed'}: {agent_summary}",
                        )
                        steps.append(step)
                        task_complete = True

                        # Add tool response to conversation
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": step.observation,
                        })
                        break

                    # Execute environment action
                    try:
                        command = _map_tool_to_command(tool_name, tool_args)
                        observation, _, _, _ = env_manager.step(command)
                    except Exception as e:
                        observation = f"Error executing action: {str(e)}"

                    # Create step
                    step = Step(
                        step=step_num,
                        thought=response_message.content or "",
                        action=tool_name,
                        action_input=tool_call.function.arguments,
                        observation=observation,
                    )
                    steps.append(step)

                    # Add tool response to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": observation,
                    })

                # If task_completed was called, exit loop
                if task_complete:
                    break

            # If we exhausted max_steps without completion
            if not task_complete:
                failure_reason = "timeout"

        # Run loop with wall clock timeout
        await asyncio.wait_for(execute_loop(), timeout=wall_clock_timeout)

    except asyncio.TimeoutError:
        failure_reason = "wall_clock_timeout"

    # Calculate duration
    duration = time.time() - start_time

    # Build trajectory
    trajectory = Trajectory(
        task_id=task_id,
        task_description=task_description,
        task_type=task_type,
        success=success,
        steps=steps,
        total_steps=len(steps),
        duration_seconds=duration,
        failure_reason=failure_reason if not success else None,
        env_done=env_manager.done,
    )

    return trajectory
