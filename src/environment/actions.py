"""ALFWorld action tools for FastMCP."""

from src.environment.server import mcp, env_manager


# Environment action tools (10 actions)

@mcp.tool
async def go_to(location: str) -> str:
    """Navigate to a specific location in the room.

    Args:
        location: The name of the location to navigate to (e.g., "cabinet 1", "table 1", "countertop 1")

    Returns:
        Observation string describing what you see after moving
    """
    obs, _score, _done, _info = env_manager.step(f"go to {location}")
    return obs


@mcp.tool
async def take(object_name: str) -> str:
    """Pick up an object from the current location.

    Args:
        object_name: The name of the object to pick up (e.g., "apple 1", "mug 1", "knife 1")

    Returns:
        Observation string describing the result of picking up the object
    """
    obs, _score, _done, _info = env_manager.step(f"take {object_name}")
    return obs


@mcp.tool
async def put(object_name: str, receptacle: str) -> str:
    """Place an object in or on a receptacle.

    Args:
        object_name: The name of the object you're holding to place
        receptacle: The receptacle to place the object in/on (e.g., "microwave 1", "table 1", "cabinet 1")

    Returns:
        Observation string describing the result of placing the object
    """
    obs, _score, _done, _info = env_manager.step(f"put {object_name} in/on {receptacle}")
    return obs


@mcp.tool
async def open_receptacle(receptacle: str) -> str:
    """Open a receptacle (cabinet, drawer, microwave, etc.).

    Args:
        receptacle: The receptacle to open (e.g., "cabinet 1", "drawer 2", "microwave 1")

    Returns:
        Observation string describing what you see after opening the receptacle
    """
    obs, _score, _done, _info = env_manager.step(f"open {receptacle}")
    return obs


@mcp.tool
async def close_receptacle(receptacle: str) -> str:
    """Close a receptacle (cabinet, drawer, microwave, etc.).

    Args:
        receptacle: The receptacle to close (e.g., "cabinet 1", "drawer 2", "microwave 1")

    Returns:
        Observation string describing the result of closing the receptacle
    """
    obs, _score, _done, _info = env_manager.step(f"close {receptacle}")
    return obs


@mcp.tool
async def clean(object_name: str, receptacle: str) -> str:
    """Clean an object using a receptacle (typically a sinkbasin).

    Args:
        object_name: The name of the object to clean
        receptacle: The receptacle to use for cleaning (typically "sinkbasin 1")

    Returns:
        Observation string describing the result of cleaning the object
    """
    obs, _score, _done, _info = env_manager.step(f"clean {object_name} with {receptacle}")
    return obs


@mcp.tool
async def heat(object_name: str, receptacle: str) -> str:
    """Heat an object using a receptacle (typically a microwave).

    Args:
        object_name: The name of the object to heat
        receptacle: The receptacle to use for heating (typically "microwave 1")

    Returns:
        Observation string describing the result of heating the object
    """
    obs, _score, _done, _info = env_manager.step(f"heat {object_name} with {receptacle}")
    return obs


@mcp.tool
async def cool(object_name: str, receptacle: str) -> str:
    """Cool an object using a receptacle (typically a fridge).

    Args:
        object_name: The name of the object to cool
        receptacle: The receptacle to use for cooling (typically "fridge 1")

    Returns:
        Observation string describing the result of cooling the object
    """
    obs, _score, _done, _info = env_manager.step(f"cool {object_name} with {receptacle}")
    return obs


@mcp.tool
async def use(object_name: str) -> str:
    """Use/toggle an object (typically a lamp or switch).

    Args:
        object_name: The name of the object to use/toggle (e.g., "desklamp 1", "lightswitch 1")

    Returns:
        Observation string describing the result of using the object
    """
    obs, _score, _done, _info = env_manager.step(f"use {object_name}")
    return obs


@mcp.tool
async def examine(object_name: str) -> str:
    """Examine an object or receptacle to see details.

    Args:
        object_name: The name of the object or receptacle to examine

    Returns:
        Observation string describing what you see when examining
    """
    obs, _score, _done, _info = env_manager.step(f"examine {object_name}")
    return obs


@mcp.tool
async def inventory() -> str:
    """Check what objects you are currently holding.

    Returns:
        Observation string listing the objects in your inventory
    """
    obs, _score, _done, _info = env_manager.step("inventory")
    return obs


# Meta tool for task completion signaling

@mcp.tool
async def task_completed(success: bool, summary: str) -> str:
    """Signal that you have completed the task and the episode should end.

    This tool does NOT interact with the environment. It only signals to the
    agent loop that you believe the task is complete.

    Args:
        success: Whether you believe you successfully completed the task objective
        summary: A brief summary of what you did and why you think the task is complete

    Returns:
        Confirmation message
    """
    env_manager.task_completed_flag = True
    env_manager.task_completed_success = success
    env_manager.task_completed_summary = summary

    status = "successfully" if success else "unsuccessfully"
    return f"Task marked as {status} completed. Summary: {summary}"
