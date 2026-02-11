"""FastMCP server for skill library management (used by teacher agent in Phase 3)."""

from fastmcp import FastMCP
from pathlib import Path

from src.skills.models import Skill
from src.skills.library import SkillLibrary

mcp = FastMCP(name="SkillLibraryServer")

# Module-level library instance (caller sets path before use)
library = SkillLibrary(Path("data/skills/skills.json"))


@mcp.tool()
def add_skill(name: str, principle: str, when_to_apply: str, iteration: int = 0) -> str:
    """Add a new skill to the library or overwrite an existing one.

    Args:
        name: Unique skill identifier
        principle: The core transferable insight
        when_to_apply: Conditions where this skill is relevant
        iteration: Iteration when skill was created (default: 0)

    Returns:
        Success message
    """
    skill = Skill(
        name=name,
        principle=principle,
        when_to_apply=when_to_apply,
        created_iteration=iteration
    )
    library.add_skill(skill)
    return f"Skill '{name}' added successfully."


@mcp.tool()
def update_skill(name: str, principle: str | None = None, when_to_apply: str | None = None) -> str:
    """Update fields of an existing skill.

    Args:
        name: Skill name
        principle: Updated principle (optional)
        when_to_apply: Updated when_to_apply (optional)

    Returns:
        Success message

    Raises:
        KeyError: If skill not found
    """
    kwargs = {}
    if principle is not None:
        kwargs["principle"] = principle
    if when_to_apply is not None:
        kwargs["when_to_apply"] = when_to_apply
    library.update_skill(name, **kwargs)
    return f"Skill '{name}' updated successfully."


@mcp.tool()
def remove_skill(name: str) -> str:
    """Remove a skill from the library.

    Args:
        name: Skill name

    Returns:
        Success message

    Raises:
        KeyError: If skill not found
    """
    library.remove_skill(name)
    return f"Skill '{name}' removed successfully."
