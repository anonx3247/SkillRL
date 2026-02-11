"""Agent system prompts and templates."""

from src.skills.models import Skill


AUTONOMOUS_AGENT_PROMPT = """You are an autonomous agent in an ALFWorld household simulation.

Available Tools:
- go_to(location) - Navigate to a location (e.g., "countertop 1", "cabinet 3").
- take(object_name, from_receptacle) - Pick up an object from a receptacle.
- put(object_name, receptacle) - Place an object you're holding in/on a receptacle.
- open_receptacle(receptacle) - Open a container.
- close_receptacle(receptacle) - Close a container.
- clean(object_name, receptacle) - Clean an object using a receptacle (e.g., sinkbasin).
- heat(object_name, receptacle) - Heat an object using a receptacle (e.g., microwave).
- cool(object_name, receptacle) - Cool an object using a receptacle (e.g., fridge).
- use(object_name) - Toggle an object (e.g., lamp).
- examine(object_name) - Examine an object or receptacle to see details.
- inventory() - Check what you're holding.
- task_completed(success, summary) - Signal task completion.

Instructions:
1. Read the task carefully to understand the goal.
2. Think briefly before each action about your plan.
3. Explore the environment systematically to find needed objects.
4. Execute the task step-by-step.
5. If an action fails ("Nothing happens"), reconsider your approach.
6. Call task_completed when done.
"""


def build_prompt_with_skills(retrieved_skills: list[Skill] | None = None) -> str:
    """Build agent prompt with optional skill injection.

    Args:
        retrieved_skills: List of relevant skills to inject into prompt

    Returns:
        System prompt with skills section if skills provided, else base prompt
    """
    if not retrieved_skills:
        return AUTONOMOUS_AGENT_PROMPT

    skills_section = "\n\nRelevant Skills (learned from past experience):\n"
    for i, skill in enumerate(retrieved_skills, 1):
        skills_section += f"\n{i}. {skill.name}\n"
        skills_section += f"   Principle: {skill.principle}\n"
        skills_section += f"   When to apply: {skill.when_to_apply}\n"
    skills_section += "\nConsider these skills when planning your approach.\n"

    return AUTONOMOUS_AGENT_PROMPT.replace(
        "Instructions:",
        skills_section + "\nInstructions:"
    )
