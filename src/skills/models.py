"""Skill data models."""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class Skill:
    """A transferable skill learned from agent experience.

    Attributes:
        name: Unique skill identifier
        principle: The core transferable insight
        when_to_apply: Conditions where this skill is relevant
        created_iteration: Iteration when skill was created
        last_used_iteration: Last iteration skill was retrieved
        usage_count: Total times retrieved
    """
    name: str
    principle: str
    when_to_apply: str
    created_iteration: int = 0
    last_used_iteration: int = 0
    usage_count: int = 0

    def to_dict(self) -> dict:
        """Convert skill to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Skill:
        """Create skill from dictionary."""
        return cls(**data)
