"""Skill library with CRUD operations and atomic JSON persistence."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .models import Skill


class SkillLibrary:
    """Manages a collection of skills with persistent storage.

    Uses atomic write pattern (temp file + fsync + os.replace) for crash resilience.
    """

    def __init__(self, storage_path: Path):
        """Initialize skill library.

        Args:
            storage_path: Path to JSON file for persistent storage
        """
        self.storage_path = storage_path
        self.skills: dict[str, Skill] = {}

    def load(self) -> None:
        """Load skills from JSON file.

        Creates empty library if file doesn't exist.
        """
        if not self.storage_path.exists():
            self.skills = {}
            return

        with open(self.storage_path, 'r') as f:
            data = json.load(f)

        self.skills = {name: Skill.from_dict(skill_data) for name, skill_data in data.items()}

    def save(self) -> None:
        """Save skills to JSON file atomically.

        Uses temp file + fsync + os.replace pattern for crash resilience.
        """
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert skills to dict format
        data = {name: skill.to_dict() for name, skill in self.skills.items()}

        # Atomic write: temp file + fsync + replace
        temp_path = self.storage_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, self.storage_path)

    def add_skill(self, skill: Skill) -> None:
        """Add or overwrite a skill.

        Args:
            skill: Skill to add
        """
        self.skills[skill.name] = skill
        self.save()

    def update_skill(self, name: str, **kwargs) -> None:
        """Update specific fields of an existing skill.

        Args:
            name: Skill name
            **kwargs: Fields to update (principle, when_to_apply, etc.)

        Raises:
            KeyError: If skill not found
        """
        if name not in self.skills:
            raise KeyError(f"Skill '{name}' not found")

        skill = self.skills[name]
        for key, value in kwargs.items():
            if hasattr(skill, key):
                setattr(skill, key, value)

        self.save()

    def remove_skill(self, name: str) -> None:
        """Remove a skill.

        Args:
            name: Skill name

        Raises:
            KeyError: If skill not found
        """
        if name not in self.skills:
            raise KeyError(f"Skill '{name}' not found")

        del self.skills[name]
        self.save()

    def get_all_skills(self) -> list[Skill]:
        """Get all skills as a list.

        Returns:
            List of all skills
        """
        return list(self.skills.values())

    def __len__(self) -> int:
        """Number of skills in library."""
        return len(self.skills)
