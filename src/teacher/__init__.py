"""Teacher module for analyzing trajectories and proposing skill updates."""

from src.teacher.prompts import (
    FAILURE_ANALYSIS_PROMPT,
    SUCCESS_ANALYSIS_PROMPT,
    format_trajectory_for_teacher,
)

__all__ = [
    "FAILURE_ANALYSIS_PROMPT",
    "SUCCESS_ANALYSIS_PROMPT",
    "format_trajectory_for_teacher",
]
