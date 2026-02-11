"""Teacher analyzer for proposing skill library updates from trajectories."""

import json
import re
import logging
from dataclasses import dataclass

from src.agent.client import DeepSeekClient
from src.trajectory.models import Trajectory
from src.skills.models import Skill
from src.teacher.prompts import (
    FAILURE_ANALYSIS_PROMPT,
    SUCCESS_ANALYSIS_PROMPT,
    format_trajectory_for_teacher,
)


logger = logging.getLogger(__name__)


@dataclass
class SkillProposal:
    """A proposed skill library update from teacher analysis.

    Attributes:
        action: Type of update ("add", "update", "remove")
        skill_name: Name of the skill
        principle: Core transferable insight (empty for "remove")
        when_to_apply: Conditions for applying skill (empty for "remove")
        reason: Justification for the proposal
        old_skill_name: For "update" actions, name of skill being updated
    """
    action: str  # "add", "update", "remove"
    skill_name: str
    principle: str
    when_to_apply: str
    reason: str
    old_skill_name: str | None = None


class TeacherAnalyzer:
    """Analyzes agent trajectories and proposes skill library updates."""

    def __init__(self, client: DeepSeekClient | None = None):
        """Initialize teacher analyzer.

        Args:
            client: DeepSeek client (creates new one if not provided)
        """
        self.client = client or DeepSeekClient()

    async def analyze_failures(
        self,
        trajectories: list[Trajectory],
        existing_skills: list[Skill],
        batch_size: int = 10,
    ) -> list[SkillProposal]:
        """Analyze failed trajectories and propose skill updates.

        Args:
            trajectories: List of trajectories to analyze
            existing_skills: Current skill library for context
            batch_size: Number of trajectories per LLM call

        Returns:
            List of skill proposals from failure analysis
        """
        # Filter to failures only
        failures = [t for t in trajectories if not t.success]
        if not failures:
            logger.info("No failures to analyze")
            return []

        logger.info(f"Analyzing {len(failures)} failed trajectories in batches of {batch_size}")

        all_proposals = []

        # Process in batches
        for i in range(0, len(failures), batch_size):
            batch = failures[i:i + batch_size]
            logger.info(f"Processing failure batch {i // batch_size + 1} ({len(batch)} trajectories)")

            # Format batch for teacher
            batch_text = "\n\n---\n\n".join(
                format_trajectory_for_teacher(t) for t in batch
            )

            # Build context message with existing skills
            skill_context = "\n".join(
                f"- {skill.name}: {skill.principle} "
                f"[usage: {skill.usage_count} retrievals, "
                f"last used: iter {skill.last_used_iteration}, "
                f"created: iter {skill.created_iteration}]"
                for skill in existing_skills
            ) if existing_skills else "No skills in library yet."

            user_message = f"""Existing skills in library:
{skill_context}

Failed trajectories to analyze:

{batch_text}

Note: Usage statistics [usage: N retrievals, last used: iter M, created: iter K] show how often each skill was retrieved. Skills with 0 or very low retrievals after several iterations may not be relevant and could be candidates for removal.

Propose skill library updates based on patterns in these failures."""

            # Call teacher LLM
            try:
                response = await self.client.chat(
                    messages=[
                        {"role": "system", "content": FAILURE_ANALYSIS_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0.7,
                )

                # Parse JSON response
                content = response.choices[0].message.content
                proposals = self._parse_proposals(content)

                # Validate and filter proposals
                valid_proposals = [
                    p for p in proposals if self._validate_proposal(p)
                ]

                all_proposals.extend(valid_proposals)
                logger.info(f"Batch {i // batch_size + 1}: {len(valid_proposals)} valid proposals")

            except Exception as e:
                logger.warning(f"Failed to process failure batch {i // batch_size + 1}: {e}")
                continue

        return all_proposals

    async def analyze_successes(
        self,
        trajectories: list[Trajectory],
        existing_skills: list[Skill],
        batch_size: int = 10,
    ) -> list[SkillProposal]:
        """Analyze successful trajectories and propose skill updates.

        Args:
            trajectories: List of trajectories to analyze
            existing_skills: Current skill library for context
            batch_size: Number of trajectories per LLM call

        Returns:
            List of skill proposals from success analysis
        """
        # Filter to successes only
        successes = [t for t in trajectories if t.success]
        if not successes:
            logger.info("No successes to analyze")
            return []

        logger.info(f"Analyzing {len(successes)} successful trajectories in batches of {batch_size}")

        all_proposals = []

        # Process in batches
        for i in range(0, len(successes), batch_size):
            batch = successes[i:i + batch_size]
            logger.info(f"Processing success batch {i // batch_size + 1} ({len(batch)} trajectories)")

            # Format batch for teacher
            batch_text = "\n\n---\n\n".join(
                format_trajectory_for_teacher(t) for t in batch
            )

            # Build context message with existing skills
            skill_context = "\n".join(
                f"- {skill.name}: {skill.principle} "
                f"[usage: {skill.usage_count} retrievals, "
                f"last used: iter {skill.last_used_iteration}, "
                f"created: iter {skill.created_iteration}]"
                for skill in existing_skills
            ) if existing_skills else "No skills in library yet."

            user_message = f"""Existing skills in library:
{skill_context}

Successful trajectories to analyze:

{batch_text}

Note: Usage statistics [usage: N retrievals, last used: iter M, created: iter K] show how often each skill was retrieved. Skills with 0 or very low retrievals after several iterations may not be relevant and could be candidates for removal.

Propose skill library updates based on patterns in these successes."""

            # Call teacher LLM
            try:
                response = await self.client.chat(
                    messages=[
                        {"role": "system", "content": SUCCESS_ANALYSIS_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0.7,
                )

                # Parse JSON response
                content = response.choices[0].message.content
                proposals = self._parse_proposals(content)

                # Validate and filter proposals
                valid_proposals = [
                    p for p in proposals if self._validate_proposal(p)
                ]

                all_proposals.extend(valid_proposals)
                logger.info(f"Batch {i // batch_size + 1}: {len(valid_proposals)} valid proposals")

            except Exception as e:
                logger.warning(f"Failed to process success batch {i // batch_size + 1}: {e}")
                continue

        return all_proposals

    async def analyze_and_propose(
        self,
        trajectories: list[Trajectory],
        existing_skills: list[Skill],
        batch_size: int = 10,
    ) -> list[SkillProposal]:
        """Analyze all trajectories (failures and successes) and propose updates.

        Args:
            trajectories: List of trajectories to analyze
            existing_skills: Current skill library for context
            batch_size: Number of trajectories per LLM call

        Returns:
            Deduplicated list of skill proposals from both analyses
        """
        logger.info(f"Analyzing {len(trajectories)} trajectories (failures + successes)")

        # Run both analyses
        failure_proposals = await self.analyze_failures(
            trajectories, existing_skills, batch_size
        )
        success_proposals = await self.analyze_successes(
            trajectories, existing_skills, batch_size
        )

        # Combine and deduplicate
        all_proposals = failure_proposals + success_proposals
        deduplicated = self._deduplicate_proposals(all_proposals)

        logger.info(
            f"Total proposals: {len(all_proposals)} "
            f"(failures: {len(failure_proposals)}, successes: {len(success_proposals)}), "
            f"after deduplication: {len(deduplicated)}"
        )

        return deduplicated

    def _parse_proposals(self, content: str) -> list[SkillProposal]:
        """Parse JSON proposals from LLM response.

        Args:
            content: LLM response content

        Returns:
            List of SkillProposal objects (empty if parsing fails)
        """
        try:
            # Try direct JSON parse
            data = json.loads(content)
            return [SkillProposal(**item) for item in data]
        except json.JSONDecodeError:
            # Try extracting from markdown code block
            match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    return [SkillProposal(**item) for item in data]
                except (json.JSONDecodeError, TypeError):
                    pass

            logger.warning("Failed to parse proposals from LLM response")
            return []

    def _validate_proposal(self, proposal: SkillProposal) -> bool:
        """Validate proposal for generality constraints.

        Args:
            proposal: Proposal to validate

        Returns:
            True if proposal passes validation, False if rejected
        """
        # Skip validation for remove actions (only need name)
        if proposal.action == "remove":
            return True

        # Check principle and when_to_apply for task-specific references
        text_to_check = f"{proposal.principle} {proposal.when_to_apply}"

        # Pattern 1: Locations/receptacles with numbers
        location_pattern = r"\b(cabinet|drawer|shelf|countertop|fridge|microwave|sinkbasin|desk|bed|sofa|toilet|bathtub|garbagecan)\s+\d+\b"
        if re.search(location_pattern, text_to_check, re.IGNORECASE):
            logger.warning(
                f"Rejected proposal '{proposal.skill_name}': "
                f"contains specific location (e.g., 'cabinet 3')"
            )
            return False

        # Pattern 2: Objects with numbers
        object_pattern = r"\b(tomato|apple|potato|lettuce|bread|egg|mug|cup|plate|bowl|fork|knife|spoon|pen|pencil|book|cd|cellphone|laptop|pillow|cloth|soapbar|spraybottle|candle|alarmclock|vase|statue|box|keychain|creditcard|remotecontrol|watch|tissuebox|toiletpaper|plunger|scrubbrush|dishsponge|spatula|ladle|butterknife)\s+\d+\b"
        if re.search(object_pattern, text_to_check, re.IGNORECASE):
            logger.warning(
                f"Rejected proposal '{proposal.skill_name}': "
                f"contains specific object instance (e.g., 'tomato 1')"
            )
            return False

        return True

    def _deduplicate_proposals(self, proposals: list[SkillProposal]) -> list[SkillProposal]:
        """Deduplicate proposals with same skill_name and action.

        Args:
            proposals: List of proposals to deduplicate

        Returns:
            Deduplicated list (keeps first occurrence of each duplicate)
        """
        seen = set()
        deduplicated = []

        for proposal in proposals:
            key = (proposal.skill_name, proposal.action)
            if key not in seen:
                seen.add(key)
                deduplicated.append(proposal)

        return deduplicated
