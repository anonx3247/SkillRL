---
phase: 03-teacher-evolution
plan: 01
subsystem: teacher
tags: [deepseek, llm, teacher, skill-evolution, trajectory-analysis]

# Dependency graph
requires:
  - phase: 02-skill-system
    provides: Trajectory and Skill models, DeepSeekClient
provides:
  - Teacher system for analyzing trajectories and proposing skill library updates
  - Generality enforcement preventing task-specific skills
  - Batch processing for efficient LLM analysis
affects: [03-02, 03-03, evolution-loop]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Teacher prompt engineering with few-shot generality constraints
    - JSON parsing with markdown code block fallback
    - Post-process regex validation for skill abstraction

key-files:
  created:
    - src/teacher/__init__.py
    - src/teacher/prompts.py
    - src/teacher/analyzer.py
  modified: []

key-decisions:
  - "Teacher uses same DeepSeek model as agent (no separate reasoning model)"
  - "Batch size default 10 trajectories per LLM call for efficiency"
  - "Reject skills mentioning specific objects/locations via regex patterns"
  - "Deduplicate proposals by (skill_name, action) tuple"

patterns-established:
  - "Teacher prompts include negative examples (BAD skills) to enforce constraints"
  - "Post-process validation as safety net for LLM failures"
  - "Trajectory compression to ~500-1000 tokens per trajectory for batch fits"

# Metrics
duration: 3min
completed: 2026-02-11
---

# Phase 3 Plan 01: Teacher Analysis System Summary

**Teacher analyzes trajectories via DeepSeek with generality enforcement, proposing abstract skill updates through regex-validated batch processing**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-11T14:09:08Z
- **Completed:** 2026-02-11T14:11:59Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Teacher system that analyzes failures and successes to propose skill library updates
- Strong generality enforcement via prompts and post-process validation
- Batch processing of trajectories (10 per call) for efficiency
- JSON output with add/update/remove actions for skill library evolution

## Task Commits

Each task was committed atomically:

1. **Task 1: Teacher prompts with generality enforcement** - `694c125` (feat)
2. **Task 2: TeacherAnalyzer with batch trajectory analysis** - `af1e502` (feat)

## Files Created/Modified
- `src/teacher/__init__.py` - Package exports for TeacherAnalyzer, SkillProposal, prompts
- `src/teacher/prompts.py` - FAILURE_ANALYSIS_PROMPT, SUCCESS_ANALYSIS_PROMPT, format_trajectory_for_teacher()
- `src/teacher/analyzer.py` - TeacherAnalyzer class with analyze_failures(), analyze_successes(), analyze_and_propose()

## Decisions Made

**Teacher prompts with negative examples:**
- Include explicit GOOD vs BAD skill examples to enforce abstraction
- "NEVER mention specific objects/locations" constraint stated prominently
- Few-shot learning guides LLM to abstract principles

**Post-process validation as safety net:**
- Regex patterns catch task-specific references (cabinet 3, tomato 1, etc)
- Reject proposals containing ALFWorld object/location patterns
- Log warnings for rejected skills for debugging

**Batch processing strategy:**
- Default batch_size=10 (tunable parameter)
- Process failures and successes separately
- Deduplicate proposals by (skill_name, action) to avoid redundancy

**JSON parsing with fallback:**
- Try direct JSON parse first
- Fall back to extracting from markdown code blocks
- Skip batches that fail parsing (log warning)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for 03-02 (Skill Library Evolution):**
- TeacherAnalyzer provides SkillProposal objects (add/update/remove)
- Proposals are validated for generality
- Can process arbitrary trajectory batches

**Concerns:**
- Teacher prompt effectiveness untested until integrated with real trajectories
- Regex validation patterns may need refinement based on LLM output quality
- Batch size (10) may need tuning based on trajectory length distribution

---
*Phase: 03-teacher-evolution*
*Completed: 2026-02-11*
