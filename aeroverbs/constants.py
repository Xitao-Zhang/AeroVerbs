"""Shared constants for AeroVerbs.

The skill vocabulary is the stable interface between the high-level agent and
the low-level policy. Ordering here defines the `skill_id` integers used by the
SkillEncoder, so it must stay consistent between data loading and inference.
"""

from typing import Dict, List

SKILLS: List[str] = [
    "around_left",
    "around_right",
    "through",
    "over",
    "under",
]

SKILL_TO_ID: Dict[str, int] = {skill: i for i, skill in enumerate(SKILLS)}
ID_TO_SKILL: Dict[int, str] = {i: skill for skill, i in SKILL_TO_ID.items()}

NUM_SKILLS: int = len(SKILLS)

# Radar preprocessing channels (see maneuver_policy_network_v1.md §1.3).
RADAR_INV_RANGE_CH = 0
RADAR_VALID_MASK_CH = 1
RADAR_ELEVATION_CH = 2
