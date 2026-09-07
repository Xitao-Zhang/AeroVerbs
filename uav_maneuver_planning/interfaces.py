from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


Vector3 = Tuple[float, float, float]

# Minimum drone-center altitude above the ground plane (z = 0).
GROUND_CLEARANCE: float = 0.2


class Skill(str, Enum):
    AROUND_LEFT = "around_left"
    AROUND_RIGHT = "around_right"
    THROUGH = "through"
    OVER = "over"
    UNDER = "under"


@dataclass(frozen=True)
class PlannerParams:
    drone_radius: float = 0.15  # drone diameter is 0.3 m, so radius = 0.15 m
    clearance_margin: float = 0.2
    semantic_margin: float = 0.0
    pre_distance: float = 1.0
    post_distance: float = 1.0
    around_margin: float = 0.3

    @property
    def planning_margin(self) -> float:
        return self.drone_radius + self.clearance_margin


@dataclass(frozen=True)
class PlanningRequest:
    skill: Skill
    object_id: str
    start: Vector3
    goal: Vector3
    params: PlannerParams = field(default_factory=PlannerParams)


@dataclass(frozen=True)
class SemanticScene:
    scene_id: str
    uav_id: str
    start: Optional[Vector3]
    goal: Optional[Vector3]
    registry: object
