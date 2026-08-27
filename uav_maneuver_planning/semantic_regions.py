import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .interfaces import GROUND_CLEARANCE, PlannerParams, PlanningRequest, SemanticScene, Skill, Vector3
from .registry import SemanticObject
from .voxel_occupancy import occupancy_for_object


@dataclass(frozen=True)
class SemanticFreeRegion:
    skill: Skill
    object_id: str
    clearance: float
    points: Tuple[Vector3, ...]
    description: str

    def sample_guides(self, num: int, seed: Optional[int] = None) -> List[Vector3]:
        if num <= 0:
            return []
        if not self.points:
            return []
        rng = np.random.default_rng(seed)
        count = min(num, len(self.points))
        indices = rng.choice(len(self.points), size=count, replace=False)
        return [self.points[int(index)] for index in indices]

    def to_json_dict(self) -> dict:
        return {
            "skill": self.skill.value,
            "object_id": self.object_id,
            "clearance": self.clearance,
            "description": self.description,
            "num_points": len(self.points),
            "points": [list(point) for point in self.points],
        }

    @classmethod
    def from_json_dict(cls, payload: dict) -> "SemanticFreeRegion":
        return cls(
            skill=Skill(payload["skill"]),
            object_id=str(payload["object_id"]),
            clearance=float(payload["clearance"]),
            description=str(payload.get("description", "")),
            points=tuple(tuple(float(value) for value in point[:3]) for point in payload.get("points", [])),
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SemanticFreeRegion":
        return cls.from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_semantic_free_region(
    scene: SemanticScene,
    request: PlanningRequest,
    resolution: float = 0.2,
    z_padding: float = 0.8,
    max_points: Optional[int] = None,
) -> SemanticFreeRegion:
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")

    target = scene.registry.get(request.object_id)
    params = request.params or PlannerParams()
    clearance = params.planning_margin
    candidates = _candidate_points_for_skill(scene, request, target, resolution, z_padding)
    free_points = [
        point
        for point in candidates
        if _point_satisfies_skill(point, request, target, clearance)
        and _point_has_clearance(point, scene.registry, clearance, pass_through_target=target if request.skill == Skill.THROUGH else None)
    ]
    free_points = _stable_unique_points(free_points, precision=6)
    if max_points is not None and max_points > 0 and len(free_points) > max_points:
        rng = np.random.default_rng(0)
        indices = sorted(rng.choice(len(free_points), size=max_points, replace=False))
        free_points = [free_points[int(index)] for index in indices]

    return SemanticFreeRegion(
        skill=request.skill,
        object_id=request.object_id,
        clearance=clearance,
        points=tuple(free_points),
        description=_description_for_skill(request.skill),
    )


def _candidate_points_for_skill(
    scene: SemanticScene,
    request: PlanningRequest,
    target: SemanticObject,
    resolution: float,
    z_padding: float,
) -> List[Vector3]:
    bbox = _object_bbox(target)
    footprint = _object_footprint_radius(target)
    clearance = request.params.planning_margin
    start = np.asarray(request.start, dtype=float)
    goal = np.asarray(request.goal, dtype=float)
    center = np.asarray(target.center, dtype=float)
    forward = _normalize_2d(goal[:2] - start[:2])
    if forward is None:
        forward = np.array([1.0, 0.0], dtype=float)
    left = np.array([-forward[1], forward[0]], dtype=float)

    if request.skill == Skill.UNDER:
        if bbox is None:
            return []
        min_pt = np.array([bbox[0][0], bbox[0][1], bbox[0][2] + clearance], dtype=float)
        max_pt = np.array([bbox[1][0], bbox[1][1], bbox[1][2] - clearance], dtype=float)
        return _grid_points(min_pt, max_pt, resolution)

    if request.skill == Skill.OVER:
        top_z = target.top_z
        if top_z is None:
            top_z = bbox[1][2] if bbox is not None else center[2] + (_object_height(target) or footprint) / 2.0
        xy_margin = footprint + request.params.around_margin
        min_pt = np.array([center[0] - xy_margin, center[1] - xy_margin, top_z + clearance], dtype=float)
        max_pt = np.array([center[0] + xy_margin, center[1] + xy_margin, top_z + clearance + z_padding], dtype=float)
        return _grid_points(min_pt, max_pt, resolution)

    if request.skill in {Skill.AROUND_LEFT, Skill.AROUND_RIGHT}:
        side_inner = footprint + clearance
        side_outer = footprint + clearance + max(1.0, request.params.around_margin * 3.0)
        along = footprint + request.params.pre_distance + request.params.post_distance
        z_half = max(0.2, z_padding * 0.5)
        z_min = max(start[2] - z_half, GROUND_CLEARANCE)
        z_max = start[2] + z_half
        points = []
        for a in _axis_values(-along, along, resolution):
            for s in _axis_values(side_inner, side_outer, resolution):
                signed_s = s if request.skill == Skill.AROUND_LEFT else -s
                xy = center[:2] + forward * a + left * signed_s
                for z in _axis_values(z_min, z_max, resolution):
                    points.append((float(xy[0]), float(xy[1]), float(z)))
        return points

    if request.skill == Skill.THROUGH:
        opening_center = np.asarray(target.opening_center or target.center, dtype=float)
        normal = _normalize_3d(np.asarray(target.normal or (1.0, 0.0, 0.0), dtype=float))
        width = target.width or footprint * 2.0
        height = target.clearance_height or target.height or _object_height(target) or 2.0
        half_width = max(0.0, width / 2.0 - clearance)
        half_height = max(0.0, height / 2.0 - clearance)
        if half_width <= 0.0 or half_height <= 0.0:
            return []
        lateral = _normalize_3d(np.cross(normal, np.array([0.0, 0.0, 1.0], dtype=float)))
        if np.linalg.norm(lateral) < 1e-8:
            lateral = np.array([0.0, 1.0, 0.0], dtype=float)
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        thickness = max(resolution, clearance)
        points = []
        for n in _axis_values(-thickness, thickness, resolution):
            for l in _axis_values(-half_width, half_width, resolution):
                for u in _axis_values(-half_height, half_height, resolution):
                    point = opening_center + normal * n + lateral * l + up * u
                    points.append(tuple(float(value) for value in point[:3]))
        return points

    return []


def _point_satisfies_skill(point: Vector3, request: PlanningRequest, target: SemanticObject, clearance: float) -> bool:
    bbox = _object_bbox(target)
    if request.skill == Skill.UNDER:
        if bbox is None:
            return False
        return (
            bbox[0][0] <= point[0] <= bbox[1][0]
            and bbox[0][1] <= point[1] <= bbox[1][1]
            and bbox[0][2] + clearance <= point[2] <= bbox[1][2] - clearance
        )
    if request.skill == Skill.OVER:
        top_z = target.top_z if target.top_z is not None else (bbox[1][2] if bbox else target.center[2])
        return point[2] >= top_z + clearance
    if request.skill == Skill.THROUGH:
        return True
    if request.skill in {Skill.AROUND_LEFT, Skill.AROUND_RIGHT}:
        start = np.asarray(request.start, dtype=float)
        goal = np.asarray(request.goal, dtype=float)
        forward = _normalize_2d(goal[:2] - start[:2])
        if forward is None:
            return False
        left = np.array([-forward[1], forward[0]], dtype=float)
        rel = np.asarray(point[:2], dtype=float) - np.asarray(target.center[:2], dtype=float)
        side = float(np.dot(rel, left))
        footprint = _object_footprint_radius(target)
        distance = float(np.linalg.norm(rel))
        if distance < footprint + clearance:
            return False
        if request.skill == Skill.AROUND_LEFT:
            return side > footprint * 0.25
        return side < -footprint * 0.25
    return False


def _point_has_clearance(
    point: Vector3,
    objects: Iterable[SemanticObject],
    clearance: float,
    pass_through_target: Optional[SemanticObject] = None,
) -> bool:
    return not any(
        _point_collides_object(point, obj, clearance)
        for obj in objects
        if pass_through_target is None or obj.object_id != pass_through_target.object_id
    )


def _point_collides_object(point: Vector3, obj: SemanticObject, margin: float) -> bool:
    occupancy = occupancy_for_object(obj)
    if occupancy is not None:
        return occupancy.contains_point(point, margin=margin)
    bbox = _object_bbox(obj)
    if bbox is not None:
        return all(bbox[0][i] - margin <= point[i] <= bbox[1][i] + margin for i in range(3))
    if obj.radius is None:
        return False
    xy_collision = math.hypot(point[0] - obj.center[0], point[1] - obj.center[1]) <= obj.radius + margin
    height = _object_height(obj)
    if height is None:
        return xy_collision
    return xy_collision and obj.center[2] - height / 2.0 - margin <= point[2] <= obj.center[2] + height / 2.0 + margin


def _grid_points(min_pt: np.ndarray, max_pt: np.ndarray, resolution: float) -> List[Vector3]:
    if np.any(max_pt < min_pt):
        return []
    points = []
    for x in _axis_values(min_pt[0], max_pt[0], resolution):
        for y in _axis_values(min_pt[1], max_pt[1], resolution):
            for z in _axis_values(min_pt[2], max_pt[2], resolution):
                points.append((float(x), float(y), float(z)))
    return points


def _axis_values(start: float, stop: float, step: float) -> np.ndarray:
    if stop < start:
        return np.array([], dtype=float)
    count = int(math.floor((stop - start) / step + 1e-9)) + 1
    values = start + np.arange(count, dtype=float) * step
    if len(values) == 0 or values[-1] < stop - step * 0.5:
        values = np.append(values, stop)
    return values


def _stable_unique_points(points: Sequence[Vector3], precision: int) -> List[Vector3]:
    seen = set()
    unique = []
    for point in points:
        key = tuple(round(float(value), precision) for value in point)
        if key in seen:
            continue
        seen.add(key)
        unique.append(tuple(float(value) for value in point))
    return unique


def _object_bbox(obj: SemanticObject) -> Optional[Tuple[Vector3, Vector3]]:
    if obj.bbox_min is not None and obj.bbox_max is not None:
        return obj.bbox_min, obj.bbox_max
    occupancy = occupancy_for_object(obj)
    if occupancy is not None:
        return occupancy.bbox_min, occupancy.bbox_max
    return None


def _object_footprint_radius(obj: SemanticObject) -> float:
    if obj.radius is not None:
        return float(obj.radius)
    bbox = _object_bbox(obj)
    if bbox is not None:
        return math.hypot(bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1]) / 2.0
    if obj.width is not None:
        return float(obj.width) / 2.0
    return 0.5


def _object_height(obj: SemanticObject) -> Optional[float]:
    if obj.height is not None:
        return float(obj.height)
    bbox = _object_bbox(obj)
    if bbox is not None:
        return bbox[1][2] - bbox[0][2]
    return None


def _normalize_2d(vector) -> Optional[np.ndarray]:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return None
    return vector / norm


def _normalize_3d(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return vector / norm


def _description_for_skill(skill: Skill) -> str:
    return {
        Skill.UNDER: "Free points inside the target footprint and below the target top volume.",
        Skill.OVER: "Free points above the target top with UAV clearance.",
        Skill.THROUGH: "Free points inside a pass-through opening volume.",
        Skill.AROUND_LEFT: "Free points on the target's left-side flyable region.",
        Skill.AROUND_RIGHT: "Free points on the target's right-side flyable region.",
    }.get(skill, "Semantic free-flight region.")
