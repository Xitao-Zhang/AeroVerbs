import heapq
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .fast_planner import FastPlannerConfig, FastPlannerManager
from .interfaces import GROUND_CLEARANCE, PlannerParams, PlanningRequest, SemanticScene, Skill, Vector3
from .registry import SemanticObject
from .semantic_regions import SemanticFreeRegion, build_semantic_free_region
from .voxel_occupancy import occupancy_for_object


GridNode = Tuple[int, int]


class SemanticLabel(str, Enum):
    AROUND_LEFT = "around_left"
    AROUND_RIGHT = "around_right"
    THROUGH = "through"
    OVER = "over"
    UNDER = "under"
    INVALID = "invalid"


@dataclass(frozen=True)
class Trajectory:
    points: Tuple[Vector3, ...]


@dataclass(frozen=True)
class TrajectoryPlan:
    trajectory: Trajectory
    semantic_label: SemanticLabel
    object_id: str


def plan_grid_trajectory(
    scene: SemanticScene,
    request: PlanningRequest,
    grid_resolution: float = 0.2,
    bounds_margin: float = 1.5,
    semantic_check: bool = True,
) -> TrajectoryPlan:
    """Plan a fixed-altitude XY path through the scene, then classify it semantically."""
    if grid_resolution <= 0.0:
        raise ValueError("grid_resolution must be positive")

    target = scene.registry.get(request.object_id)
    obstacles = _solid_obstacles(scene.registry)
    obstacle_func = _make_obstacle_func(obstacles, request.params.planning_margin)
    bounds = _make_bounds(request.start, request.goal, scene.registry, bounds_margin)
    preferred_label = _preferred_label_from_skill(request.skill)

    grid_path = _astar_grid_path(
        start=request.start,
        goal=request.goal,
        target=target,
        obstacles=obstacles,
        obstacle_func=obstacle_func,
        grid_resolution=grid_resolution,
        bounds=bounds,
        preferred_label=preferred_label,
    )
    trajectory = Trajectory(
        points=tuple(_grid_path_to_world_path(grid_path, request.start, request.goal, grid_resolution, bounds))
    )
    semantic_label = _classify_planned_points(trajectory.points, target, request, scene, semantic_check)
    return TrajectoryPlan(trajectory=trajectory, semantic_label=semantic_label, object_id=request.object_id)


def plan_visfly_prm_trajectory(
    scene: SemanticScene,
    request: PlanningRequest,
    num_samples: int = 500,
    k: int = 10,
    bounds_margin: float = 1.5,
    max_attempts: int = 30,
    seed: int = 7,
    interpolation_step: float = 0.15,
    semantic_check: bool = True,
) -> TrajectoryPlan:
    """Plan candidate paths with VisFly's PRMPlanner and keep the first semantically valid one."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    target = scene.registry.get(request.object_id)
    obstacles = _solid_obstacles(scene.registry)
    bounds = _make_prm_bounds(request.start, request.goal, scene.registry, bounds_margin)
    obstacle_func = _make_obstacle_func(obstacles, request.params.planning_margin)
    prm_planner_class = _load_visfly_prm_planner()
    expected_label = _preferred_label_from_skill(request.skill)
    last_label = SemanticLabel.INVALID

    for attempt in range(max_attempts):
        raw_path = _run_visfly_prm_attempt(
            prm_planner_class=prm_planner_class,
            bounds=bounds,
            num_samples=num_samples,
            obstacle_func=obstacle_func,
            k=k,
            start=request.start,
            goal=request.goal,
            seed=seed + attempt,
        )
        points = _densify_polyline(raw_path, interpolation_step)
        if not _path_is_collision_free(points, obstacle_func):
            continue

        semantic_label = classify_trajectory(
            points,
            target=target,
            skill=request.skill,
            start=request.start,
            goal=request.goal,
            params=request.params,
            scene=scene if semantic_check else None,
        )
        last_label = semantic_label
        if not semantic_check:
            return TrajectoryPlan(
                trajectory=Trajectory(points=tuple(points)),
                semantic_label=semantic_label,
                object_id=request.object_id,
            )
        if expected_label is None or semantic_label == expected_label:
            return TrajectoryPlan(
                trajectory=Trajectory(points=tuple(points)),
                semantic_label=semantic_label,
                object_id=request.object_id,
            )

    expected_text = expected_label.value if expected_label is not None else request.skill.value
    raise RuntimeError(
        f"VisFly PRM did not find a {expected_text} trajectory after {max_attempts} attempts; "
        f"last semantic label was {last_label.value}"
    )


def plan_fast_esdf_trajectory(
    scene: SemanticScene,
    request: PlanningRequest,
    config: Optional[FastPlannerConfig] = None,
    semantic_check: bool = True,
) -> TrajectoryPlan:
    """Plan with a non-ROS Fast-Planner-style ESDF + kinodynamic A* + B-spline pipeline."""
    target = scene.registry.get(request.object_id)
    obstacles = _planning_obstacles_for_skill(scene.registry, request)
    config = config or FastPlannerConfig(clearance=request.params.planning_margin)
    manager = FastPlannerManager(scene, config, obstacles=obstacles)
    expected_label = _preferred_label_from_skill(request.skill)
    last_label = SemanticLabel.INVALID
    last_points: Optional[List[Vector3]] = None
    last_error: Optional[Exception] = None

    for route in _fast_planner_candidate_routes(request, target, manager, scene):
        try:
            points = manager.plan_waypoints(route)
        except RuntimeError as exc:
            last_error = exc
            continue
        if not _path_above_ground(points):
            last_error = RuntimeError("path dips below ground clearance")
            continue
        last_points = points
        semantic_label = classify_trajectory(
            points,
            target=target,
            skill=request.skill,
            start=request.start,
            goal=request.goal,
            params=request.params,
            scene=scene if semantic_check else None,
        )
        last_label = semantic_label
        if not semantic_check:
            return TrajectoryPlan(
                trajectory=Trajectory(points=tuple(points)),
                semantic_label=semantic_label,
                object_id=request.object_id,
            )
        if expected_label is None or semantic_label == expected_label:
            return TrajectoryPlan(
                trajectory=Trajectory(points=tuple(points)),
                semantic_label=semantic_label,
                object_id=request.object_id,
            )

    if expected_label is None and last_points is not None:
        return TrajectoryPlan(
            trajectory=Trajectory(points=tuple(last_points)),
            semantic_label=last_label,
            object_id=request.object_id,
        )
    expected_text = expected_label.value if expected_label is not None else request.skill.value
    detail = f"; last planner error was {last_error}" if last_error is not None else ""
    raise RuntimeError(
        f"Fast-Planner ESDF did not find a {expected_text} trajectory; "
        f"last semantic label was {last_label.value}{detail}"
    )


def classify_trajectory(
    points: Iterable[Vector3],
    target: SemanticObject,
    skill: Skill,
    start: Vector3,
    goal: Vector3,
    params: Optional[PlannerParams] = None,
    scene: Optional[SemanticScene] = None,
    region: Optional[SemanticFreeRegion] = None,
) -> SemanticLabel:
    params = params or PlannerParams()
    path = tuple(points)
    region = region or _build_validation_region(scene, target, skill, start, goal, params)
    if region is not None:
        expected = _preferred_label_from_skill(skill)
        if expected is not None and validate_semantic_path(path, region, target, start, goal, params, scene=scene):
            return expected
        return SemanticLabel.INVALID
    if skill == Skill.AROUND_LEFT:
        return _classify_around(path, target, start, goal, SemanticLabel.AROUND_LEFT)
    if skill == Skill.AROUND_RIGHT:
        return _classify_around(path, target, start, goal, SemanticLabel.AROUND_RIGHT)
    if skill == Skill.THROUGH:
        return SemanticLabel.THROUGH if _passes_through_opening(path, target, params) else SemanticLabel.INVALID
    if skill == Skill.OVER:
        return SemanticLabel.OVER if _passes_over_object(path, target, params) else SemanticLabel.INVALID
    if skill == Skill.UNDER:
        return SemanticLabel.UNDER if _passes_under_object(path, target, params) else SemanticLabel.INVALID
    return SemanticLabel.INVALID


def _classify_planned_points(
    points: Iterable[Vector3],
    target: SemanticObject,
    request: PlanningRequest,
    scene: SemanticScene,
    semantic_check: bool,
) -> SemanticLabel:
    return classify_trajectory(
        points,
        target=target,
        skill=request.skill,
        start=request.start,
        goal=request.goal,
        params=request.params,
        scene=scene if semantic_check else None,
    )


def validate_semantic_path(
    points: Iterable[Vector3],
    region: SemanticFreeRegion,
    target: SemanticObject,
    start: Vector3,
    goal: Vector3,
    params: Optional[PlannerParams] = None,
    scene: Optional[SemanticScene] = None,
    min_region_points: int = 2,
) -> bool:
    params = params or PlannerParams()
    path = tuple(points)
    if len(path) < 2 or not region.points:
        return False
    if not _path_has_clearance(path, scene, target, params.planning_margin, region.skill):
        return False

    inside_indices = _path_region_indices(path, region)
    if len(inside_indices) < min_region_points:
        return False
    if _longest_consecutive_run(inside_indices) < min_region_points:
        return False
    return _path_region_direction_matches(path, inside_indices, region.skill, target, start, goal)


def path_is_collision_free(
    points: Iterable[Vector3],
    scene: SemanticScene,
    params: Optional[PlannerParams] = None,
) -> bool:
    params = params or PlannerParams()
    obstacle_func = _make_obstacle_func(_solid_obstacles(scene.registry), params.planning_margin)
    return _path_is_collision_free(points, obstacle_func)


def _build_validation_region(
    scene: Optional[SemanticScene],
    target: SemanticObject,
    skill: Skill,
    start: Vector3,
    goal: Vector3,
    params: PlannerParams,
) -> Optional[SemanticFreeRegion]:
    if scene is None:
        return None
    request = PlanningRequest(skill=skill, object_id=target.object_id, start=start, goal=goal, params=params)
    try:
        region = build_semantic_free_region(scene, request)
    except (ValueError, KeyError):
        return None
    return region if region.points else None


def _path_region_indices(points: Tuple[Vector3, ...], region: SemanticFreeRegion) -> List[int]:
    radius = max(region.clearance, 1e-6)
    region_points = np.asarray(region.points, dtype=float)
    indices = []
    for index, point in enumerate(points):
        point_arr = np.asarray(point, dtype=float)
        distances = np.linalg.norm(region_points - point_arr, axis=1)
        if float(distances.min()) <= radius:
            indices.append(index)
    return indices


def _longest_consecutive_run(indices: List[int]) -> int:
    if not indices:
        return 0
    best = 1
    current = 1
    for prev, item in zip(indices, indices[1:]):
        if item == prev + 1:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def _path_has_clearance(
    points: Tuple[Vector3, ...],
    scene: Optional[SemanticScene],
    target: SemanticObject,
    clearance: float,
    skill: Skill,
) -> bool:
    objects = _validation_obstacles_for_skill(scene.registry, skill, target) if scene is not None else [target]
    for obj in objects:
        if not _path_has_object_clearance(points, obj, clearance):
            return False
    return True


def _path_has_object_clearance(points: Tuple[Vector3, ...], target: SemanticObject, clearance: float) -> bool:
    for start, end in zip(points, points[1:]):
        if _segment_collides_object(start, end, target, clearance):
            return False
    return True


def _segment_collides_object(start: Vector3, end: Vector3, target: SemanticObject, clearance: float) -> bool:
    occupancy = occupancy_for_object(target)
    if occupancy is not None:
        return occupancy.collides_segment(start, end, margin=clearance)
    start_arr = np.asarray(start, dtype=float)
    end_arr = np.asarray(end, dtype=float)
    distance = float(np.linalg.norm(end_arr - start_arr))
    samples = max(1, math.ceil(distance / max(clearance * 0.5, 0.05)))
    for step in range(samples + 1):
        point = start_arr * (1.0 - step / samples) + end_arr * (step / samples)
        if _point_collides_object(tuple(float(value) for value in point[:3]), target, clearance):
            return True
    return False


def _path_region_direction_matches(
    points: Tuple[Vector3, ...],
    inside_indices: List[int],
    skill: Skill,
    target: SemanticObject,
    start: Vector3,
    goal: Vector3,
) -> bool:
    first = inside_indices[0]
    last = inside_indices[-1]
    before = np.asarray(points[max(0, first - 1)], dtype=float)
    after = np.asarray(points[min(len(points) - 1, last + 1)], dtype=float)
    travel = after - before
    if np.linalg.norm(travel) < 1e-8:
        travel = np.asarray(goal, dtype=float) - np.asarray(start, dtype=float)

    nominal = np.asarray(goal, dtype=float) - np.asarray(start, dtype=float)
    if np.linalg.norm(nominal) < 1e-8:
        return False
    nominal = nominal / np.linalg.norm(nominal)

    if skill in {Skill.UNDER, Skill.OVER, Skill.THROUGH}:
        return float(np.dot(travel, nominal)) > 0.0

    forward_xy = np.asarray(goal[:2], dtype=float) - np.asarray(start[:2], dtype=float)
    norm = float(np.linalg.norm(forward_xy))
    if norm < 1e-8:
        return False
    left_xy = np.array([-forward_xy[1] / norm, forward_xy[0] / norm], dtype=float)
    region_points = [points[index] for index in inside_indices]
    side_values = [
        float(np.dot(np.asarray(point[:2], dtype=float) - np.asarray(target.center[:2], dtype=float), left_xy))
        for point in region_points
    ]
    if skill == Skill.AROUND_LEFT:
        return min(side_values) > 0.0
    if skill == Skill.AROUND_RIGHT:
        return max(side_values) < 0.0
    return False


def _path_above_ground(points: Iterable[Vector3], min_z: float = GROUND_CLEARANCE) -> bool:
    return all(float(point[2]) >= min_z for point in points)


def _fast_planner_candidate_routes(
    request: PlanningRequest,
    target: SemanticObject,
    manager: FastPlannerManager,
    scene: Optional[SemanticScene] = None,
) -> List[List[Vector3]]:
    routes: List[List[Vector3]] = [[request.start, request.goal]]
    routes = _region_sampled_routes(request, target, manager, scene) + routes
    # Around maneuvers sample guides uniformly from the free region (so trajectories
    # do not all collapse onto one fixed side point). Through/over/under still use
    # their single semantic guide point (opening center / top / underside).
    if request.skill not in {Skill.AROUND_LEFT, Skill.AROUND_RIGHT}:
        guide_points = _semantic_guide_points(request, target)
        for guide in guide_points:
            if manager.point_is_free(guide):
                routes.insert(0, [request.start, guide, request.goal])
    return _dedupe_routes(routes)


def _region_sampled_routes(
    request: PlanningRequest,
    target: SemanticObject,
    manager: FastPlannerManager,
    scene: Optional[SemanticScene],
    num_guides: int = 32,
    seed: Optional[int] = None,
) -> List[List[Vector3]]:
    if scene is None:
        return []
    region = _build_validation_region(scene, target, request.skill, request.start, request.goal, request.params)
    if region is None:
        return []
    guides = [guide for guide in region.sample_guides(num_guides, seed=seed) if _route_guide_is_usable(guide, manager)]
    routes = [[request.start, guide, request.goal] for guide in guides]
    if request.skill in {Skill.UNDER, Skill.THROUGH}:
        routes.extend(_entry_mid_exit_routes(request, guides))
    return routes


def _entry_mid_exit_routes(request: PlanningRequest, guides: List[Vector3]) -> List[List[Vector3]]:
    if not guides:
        return []
    forward = np.asarray(_normalize_3d(_sub_3d(request.goal, request.start)), dtype=float)
    routes = []
    for guide in guides:
        guide_arr = np.asarray(guide, dtype=float)
        entry = guide_arr - forward * request.params.pre_distance
        exit_point = guide_arr + forward * request.params.post_distance
        routes.append(
            [
                request.start,
                tuple(float(value) for value in entry[:3]),
                guide,
                tuple(float(value) for value in exit_point[:3]),
                request.goal,
            ]
        )
    return routes


def _route_guide_is_usable(guide: Vector3, manager: FastPlannerManager) -> bool:
    # Region points are already checked against object geometry. The Fast-Planner
    # ESDF can over-inflate concave free spaces such as table undersides, so do
    # not reject guides solely because the global inflated map marks them occupied.
    return manager.sdf_map.is_in_map(guide)


def _dedupe_routes(routes: List[List[Vector3]]) -> List[List[Vector3]]:
    seen = set()
    unique = []
    for route in routes:
        key = tuple(tuple(round(float(value), 4) for value in point) for point in route)
        if key in seen:
            continue
        seen.add(key)
        unique.append(route)
    return unique


def _semantic_guide_points(request: PlanningRequest, target: SemanticObject) -> List[Vector3]:
    start = request.start
    goal = request.goal
    params = request.params
    guides: List[Vector3] = []
    footprint = _object_footprint_radius(target)
    forward = _normalize_2d((goal[0] - start[0], goal[1] - start[1]))
    if forward is None:
        forward = (1.0, 0.0)
    left = (-forward[1], forward[0])
    side_distance = footprint + params.planning_margin + max(params.around_margin * 0.5, 0.25)

    if request.skill == Skill.AROUND_LEFT:
        guides.append((target.center[0] + left[0] * side_distance, target.center[1] + left[1] * side_distance, start[2]))
    elif request.skill == Skill.AROUND_RIGHT:
        guides.append((target.center[0] - left[0] * side_distance, target.center[1] - left[1] * side_distance, start[2]))
    elif request.skill == Skill.THROUGH:
        guides.append(target.opening_center or target.center)
    elif request.skill == Skill.OVER:
        top_z = target.top_z
        if top_z is None:
            bbox = _object_bbox(target)
            top_z = bbox[1][2] if bbox is not None else target.center[2] + (_object_height(target) or 0.0) / 2.0
        guides.append((target.center[0], target.center[1], top_z + params.planning_margin + 0.2))
    elif request.skill == Skill.UNDER:
        bbox = _object_bbox(target)
        if bbox is not None:
            z = bbox[0][2] + params.planning_margin + 0.1
        elif target.bottom_z is not None:
            z = target.bottom_z - params.planning_margin - 0.1
        else:
            z = start[2]
        guides.append((target.center[0], target.center[1], z))
    return guides


def _classify_around(
    path: Tuple[Vector3, ...],
    target: SemanticObject,
    start: Vector3,
    goal: Vector3,
    expected: SemanticLabel,
) -> SemanticLabel:
    if len(path) < 3:
        return SemanticLabel.INVALID

    footprint = _object_footprint_radius(target)
    forward = _normalize_2d((goal[0] - start[0], goal[1] - start[1]))
    if forward is None:
        return SemanticLabel.INVALID
    left = (-forward[1], forward[0])

    forward_coords: List[float] = []
    near_side_coords: List[float] = []
    near_radius = max(footprint * 3.0, footprint + 0.5)

    for point in path:
        rel = (point[0] - target.center[0], point[1] - target.center[1])
        distance = math.hypot(rel[0], rel[1])
        if distance <= footprint:
            return SemanticLabel.INVALID
        forward_coords.append(_dot_2d(rel, forward))
        if distance <= near_radius:
            near_side_coords.append(_dot_2d(rel, left))

    if not near_side_coords:
        return SemanticLabel.INVALID
    if min(forward_coords) >= -footprint or max(forward_coords) <= footprint:
        return SemanticLabel.INVALID

    mean_side = sum(near_side_coords) / len(near_side_coords)
    threshold = max(0.05, footprint * 0.25)
    if mean_side > threshold:
        return SemanticLabel.AROUND_LEFT if expected == SemanticLabel.AROUND_LEFT else SemanticLabel.INVALID
    if mean_side < -threshold:
        return SemanticLabel.AROUND_RIGHT if expected == SemanticLabel.AROUND_RIGHT else SemanticLabel.INVALID
    return SemanticLabel.INVALID


def _passes_through_opening(path: Tuple[Vector3, ...], target: SemanticObject, params: PlannerParams) -> bool:
    center = target.opening_center or target.center
    normal = _normalize_3d(target.normal or (1.0, 0.0, 0.0))
    half_width = (target.width or _object_footprint_radius(target) * 2.0) / 2.0
    half_height = (target.clearance_height or target.height or _object_height(target) or 2.0) / 2.0

    for a, b in zip(path, path[1:]):
        da = _dot_3d(_sub_3d(a, center), normal)
        db = _dot_3d(_sub_3d(b, center), normal)
        if da == 0.0 or da * db > 0.0:
            continue
        t = abs(da) / (abs(da) + abs(db))
        cross = _lerp_3d(a, b, t)
        lateral = math.hypot(cross[0] - center[0], cross[1] - center[1])
        vertical = abs(cross[2] - center[2])
        if lateral <= half_width - params.planning_margin and vertical <= half_height - params.planning_margin:
            return True
    return False


def _passes_over_object(path: Tuple[Vector3, ...], target: SemanticObject, params: PlannerParams) -> bool:
    top_z = target.top_z
    if top_z is None:
        bbox = _object_bbox(target)
        top_z = bbox[1][2] if bbox is not None else target.center[2] + (_object_height(target) or 0.0) / 2.0
    required_z = top_z + params.planning_margin
    return any(_near_object_xy(point, target, params.around_margin) and point[2] >= required_z for point in path)


def _passes_under_object(path: Tuple[Vector3, ...], target: SemanticObject, params: PlannerParams) -> bool:
    bbox = _object_bbox(target)
    occupancy = occupancy_for_object(target)
    if bbox is not None and occupancy is not None:
        min_z = bbox[0][2] + params.planning_margin
        max_z = bbox[1][2] - params.planning_margin
        for point in path:
            if not _point_inside_bbox_xy(point, bbox):
                continue
            if not min_z <= point[2] <= max_z:
                continue
            if not _point_collides_object(point, target, params.planning_margin):
                return True
        return False

    ceiling_z = target.bottom_z
    if ceiling_z is None:
        ceiling_z = bbox[0][2] if bbox is not None else None
    if ceiling_z is None:
        return False
    max_z = ceiling_z - params.planning_margin
    return any(_near_object_xy(point, target, params.around_margin) and point[2] <= max_z for point in path)


def _solid_obstacles(registry) -> List[SemanticObject]:
    passthrough_types = {"door", "window", "opening"}
    return [obj for obj in registry if obj.object_type not in passthrough_types]


def _planning_obstacles_for_skill(registry, request: PlanningRequest) -> List[SemanticObject]:
    obstacles = _solid_obstacles(registry)
    if request.skill in {Skill.UNDER, Skill.THROUGH}:
        return [obj for obj in obstacles if obj.object_id != request.object_id]
    return obstacles


def _validation_obstacles_for_skill(
    registry,
    skill: Skill,
    target: SemanticObject,
) -> List[SemanticObject]:
    obstacles = _solid_obstacles(registry)
    if skill in {Skill.UNDER, Skill.THROUGH}:
        return [obj for obj in obstacles if obj.object_id != target.object_id]
    return obstacles


def _make_bounds(
    start: Vector3,
    goal: Vector3,
    registry,
    padding: float,
) -> Tuple[float, float, float, float]:
    xs = [start[0], goal[0]]
    ys = [start[1], goal[1]]
    for obj in registry:
        footprint = _object_footprint_radius(obj)
        xs.extend([obj.center[0] - footprint, obj.center[0] + footprint])
        ys.extend([obj.center[1] - footprint, obj.center[1] + footprint])
    return min(xs) - padding, max(xs) + padding, min(ys) - padding, max(ys) + padding


def _make_prm_bounds(
    start: Vector3,
    goal: Vector3,
    registry,
    padding: float,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    min_x, max_x, min_y, max_y = _make_bounds(start, goal, registry, padding)
    z = start[2]
    return (min_x, max_x), (min_y, max_y), (z, z)


def _make_obstacle_func(obstacles: Iterable[SemanticObject], margin: float):
    obstacle_list = tuple(obstacles)

    def is_obstacle(point, scene_id=None):
        del scene_id
        return any(_point_collides_object(tuple(np.asarray(point, dtype=float)[:3]), obj, margin) for obj in obstacle_list)

    return is_obstacle


def _point_collides_object(point: Vector3, obj: SemanticObject, margin: float) -> bool:
    occupancy = occupancy_for_object(obj)
    if occupancy is not None:
        return occupancy.contains_point(point, margin=margin)

    bbox = _object_bbox(obj)
    if bbox is not None:
        return all(bbox[0][i] - margin <= point[i] <= bbox[1][i] + margin for i in range(3))

    radius = obj.radius
    height = _object_height(obj)
    if radius is None:
        return False
    xy_collision = math.hypot(point[0] - obj.center[0], point[1] - obj.center[1]) <= radius + margin
    if height is None:
        return xy_collision
    z_min = obj.center[2] - height / 2.0 - margin
    z_max = obj.center[2] + height / 2.0 + margin
    return xy_collision and z_min <= point[2] <= z_max


def _object_bbox(obj: SemanticObject) -> Optional[Tuple[Vector3, Vector3]]:
    if obj.bbox_min is not None and obj.bbox_max is not None:
        return obj.bbox_min, obj.bbox_max
    return None


def _object_footprint_radius(obj: SemanticObject) -> float:
    if obj.radius is not None:
        return obj.radius
    bbox = _object_bbox(obj)
    if bbox is not None:
        dx = bbox[1][0] - bbox[0][0]
        dy = bbox[1][1] - bbox[0][1]
        return math.hypot(dx, dy) / 2.0
    if obj.width is not None:
        return obj.width / 2.0
    return 0.5


def _object_height(obj: SemanticObject) -> Optional[float]:
    if obj.height is not None:
        return obj.height
    bbox = _object_bbox(obj)
    if bbox is not None:
        return bbox[1][2] - bbox[0][2]
    return None


def _near_object_xy(point: Vector3, obj: SemanticObject, extra_margin: float) -> bool:
    footprint = _object_footprint_radius(obj) + extra_margin
    return math.hypot(point[0] - obj.center[0], point[1] - obj.center[1]) <= footprint


def _point_inside_bbox_xy(point: Vector3, bbox: Tuple[Vector3, Vector3]) -> bool:
    return bbox[0][0] <= point[0] <= bbox[1][0] and bbox[0][1] <= point[1] <= bbox[1][1]


def _load_visfly_prm_planner():
    try:
        from VisFly.utils.pathFinder import PRMPlanner
    except ImportError as exc:
        raise ImportError(
            "VisFly PRM planning requires the VisFly dependencies. "
            "Run this from the visfly conda environment."
        ) from exc
    return PRMPlanner


def _run_visfly_prm_attempt(
    prm_planner_class,
    bounds,
    num_samples: int,
    obstacle_func,
    k: int,
    start: Vector3,
    goal: Vector3,
    seed: int,
) -> List[Vector3]:
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        planner = prm_planner_class(
            bounds=bounds,
            num_samples=num_samples,
            obstacle_func=obstacle_func,
            k=k,
            scene_id=0,
        )
        planner.pre_plan()
        raw_path = planner.plan(np.array(start, dtype=float), np.array(goal, dtype=float))
    finally:
        np.random.set_state(state)

    return [tuple(float(coord) for coord in np.asarray(point, dtype=float)[:3]) for point in raw_path]


def _densify_polyline(points: List[Vector3], step: float) -> List[Vector3]:
    if step <= 0.0:
        raise ValueError("interpolation_step must be positive")
    if not points:
        return []

    dense = [points[0]]
    for start, end in zip(points, points[1:]):
        start_arr = np.asarray(start, dtype=float)
        end_arr = np.asarray(end, dtype=float)
        distance = float(np.linalg.norm(end_arr - start_arr))
        segments = max(1, math.ceil(distance / step))
        for i in range(1, segments + 1):
            t = i / segments
            point = start_arr * (1.0 - t) + end_arr * t
            dense.append(tuple(float(coord) for coord in point[:3]))
    return dense


def _path_is_collision_free(points: Iterable[Vector3], obstacle_func) -> bool:
    return all(not obstacle_func(point, 0) for point in points)


def _astar_grid_path(
    start: Vector3,
    goal: Vector3,
    target: SemanticObject,
    obstacles: Iterable[SemanticObject],
    obstacle_func,
    grid_resolution: float,
    bounds: Tuple[float, float, float, float],
    preferred_label: Optional[SemanticLabel],
) -> List[GridNode]:
    start_node = _world_to_grid(start[0], start[1], grid_resolution, bounds)
    goal_node = _world_to_grid(goal[0], goal[1], grid_resolution, bounds)
    del obstacles
    if _node_in_collision(start_node, obstacle_func, start[2], grid_resolution, bounds):
        raise ValueError("start is inside an inflated obstacle")
    if _node_in_collision(goal_node, obstacle_func, start[2], grid_resolution, bounds):
        raise ValueError("goal is inside an inflated obstacle")

    open_heap = [(0.0, 0.0, 0, start_node)]
    came_from: Dict[GridNode, Optional[GridNode]] = {start_node: None}
    g_score: Dict[GridNode, float] = {start_node: 0.0}
    tie = 0

    while open_heap:
        _, _, _, current = heapq.heappop(open_heap)
        if current == goal_node:
            return _reconstruct_path(came_from, current)

        for neighbor, move_cost in _neighbors(current):
            if not _node_in_bounds(neighbor, bounds, grid_resolution):
                continue
            if _node_in_collision(neighbor, obstacle_func, start[2], grid_resolution, bounds):
                continue
            if _edge_in_collision(current, neighbor, obstacle_func, start[2], grid_resolution, bounds):
                continue

            side_bias = _side_priority_bias(
                neighbor,
                target,
                start,
                goal,
                preferred_label,
                grid_resolution,
                bounds,
            )
            tentative_g = g_score[current] + move_cost * grid_resolution + side_bias
            if tentative_g >= g_score.get(neighbor, math.inf):
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative_g
            heuristic = _grid_distance(neighbor, goal_node) * grid_resolution
            tie += 1
            heapq.heappush(open_heap, (tentative_g + heuristic, heuristic, tie, neighbor))

    raise RuntimeError("failed to find a grid path")


def _neighbors(node: GridNode) -> Iterable[Tuple[GridNode, float]]:
    x, y = node
    for dx, dy, cost in (
        (1, -1, math.sqrt(2.0)),
        (0, -1, 1.0),
        (1, 0, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
        (0, 1, 1.0),
        (-1, 0, 1.0),
        (-1, 1, math.sqrt(2.0)),
    ):
        yield (x + dx, y + dy), cost


def _preferred_label_from_skill(skill: Skill) -> Optional[SemanticLabel]:
    if skill == Skill.AROUND_LEFT:
        return SemanticLabel.AROUND_LEFT
    if skill == Skill.AROUND_RIGHT:
        return SemanticLabel.AROUND_RIGHT
    if skill == Skill.THROUGH:
        return SemanticLabel.THROUGH
    if skill == Skill.OVER:
        return SemanticLabel.OVER
    if skill == Skill.UNDER:
        return SemanticLabel.UNDER
    return None


def _side_priority_bias(
    node: GridNode,
    obj: SemanticObject,
    start: Vector3,
    goal: Vector3,
    preferred_label: Optional[SemanticLabel],
    grid_resolution: float,
    bounds: Tuple[float, float, float, float],
) -> float:
    if preferred_label not in {SemanticLabel.AROUND_LEFT, SemanticLabel.AROUND_RIGHT}:
        return 0.0

    forward = _normalize_2d((goal[0] - start[0], goal[1] - start[1]))
    if forward is None:
        return 0.0
    left = (-forward[1], forward[0])
    x, y, _ = _grid_to_world(node, 0.0, grid_resolution, bounds)
    rel = (x - obj.center[0], y - obj.center[1])
    side_coord = _dot_2d(rel, left)
    distance = math.hypot(rel[0], rel[1])
    footprint = _object_footprint_radius(obj)
    influence_radius = max(footprint * 4.0, footprint + 1.5)
    if distance > influence_radius:
        return 0.0

    if preferred_label == SemanticLabel.AROUND_RIGHT:
        undesired_side_amount = max(0.0, side_coord)
    else:
        undesired_side_amount = max(0.0, -side_coord)
    return undesired_side_amount * 0.5


def _reconstruct_path(came_from: Dict[GridNode, Optional[GridNode]], current: GridNode) -> List[GridNode]:
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _grid_path_to_world_path(
    grid_path: List[GridNode],
    start: Vector3,
    goal: Vector3,
    grid_resolution: float,
    bounds: Tuple[float, float, float, float],
) -> List[Vector3]:
    points = [_grid_to_world(node, start[2], grid_resolution, bounds) for node in grid_path]
    points[0] = start
    points[-1] = goal
    return points


def _world_to_grid(x: float, y: float, grid_resolution: float, bounds: Tuple[float, float, float, float]) -> GridNode:
    min_x, _, min_y, _ = bounds
    return round((x - min_x) / grid_resolution), round((y - min_y) / grid_resolution)


def _grid_to_world(node: GridNode, z: float, grid_resolution: float, bounds: Tuple[float, float, float, float]) -> Vector3:
    min_x, _, min_y, _ = bounds
    return (
        min_x + node[0] * grid_resolution,
        min_y + node[1] * grid_resolution,
        z,
    )


def _node_in_bounds(node: GridNode, bounds: Tuple[float, float, float, float], grid_resolution: float) -> bool:
    min_x, max_x, min_y, max_y = bounds
    max_ix = round((max_x - min_x) / grid_resolution)
    max_iy = round((max_y - min_y) / grid_resolution)
    return 0 <= node[0] <= max_ix and 0 <= node[1] <= max_iy


def _node_in_collision(
    node: GridNode,
    obstacle_func,
    z: float,
    grid_resolution: float,
    bounds: Tuple[float, float, float, float],
) -> bool:
    point = _grid_to_world(node, z, grid_resolution, bounds)
    return bool(obstacle_func(point, 0))


def _edge_in_collision(
    a: GridNode,
    b: GridNode,
    obstacle_func,
    z: float,
    grid_resolution: float,
    bounds: Tuple[float, float, float, float],
) -> bool:
    start = _grid_to_world(a, z, grid_resolution, bounds)
    end = _grid_to_world(b, z, grid_resolution, bounds)
    return _segment_hits_obstacle(start, end, obstacle_func, step=max(grid_resolution * 0.5, 0.05))


def _segment_hits_obstacle(start: Vector3, end: Vector3, obstacle_func, step: float) -> bool:
    start_arr = np.asarray(start, dtype=float)
    end_arr = np.asarray(end, dtype=float)
    distance = float(np.linalg.norm(end_arr - start_arr))
    samples = max(1, math.ceil(distance / step))
    for i in range(samples + 1):
        point = start_arr * (1.0 - i / samples) + end_arr * (i / samples)
        if obstacle_func(tuple(float(v) for v in point), 0):
            return True
    return False


def _segment_distance_to_point(a: Tuple[float, float], b: Tuple[float, float], p: Tuple[float, float]) -> float:
    ab = (b[0] - a[0], b[1] - a[1])
    ap = (p[0] - a[0], p[1] - a[1])
    denom = _dot_2d(ab, ab)
    if denom == 0.0:
        return math.hypot(ap[0], ap[1])
    t = max(0.0, min(1.0, _dot_2d(ap, ab) / denom))
    closest = (a[0] + ab[0] * t, a[1] + ab[1] * t)
    return math.hypot(p[0] - closest[0], p[1] - closest[1])


def _grid_distance(a: GridNode, b: GridNode) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _normalize_2d(vector: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    norm = math.hypot(vector[0], vector[1])
    if norm == 0.0:
        return None
    return vector[0] / norm, vector[1] / norm


def _normalize_3d(vector: Vector3) -> Vector3:
    norm = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    if norm == 0.0:
        return (1.0, 0.0, 0.0)
    return vector[0] / norm, vector[1] / norm, vector[2] / norm


def _dot_2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _dot_3d(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub_3d(a: Vector3, b: Vector3) -> Vector3:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _lerp_3d(a: Vector3, b: Vector3, t: float) -> Vector3:
    return (
        a[0] * (1.0 - t) + b[0] * t,
        a[1] * (1.0 - t) + b[1] * t,
        a[2] * (1.0 - t) + b[2] * t,
    )
