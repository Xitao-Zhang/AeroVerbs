from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..interfaces import SemanticScene, Vector3
from .bspline import NonUniformBspline
from .bspline_optimizer import BsplineOptimizer
from .edt_environment import EDTEnvironment
from .kinodynamic_astar import KinodynamicAstar, KinodynamicAstarParams
from .sdf_map import SDFMap, SDFMapParams


@dataclass(frozen=True)
class FastPlannerConfig:
    resolution: float = 0.2
    bounds_margin: float = 1.5
    max_vel: float = 3.0
    max_acc: float = 2.0
    max_jerk: float = 4.0
    control_points_distance: float = 0.5
    clearance: float = 0.3
    max_tau: float = 0.6
    init_max_tau: float = 0.8
    w_time: float = 10.0
    horizon: float = 20.0
    lambda_heu: float = 5.0
    time_resolution: float = 0.8
    allocate_num: int = 100000
    max_iter: int = 2000
    check_num: int = 5
    sample_step: float = 0.05
    output_step: float = 0.1
    optimize: bool = True
    optimization_dist0: float = 0.4
    optimization_max_iter: int = 80


class FastPlannerManager:
    """Non-ROS Fast-Planner manager preserving the original planning pipeline."""

    def __init__(
        self,
        scene: SemanticScene,
        config: FastPlannerConfig,
        obstacles: Iterable = (),
        sdf_map: Optional[SDFMap] = None,
    ):
        self.scene = scene
        self.config = config
        sdf_params = SDFMapParams(
            resolution=config.resolution,
            obstacles_inflation=config.clearance,
            bounds_margin=config.bounds_margin,
        )
        self.sdf_map = sdf_map or SDFMap.from_scene(
            scene, sdf_params, obstacles=tuple(obstacles)
        )
        self.edt_environment = EDTEnvironment(self.sdf_map)
        self.kino_params = KinodynamicAstarParams(
            max_tau=config.max_tau,
            init_max_tau=config.init_max_tau,
            max_vel=config.max_vel,
            max_acc=config.max_acc,
            w_time=config.w_time,
            horizon=config.horizon,
            lambda_heu=config.lambda_heu,
            resolution_astar=config.resolution,
            time_resolution=config.time_resolution,
            allocate_num=config.allocate_num,
            max_iter=config.max_iter,
            check_num=config.check_num,
        )
        self.optimizer = BsplineOptimizer(
            self.edt_environment,
            max_vel=config.max_vel,
            max_acc=config.max_acc,
            dist0=config.optimization_dist0,
            max_iter=config.optimization_max_iter,
        )

    def plan_waypoints(
        self,
        waypoints: Sequence[Vector3],
        start_vel=None,
        start_acc=None,
    ) -> List[Vector3]:
        if len(waypoints) < 2:
            raise ValueError("waypoints must contain at least start and goal")
        measured_vel = np.zeros(3, dtype=float) if start_vel is None else np.asarray(start_vel, dtype=float)
        measured_acc = np.zeros(3, dtype=float) if start_acc is None else np.asarray(start_acc, dtype=float)
        all_points: List[Vector3] = []
        for segment_index, (start, goal) in enumerate(zip(waypoints, waypoints[1:])):
            segment = self.kinodynamic_replan(
                np.asarray(start, dtype=float),
                measured_vel if segment_index == 0 else np.zeros(3, dtype=float),
                measured_acc if segment_index == 0 else np.zeros(3, dtype=float),
                np.asarray(goal, dtype=float),
                np.zeros(3, dtype=float),
            )
            if all_points:
                all_points.extend(segment[1:])
            else:
                all_points.extend(segment)
        if self.config.optimize and len(waypoints) > 2 and len(all_points) >= 4:
            return self._smooth_joined_path(all_points)
        return all_points

    def kinodynamic_replan(self, start_pt, start_vel, start_acc, end_pt, end_vel) -> List[Vector3]:
        if np.linalg.norm(start_pt - end_pt) < 0.2:
            return [tuple(float(v) for v in start_pt), tuple(float(v) for v in end_pt)]

        finder = KinodynamicAstar(self.edt_environment, self.kino_params)
        status = finder.search(start_pt, start_vel, start_acc, end_pt, end_vel, init=True)
        if status == KinodynamicAstar.NO_PATH:
            finder.reset()
            status = finder.search(start_pt, start_vel, start_acc, end_pt, end_vel, init=False)
            if status == KinodynamicAstar.NO_PATH:
                raise RuntimeError("Fast-Planner kinodynamic search failed")

        raw_points = finder.get_kino_traj(self.config.sample_step)
        if len(raw_points) < 2:
            raise RuntimeError("Fast-Planner search returned too few points")
        if not self.config.optimize:
            return [tuple(float(v) for v in p[:3]) for p in raw_points]

        ts = max(self.config.control_points_distance / self.config.max_vel, 1e-3)
        sample_points, derivatives = finder.get_samples(ts)
        if len(sample_points) < 2:
            sample_points = raw_points
            derivatives = [start_vel, end_vel, start_acc, np.zeros(3)]

        ctrl_pts = NonUniformBspline.parameterize_to_bspline(ts, sample_points, derivatives)
        cost_function = BsplineOptimizer.NORMAL_PHASE
        if status != KinodynamicAstar.REACH_END:
            cost_function |= BsplineOptimizer.ENDPOINT
        ctrl_pts = self.optimizer.bspline_optimize_traj(ctrl_pts, ts, cost_function, 1, 1)
        bspline = NonUniformBspline(ctrl_pts, 3, ts)
        bspline.set_physical_limits(self.config.max_vel, self.config.max_acc)
        for _ in range(3):
            if bspline.check_feasibility():
                break
            bspline.reallocate_time()
        return bspline.sample(self.config.output_step)

    def point_is_free(self, point: Vector3) -> bool:
        return self.sdf_map.get_inflate_occupancy(point) == 0

    def _smooth_joined_path(self, points: Sequence[Vector3]) -> List[Vector3]:
        """Re-optimize a multi-segment route as one spline to remove guide-point kinks."""
        samples = _resample_polyline(points, max(self.config.control_points_distance, self.config.resolution))
        if len(samples) < 4:
            return list(points)

        ts = max(self.config.control_points_distance / self.config.max_vel, 1e-3)
        zero = np.zeros(3, dtype=float)
        derivatives = [zero, zero, zero, zero]
        ctrl_pts = NonUniformBspline.parameterize_to_bspline(ts, samples, derivatives)
        unoptimized = NonUniformBspline(ctrl_pts, 3, ts)
        unoptimized.set_physical_limits(self.config.max_vel, self.config.max_acc)

        ctrl_pts_opt = self.optimizer.bspline_optimize_traj(ctrl_pts, ts, BsplineOptimizer.NORMAL_PHASE, 1, 1)
        optimized = NonUniformBspline(ctrl_pts_opt, 3, ts)
        optimized.set_physical_limits(self.config.max_vel, self.config.max_acc)
        for _ in range(3):
            if optimized.check_feasibility():
                break
            optimized.reallocate_time()

        candidate = optimized.sample(self.config.output_step)
        if _points_are_free(candidate, self):
            return _with_exact_endpoints(candidate, points[0], points[-1])

        fallback = unoptimized.sample(self.config.output_step)
        if _points_are_free(fallback, self):
            return _with_exact_endpoints(fallback, points[0], points[-1])
        return list(points)


def route_points_cost(points: Sequence[Vector3]) -> float:
    total = 0.0
    for a, b in zip(points, points[1:]):
        total += float(np.linalg.norm(np.asarray(b, dtype=float) - np.asarray(a, dtype=float)))
    return total


def _resample_polyline(points: Sequence[Vector3], step: float) -> List[np.ndarray]:
    arrays = [np.asarray(point, dtype=float) for point in points]
    if len(arrays) < 2:
        return arrays
    result = [arrays[0]]
    carried = 0.0
    for start, end in zip(arrays, arrays[1:]):
        segment = end - start
        length = float(np.linalg.norm(segment))
        if length < 1e-9:
            continue
        direction = segment / length
        distance = step - carried if carried > 1e-9 else step
        while distance < length + 1e-9:
            result.append(start + direction * distance)
            distance += step
        carried = max(0.0, length - (distance - step))
    if np.linalg.norm(result[-1] - arrays[-1]) > 1e-9:
        result.append(arrays[-1])
    return result


def _points_are_free(points: Sequence[Vector3], manager: FastPlannerManager) -> bool:
    return all(manager.point_is_free(point) for point in points)


def _with_exact_endpoints(points: Sequence[Vector3], start: Vector3, goal: Vector3) -> List[Vector3]:
    result = [tuple(float(v) for v in point[:3]) for point in points]
    if result:
        result[0] = tuple(float(v) for v in start)
        result[-1] = tuple(float(v) for v in goal)
    return result
