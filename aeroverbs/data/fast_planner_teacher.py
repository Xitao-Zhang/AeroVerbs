"""Fast-Planner-backed privileged teacher for DAgger collection.

The planner consumes AeroVerbs' precomputed per-scene SDF archive directly.
It never voxelizes meshes or recomputes a distance transform during rollout.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from uav_maneuver_planning.fast_planner import FastPlannerConfig, FastPlannerManager
from uav_maneuver_planning.fast_planner.sdf_map import SDFMap
from uav_maneuver_planning.scenes import create_scene

from .labels import body_frame_basis
from .teacher import TeacherLabeler


DEFAULT_PREFERRED_CLEARANCE_BY_SKILL = {
    "around_left": 0.50,
    "around_right": 0.50,
    "over": 0.20,
    "through": 0.15,
    "under": 0.15,
}


@dataclass(frozen=True)
class FastPlannerTeacherConfig:
    """Runtime and safety settings for online teacher replanning."""

    collision_radius: float = 0.15
    preferred_clearance: Optional[float] = None
    replan_interval: int = 5
    timeout_s: float = 2.0
    bounds_margin: float = 1.5
    max_iter: int = 2000
    optimization_max_iter: int = 80
    optimize: bool = True
    frame_dt: float = 0.1
    max_start_speed: float = 3.0
    cached_path_max_distance: float = 0.75


class FastPlannerTeacherLabeler:
    """Label student states with periodically cached Fast-Planner paths."""

    def __init__(
        self,
        sample_dir: str,
        offsets: Optional[List[int]] = None,
        data_root: Optional[str] = None,
        config: Optional[FastPlannerTeacherConfig] = None,
        fallback_teacher: Optional[TeacherLabeler] = None,
        planner_factory: Optional[Callable] = None,
    ) -> None:
        self.sample_dir = os.path.abspath(sample_dir)
        self.data_root = data_root or os.path.dirname(os.path.dirname(self.sample_dir))
        self.config = config or FastPlannerTeacherConfig()
        if self.config.collision_radius < 0.0:
            raise ValueError("collision_radius must be non-negative")
        if self.config.replan_interval < 1:
            raise ValueError("replan_interval must be at least one")
        if self.config.timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        self.offsets = [int(value) for value in (offsets or [3, 6, 10, 15, 20])]

        with open(os.path.join(self.sample_dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        with open(os.path.join(self.sample_dir, "frames.json"), encoding="utf-8") as f:
            frames = json.load(f)
        self.skill = self.meta.get("skill") or os.path.basename(os.path.dirname(self.sample_dir))
        self.scene_id = self.meta.get("scene_id") or os.path.basename(self.sample_dir)
        self.preferred_clearance = float(
            self.config.preferred_clearance
            if self.config.preferred_clearance is not None
            else DEFAULT_PREFERRED_CLEARANCE_BY_SKILL.get(
                self.skill, self.config.collision_radius
            )
        )
        if self.preferred_clearance < self.config.collision_radius:
            raise ValueError("preferred_clearance must be at least collision_radius")
        self.expert_positions = np.asarray(
            [frame["camera_position_std"] for frame in frames], dtype=np.float32
        )
        if len(self.expert_positions) < 1:
            raise ValueError(f"sample has no expert frames: {self.sample_dir}")
        self.goal = self.expert_positions[-1].copy()

        self.region_points, self.guide_center, self.guide_normal = self._load_semantic_inputs()
        self.sdf_path = os.path.join(
            self.data_root, "_sdf", self.skill, f"{self.scene_id}.npz"
        )
        self.sdf_map = self._load_precomputed_sdf()
        self.guide, self.guide_source = self._select_semantic_guide()
        scene = create_scene(
            scene_id=self.scene_id,
            start=tuple(float(v) for v in self.expert_positions[0]),
            goal=tuple(float(v) for v in self.goal),
        )
        self.planner_config = FastPlannerConfig(
            resolution=0.2,
            clearance=self.config.collision_radius,
            max_iter=self.config.max_iter,
            optimize=self.config.optimize,
            optimization_dist0=max(
                self.preferred_clearance - self.config.collision_radius,
                0.0,
            ),
            optimization_max_iter=self.config.optimization_max_iter,
            output_step=0.1,
        )
        factory = planner_factory or (
            lambda planner_scene, planner_config, sdf_map: FastPlannerManager(
                planner_scene, planner_config, sdf_map=sdf_map
            )
        )
        self.planner = factory(scene, self.planner_config, self.sdf_map)
        self.fallback_teacher = fallback_teacher or TeacherLabeler(
            self.sample_dir, offsets=self.offsets, data_root=self.data_root
        )

        self._cached_path: Optional[np.ndarray] = None
        self._cached_index = 0
        self._label_frames = 0
        self._replan_attempts = 0
        self._replan_successes = 0
        self._replan_failures = 0
        self._replan_timeouts = 0
        self._fallback_frames = 0
        self._cached_after_failed_replan_frames = 0
        self._plan_latencies: List[float] = []
        self._last_error = ""
        self._previous_position: Optional[np.ndarray] = None
        self._last_label_source = "invalid"
        self._last_label_weight = 0.0
        self._label_source_counts: Dict[str, int] = {}

    @property
    def last_label_source(self) -> str:
        return self._last_label_source

    @property
    def last_label_weight(self) -> float:
        return self._last_label_weight

    def _record_label_source(self, source: str, weight: float) -> None:
        self._last_label_source = str(source)
        self._last_label_weight = float(weight)
        self._label_source_counts[source] = self._label_source_counts.get(source, 0) + 1

    def _load_precomputed_sdf(self) -> SDFMap:
        if not os.path.exists(self.sdf_path):
            raise FileNotFoundError(f"missing precomputed SDF: {self.sdf_path}")
        with np.load(self.sdf_path) as archive:
            missing = {"grid", "origin", "voxel_size"} - set(archive.files)
            if missing:
                raise ValueError(
                    f"precomputed SDF is missing keys {sorted(missing)}: {self.sdf_path}"
                )
            grid = archive["grid"]
            origin = archive["origin"]
            voxel_size = float(np.asarray(archive["voxel_size"]).reshape(-1)[0])
        return SDFMap.from_precomputed(
            grid_zyx=grid,
            origin=origin,
            voxel_size=voxel_size,
            inflation=self.config.collision_radius,
            bounds_points=self.expert_positions,
            bounds_margin=self.config.bounds_margin,
        )

    def _load_semantic_inputs(
        self,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        region_path = os.path.join(self.sample_dir, "heatmap_gate_region.json")
        summary_path = os.path.join(self.sample_dir, "heatmap_gate_summary.json")
        points = np.zeros((0, 3), dtype=np.float32)
        center = None
        normal = None
        if os.path.exists(region_path):
            with open(region_path, encoding="utf-8") as f:
                region = json.load(f)
            points = np.asarray(region.get("points", []), dtype=np.float32).reshape(-1, 3)
            points = points[np.all(np.isfinite(points), axis=1)]
        if os.path.exists(summary_path):
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
            if summary.get("center_std") is not None:
                center = np.asarray(summary["center_std"], dtype=np.float32)
            if summary.get("normal_std") is not None:
                normal = np.asarray(summary["normal_std"], dtype=np.float32)
        if center is None and len(points):
            center = points.mean(axis=0)
        if center is None:
            return points, None, None
        if normal is None or float(np.linalg.norm(normal)) < 1e-8:
            normal = self.goal - self.expert_positions[0]
        norm = float(np.linalg.norm(normal))
        if norm < 1e-8:
            return points, center, None
        return points, center, normal / norm

    def _expert_plane_crossing(self) -> Optional[np.ndarray]:
        if self.guide_center is None or self.guide_normal is None:
            return None
        signed = (self.expert_positions - self.guide_center) @ self.guide_normal
        crossings = np.flatnonzero(signed[:-1] * signed[1:] <= 0.0)
        if len(crossings):
            index = int(crossings[0])
            denom = float(signed[index + 1] - signed[index])
            alpha = 0.0 if abs(denom) < 1e-8 else float(-signed[index] / denom)
            return (
                self.expert_positions[index]
                + np.clip(alpha, 0.0, 1.0)
                * (self.expert_positions[index + 1] - self.expert_positions[index])
            ).astype(np.float32)
        return self.expert_positions[int(np.argmin(np.abs(signed)))].copy()

    def _select_semantic_guide(self) -> Tuple[Optional[np.ndarray], str]:
        anchor = self._expert_plane_crossing()
        source = "expert_crossing"
        if anchor is None:
            anchor = None if self.guide_center is None else self.guide_center.copy()
            source = "summary_center"
        if anchor is None:
            return None, "none"
        if not len(self.region_points):
            return anchor.astype(np.float32), source

        distances = np.linalg.norm(self.region_points - anchor[None, :], axis=1)
        raw_clearances = np.zeros((len(self.region_points),), dtype=np.float32)
        for index, point in enumerate(self.region_points):
            if self.sdf_map.is_in_map(point):
                inflated_distance, _ = self.sdf_map.get_dist_with_grad_trilinear(point)
                raw_clearances[index] = float(inflated_distance) + self.config.collision_radius
        deficits = np.maximum(self.preferred_clearance - raw_clearances, 0.0)
        costs = distances + 2.0 * deficits
        temperature = 0.10
        weights = np.exp(-(costs - costs.min()) / temperature)
        weights /= max(float(weights.sum()), 1e-8)
        region_hint = np.sum(self.region_points * weights[:, None], axis=0)
        guide = 0.75 * anchor + 0.25 * region_hint
        return guide.astype(np.float32), "expert_crossing_region_soft"

    def _route_for(self, position: np.ndarray) -> List[Tuple[float, float, float]]:
        route = [tuple(float(v) for v in position)]
        if self._needs_guide(position):
            route.append(tuple(float(v) for v in self.guide))
        route.append(tuple(float(v) for v in self.goal))
        return route

    def _needs_guide(self, position: np.ndarray) -> bool:
        if self.guide is None or self.guide_normal is None:
            return False
        if (
            self.skill not in ("around_left", "around_right")
            and float(np.linalg.norm(position - self.guide)) < 0.2
        ):
            return False
        current_side = float(np.dot(position - self.guide, self.guide_normal))
        goal_side = float(np.dot(self.goal - self.guide, self.guide_normal))
        return current_side * goal_side <= 0.0

    def _try_replan(self, position: np.ndarray, start_velocity: np.ndarray) -> bool:
        self._replan_attempts += 1
        started = time.perf_counter()
        try:
            points = self.planner.plan_waypoints(
                self._route_for(position),
                start_vel=start_velocity,
                start_acc=np.zeros(3, dtype=np.float32),
            )
            elapsed = time.perf_counter() - started
            self._plan_latencies.append(elapsed)
            if elapsed > self.config.timeout_s:
                self._replan_timeouts += 1
                raise TimeoutError(
                    f"planner took {elapsed:.3f}s, budget is {self.config.timeout_s:.3f}s"
                )
            path = np.asarray(points, dtype=np.float32).reshape(-1, 3)
            if len(path) < 2 or not np.all(np.isfinite(path)):
                raise RuntimeError("Fast-Planner returned an invalid path")
        except Exception as exc:
            if not self._plan_latencies:
                self._plan_latencies.append(time.perf_counter() - started)
            self._replan_failures += 1
            self._last_error = repr(exc)
            if not self._cached_path_is_usable(position):
                self._cached_path = None
                self._cached_index = 0
            return False
        self._replan_successes += 1
        self._last_error = ""
        self._cached_path = path
        self._cached_index = 0
        return True

    def _cached_path_is_usable(self, position: np.ndarray) -> bool:
        """Keep a previous plan only while it is nearby and collision-free."""
        path = self._cached_path
        if path is None or not len(path):
            return False
        start = int(np.clip(self._cached_index, 0, len(path) - 1))
        nearest = start + int(
            np.argmin(np.linalg.norm(path[start:] - position[None, :], axis=1))
        )
        if float(np.linalg.norm(path[nearest] - position)) > float(
            self.config.cached_path_max_distance
        ):
            return False
        validation_end = min(len(path), nearest + max(self.offsets, default=0) + 1)
        for point in path[nearest:validation_end]:
            if not self.sdf_map.is_in_map(point):
                return False
            distance, _ = self.sdf_map.get_dist_with_grad_trilinear(point)
            if not np.isfinite(distance) or float(distance) <= 0.0:
                return False
        self._cached_index = max(self._cached_index, nearest)
        return True

    def _target_crosses_around_guide(
        self, position: np.ndarray, target: np.ndarray
    ) -> bool:
        if self.skill not in ("around_left", "around_right"):
            return False
        if not self._needs_guide(position):
            return False
        if self.guide is None or self.guide_normal is None:
            return False
        goal_side = float(np.dot(self.goal - self.guide, self.guide_normal))
        target_side = float(np.dot(target - self.guide, self.guide_normal))
        return abs(goal_side) > 1e-8 and target_side * goal_side > 1e-6

    def _planned_waypoints(
        self, position: np.ndarray, forward: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        path = self._cached_path
        if path is None or not self._cached_path_is_usable(position):
            self._cached_path = None
            self._cached_index = 0
            return np.zeros((len(self.offsets), 3), dtype=np.float32), np.zeros(
                (len(self.offsets),), dtype=np.float32
            )
        start = int(np.clip(self._cached_index, 0, len(path) - 1))
        nearest = start + int(np.argmin(np.linalg.norm(path[start:] - position[None, :], axis=1)))
        self._cached_index = max(self._cached_index, nearest)
        body_x, body_y = body_frame_basis(np.asarray(forward, dtype=np.float32)[:2])
        waypoints = np.zeros((len(self.offsets), 3), dtype=np.float32)
        valid = np.zeros((len(self.offsets),), dtype=np.float32)
        for index, offset in enumerate(self.offsets):
            target_index = nearest + offset
            if target_index >= len(path):
                continue
            target = path[target_index]
            if self._target_crosses_around_guide(position, target):
                if index > 2:
                    continue
                target = self.guide
            delta = target - position
            waypoints[index] = np.array(
                [
                    float(np.dot(delta[:2], body_x)),
                    float(np.dot(delta[:2], body_y)),
                    float(delta[2]),
                ],
                dtype=np.float32,
            )
            valid[index] = 1.0
        return waypoints, valid

    def label(
        self,
        student_position: np.ndarray,
        student_forward: np.ndarray,
        prefix_points: List[np.ndarray] | np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        position = np.asarray(student_position, dtype=np.float32)
        forward = np.asarray(student_forward, dtype=np.float32)
        frame_index = self._label_frames
        self._label_frames += 1

        start_velocity = np.zeros(3, dtype=np.float32)
        if self._previous_position is not None:
            start_velocity = (position - self._previous_position) / max(
                float(self.config.frame_dt), 1e-6
            )
            speed = float(np.linalg.norm(start_velocity))
            if speed > self.config.max_start_speed:
                start_velocity *= self.config.max_start_speed / speed
        self._previous_position = position.copy()

        attempted_replan = frame_index % self.config.replan_interval == 0
        replanned = False
        if attempted_replan:
            replanned = self._try_replan(position, start_velocity)
        waypoints, valid = self._planned_waypoints(position, forward)
        if not np.any(valid):
            self._fallback_frames += 1
            self._record_label_source("invalid", 0.0)
            completion = self.fallback_teacher.label_completion(
                prefix_points, current_position=position
            )
            return waypoints, valid, completion
        if attempted_replan and not replanned:
            self._cached_after_failed_replan_frames += 1
        self._record_label_source(
            "fast_planner_replan" if replanned else "fast_planner_cache", 1.0
        )
        completion = self.fallback_teacher.label_completion(
            prefix_points, current_position=position
        )
        return waypoints, valid, completion

    def stats(self) -> Dict[str, object]:
        latencies = np.asarray(self._plan_latencies, dtype=float)
        return {
            "teacher": "fast_planner_online",
            "sdf_source": "precomputed_npz",
            "sdf_path": self.sdf_path,
            "optimizer_enabled": bool(self.planner_config.optimize),
            "collision_radius_m": float(self.config.collision_radius),
            "preferred_clearance_m": float(self.preferred_clearance),
            "optimizer_clearance_target_m": float(self.planner_config.optimization_dist0),
            "semantic_guide_source": self.guide_source,
            "semantic_guide": None if self.guide is None else self.guide.tolist(),
            "replan_interval_frames": int(self.config.replan_interval),
            "label_frames": int(self._label_frames),
            "replan_attempts": int(self._replan_attempts),
            "replan_successes": int(self._replan_successes),
            "replan_failures": int(self._replan_failures),
            "replan_timeouts": int(self._replan_timeouts),
            "fallback_frames": int(self._fallback_frames),
            "fallback_fraction": float(self._fallback_frames / max(self._label_frames, 1)),
            "cached_after_failed_replan_frames": int(
                self._cached_after_failed_replan_frames
            ),
            "label_source_counts": dict(self._label_source_counts),
            "mean_replan_s": float(latencies.mean()) if len(latencies) else 0.0,
            "p95_replan_s": float(np.percentile(latencies, 95)) if len(latencies) else 0.0,
            "max_replan_s": float(latencies.max()) if len(latencies) else 0.0,
            "last_error": self._last_error,
        }
