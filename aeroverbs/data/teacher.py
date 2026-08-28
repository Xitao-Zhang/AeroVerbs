"""Privileged teacher used by the DAgger loop.

For every student state the teacher produces:
- future body-frame waypoint labels (``[K, 3]`` + valid mask), by matching the
  student position to the nearest expert frame and reading off future expert
  positions;
- a completion label from the unified :class:`CompletionJudge` (semantic done
  AND far side AND safe clearance).

When the student is outside the expert-trajectory tube and the current or
trend-predicted future state is dangerously close to the object surface, the
teacher emits a short recovery trajectory instead of the raw expert waypoints:
``safe subgoal -> blend -> back into the tube``. The safe subgoal is
skill-specific (around: radial push in XY; through: steer to the opening
centre; over: climb above the bbox; under: stay below the table top).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..config import CompletionConfig
from .completion import CompletionJudge, make_clearance_provider
from .labels import body_frame_basis
from .semantic_judge import SemanticCompletionJudge


@dataclass
class PushConfig:
    """Geometry and parameters for the per-skill recovery correction."""

    skill: str
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    center: np.ndarray
    opening_center: Optional[np.ndarray] = None
    push_distance: float = 0.5
    push_threshold: float = 0.45
    push_waypoint_count: int = 3
    push_blend_width: float = 0.15
    under_clearance: float = 0.4
    tube_radius: float = 0.5


class TeacherLabeler:
    def __init__(
        self,
        sample_dir: str,
        offsets: Optional[List[int]] = None,
        push_config: Optional[PushConfig] = None,
        data_root: Optional[str] = None,
        completion_cfg: Optional[CompletionConfig] = None,
        # Backwards-compatible legacy parameters (used when push_config is None).
        object_center_xy: Optional[np.ndarray] = None,
        push_distance: float = 0.5,
        push_threshold: float = 0.45,
        push_waypoint_count: int = 3,
    ):
        self.sample_dir = sample_dir
        self.offsets = [int(o) for o in (offsets or [3, 6, 10, 15, 20])]

        if push_config is not None:
            self.push_config = push_config
        elif object_center_xy is not None:
            center = np.asarray(object_center_xy, dtype=np.float32)
            half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            self.push_config = PushConfig(
                skill="around",
                bbox_min=center - half,
                bbox_max=center + half,
                center=center,
                push_distance=push_distance,
                push_threshold=push_threshold,
                push_waypoint_count=push_waypoint_count,
            )
        else:
            self.push_config = None

        with open(os.path.join(sample_dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        with open(os.path.join(sample_dir, "frames.json"), encoding="utf-8") as f:
            frames = json.load(f)

        self.expert_positions = np.asarray(
            [fr["camera_position_std"] for fr in frames], dtype=np.float32
        )
        expert_forwards = np.asarray(
            [fr["camera_forward_std"][:2] for fr in frames], dtype=np.float32
        )
        norms = np.linalg.norm(expert_forwards, axis=1, keepdims=True)
        self.expert_forwards = expert_forwards / np.maximum(norms, 1e-8)

        self.judge = SemanticCompletionJudge(self.expert_positions, self.meta)
        self._last_expert_index = 0
        self._match_window = max(max(self.offsets) + 10, 30)
        clearance_provider = make_clearance_provider(sample_dir, data_root, self.meta)
        self.completion_judge = CompletionJudge(
            self.meta,
            clearance_provider=clearance_provider,
            cfg=completion_cfg or CompletionConfig(),
        )

    def nearest_expert_index(self, position: np.ndarray) -> Tuple[int, float]:
        """Return a monotonic ``(expert_frame_index, euclidean_distance)``.

        DAgger labels are generated along one student rollout. Searching the
        whole expert trajectory independently at every frame can jump backward
        or across nearby path segments, producing zig-zag teacher targets. Keep
        the match monotonic and only search a short forward window from the last
        matched frame.
        """
        pos = np.asarray(position, dtype=np.float32)
        start = int(np.clip(self._last_expert_index, 0, len(self.expert_positions) - 1))
        end = min(len(self.expert_positions), start + self._match_window + 1)
        local = self.expert_positions[start:end]
        diff_local = local - pos[None, :]
        dist = np.linalg.norm(diff_local[:, :2], axis=1) * 2.0 + np.abs(diff_local[:, 2])
        i0 = start + int(np.argmin(dist))
        self._last_expert_index = max(self._last_expert_index, i0)
        d_expert = float(np.linalg.norm(self.expert_positions[i0] - pos))
        return i0, d_expert

    @staticmethod
    def _to_body(delta_world: np.ndarray, body_x: np.ndarray, body_y: np.ndarray) -> np.ndarray:
        return np.array(
            [
                float(np.dot(delta_world[:2], body_x)),
                float(np.dot(delta_world[:2], body_y)),
                float(delta_world[2]),
            ],
            dtype=np.float32,
        )

    def label_waypoints(
        self,
        student_position: np.ndarray,
        student_forward: np.ndarray,
        surface_distance: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        position = np.asarray(student_position, dtype=np.float32)
        body_x, body_y = body_frame_basis(np.asarray(student_forward, dtype=np.float32)[:2])

        i0, d_expert = self.nearest_expert_index(position)
        K = len(self.offsets)
        waypoints = np.zeros((K, 3), dtype=np.float32)
        valid = np.zeros((K,), dtype=np.float32)
        for k, off in enumerate(self.offsets):
            j = i0 + off
            if j >= len(self.expert_positions):
                continue
            delta_world = self.expert_positions[j] - position
            waypoints[k, 0] = float(np.dot(delta_world[:2], body_x))
            waypoints[k, 1] = float(np.dot(delta_world[:2], body_y))
            waypoints[k, 2] = float(delta_world[2])
            valid[k] = 1.0

        # Recovery correction. It is only applied when the student is outside
        # the expert-trajectory tube (so we do not pull a student that is
        # already tracking the expert path) and the current or trend-predicted
        # future distance to the object surface is dangerous.
        if self.push_config is not None and surface_distance is not None:
            alpha = self._push_alpha(float(d_expert), float(surface_distance))
            if alpha > 0.0:
                self._apply_push_away(position, body_x, body_y, waypoints, valid, i0, alpha)

        return waypoints, valid

    @staticmethod
    def _smoothstep(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _push_alpha(self, d_expert: float, surface_distance: float) -> float:
        """Continuous recovery-label blend weight in ``[0, 1]``."""
        cfg = self.push_config
        width = max(float(cfg.push_blend_width), 1e-6)
        danger_alpha = self._smoothstep((cfg.push_threshold - surface_distance) / width)
        tube_alpha = self._smoothstep((d_expert - cfg.tube_radius) / max(cfg.tube_radius, 1e-6))
        return float(danger_alpha * tube_alpha)

    def _apply_push_away(
        self,
        position: np.ndarray,
        body_x: np.ndarray,
        body_y: np.ndarray,
        waypoints: np.ndarray,
        valid: np.ndarray,
        i0: int,
        alpha: float = 1.0,
    ) -> None:
        """Replace the first waypoints with a short recovery trajectory.

        ``safe subgoal -> blend(g_safe, tube point) -> tube point``. The tube
        point is the expert future position clamped to at most
        ``2 * tube_radius`` from the student, so the recovery target is "back
        into the tube", not the exact expert point.
        """
        g_safe = self._safe_subgoal(position, body_x, body_y, waypoints, valid)
        if g_safe is None:
            return

        cfg = self.push_config
        n_push = min(cfg.push_waypoint_count, int(waypoints.shape[0]))
        for k in range(n_push):
            if k == 0:
                target = g_safe
            else:
                off = self.offsets[k]
                j = i0 + off
                if j >= len(self.expert_positions):
                    continue
                p_tube = self._tube_point(position, self.expert_positions[j])
                if k == 1:
                    target = 0.5 * g_safe + 0.5 * p_tube
                else:
                    target = p_tube

            delta_world = np.asarray(target, dtype=np.float32) - position
            recovery = self._to_body(delta_world, body_x, body_y)
            if valid[k] > 0.5:
                waypoints[k] = (1.0 - alpha) * waypoints[k] + alpha * recovery
            else:
                waypoints[k] = recovery
            valid[k] = 1.0

    def _tube_point(self, position: np.ndarray, p_expert: np.ndarray) -> np.ndarray:
        """Clamp an expert future point to a tube around the student position."""
        p_expert = np.asarray(p_expert, dtype=np.float32)
        rel = p_expert - position
        dist = float(np.linalg.norm(rel))
        if dist < 1e-9:
            return p_expert
        limit = self.push_config.tube_radius * 2.0
        if dist <= limit:
            return p_expert
        return position + rel / dist * limit

    def _safe_subgoal(
        self,
        position: np.ndarray,
        body_x: np.ndarray,
        body_y: np.ndarray,
        waypoints: np.ndarray,
        valid: np.ndarray,
    ) -> Optional[np.ndarray]:
        skill = self.push_config.skill
        if skill in ("around_left", "around_right", "around"):
            return self._safe_subgoal_around(position)
        if skill == "through":
            return self._safe_subgoal_through(position)
        if skill == "over":
            return self._safe_subgoal_over(position, body_x, body_y, waypoints, valid)
        if skill == "under":
            return self._safe_subgoal_under(position, body_x, body_y, waypoints, valid)
        return None

    def _safe_subgoal_around(self, position: np.ndarray) -> Optional[np.ndarray]:
        """Radial point ``push_distance`` outside the object footprint."""
        cfg = self.push_config
        pos_xy = position[:2]
        center_xy = cfg.center[:2]
        rel = pos_xy - center_xy
        rel_norm = float(np.linalg.norm(rel))
        if rel_norm < 1e-8:
            return None  # exactly on the object center; no meaningful direction

        footprint = max(
            float(np.linalg.norm(cfg.bbox_min[:2] - center_xy)),
            float(np.linalg.norm(cfg.bbox_max[:2] - center_xy)),
        )
        safe_dist = footprint + cfg.push_distance
        target_xy = center_xy + (rel / rel_norm) * safe_dist
        return np.array([target_xy[0], target_xy[1], position[2]], dtype=np.float32)

    def _safe_subgoal_through(self, position: np.ndarray) -> np.ndarray:
        """The opening centre (fallback: object bbox centre)."""
        cfg = self.push_config
        target = cfg.opening_center if cfg.opening_center is not None else cfg.center
        return np.asarray(target, dtype=np.float32)

    def _safe_subgoal_over(self, position, body_x, body_y, waypoints, valid) -> np.ndarray:
        """Keep the expert horizontal direction, force z above the bbox."""
        cfg = self.push_config
        target_z = float(cfg.bbox_max[2]) + cfg.push_distance
        target_xy = self._expert_horizontal_target(position, body_x, body_y, waypoints, valid)
        return np.array([target_xy[0], target_xy[1], target_z], dtype=np.float32)

    def _safe_subgoal_under(self, position, body_x, body_y, waypoints, valid) -> np.ndarray:
        """Keep the expert horizontal direction, cap z below the table top."""
        cfg = self.push_config
        max_z = float(cfg.bbox_max[2]) - cfg.under_clearance
        target_xy = self._expert_horizontal_target(position, body_x, body_y, waypoints, valid)
        expert_z = None
        if valid[0] == 1.0:
            expert_z = float(position[2] + waypoints[0, 2])
        target_z = min(expert_z, max_z) if expert_z is not None else max_z
        return np.array([target_xy[0], target_xy[1], target_z], dtype=np.float32)

    def _expert_horizontal_target(
        self,
        position: np.ndarray,
        body_x: np.ndarray,
        body_y: np.ndarray,
        waypoints: np.ndarray,
        valid: np.ndarray,
    ) -> np.ndarray:
        """World XY of the first expert waypoint; fallback: current heading."""
        if valid[0] == 1.0:
            delta_body = waypoints[0]
            delta_world_xy = delta_body[0] * body_x + delta_body[1] * body_y
            return position[:2] + delta_world_xy
        return position[:2] + body_x * self.push_config.push_distance

    def label_completion(
        self,
        prefix_points: List[np.ndarray] | np.ndarray,
        current_position: Optional[np.ndarray] = None,
    ) -> int:
        if current_position is None:
            prefix = np.asarray(prefix_points, dtype=float)
            if prefix.ndim != 2 or prefix.shape[0] == 0:
                return 0
            current_position = prefix[-1]
        result = self.completion_judge.update(prefix_points, current_position)
        return 1 if result.completion else 0

    def label(
        self,
        student_position: np.ndarray,
        student_forward: np.ndarray,
        prefix_points: List[np.ndarray] | np.ndarray,
        surface_distance: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        waypoints, valid = self.label_waypoints(
            student_position, student_forward, surface_distance=surface_distance
        )
        completion = self.label_completion(prefix_points, current_position=student_position)
        return waypoints, valid, completion
