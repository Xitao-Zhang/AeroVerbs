"""Semantic completion judge based on a lightweight trajectory signature.

The signature is a practical approximation of a *uniform visibility
deformation* class: it describes how a trajectory deforms around / through /
over / under the target object. A student prefix is considered semantically
complete when its signature matches the teacher trajectory's signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SceneTarget:
    center: np.ndarray
    bbox_min: Optional[np.ndarray] = None
    bbox_max: Optional[np.ndarray] = None
    opening_center: Optional[np.ndarray] = None
    normal: Optional[np.ndarray] = None
    width: Optional[float] = None
    height: Optional[float] = None
    clearance_height: Optional[float] = None

    @classmethod
    def from_meta(cls, meta: dict) -> "SceneTarget":
        obj = None
        for candidate in meta.get("objects", []):
            if candidate.get("type") in ("obstacle", "door", "table", "window", "ring"):
                obj = candidate
                break
        if obj is None and meta.get("objects"):
            obj = meta["objects"][0]
        if obj is None:
            return cls(center=np.array([0.0, 0.0, 0.0]))

        center = np.asarray(obj.get("center", [0.0, 0.0, 0.0]), dtype=float)
        bbox_min = np.asarray(obj["bbox_min"], dtype=float) if obj.get("bbox_min") is not None else None
        bbox_max = np.asarray(obj["bbox_max"], dtype=float) if obj.get("bbox_max") is not None else None
        opening_center = (
            np.asarray(obj["opening_center"], dtype=float)
            if obj.get("opening_center") is not None
            else None
        )
        normal = (
            np.asarray(obj["normal"], dtype=float)
            if obj.get("normal") is not None
            else None
        )
        return cls(
            center=center,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            opening_center=opening_center,
            normal=normal,
            width=float(obj["width"]) if obj.get("width") is not None else None,
            height=float(obj["height"]) if obj.get("height") is not None else None,
            clearance_height=(
                float(obj["clearance_height"])
                if obj.get("clearance_height") is not None
                else None
            ),
        )


def _winding_class(points_xy: np.ndarray, center_xy: np.ndarray) -> str:
    """Classify the 2D winding direction of a trajectory around the object."""
    if len(points_xy) < 2:
        return "none"
    rel = points_xy - center_xy[None, :]
    dist = np.linalg.norm(rel, axis=1)
    valid = dist > 0.05
    if int(valid.sum()) < 2:
        return "none"
    angles = np.arctan2(rel[valid, 1], rel[valid, 0])
    deltas = np.diff(angles)
    deltas = (deltas + np.pi) % (2.0 * np.pi) - np.pi
    total = float(np.sum(deltas))
    # In this dataset the path wraps clockwise around the object for
    # around_left (negative accumulated angle) and counter-clockwise for
    # around_right (positive accumulated angle).
    if total < -0.7 * np.pi:
        return "around_left"
    if total > 0.7 * np.pi:
        return "around_right"
    return "none"


def _vertical_class(points: np.ndarray, target: SceneTarget) -> str:
    """Classify whether a trajectory passes over / under / through the object."""
    if target.bbox_min is None or target.bbox_max is None or len(points) == 0:
        return "none"
    bmin = target.bbox_min
    bmax = target.bbox_max
    margin = 0.2
    rel_xy = points[:, :2] - target.center[None, :2]
    footprint = max(float(np.linalg.norm(bmax[:2] - bmin[:2]) / 2.0), 0.3)
    near = np.linalg.norm(rel_xy, axis=1) <= footprint + 0.8

    if int(near.sum()) == 0:
        return "none"

    z_near = points[near, 2]
    if float(z_near.max()) > float(bmax[2]) + margin:
        return "over"
    if float(z_near.min()) < float(bmin[2]) + margin and float(z_near.max()) < float(bmax[2]) - margin:
        return "under"

    # Cross the object's central vertical plane within the object's xy extent.
    plane_x = float(target.center[0])
    within_y = (points[:, 1] >= bmin[1] - margin) & (points[:, 1] <= bmax[1] + margin)
    if np.any((points[:, 0] < plane_x) & (points[:, 0] >= plane_x - 0.3) & within_y) and \
       np.any((points[:, 0] > plane_x) & (points[:, 0] <= plane_x + 0.3) & within_y):
        return "through"
    return "none"


def trajectory_signature(points: List[np.ndarray] | np.ndarray, target: SceneTarget) -> Tuple[str, str]:
    """Return ``(winding_class, vertical_class)`` for a trajectory prefix."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return "none", "none"
    winding = _winding_class(pts[:, :2], target.center[:2])
    vertical = _vertical_class(pts, target)
    return winding, vertical


class SemanticCompletionJudge:
    """Compare student prefixes against the teacher signature."""

    def __init__(self, teacher_points: List[np.ndarray] | np.ndarray, meta: dict):
        self.target = SceneTarget.from_meta(meta)
        self.teacher_signature = trajectory_signature(teacher_points, self.target)

    def is_complete(self, prefix_points: List[np.ndarray] | np.ndarray) -> bool:
        return trajectory_signature(prefix_points, self.target) == self.teacher_signature
