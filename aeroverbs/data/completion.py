"""Unified maneuver completion judge.

The single source of truth for::

    completion(t) =
        semantic_done(t)               # per-skill relationship established (latched)
        AND far_side(t)                # current point is past the object along start->goal
        AND clearance(t) >= d_exit     # clear of the object surface
        AND ground_ok(t)               # not below the ground-clearance height
        AND the condition held for sustain_frames consecutive frames

This module is intentionally NumPy-only (trimesh is imported lazily) so it can
be used by offline relabeling scripts, the DAgger teacher and rollout without a
PyTorch dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import CompletionConfig
from .semantic_judge import SceneTarget, _winding_class

_AROUND_SKILLS = {"around_left", "around_right"}


# ---------------------------------------------------------------------------
# Task geometry: start and goal must be on opposite sides of the object.
# ---------------------------------------------------------------------------

@dataclass
class TaskGeometry:
    skill: str
    target: SceneTarget
    travel_axis: np.ndarray  # unit 2D vector pointing from near side to far side
    proj_min: float  # min object projection on travel_axis
    proj_max: float  # max object projection on travel_axis
    start_proj: float  # projection of meta start_std (relative to object center)
    goal_proj: float  # projection of meta goal_std
    valid_sides: bool  # start is on the near side, goal is on the far side

    @classmethod
    def from_meta(cls, meta: dict, mesh=None) -> "TaskGeometry":
        skill = meta.get("skill") or meta.get("semantic_label") or ""
        target = SceneTarget.from_meta(meta)

        start = np.asarray(meta.get("start_std", [0.0, 0.0, 0.0]), dtype=float)
        goal = np.asarray(meta.get("goal_std", [1.0, 0.0, 0.0]), dtype=float)
        axis_xy = goal[:2] - start[:2]
        norm = float(np.linalg.norm(axis_xy))
        if norm < 1e-8:
            axis_xy = np.array([1.0, 0.0], dtype=float)
        else:
            axis_xy = axis_xy / norm
        axis_xy = axis_xy.astype(float)

        proj_min, proj_max = _object_projection_range(target, axis_xy, mesh)

        start_rel = start[:2] - target.center[:2]
        goal_rel = goal[:2] - target.center[:2]
        start_proj = float(np.dot(start_rel, axis_xy))
        goal_proj = float(np.dot(goal_rel, axis_xy))

        # The task direction should go from the near side to the far side.
        # If meta has it reversed, flip the axis so far_side(t) keeps its
        # meaning of "past the object on the goal side".
        if goal_proj < start_proj:
            axis_xy = -axis_xy
            start_proj, goal_proj = -start_proj, -goal_proj
            proj_min, proj_max = _object_projection_range(target, axis_xy, mesh)

        valid_sides = bool(start_proj < proj_min - 0.05 and goal_proj > proj_max + 0.05)
        return cls(
            skill=skill,
            target=target,
            travel_axis=axis_xy,
            proj_min=float(proj_min),
            proj_max=float(proj_max),
            start_proj=float(start_proj),
            goal_proj=float(goal_proj),
            valid_sides=valid_sides,
        )


def _object_projection_range(
    target: SceneTarget, axis_xy: np.ndarray, mesh=None
) -> Tuple[float, float]:
    """Return ``(min, max)`` of object points projected onto ``axis_xy``.

    Uses mesh vertices when available (more accurate for rotated geometry) and
    falls back to the object bbox corners.
    """
    if mesh is not None and hasattr(mesh, "vertices") and len(mesh.vertices) > 0:
        verts = np.asarray(mesh.vertices, dtype=float)
        rel = verts[:, :2] - target.center[:2]
        projs = rel @ axis_xy
        return float(projs.min()), float(projs.max())

    if target.bbox_min is not None and target.bbox_max is not None:
        bmin = target.bbox_min[:2] - target.center[:2]
        bmax = target.bbox_max[:2] - target.center[:2]
        corners = np.array(
            [
                [bmin[0], bmin[1]],
                [bmin[0], bmax[1]],
                [bmax[0], bmin[1]],
                [bmax[0], bmax[1]],
            ],
            dtype=float,
        )
        projs = corners @ axis_xy
        return float(projs.min()), float(projs.max())

    return -0.3, 0.3


# ---------------------------------------------------------------------------
# Clearance providers (non-differentiable, used for labels and rollout).
# ---------------------------------------------------------------------------

class ClearanceProvider:
    """Return positive distance from point(s) to the nearest object surface."""

    def clearance(self, points: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class BBoxClearanceProvider(ClearanceProvider):
    """Fallback clearance from meta.json object bounding boxes."""

    def __init__(self, meta: dict):
        self.bboxes: List[Tuple[np.ndarray, np.ndarray]] = []
        for obj in meta.get("objects", []):
            bmin = obj.get("bbox_min")
            bmax = obj.get("bbox_max")
            if bmin is not None and bmax is not None:
                self.bboxes.append(
                    (np.asarray(bmin, dtype=float), np.asarray(bmax, dtype=float))
                )

    def clearance(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if not self.bboxes:
            return np.full(len(points), 1e9, dtype=float)
        dists = np.full(len(points), 1e9, dtype=float)
        for bmin, bmax in self.bboxes:
            closest = np.clip(points, bmin, bmax)
            d = np.linalg.norm(points - closest, axis=1)
            dists = np.minimum(dists, d)
        return dists


class MeshClearanceProvider(ClearanceProvider):
    """Clearance from an object-only trimesh mesh via ``ProximityQuery``."""

    def __init__(self, mesh):
        self.mesh = mesh
        self._query = None
        self._failed = False
        if mesh is not None:
            try:
                from trimesh.proximity import ProximityQuery

                self._query = ProximityQuery(mesh)
            except Exception:
                self._query = None
                self._failed = True

    def clearance(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if self._query is None:
            return np.full(len(points), 1e9, dtype=float)
        try:
            out = self._query.on_surface(points)
            if isinstance(out, tuple):
                if len(out) > 1:
                    dists = np.asarray(out[1], dtype=float).reshape(-1)
                else:
                    dists = np.asarray(out[0], dtype=float).reshape(-1)
            else:
                dists = np.asarray(out, dtype=float).reshape(-1)
            return dists
        except Exception:
            self._query = None  # fall through to bbox on subsequent calls
            return np.full(len(points), 1e9, dtype=float)


class FallbackClearanceProvider(ClearanceProvider):
    """Use a primary provider and fall back to a secondary one on failure."""

    def __init__(self, primary: ClearanceProvider, fallback: ClearanceProvider):
        self.primary = primary
        self.fallback = fallback

    def clearance(self, points: np.ndarray) -> np.ndarray:
        d = self.primary.clearance(points)
        if np.any(d >= 1e8):
            d = np.minimum(d, self.fallback.clearance(points))
        return d


def load_object_mesh(sample_dir: str, data_root: Optional[str], meta: dict):
    """Load the object-only scene glb used for collision detection.

    Mirrors ``scripts/rollout._load_collision_mesh`` and returns ``None`` when
    the mesh is unavailable so callers can fall back to bbox clearance.
    """
    import trimesh

    skill = meta.get("skill") or os.path.basename(os.path.dirname(sample_dir))
    scene_id = meta.get("scene_id") or os.path.basename(sample_dir)
    candidate = None
    if data_root:
        candidate = os.path.join(data_root, "_scenes", skill, scene_id, f"{scene_id}.glb")
    if candidate is None or not os.path.exists(candidate):
        candidate = meta.get("scene_glb")
    if not candidate or not os.path.exists(candidate):
        return None

    loaded = trimesh.load(str(candidate), force="scene")
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        return None
    return mesh


def make_clearance_provider(
    sample_dir: str, data_root: Optional[str], meta: dict, mesh=None
) -> ClearanceProvider:
    """Create a clearance provider for a sample, preferring mesh over bbox."""
    if mesh is None:
        mesh = load_object_mesh(sample_dir, data_root, meta)
    bbox_provider = BBoxClearanceProvider(meta)
    if mesh is not None:
        provider = MeshClearanceProvider(mesh)
        # Probe once so we can fall back if proximity is unusable.
        try:
            provider.clearance(np.zeros((1, 3), dtype=float))
        except Exception:
            provider = None
        if provider is not None:
            return FallbackClearanceProvider(provider, bbox_provider)
    return bbox_provider


# ---------------------------------------------------------------------------
# Per-skill semantic predicates.
# ---------------------------------------------------------------------------

def _footprint_radius(target: SceneTarget) -> float:
    if target.bbox_min is not None and target.bbox_max is not None:
        return max(float(np.linalg.norm(target.bbox_max[:2] - target.bbox_min[:2]) / 2.0), 0.3)
    return 0.5


def _near_footprint(points_xy: np.ndarray, target: SceneTarget) -> np.ndarray:
    rel = points_xy - target.center[:2][None, :]
    dist = np.linalg.norm(rel, axis=1)
    return dist <= _footprint_radius(target) + 0.8


def _crossed_opening(points: np.ndarray, target: SceneTarget, margin: float) -> bool:
    """True once the trajectory has crossed the opening plane within the opening."""
    if len(points) < 2:
        return False
    if target.opening_center is not None:
        plane_x = float(target.opening_center[0])
        y_center = float(target.opening_center[1])
        half_y = 0.5  # opening half-width fallback
        if target.bbox_min is not None and target.bbox_max is not None:
            half_y = max(float(target.bbox_max[1] - target.bbox_min[1]) / 2.0, 0.5)
    else:
        plane_x = float(target.center[0])
        y_center = float(target.center[1])
        half_y = 0.8
        if target.bbox_min is not None and target.bbox_max is not None:
            half_y = max(float(target.bbox_max[1] - target.bbox_min[1]) / 2.0, 0.4) + 0.3

    within_y = np.abs(points[:, 1] - y_center) <= half_y
    left = (points[:, 0] < plane_x) & (points[:, 0] >= plane_x - 0.3) & within_y
    right = (points[:, 0] > plane_x) & (points[:, 0] <= plane_x + 0.3) & within_y
    return bool(np.any(left)) and bool(np.any(right))


def _went_over(points: np.ndarray, target: SceneTarget, margin: float) -> bool:
    if target.bbox_max is None or len(points) == 0:
        return False
    near = _near_footprint(points[:, :2], target)
    if int(near.sum()) == 0:
        return False
    return bool(np.any(points[near, 2] > float(target.bbox_max[2]) + margin))


def _went_under(points: np.ndarray, target: SceneTarget, margin: float) -> bool:
    if target.bbox_max is None or len(points) == 0:
        return False
    near = _near_footprint(points[:, :2], target)
    if int(near.sum()) == 0:
        return False
    return bool(np.any(points[near, 2] < float(target.bbox_max[2]) - margin))


def semantic_done_predicate(
    prefix_points: np.ndarray,
    skill: str,
    target: SceneTarget,
    margin: float,
    over_margin: Optional[float] = None,
) -> bool:
    """Pure per-skill relationship predicate (no latching, no far-side check)."""
    pts = np.asarray(prefix_points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) < 2:
        return False
    if skill in _AROUND_SKILLS:
        winding = _winding_class(pts[:, :2], target.center[:2])
        return winding == skill
    if skill == "through":
        return _crossed_opening(pts, target, margin)
    if skill == "over":
        return _went_over(pts, target, margin if over_margin is None else over_margin)
    if skill == "under":
        return _went_under(pts, target, margin)
    return False


# ---------------------------------------------------------------------------
# Completion judge.
# ---------------------------------------------------------------------------

@dataclass
class CompletionResult:
    semantic_done: bool
    far_side: bool
    clearance: float
    safe_clearance: bool
    ground_ok: bool
    sustained: bool
    completion: bool


class CompletionJudge:
    """Online + offline unified completion judge.

    ``semantic_done`` and ``completion`` are latched so training labels are
    monotonic: once the maneuver is complete it stays complete.
    """

    def __init__(
        self,
        meta: dict,
        clearance_provider: Optional[ClearanceProvider] = None,
        cfg: Optional[CompletionConfig] = None,
        mesh=None,
    ):
        self.cfg = cfg or CompletionConfig()
        self.meta = meta
        self.skill = meta.get("skill") or meta.get("semantic_label") or ""
        self.geometry = TaskGeometry.from_meta(meta, mesh=mesh)
        self.clearance_provider = clearance_provider or BBoxClearanceProvider(meta)

        self.semantic_done_latched = False
        self.completion_latched = False
        self._window: List[bool] = []

    # -- per-frame ----------------------------------------------------------

    def _clearance(self, point: np.ndarray) -> float:
        point = np.asarray(point, dtype=float).reshape(-1, 3)
        return float(self.clearance_provider.clearance(point)[0])

    def _far_side(self, point: np.ndarray) -> bool:
        point = np.asarray(point, dtype=float).reshape(3)
        rel = point[:2] - self.geometry.target.center[:2]
        proj = float(np.dot(rel, self.geometry.travel_axis))
        return proj >= self.geometry.proj_max + self.cfg.far_eps

    def update(self, prefix_points, current_position) -> CompletionResult:
        if not self.semantic_done_latched:
            self.semantic_done_latched = semantic_done_predicate(
                prefix_points,
                self.skill,
                self.geometry.target,
                self.cfg.vertical_margin,
                over_margin=getattr(self.cfg, "over_vertical_margin", None),
            )

        pos = np.asarray(current_position, dtype=float).reshape(3)
        clearance = self._clearance(pos)
        far_side = self._far_side(pos)
        safe_clearance = bool(clearance >= self.cfg.d_exit)
        ground_ok = bool(pos[2] >= self.cfg.ground_z_min)

        condition = bool(
            self.semantic_done_latched and far_side and safe_clearance and ground_ok
        )
        self._window.append(condition)
        if len(self._window) > int(self.cfg.sustain_frames):
            self._window.pop(0)
        sustained = (
            len(self._window) >= int(self.cfg.sustain_frames) and all(self._window)
        )
        self.completion_latched = self.completion_latched or sustained

        return CompletionResult(
            semantic_done=self.semantic_done_latched,
            far_side=far_side,
            clearance=clearance,
            safe_clearance=safe_clearance,
            ground_ok=ground_ok,
            sustained=sustained,
            completion=self.completion_latched,
        )

    # -- sequence convenience ----------------------------------------------

    def label_sequence(self, positions) -> Dict[str, np.ndarray]:
        """Return per-frame arrays for an entire trajectory.

        Returns arrays of shape ``[T]`` for keys ``semantic_done``, ``far_side``,
        ``clearance``, ``safe_clearance``, ``ground_ok``, ``completion``.
        """
        positions = np.asarray(positions, dtype=float)
        T = positions.shape[0]
        sem = np.zeros(T, dtype=bool)
        far = np.zeros(T, dtype=bool)
        clear = np.zeros(T, dtype=np.float32)
        safe = np.zeros(T, dtype=bool)
        ground = np.zeros(T, dtype=bool)
        comp = np.zeros(T, dtype=np.float32)

        prefix: List[np.ndarray] = []
        for t in range(T):
            prefix.append(positions[t])
            r = self.update(prefix, positions[t])
            sem[t] = r.semantic_done
            far[t] = r.far_side
            clear[t] = r.clearance
            safe[t] = r.safe_clearance
            ground[t] = r.ground_ok
            comp[t] = 1.0 if r.completion else 0.0

        return {
            "semantic_done": sem,
            "far_side": far,
            "clearance": clear,
            "safe_clearance": safe,
            "ground_ok": ground,
            "completion": comp,
        }
