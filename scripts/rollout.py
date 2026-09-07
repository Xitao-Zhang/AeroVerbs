#!/usr/bin/env python
"""Closed-loop rollout of the trained AeroVerbs policy.

The drone is stepped kinematically at 1 m/s and 10 Hz: every step it renders
the 40x500 trimesh radar, preprocesses it, queries the policy for future body
frame waypoints, turns the selected waypoint (default t+10) into a world
direction, steps 0.1 m toward it and updates the forward vector. A rollout ends
as success / collision / timeout and its trajectory + render are written under
``outputs/rollout/<experiment>/<run_name>/``.

Usage::

    python scripts/rollout.py --checkpoint outputs/bc_baseline/20260820-150130/final.pt \
        --data-root /root/autodl-tmp/data/trajctorys \
        --skill around_left --scene-id auto_around_left_asset_frl_apartment_bike_01_00000 \
        --output-dir outputs/rollout/bc_baseline/eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aeroverbs.config import Config
from aeroverbs.constants import SKILL_TO_ID
from aeroverbs.data.completion import CompletionJudge, make_clearance_provider
from aeroverbs.data.dataset import build_sample_index
from aeroverbs.data.gate_features import (
    GateGeometry,
    waypoints_cross_gate,
    gate_features_for_pose,
)
from aeroverbs.data.transforms import preprocess_radar
from aeroverbs.env.radar import load_scene_mesh, render_scene_glb, sample_lidar_raycast
from aeroverbs.training.trainer import make_model
from build_gate import build_gate_for_sample_dir
from compute_downward_range import compute_downward_ranges

# Imported lazily so rollout works even when matplotlib is missing (renders are
# then skipped with a warning instead of aborting the run).


def parse_args():
    ap = argparse.ArgumentParser(description="Rollout AeroVerbs policy in a closed loop.")
    ap.add_argument("--checkpoint", required=True, help="Path to final.pt (or another checkpoint).")
    ap.add_argument("--data-root", default="/root/autodl-tmp/data/trajctorys", help="Trajectory data root.")
    ap.add_argument("--skill", default=None, help="Skill/task, e.g. around_left. If omitted, all skills are used.")
    ap.add_argument("--scene-id", default=None, help="Only run one scene id under --skill (comma separated for several).")
    ap.add_argument("--output-dir", default=None, help="Output root. Defaults to outputs/rollout/<experiment>/<run_name>.")
    ap.add_argument("--experiment", default="bc_baseline", help="Experiment folder name for default output dir.")
    ap.add_argument("--run-name", default=None, help="Run folder name for default output dir (defaults to timestamp).")
    ap.add_argument("--max-steps", type=int, default=200, help="Maximum 10 Hz control steps (default 200 => 20 s).")
    ap.add_argument("--step-size", type=float, default=0.1, help="Kinematic step size in meters (default 0.1 m).")
    ap.add_argument("--waypoint-index", type=int, default=2, help="Which predicted waypoint to chase (0..4; default 2 => t+10).")
    ap.add_argument("--turn-rate", type=float, default=0.45, help="Max heading change per step in radians (0 disables smoothing, pure pursuit).")
    ap.add_argument(
        "--drone-radius",
        type=float,
        default=0.15,
        help="Physical collision sphere radius in meters (default 0.15).",
    )
    ap.add_argument("--no-collision-check", action="store_true", help="Disable collision-based early termination during rollout.")
    ap.add_argument("--goal-radius", type=float, default=1.0, help="Success distance to goal in meters.")
    ap.add_argument("--gate-correction-start-m", type=float, default=1.2, help="Only blend toward the gate when this close to the gate plane.")
    ap.add_argument("--gate-correction-blend", type=float, default=0.25, help="Blend weight for late gate correction; 0 disables correction.")
    ap.add_argument("--render-voxel-size", type=float, default=0.2, help="Voxel size for trajectory render.")
    ap.add_argument("--render-voxel-stride", type=int, default=2, help="Render every Nth occupied voxel.")
    ap.add_argument("--render-region-stride", type=int, default=2, help="Render every Nth region point.")
    ap.add_argument("--render-region", action="store_true", help="Include the semantic free-flight region in trajectory renders.")
    ap.add_argument("--render-gate", action="store_true", help="Include the 3D gate in trajectory renders.")
    ap.add_argument("--render-ground", action="store_true", help="Include ground voxels in trajectory renders.")
    ap.add_argument("--no-render", action="store_true", help="Skip trajectory PNG rendering.")
    ap.add_argument("--seed", type=int, default=0, help="Unused (policy is deterministic); kept for future noise injection.")
    return ap.parse_args()


def _load_model(checkpoint: str, device: torch.device):
    state = torch.load(checkpoint, map_location=device)
    cfg = Config()
    if isinstance(state.get("config"), dict):
        raw = state["config"]
        for section in ("model", "data", "train"):
            sub = raw.get(section)
            if isinstance(sub, dict):
                target = getattr(cfg, section)
                for key, value in sub.items():
                    if hasattr(target, key):
                        setattr(target, key, value)
    model = make_model(cfg)
    if isinstance(state.get("model"), dict):
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, cfg


def _load_or_build_gate(sample_dir: str, cfg: Config) -> GateGeometry:
    summary_path = os.path.join(sample_dir, getattr(cfg.data, "gate_summary_name", "heatmap_gate_summary.json"))
    if not os.path.exists(summary_path):
        gate = build_gate_for_sample_dir(Path(sample_dir), output_root=None)
        if gate is None:
            raise FileNotFoundError(f"missing and failed to build semantic gate: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        return GateGeometry.from_summary(json.load(f))


def _list_samples(data_root: str, skill: Optional[str], scene_id: Optional[str]) -> List[str]:
    if skill and scene_id:
        scene_ids = [s.strip() for s in scene_id.split(",") if s.strip()]
        samples = []
        for sid in scene_ids:
            d = os.path.join(data_root, skill, sid)
            if not os.path.isdir(d):
                raise FileNotFoundError(f"sample dir not found: {d}")
            samples.append(d)
        return samples
    skills = [skill] if skill else None
    if skills is None:
        from aeroverbs.constants import SKILLS

        skills = SKILLS
    return build_sample_index(data_root, skills)


def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return v / n


def _horizontal_forward(v) -> np.ndarray:
    """Return a horizontal (z = 0) unit forward vector from a 3D vector."""
    v = np.asarray(v, dtype=np.float32)
    fx, fy = float(v[0]), float(v[1])
    n = float(np.hypot(fx, fy))
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return np.array([fx / n, fy / n, 0.0], dtype=np.float32)


def _body_basis(forward):
    """Return (body_x, body_y, body_z) world basis from a forward vector.

    Training labels are computed from the 2D camera forward (``body_frame_basis``
    in ``aeroverbs/data/labels.py``): body_x is the horizontal forward, body_y is
    the horizontal left, body_z is world z. Keep the same convention here so the
    model's body-frame waypoints project back to world exactly as they were
    learned.
    """
    fx, fy = float(forward[0]), float(forward[1])
    norm = float(np.hypot(fx, fy))
    if norm < 1e-8:
        fx, fy, norm = 1.0, 0.0, 1.0
    body_x = np.array([fx / norm, fy / norm, 0.0], dtype=np.float32)
    body_y = np.array([-fy / norm, fx / norm, 0.0], dtype=np.float32)
    body_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return body_x, body_y, body_z


def _signed_angle_2d(a, b) -> float:
    """Signed angle from 2D unit vector ``a`` to ``b`` in radians."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    dot = float(np.clip(np.dot(a[:2], b[:2]), -1.0, 1.0))
    cross = float(a[0] * b[1] - a[1] * b[0])
    return float(np.arctan2(cross, dot))


def _compute_action(prev_position, prev_forward, position, forward):
    """Compute the executed kinematic action from previous to current frame.

    The action is expressed in the previous body frame:
    ``[dx_forward, dy_left, dz_up, delta_yaw]``.
    """
    delta_world = np.asarray(position, dtype=np.float32) - np.asarray(prev_position, dtype=np.float32)
    body_x, body_y, _ = _body_basis(prev_forward)
    dx = float(np.dot(delta_world[:2], body_x[:2]))
    dy = float(np.dot(delta_world[:2], body_y[:2]))
    dz = float(delta_world[2])
    delta_yaw = _signed_angle_2d(prev_forward, forward)
    return np.array([dx, dy, dz, delta_yaw], dtype=np.float32)


def _online_downward_range(ray_mesh, position: np.ndarray, max_depth: float) -> np.ndarray:
    ranges, valid = compute_downward_ranges(
        ray_mesh,
        np.asarray(position, dtype=np.float32).reshape(1, 3),
        max_depth=float(max_depth),
    )
    return np.asarray([ranges[0], float(valid[0])], dtype=np.float32)


def _gate_signed_distance(gate: GateGeometry, position: np.ndarray) -> float:
    return float(np.dot(np.asarray(position, dtype=np.float32) - gate.center, gate.normal))


def _needs_gate_correction(
    gate: GateGeometry,
    position: np.ndarray,
    forward: np.ndarray,
    waypoints: np.ndarray,
    args,
) -> bool:
    if gate.valid <= 0.5 or float(args.gate_correction_blend) <= 0.0:
        return False
    signed = _gate_signed_distance(gate, position)
    if signed < -float(args.gate_correction_start_m) or signed > 0.0:
        return False
    return not waypoints_cross_gate(gate, position, forward, waypoints, margin=0.05)


def _segment_crosses_gate(
    gate: GateGeometry,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    margin: float = 0.05,
) -> bool:
    points = np.asarray([p0, p1], dtype=np.float32)
    signed = (points - gate.center[None, :]) @ gate.normal
    s0 = float(signed[0])
    s1 = float(signed[1])
    if s0 == s1 or s0 * s1 > 0.0:
        return False
    alpha = s0 / (s0 - s1)
    if alpha < -1e-6 or alpha > 1.0 + 1e-6:
        return False
    cross = points[0] + alpha * (points[1] - points[0])
    rel = cross - gate.center
    u = float(np.dot(rel, gate.left_axis))
    v = float(np.dot(rel, gate.up_axis))
    return abs(u) <= gate.width * 0.5 + margin and abs(v) <= gate.height * 0.5 + margin


class CollisionChecker:
    """Collision sphere vs object mesh using trimesh proximity when available.

    The object-only mesh (``<data_root>/_scenes/<skill>/<scene_id>/<scene_id>.glb``)
    is used, so the ground plane does not produce false collisions. If scipy /
    proximity is unavailable the checker falls back to the per-object bounding
    boxes from meta.json expanded by the drone radius.
    """

    def __init__(self, mesh, radius: float, meta: dict):
        self.mesh = mesh
        self.radius = float(radius)
        self._proximity_query = None
        self._proximity_failed = False
        self._fallback_bboxes = []
        if mesh is not None:
            try:
                from trimesh.proximity import ProximityQuery

                self._proximity_query = ProximityQuery(mesh)
            except Exception:
                self._proximity_query = None
                self._proximity_failed = True

        for obj in meta.get("objects", []):
            bmin = obj.get("bbox_min")
            bmax = obj.get("bbox_max")
            if bmin is not None and bmax is not None:
                self._fallback_bboxes.append(
                    (np.asarray(bmin, dtype=float), np.asarray(bmax, dtype=float))
                )

    def collides(self, point: np.ndarray) -> bool:
        point = np.asarray(point, dtype=float)
        if point[2] < 0.2:
            return True  # ground clearance

        if self._proximity_query is not None:
            try:
                out = self._proximity_query.on_surface([point])
                distances = out[1] if len(out) > 1 else out[0]
                return bool(float(np.min(distances)) < self.radius)
            except Exception:
                self._proximity_query = None  # fall through to bbox fallback once

        for bmin, bmax in self._fallback_bboxes:
            if np.all(point >= bmin - self.radius) and np.all(point <= bmax + self.radius):
                return True
        return False


class AroundSemanticValidator:
    """First-version semantic success only for around_left / around_right.

    ``side_coord = dot(point_xy - object_center_xy, left_xy)``; positive means
    the point is on the left side of the object (see meta.json's
    ``around_side_validation``). around_left requires every near-object point to
    be on the left (side > 0); around_right requires every near-object point to
    be on the right (side < 0).
    """

    def __init__(self, meta: dict):
        self.skill = meta.get("skill") or meta.get("semantic_label")
        av = meta.get("around_side_validation")
        if av and av.get("left_xy") is not None:
            self.left_xy = np.asarray(av["left_xy"], dtype=float)
            self.object_center_xy = np.asarray(av["object_center_xy"], dtype=float)
            self.near_radius = float(av.get("near_radius", 1.5))
        else:
            start = np.asarray(meta["start_std"], dtype=float)
            goal = np.asarray(meta["goal_std"], dtype=float)
            forward = _norm(goal[:2] - start[:2])
            self.left_xy = np.array([-forward[1], forward[0]], dtype=float)
            obj = meta.get("objects", [{}])[0]
            self.object_center_xy = np.asarray(obj.get("center", [0.0, 0.0, 0.0]), dtype=float)[:2]
            bbox = obj.get("bbox_min")
            if bbox is not None:
                footprint = float(np.linalg.norm(np.asarray(obj["bbox_max"])[:2] - np.asarray(bbox)[:2]) / 2.0)
            else:
                footprint = float(obj.get("radius") or 0.5)
            self.near_radius = max(footprint + 0.6, 1.2)

    def validate(self, points: List[np.ndarray]) -> Tuple[bool, str]:
        pts = np.asarray(points, dtype=float)
        if len(pts) == 0:
            return False, "empty trajectory"
        rel = pts[:, :2] - self.object_center_xy[None, :]
        side = rel @ self.left_xy
        dist = np.linalg.norm(rel, axis=1)
        near = dist <= self.near_radius
        if int(near.sum()) < 2:
            return False, f"only {int(near.sum())} near-object points"
        near_side = side[near]
        if self.skill == "around_left":
            ok = bool(np.min(near_side) > 0.0)
            detail = f"min_side={float(np.min(near_side)):.3f}"
        elif self.skill == "around_right":
            ok = bool(np.max(near_side) < 0.0)
            detail = f"max_side={float(np.max(near_side)):.3f}"
        else:
            return False, f"unsupported semantic skill {self.skill}"
        return ok, detail


def _load_collision_mesh(sample_dir: str, data_root: str, meta: dict):
    """Load the object-only scene glb for collision detection."""
    import trimesh

    skill = meta.get("skill") or os.path.basename(os.path.dirname(sample_dir))
    scene_id = meta.get("scene_id") or os.path.basename(sample_dir)
    candidate = os.path.join(data_root, "_scenes", skill, scene_id, f"{scene_id}.glb")
    if not os.path.exists(candidate):
        meta_glb = meta.get("scene_glb")
        if meta_glb and os.path.exists(meta_glb):
            candidate = meta_glb
        else:
            return None
    loaded = trimesh.load(candidate, force="scene")
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        return None
    return mesh


def _render_trajectory(
    sample_dir: str,
    points: List[np.ndarray],
    skill: str,
    render_path: str,
    radar_mesh,
    args,
) -> Optional[str]:
    if args.no_render:
        return None
    try:
        import matplotlib.pyplot as plt

        from view_voxel_trajectory import (
            HeatmapGate,
            SemanticFreeRegion,
            _drop_ground_voxels,
            make_figure,
            voxel_occupancy_from_mesh,
        )
    except Exception as exc:
        print(f"  [render] skipped: {exc}")
        return None

    region = None
    if getattr(args, "render_region", False):
        region_path = os.path.join(sample_dir, f"semantic_region_{skill}.json")
        if os.path.exists(region_path):
            try:
                region = SemanticFreeRegion.load(Path(region_path))
            except Exception:
                region = None

    gate = None
    if getattr(args, "render_gate", False):
        gate_path = os.path.join(sample_dir, "heatmap_gate_summary.json")
        if os.path.exists(gate_path):
            try:
                gate = HeatmapGate.load(Path(gate_path))
            except Exception:
                gate = None

    try:
        occupancy = voxel_occupancy_from_mesh(radar_mesh, voxel_size=args.render_voxel_size)
        if not getattr(args, "render_ground", False):
            occupancy = _drop_ground_voxels(occupancy)
        fig = make_figure(
            plan={"points": [list(map(float, p)) for p in points], "semantic_label": skill},
            occupancy=occupancy,
            region=region,
            gate=gate,
            voxel_stride=max(1, args.render_voxel_stride),
            region_stride=max(1, args.render_region_stride),
        )
        Path(render_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(render_path)
        plt.close(fig)
        return render_path
    except Exception as exc:
        print(f"  [render] failed: {exc}")
        try:
            plt.close("all")
        except Exception:
            pass
        return None


def _run_one_sample(sample_dir: str, model, cfg: Config, device: torch.device, args):
    sample_name = os.path.basename(sample_dir)
    meta = json.load(open(os.path.join(sample_dir, "meta.json"), encoding="utf-8"))
    skill = meta.get("skill") or os.path.basename(os.path.dirname(sample_dir))
    if skill not in SKILL_TO_ID:
        raise ValueError(f"unknown skill {skill}")

    scene_glb = render_scene_glb(sample_dir)
    radar_mesh = load_scene_mesh(scene_glb)
    collision_mesh = _load_collision_mesh(sample_dir, args.data_root, meta)
    checker = CollisionChecker(collision_mesh, args.drone_radius, meta)
    validator = AroundSemanticValidator(meta) if skill in ("around_left", "around_right") else None
    completion_provider = make_clearance_provider(sample_dir, args.data_root, meta)
    completion_judge = CompletionJudge(
        meta,
        clearance_provider=completion_provider,
        cfg=getattr(cfg, "completion", None),
        mesh=collision_mesh,
    )
    gate = _load_or_build_gate(sample_dir, cfg)

    # One-shot skill input.  The released policy uses gate geometry rather than
    # the legacy heatmap latent, so no intention_heatmap.npy is required.
    skill_id = torch.tensor([SKILL_TO_ID[skill]], dtype=torch.long, device=device)
    z_intent, z_skill, hidden = model.initialize(None, skill_id)

    frames = json.load(open(os.path.join(sample_dir, "frames.json"), encoding="utf-8"))
    if frames:
        position = np.asarray(frames[0].get("camera_position_std"), dtype=np.float32)
        forward = np.asarray(frames[0].get("camera_forward_std"), dtype=np.float32)
    else:
        position = np.asarray(meta["start_std"], dtype=np.float32)
        goal = np.asarray(meta["goal_std"], dtype=np.float32)
        forward = _horizontal_forward(goal - position)

    # The radar and the body-frame labels are both defined with a horizontal
    # forward (z = 0). Keep the stored forward horizontal as well.
    forward = _horizontal_forward(forward)
    goal = np.asarray(meta["goal_std"], dtype=np.float32)
    start_position = position.copy()
    points: List[np.ndarray] = [position.copy()]
    status = "timeout"
    reason = ""
    last_completion_prob = 0.0
    last_progress_prob = 0.0
    gate_crossed = False
    gate_fallback_steps = 0
    semantic_ok = False
    semantic_detail = ""
    completion_detail = ""
    collision_point = None

    # Action history fed to the policy at each step. Initialized to zero /
    # masked out; after each step the actually executed kinematic action is
    # appended so the policy sees the same input distribution as in training.
    action_history_len = int(getattr(cfg.data, "action_history_len", 10))
    action_history = np.zeros((action_history_len, 4), dtype=np.float32)
    action_history_mask = np.zeros((action_history_len,), dtype=np.float32)
    prev_position = position.copy()
    prev_forward = forward.copy()

    t_start = time.time()
    for step in range(args.max_steps):
        raw_range, _, _, _, _ = sample_lidar_raycast(
            ray_mesh=radar_mesh,
            eye_std=position,
            forward_std=forward,
            lidar_height=cfg.data.radar_size[0],
            lidar_width=cfg.data.radar_size[1],
            lidar_pitch_min=cfg.data.fov_deg[0],
            lidar_pitch_max=cfg.data.fov_deg[1],
            lidar_max_depth=cfg.data.r_max,
        )
        radar = preprocess_radar(
            torch.from_numpy(raw_range),
            r_max=cfg.data.r_max,
            fov_deg=tuple(cfg.data.fov_deg),
        ).unsqueeze(0).to(device)  # [1, 3, H, W]
        downward_range = torch.from_numpy(
            _online_downward_range(radar_mesh, position, cfg.data.r_max)
        ).unsqueeze(0).to(device)
        gate_features = torch.from_numpy(
            gate_features_for_pose(gate, position, forward)
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            waypoints, completion_logit, progress_logit, hidden = model.step(
                radar,
                torch.from_numpy(action_history).unsqueeze(0).to(device),
                torch.from_numpy(action_history_mask).unsqueeze(0).to(device),
                downward_range,
                gate_features,
                z_intent,
                z_skill,
                hidden,
            )
        waypoints = waypoints[0].cpu().numpy()  # [K, 3]
        last_completion_prob = float(torch.sigmoid(completion_logit[0]).item())
        last_progress_prob = float(torch.sigmoid(progress_logit[0]).item())
        delta_body = waypoints[args.waypoint_index]

        body_x, left, up = _body_basis(forward)
        delta_world = delta_body[0] * body_x + delta_body[1] * left + delta_body[2] * up
        if (not gate_crossed) and _needs_gate_correction(gate, position, forward, waypoints, args):
            gate_delta = gate.center - position
            blend = float(np.clip(args.gate_correction_blend, 0.0, 1.0))
            delta_world = (1.0 - blend) * delta_world + blend * gate_delta
            gate_fallback_steps += 1
        dist = float(np.linalg.norm(delta_world))
        if dist < 1e-8:
            reason = "model predicted zero waypoint"
            break
        step_dir = delta_world / dist
        next_position = position + step_dir * args.step_size

        if args.turn_rate > 0:
            # Smooth the heading change: fly along the current forward and turn
            # toward the target direction at a bounded rate. This keeps the
            # kinematic path closer to the smooth training trajectories instead
            # of cutting chords straight toward far waypoints.
            current_heading = float(np.arctan2(forward[1], forward[0]))
            target_heading = float(np.arctan2(step_dir[1], step_dir[0]))
            dheading = (target_heading - current_heading + np.pi) % (2.0 * np.pi) - np.pi
            dheading = float(np.clip(dheading, -args.turn_rate, args.turn_rate))
            new_heading = current_heading + dheading
            new_forward = np.array(
                [np.cos(new_heading), np.sin(new_heading), 0.0], dtype=np.float32
            )
            next_position = position + new_forward * args.step_size
            next_position[2] = position[2] + step_dir[2] * args.step_size
            next_step_dir = new_forward.copy()
            next_step_dir[2] = step_dir[2]
        else:
            next_step_dir = step_dir

        if (not gate_crossed) and _segment_crosses_gate(gate, position, next_position, margin=0.05):
            gate_crossed = True

        if (not args.no_collision_check) and checker.collides(next_position):
            collision_point = next_position.copy()
            points.append(collision_point)
            status = "collision"
            reason = f"collision at step {step + 1}, point={next_position.tolist()}"
            break

        position = next_position
        forward = _horizontal_forward(next_step_dir)
        points.append(position.copy())

        # Update the executed-action history for the next policy step.
        action = _compute_action(prev_position, prev_forward, position, forward)
        action_history = np.concatenate([action_history[1:], action[None, :]], axis=0)
        action_history_mask = np.concatenate(
            [action_history_mask[1:], np.ones((1,), dtype=np.float32)], axis=0
        )
        prev_position = position.copy()
        prev_forward = forward.copy()

        if validator is not None:
            semantic_ok, semantic_detail = validator.validate(points)
        completion_result = completion_judge.update(points, position)
        semantic_ok = bool(completion_result.semantic_done)
        completion_detail = (
            f"semantic_done={completion_result.semantic_done}, "
            f"far_side={completion_result.far_side}, "
            f"clearance={completion_result.clearance:.3f}, "
            f"safe_clearance={completion_result.safe_clearance}, "
            f"ground_ok={completion_result.ground_ok}, "
            f"sustained={completion_result.sustained}"
        )

        goal_distance = float(np.linalg.norm(position - goal))
        if completion_result.completion:
            status = "success"
            reason = f"completion_judge: {completion_detail}; goal_distance={goal_distance:.3f}"
            break

    elapsed = time.time() - t_start
    if status == "timeout":
        goal_distance = float(np.linalg.norm(position - goal))
        reason = (
            f"max_steps={args.max_steps} reached; goal_distance={goal_distance:.3f}, "
            f"semantic_ok={semantic_ok} ({semantic_detail or completion_detail or 'n/a'})"
        )

    # Persist trajectory in the view script's format.
    trajectory = {
        "points": [[float(v) for v in p] for p in points],
        "semantic_label": skill,
    }
    rel_traj = os.path.join("trajectories", f"{skill}__{sample_name}.json")
    traj_path = os.path.join(args.output_dir, rel_traj)
    Path(traj_path).parent.mkdir(parents=True, exist_ok=True)
    with open(traj_path, "w", encoding="utf-8") as f:
        json.dump(trajectory, f)

    render_rel = os.path.join("renders", f"{skill}__{sample_name}.png")
    render_path = os.path.join(args.output_dir, render_rel)
    rendered = _render_trajectory(sample_dir, points, skill, render_path, radar_mesh, args)

    result = {
        "skill": skill,
        "scene_id": sample_name,
        "status": status,
        "steps": len(points) - 1,
        "trajectory_points": len(points),
        "goal_distance": float(np.linalg.norm(position - goal)),
        "start_position": start_position.tolist(),
        "final_position": position.tolist(),
        "goal": goal.tolist(),
        "semantic_ok": semantic_ok,
        "semantic_detail": semantic_detail,
        "completion_detail": completion_detail,
        "last_completion_prob": last_completion_prob,
        "last_progress_prob": last_progress_prob,
        "gate_crossed": gate_crossed,
        "gate_fallback_steps": gate_fallback_steps,
        "collision_point": None if collision_point is None else collision_point.tolist(),
        "reason": reason,
        "trajectory": rel_traj,
        "render": rendered if rendered is None else render_rel,
        "elapsed_s": round(elapsed, 3),
    }
    return result


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model, cfg = _load_model(args.checkpoint, device)
    print(f"loaded checkpoint: {args.checkpoint}")

    if args.output_dir is None:
        run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
        args.output_dir = os.path.join("outputs", "rollout", args.experiment, run_name)
    os.makedirs(os.path.join(args.output_dir, "trajectories"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "renders"), exist_ok=True)
    print(f"output dir: {args.output_dir}")

    samples = _list_samples(args.data_root, args.skill, args.scene_id)
    print(f"rollout samples: {len(samples)}")
    results_path = os.path.join(args.output_dir, "results.jsonl")

    success = 0
    with open(results_path, "a", encoding="utf-8") as f:
        for i, sample_dir in enumerate(samples, 1):
            sample_name = os.path.basename(sample_dir)
            print(f"[{i}/{len(samples)}] {sample_name}")
            try:
                result = _run_one_sample(sample_dir, model, cfg, device, args)
            except Exception as exc:
                result = {
                    "skill": args.skill or os.path.basename(os.path.dirname(sample_dir)),
                    "scene_id": sample_name,
                    "status": "error",
                    "reason": repr(exc),
                }
                print(f"  ERROR: {exc!r}")
            f.write(json.dumps(result) + "\n")
            f.flush()
            if result.get("status") == "success":
                success += 1
            print(
                f"  status={result.get('status')} steps={result.get('steps')} "
                f"goal_dist={result.get('goal_distance', float('nan')):.3f} "
                f"semantic_ok={result.get('semantic_ok')}"
            )

    print(f"success: {success}/{len(samples)} ({success / max(len(samples), 1):.1%})")
    print(f"results: {results_path}")


if __name__ == "__main__":
    main()
