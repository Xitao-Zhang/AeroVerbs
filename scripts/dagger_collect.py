#!/usr/bin/env python
"""Collect DAgger samples: run the student policy closed-loop and ask the
privileged teacher to label every student-visited frame.

Each collected sample has the same on-disk layout as an expert sample, so the
existing :class:`TrajectoryDataset` can load it directly. The only difference
is that the sample contains ``teacher_waypoints.npy``, ``teacher_waypoint_valid.npy``
and ``teacher_completion.npy``, which the dataset prefers over online labels.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from aeroverbs.constants import SKILL_TO_ID
from aeroverbs.data.gate_features import GateGeometry, gate_features_for_pose
from aeroverbs.data.dataset import build_sample_index
from aeroverbs.data.fast_planner_teacher import (
    FastPlannerTeacherConfig,
    FastPlannerTeacherLabeler,
)
from aeroverbs.data.transforms import preprocess_radar
from aeroverbs.env.radar import load_scene_mesh, render_scene_glb, sample_lidar_raycast
from compute_downward_range import compute_downward_ranges

# Reuse rollout helpers where possible.
from rollout import (
    CollisionChecker,
    _body_basis,
    _compute_action,
    _horizontal_forward,
    _load_collision_mesh,
    _load_model,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Collect DAgger samples from a student policy.")
    ap.add_argument("--checkpoint", required=True, help="Student checkpoint (new policy structure).")
    ap.add_argument("--data-root", default="/root/autodl-tmp/data/trajctorys", help="Expert data root.")
    ap.add_argument("--skills", default="around_left,around_right", help="Comma-separated skills.")
    ap.add_argument("--scene-ids", default=None, help="Optional comma-separated scene ids.")
    ap.add_argument("--num-per-skill", type=int, default=5, help="Max samples to collect per skill.")
    ap.add_argument("--output-dir", required=True, help="Where to write DAgger sample dirs.")
    ap.add_argument("--max-steps", type=int, default=200, help="Max rollout steps per sample.")
    ap.add_argument("--step-size", type=float, default=0.1, help="Kinematic step size in meters.")
    ap.add_argument("--waypoint-index", type=int, default=2, help="Which predicted waypoint to chase.")
    ap.add_argument("--turn-rate", type=float, default=0.45, help="Heading smoothing (0 disables).")
    ap.add_argument("--drone-radius", type=float, default=0.15, help="Collision sphere radius.")
    ap.add_argument(
        "--teacher-preferred-clearance",
        type=float,
        default=None,
        help="Optional global Teacher preferred-clearance override; omitted uses per-skill defaults.",
    )
    ap.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Fixed seed for deterministic category-stratified sample ordering.",
    )
    ap.add_argument(
        "--teacher-replan-interval",
        type=int,
        default=5,
        help="Run Fast-Planner every N recorded frames and reuse its path between replans.",
    )
    ap.add_argument(
        "--teacher-timeout",
        type=float,
        default=2.0,
        help="Post-call Fast-Planner time budget in seconds; over-budget plans fall back.",
    )
    ap.add_argument(
        "--teacher-max-iter",
        type=int,
        default=2000,
        help="Maximum kinodynamic A* expansions per search.",
    )
    ap.add_argument("--device", default="auto", help="Device (default auto).")
    return ap.parse_args()


def _sample_category(sample_dir: str) -> str:
    """Return a stable object/opening category for stratified collection."""
    meta_path = os.path.join(sample_dir, "meta.json")
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        meta = {}
    objects = meta.get("objects") or []
    obj = objects[0] if objects else {}
    asset_path = obj.get("asset_path")
    if asset_path:
        return Path(asset_path).stem
    object_type = obj.get("type")
    if object_type and str(object_type).lower() != "obstacle":
        return str(object_type)
    scene_id = str(meta.get("scene_id") or os.path.basename(sample_dir))
    skill = meta.get("skill")
    procedural_prefix = f"auto_{skill}_" if skill else None
    if procedural_prefix and scene_id.startswith(procedural_prefix):
        category = scene_id[len(procedural_prefix):]
        category_parts = category.rsplit("_", 1)
        if len(category_parts) == 2 and category_parts[1].isdigit():
            category = category_parts[0]
        if category:
            return category
    object_id = meta.get("object_id") or obj.get("id")
    if object_id:
        return str(object_id).rsplit("_", 1)[0]
    return os.path.basename(sample_dir).rsplit("_", 1)[0]


def _stratified_sample_order(samples: List[str], seed: int) -> List[str]:
    """Deterministically interleave shuffled object categories."""
    groups = {}
    for sample in sorted(samples):
        groups.setdefault(_sample_category(sample), []).append(sample)
    rng = random.Random(int(seed))
    categories = sorted(groups)
    rng.shuffle(categories)
    for category in categories:
        rng.shuffle(groups[category])

    ordered = []
    while True:
        added = False
        for category in categories:
            if groups[category]:
                ordered.append(groups[category].pop())
                added = True
        if not added:
            return ordered


def _skill_seed(seed: int, skill: str) -> int:
    return int(seed) + sum((index + 1) * ord(char) for index, char in enumerate(skill))


def _list_samples(
    data_root: str,
    skills: List[str],
    scene_ids: Optional[str],
    sample_seed: int = 42,
) -> List[str]:
    if scene_ids:
        ids = [s.strip() for s in scene_ids.split(",") if s.strip()]
        samples = []
        for skill in skills:
            for sid in ids:
                d = os.path.join(data_root, skill, sid)
                if os.path.isdir(d):
                    samples.append(d)
        return samples
    samples = []
    for skill in skills:
        candidates = build_sample_index(data_root, [skill])
        samples.extend(
            _stratified_sample_order(candidates, seed=_skill_seed(sample_seed, skill))
        )
    return samples


def _teacher_completion_is_terminal(completion: int) -> bool:
    """Stop collection once the unified semantic completion judge fires."""
    return bool(completion)


def _save_sample(sample_dir: str, frames: List[dict], raw_ranges: List[np.ndarray],
                 downward_ranges: List[np.ndarray],
                 teacher_wp: List[np.ndarray], teacher_valid: List[np.ndarray],
                 teacher_comp: List[np.ndarray], meta: dict,
                 planner_stats: Optional[dict] = None,
                 teacher_sources: Optional[List[str]] = None,
                 teacher_weights: Optional[List[float]] = None) -> str:
    os.makedirs(os.path.join(sample_dir, "frames", "lidar_range_npy"), exist_ok=True)
    for i, raw in enumerate(raw_ranges):
        np.save(os.path.join(sample_dir, "frames", "lidar_range_npy", f"{i:03d}.npy"), raw)
    with open(os.path.join(sample_dir, "frames.json"), "w", encoding="utf-8") as f:
        json.dump(frames, f)
    with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    np.save(os.path.join(sample_dir, "downward_range.npy"), np.asarray(downward_ranges, dtype=np.float32))
    np.save(os.path.join(sample_dir, "teacher_waypoints.npy"), np.asarray(teacher_wp, dtype=np.float32))
    np.save(os.path.join(sample_dir, "teacher_waypoint_valid.npy"), np.asarray(teacher_valid, dtype=np.float32))
    np.save(os.path.join(sample_dir, "teacher_completion.npy"), np.asarray(teacher_comp, dtype=np.float32))
    if teacher_sources is not None:
        np.save(
            os.path.join(sample_dir, "teacher_label_source.npy"),
            np.asarray(teacher_sources, dtype="U32"),
        )
    if teacher_weights is not None:
        np.save(
            os.path.join(sample_dir, "teacher_frame_weight.npy"),
            np.asarray(teacher_weights, dtype=np.float32),
        )
    if planner_stats is not None:
        with open(
            os.path.join(sample_dir, "teacher_planner_stats.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(planner_stats, f, indent=2)
    return sample_dir


def _collect_one(sample_dir: str, model, cfg, device, args, skill_counter: dict) -> str:
    meta = json.load(open(os.path.join(sample_dir, "meta.json"), encoding="utf-8"))
    skill = meta.get("skill") or os.path.basename(os.path.dirname(sample_dir))
    scene_glb = render_scene_glb(sample_dir)
    radar_mesh = load_scene_mesh(scene_glb)
    collision_mesh = _load_collision_mesh(sample_dir, args.data_root, meta)
    checker = CollisionChecker(collision_mesh, args.drone_radius, meta)
    teacher = FastPlannerTeacherLabeler(
        sample_dir,
        offsets=list(getattr(cfg.data, "future_offsets", [3, 6, 10, 15, 20])),
        data_root=args.data_root,
        config=FastPlannerTeacherConfig(
            collision_radius=args.drone_radius,
            preferred_clearance=args.teacher_preferred_clearance,
            replan_interval=args.teacher_replan_interval,
            timeout_s=args.teacher_timeout,
            max_iter=args.teacher_max_iter,
            optimize=True,
        ),
    )

    skill_id = torch.tensor([SKILL_TO_ID[skill]], dtype=torch.long, device=device)
    z_intent, z_skill, hidden = model.initialize(None, skill_id)

    frames = json.load(open(os.path.join(sample_dir, "frames.json"), encoding="utf-8"))
    position = np.asarray(frames[0]["camera_position_std"], dtype=np.float32)
    forward = _horizontal_forward(frames[0]["camera_forward_std"])

    H = int(getattr(cfg.data, "action_history_len", 10))
    action_history = np.zeros((H, 4), dtype=np.float32)
    action_history_mask = np.zeros((H,), dtype=np.float32)
    prev_position = position.copy()
    prev_forward = forward.copy()

    out_frames: List[dict] = []
    raw_ranges: List[np.ndarray] = []
    teacher_wp: List[np.ndarray] = []
    teacher_valid: List[np.ndarray] = []
    teacher_comp: List[np.ndarray] = []
    teacher_sources: List[str] = []
    teacher_weights: List[float] = []
    downward_ranges: List[np.ndarray] = []
    prefix_points: List[np.ndarray] = []
    gate_summary_path = os.path.join(sample_dir, "heatmap_gate_summary.json")
    gate = GateGeometry.from_summary(json.load(open(gate_summary_path, encoding="utf-8")))

    def render_radar(pos, fwd):
        raw, _, _, _, _ = sample_lidar_raycast(
            ray_mesh=radar_mesh,
            eye_std=pos,
            forward_std=fwd,
            lidar_height=cfg.data.radar_size[0],
            lidar_width=cfg.data.radar_size[1],
            lidar_pitch_min=cfg.data.fov_deg[0],
            lidar_pitch_max=cfg.data.fov_deg[1],
            lidar_max_depth=cfg.data.r_max,
        )
        return raw

    def record_frame(step_index: int, pos, fwd, raw):
        out_frames.append(
            {
                "frame_id": step_index,
                "timestamp": float(step_index) / 10.0,
                "lidar_range_npy": f"frames/lidar_range_npy/{step_index:03d}.npy",
                "camera_position_std": pos.astype(float).tolist(),
                "camera_forward_std": fwd.astype(float).tolist(),
            }
        )
        raw_ranges.append(raw)
        down_dist, down_valid = compute_downward_ranges(
            radar_mesh, np.asarray(pos, dtype=np.float32).reshape(1, 3), max_depth=cfg.data.r_max
        )
        downward_ranges.append(np.asarray([down_dist[0], float(down_valid[0])], dtype=np.float32))
        prefix_points.append(pos.copy())
        wp, valid, comp = teacher.label(pos, fwd, prefix_points)
        teacher_wp.append(wp)
        teacher_valid.append(valid * teacher.last_label_weight)
        teacher_comp.append(np.array([comp], dtype=np.float32))
        teacher_sources.append(teacher.last_label_source)
        teacher_weights.append(teacher.last_label_weight)
        return comp

    # Record the initial frame (student has not acted yet, action history is zero).
    completed = _teacher_completion_is_terminal(
        record_frame(0, position, forward, render_radar(position, forward))
    )

    for step in range(args.max_steps):
        if completed:
            break
        raw_range = raw_ranges[-1]
        radar = preprocess_radar(
            torch.from_numpy(raw_range),
            r_max=cfg.data.r_max,
            fov_deg=tuple(cfg.data.fov_deg),
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            gate_features = torch.from_numpy(
                gate_features_for_pose(gate, position, forward)
            ).unsqueeze(0).to(device)
            downward_range = torch.from_numpy(downward_ranges[-1]).unsqueeze(0).to(device)
            waypoints, _, _, hidden = model.step(
                radar,
                torch.from_numpy(action_history).unsqueeze(0).to(device),
                torch.from_numpy(action_history_mask).unsqueeze(0).to(device),
                downward_range,
                gate_features,
                z_intent,
                z_skill,
                hidden,
            )
        delta_body = waypoints[0, args.waypoint_index].cpu().numpy()
        body_x, left, up = _body_basis(forward)
        delta_world = delta_body[0] * body_x + delta_body[1] * left + delta_body[2] * up
        dist = float(np.linalg.norm(delta_world))
        if dist < 1e-8:
            break
        step_dir = delta_world / dist

        if args.turn_rate > 0:
            current_heading = float(np.arctan2(forward[1], forward[0]))
            target_heading = float(np.arctan2(step_dir[1], step_dir[0]))
            dheading = (target_heading - current_heading + np.pi) % (2.0 * np.pi) - np.pi
            dheading = float(np.clip(dheading, -args.turn_rate, args.turn_rate))
            new_heading = current_heading + dheading
            new_forward = np.array([np.cos(new_heading), np.sin(new_heading), 0.0], dtype=np.float32)
            next_position = position + new_forward * args.step_size
            next_position[2] = position[2] + step_dir[2] * args.step_size
            next_step_dir = new_forward.copy()
            next_step_dir[2] = step_dir[2]
        else:
            next_step_dir = step_dir
            next_position = position + step_dir * args.step_size

        if checker.collides(next_position):
            break  # stop before the collision frame; its prefix remains valid training data

        position = next_position
        forward = _horizontal_forward(next_step_dir)

        # Record the new frame together with the radar the student sees there.
        completed = _teacher_completion_is_terminal(
            record_frame(len(out_frames), position, forward, render_radar(position, forward))
        )
        if completed:
            break

        action = _compute_action(prev_position, prev_forward, position, forward)
        action_history = np.concatenate([action_history[1:], action[None, :]], axis=0)
        action_history_mask = np.concatenate(
            [action_history_mask[1:], np.ones((1,), dtype=np.float32)], axis=0
        )
        prev_position = position.copy()
        prev_forward = forward.copy()

    n = skill_counter.get(skill, 0)
    skill_counter[skill] = n + 1
    out_name = f"{os.path.basename(sample_dir)}__dagger{n:05d}"
    out_dir = os.path.join(args.output_dir, skill, out_name)
    _save_sample(
        out_dir, out_frames, raw_ranges, downward_ranges,
        teacher_wp, teacher_valid, teacher_comp,
        meta,
        planner_stats=teacher.stats(),
        teacher_sources=teacher_sources,
        teacher_weights=teacher_weights,
    )
    for name in ("heatmap_gate_summary.json", "heatmap_gate_region.json"):
        src = os.path.join(sample_dir, name)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(out_dir, name))
    planner_stats = teacher.stats()
    print(
        f"  saved {out_dir} frames={len(out_frames)} "
        f"replans={planner_stats['replan_successes']}/{planner_stats['replan_attempts']} "
        f"fallback={planner_stats['fallback_frames']} "
        f"mean_plan={planner_stats['mean_replan_s']:.3f}s"
    )
    return out_dir


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg = _load_model(args.checkpoint, device)
    skills = [s.strip() for s in args.skills.split(",") if s.strip()]
    samples = _list_samples(args.data_root, skills, args.scene_ids, args.sample_seed)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"collecting DAgger samples with checkpoint={args.checkpoint}")
    print(f"skills={skills}, candidate samples={len(samples)}, max per skill={args.num_per_skill}")
    print(f"category-stratified sample seed={args.sample_seed}")

    skill_counter = {skill: 0 for skill in skills}
    collected = 0
    for sample_dir in samples:
        skill = os.path.basename(os.path.dirname(sample_dir))
        if skill_counter.get(skill, 0) >= args.num_per_skill:
            continue
        print(f"[{skill}/{_sample_category(sample_dir)}] {os.path.basename(sample_dir)}")
        try:
            _collect_one(sample_dir, model, cfg, device, args, skill_counter)
            collected += 1
        except Exception as exc:
            print(f"  FAILED: {exc!r}")

    print(f"collected {collected} DAgger samples under {args.output_dir}")


if __name__ == "__main__":
    main()
