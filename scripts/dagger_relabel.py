#!/usr/bin/env python
"""Relabel saved student-state DAgger trajectories with the current Teacher.

Large immutable payloads are reused through symlinks. New Teacher arrays and
planner statistics are written under a separate output root, so the original
collection is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Callable, Iterable, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from aeroverbs.data.fast_planner_teacher import (
    FastPlannerTeacherConfig,
    FastPlannerTeacherLabeler,
)


OLD_TEACHER_FILES = {
    "teacher_waypoints.npy",
    "teacher_waypoint_valid.npy",
    "teacher_completion.npy",
    "teacher_label_source.npy",
    "teacher_frame_weight.npy",
    "teacher_planner_stats.json",
    "relabel_metadata.json",
}


def parse_args():
    ap = argparse.ArgumentParser(description="Relabel existing DAgger student trajectories.")
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--data-root", default="/root/autodl-tmp/data/trajctorys")
    ap.add_argument("--skills", default="around_left,around_right,through,over,under")
    ap.add_argument("--drone-radius", type=float, default=0.15)
    ap.add_argument("--teacher-preferred-clearance", type=float, default=None)
    ap.add_argument("--teacher-replan-interval", type=int, default=5)
    ap.add_argument("--teacher-timeout", type=float, default=2.0)
    ap.add_argument("--teacher-max-iter", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0, help="Optional total sample limit for smoke tests.")
    ap.add_argument("--skip-existing", action="store_true")
    return ap.parse_args()


def _link_sample_payload(source_dir: str, output_dir: str) -> None:
    if os.path.lexists(output_dir):
        raise FileExistsError(f"relabel output already exists: {output_dir}")
    os.makedirs(output_dir)
    for name in sorted(os.listdir(source_dir)):
        if name in OLD_TEACHER_FILES:
            continue
        source = os.path.abspath(os.path.join(source_dir, name))
        os.symlink(source, os.path.join(output_dir, name), target_is_directory=os.path.isdir(source))


def _sample_dirs(input_root: str, skills: Iterable[str]) -> Iterable[str]:
    for skill in skills:
        skill_dir = os.path.join(input_root, skill)
        if not os.path.isdir(skill_dir):
            continue
        for name in sorted(os.listdir(skill_dir)):
            path = os.path.join(skill_dir, name)
            if os.path.isdir(path):
                yield path


def _relabel_one(
    source_dir: str,
    output_dir: str,
    data_root: str,
    config: FastPlannerTeacherConfig,
    teacher_factory: Callable = FastPlannerTeacherLabeler,
) -> dict:
    with open(os.path.join(source_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    skill = meta.get("skill") or os.path.basename(os.path.dirname(source_dir))
    scene_id = meta.get("scene_id")
    if not scene_id:
        raise ValueError(f"missing scene_id: {source_dir}")
    expert_dir = os.path.join(data_root, skill, scene_id)
    if not os.path.isdir(expert_dir):
        raise FileNotFoundError(f"missing original expert sample: {expert_dir}")

    teacher = teacher_factory(expert_dir, data_root=data_root, config=config)
    with open(os.path.join(source_dir, "frames.json"), encoding="utf-8") as f:
        frames = json.load(f)

    prefix = []
    waypoints = []
    validity = []
    completion = []
    sources = []
    weights = []
    for frame in frames:
        position = np.asarray(frame["camera_position_std"], dtype=np.float32)
        forward = np.asarray(frame["camera_forward_std"], dtype=np.float32)
        prefix.append(position.copy())
        wp, valid, comp = teacher.label(position, forward, prefix)
        weight = float(teacher.last_label_weight)
        waypoints.append(wp)
        validity.append(valid * weight)
        completion.append([comp])
        sources.append(teacher.last_label_source)
        weights.append(weight)

    _link_sample_payload(source_dir, output_dir)
    np.save(os.path.join(output_dir, "teacher_waypoints.npy"), np.asarray(waypoints, dtype=np.float32))
    np.save(os.path.join(output_dir, "teacher_waypoint_valid.npy"), np.asarray(validity, dtype=np.float32))
    np.save(os.path.join(output_dir, "teacher_completion.npy"), np.asarray(completion, dtype=np.float32))
    np.save(os.path.join(output_dir, "teacher_label_source.npy"), np.asarray(sources, dtype="U32"))
    np.save(os.path.join(output_dir, "teacher_frame_weight.npy"), np.asarray(weights, dtype=np.float32))
    stats = teacher.stats()
    stats["relabel_source_sample"] = os.path.abspath(source_dir)
    stats["original_expert_sample"] = os.path.abspath(expert_dir)
    with open(os.path.join(output_dir, "teacher_planner_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(output_dir, "relabel_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_sample": os.path.abspath(source_dir),
                "expert_sample": os.path.abspath(expert_dir),
                "label_source_counts": dict(Counter(sources)),
            },
            f,
            indent=2,
        )
    return stats


def main() -> int:
    args = parse_args()
    if os.path.abspath(args.input_root) == os.path.abspath(args.output_root):
        raise SystemExit("--output-root must differ from --input-root")
    skills = [value.strip() for value in args.skills.split(",") if value.strip()]
    config = FastPlannerTeacherConfig(
        collision_radius=args.drone_radius,
        preferred_clearance=args.teacher_preferred_clearance,
        replan_interval=args.teacher_replan_interval,
        timeout_s=args.teacher_timeout,
        max_iter=args.teacher_max_iter,
        optimize=True,
    )
    processed = 0
    totals = Counter()
    for source_dir in _sample_dirs(args.input_root, skills):
        if args.limit > 0 and processed >= args.limit:
            break
        skill = os.path.basename(os.path.dirname(source_dir))
        output_dir = os.path.join(args.output_root, skill, os.path.basename(source_dir))
        if os.path.lexists(output_dir) and args.skip_existing:
            print(f"skip existing: {output_dir}")
            continue
        print(f"[{processed + 1}] relabel {source_dir}")
        stats = _relabel_one(source_dir, output_dir, args.data_root, config)
        totals.update(stats.get("label_source_counts", {}))
        processed += 1
        print(
            f"  sources={stats.get('label_source_counts', {})} "
            f"replans={stats['replan_successes']}/{stats['replan_attempts']}"
        )
    print(f"relabelled {processed} samples under {args.output_root}")
    print(f"label sources: {dict(totals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
