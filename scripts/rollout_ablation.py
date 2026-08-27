#!/usr/bin/env python
"""Ablation rollout: flip semantic input and/or intention heatmap for around tasks.

For an around_left / around_right scene, three perturbations are supported:
    semantic_flip : keep heatmap, swap skill id (around_left <-> around_right)
    heatmap_flip  : keep skill, mirror the heatmap horizontally
    both_flip     : swap skill id AND mirror the heatmap

The *scene* (radar, collision geometry, start/goal, and the external semantic
judge) stays untouched; only the policy's one-shot inputs are modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import rollout as R  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="Ablation rollout for around_left/around_right.")
    ap.add_argument("--checkpoint", required=True, help="Path to final.pt.")
    ap.add_argument("--data-root", default="/root/autodl-tmp/data/trajctorys")
    ap.add_argument("--skill", required=True, help="Scene skill: around_left or around_right.")
    ap.add_argument("--scene-id", required=True, help="Single scene id under --skill.")
    ap.add_argument("--ablation", required=True,
                    choices=["semantic_flip", "heatmap_flip", "both_flip"])
    ap.add_argument("--output-dir", required=True, help="Output root for this experiment.")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--step-size", type=float, default=0.1)
    ap.add_argument("--waypoint-index", type=int, default=2)
    ap.add_argument("--turn-rate", type=float, default=0.45)
    ap.add_argument("--drone-radius", type=float, default=0.35)
    ap.add_argument("--disable-collision-check", action="store_true")
    ap.add_argument("--goal-radius", type=float, default=1.0)
    ap.add_argument("--render-voxel-size", type=float, default=0.2)
    ap.add_argument("--render-voxel-stride", type=int, default=2)
    ap.add_argument("--render-region-stride", type=int, default=2)
    ap.add_argument("--no-render", action="store_true")
    return ap.parse_args()


def _flip_skill(skill: str) -> str:
    return {
        "around_left": "around_right",
        "around_right": "around_left",
    }.get(skill, skill)


def _patch_rollout_for_ablation(ablation: str):
    """Patch rollout module globals so _run_one_sample uses flipped inputs.

    The original function stays unchanged; we only swap the two global
    dependencies it uses for one-shot inputs:
      * ``rollout.SKILL_TO_ID`` (skill id mapping)
      * ``rollout.load_heatmap_npy`` (heatmap loader)
    """
    if ablation in ("semantic_flip", "both_flip"):
        base_map = R.SKILL_TO_ID

        class FlippedSkillMap(dict):
            def __getitem__(self, key):
                return base_map[_flip_skill(key)]

        R.SKILL_TO_ID = FlippedSkillMap(base_map)

    if ablation in ("heatmap_flip", "both_flip"):
        orig_load = R.load_heatmap_npy

        def flipped_load(path, size):
            hm = orig_load(path, size=size)
            return torch.flip(hm, dims=[-1])  # mirror horizontally (width axis)

        R.load_heatmap_npy = flipped_load


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model, cfg = R._load_model(args.checkpoint, device)
    print(f"loaded checkpoint: {args.checkpoint}")
    print(f"ablation: {args.ablation}  scene: {args.skill}/{args.scene_id}")

    _patch_rollout_for_ablation(args.ablation)

    os.makedirs(os.path.join(args.output_dir, "trajectories"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "renders"), exist_ok=True)
    print(f"output dir: {args.output_dir}")

    sample_dirs = R._list_samples(args.data_root, args.skill, args.scene_id)
    print(f"rollout samples: {len(sample_dirs)}")
    results_path = os.path.join(args.output_dir, "results.jsonl")

    success = 0
    with open(results_path, "a", encoding="utf-8") as f:
        for i, sample_dir in enumerate(sample_dirs, 1):
            sample_name = os.path.basename(sample_dir)
            print(f"[{i}/{len(sample_dirs)}] {sample_name}")
            try:
                result = R._run_one_sample(sample_dir, model, cfg, device, args)
                result["ablation"] = args.ablation
                result["scene_skill"] = args.skill
                result["model_skill"] = _flip_skill(args.skill) if args.ablation in ("semantic_flip", "both_flip") else args.skill
            except Exception as exc:
                result = {
                    "skill": args.skill,
                    "scene_id": sample_name,
                    "ablation": args.ablation,
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

    print(f"success: {success}/{len(sample_dirs)} ({success / max(len(sample_dirs), 1):.1%})")
    print(f"results: {results_path}")


if __name__ == "__main__":
    main()
