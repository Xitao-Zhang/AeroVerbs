#!/usr/bin/env python
"""Verify that trimesh raycast radar matches the stored training radar.

Loads a sample's scene ``.glb`` with trimesh, regenerates the first frame's
radar range image from the stored pose, and compares it against the saved
``frames/lidar_range_npy/000.npy``. A near-zero diff proves the rollout radar
renderer is bit-identical to the collection pipeline (no Habitat needed).

Usage::

    python scripts/verify_radar.py --sample-dir /root/autodl-tmp/data/trajctorys/around_left/<scene_id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aeroverbs.env.radar import load_scene_mesh, render_scene_glb, sample_lidar_raycast


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify trimesh radar vs stored radar")
    ap.add_argument("--sample-dir", required=True, help="Path to a trajectory sample dir")
    args = ap.parse_args()

    sample_dir = os.path.abspath(args.sample_dir)
    scene_glb = render_scene_glb(sample_dir)

    meta = json.load(open(os.path.join(sample_dir, "meta.json"), encoding="utf-8"))
    frames = json.load(open(os.path.join(sample_dir, "frames.json"), encoding="utf-8"))
    fr0 = frames[0]
    eye = np.asarray(fr0["camera_position_std"], dtype=np.float32)
    fwd = np.asarray(fr0["camera_forward_std"], dtype=np.float32)
    h, w = meta["lidar_range_image_size"]
    pmin, pmax = meta["lidar_vertical_fov_deg"]
    rmax = meta["lidar_max_depth_m"]

    print(f"scene glb : {scene_glb}")
    print(f"glb exists: {os.path.exists(scene_glb)}")

    mesh = load_scene_mesh(scene_glb)
    print(f"mesh      : {len(mesh.triangles)} triangles")

    rng, points, valid, _, _ = sample_lidar_raycast(mesh, eye, fwd, h, w, pmin, pmax, rmax)
    saved = np.load(os.path.join(sample_dir, "frames", "lidar_range_npy", "000.npy"))

    diff = np.abs(rng.astype(np.float32) - saved.astype(np.float32))
    print(f"generated : shape={rng.shape}, hits={int(valid.sum())}, min={rng.min():.3f}, max={rng.max():.3f}")
    print(f"saved     : shape={saved.shape}, hits={int((saved < rmax).sum())}, min={saved.min():.3f}, max={saved.max():.3f}")
    print(f"max abs diff        : {diff.max():.6f}")
    print(f"mean abs diff       : {diff.mean():.6f}")
    print(f"pixels diff > 1e-3  : {(diff > 1e-3).mean():.4%}")
    print("VERIFY:", "PASS" if diff.max() < 1e-3 else "FAIL")


if __name__ == "__main__":
    main()
