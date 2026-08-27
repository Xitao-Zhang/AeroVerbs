#!/usr/bin/env python
"""Rewrite absolute paths in frames.json to sample-relative paths.

The collection scripts stored absolute paths (e.g.
``/data/UAV_Basic_Skills/trajctorys/<skill>/<scene>/frames/lidar_range_npy/000.npy``)
inside ``frames.json``. Those paths break when the data is moved to another
machine. This script rewrites the two frame-level paths to be relative to the
sample directory::

    lidar_range_npy  -> frames/lidar_range_npy/000.npy
    lidar_preview    -> frames/lidar_png/000.png

Only ``frames.json`` is touched. ``meta.json`` absolute paths (scene_glb,
initial_rgb, asset_path, ...) are left as-is: they are only read by the
abandoned collection/render pipeline, never by training.

Usage::

    python scripts/rewrite_data_paths.py <data_root>
"""

from __future__ import annotations

import json
import os
import sys


def rewrite_one(frames_path: str) -> bool:
    with open(frames_path, encoding="utf-8") as f:
        frames = json.load(f)
    changed = False
    for fr in frames:
        fid = fr.get("frame_id")
        if fid is None:
            continue
        range_rel = f"frames/lidar_range_npy/{int(fid):03d}.npy"
        preview_rel = f"frames/lidar_png/{int(fid):03d}.png"
        if fr.get("lidar_range_npy") != range_rel:
            fr["lidar_range_npy"] = range_rel
            changed = True
        if fr.get("lidar_preview") != preview_rel:
            fr["lidar_preview"] = preview_rel
            changed = True
    if changed:
        with open(frames_path, "w", encoding="utf-8") as f:
            json.dump(frames, f, ensure_ascii=False)
    return changed


def main() -> None:
    data_root = sys.argv[1] if len(sys.argv) > 1 else "data/trajctorys"
    n_files = 0
    n_frames = 0
    for skill in sorted(os.listdir(data_root)):
        skill_dir = os.path.join(data_root, skill)
        if not os.path.isdir(skill_dir) or skill.startswith("_"):
            continue
        for sample in sorted(os.listdir(skill_dir)):
            sample_dir = os.path.join(skill_dir, sample)
            fp = os.path.join(sample_dir, "frames.json")
            if not os.path.isfile(fp):
                continue
            if rewrite_one(fp):
                n_files += 1
            with open(fp, encoding="utf-8") as f:
                n_frames += len(json.load(f))
    print(f"rewrote {n_files} frames.json files ({n_frames} frames total)")


if __name__ == "__main__":
    main()
