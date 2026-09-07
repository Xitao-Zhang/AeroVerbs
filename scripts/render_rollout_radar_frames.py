#!/usr/bin/env python3
"""Render radar range images along previously saved rollout trajectories.

The range sampling is identical to the rollout/collection raycast path.  For
each trajectory point this script saves the raw range image (``.npy``), a
valid-hit mask, and a grayscale visualization (``.png``). Images and arrays
are stored in separate subdirectories.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aeroverbs.env.radar import load_scene_mesh, render_scene_glb, sample_lidar_raycast


DEFAULT_ROLLOUT = REPO_ROOT / "outputs/rollout/procedural_three_tasks_dynamic_through"
DEFAULT_SAMPLES = REPO_ROOT / "generated_scenes/procedural_test_tasks"


def horizontal_normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).copy()
    vec[2] = 0.0
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.asarray(fallback, dtype=np.float32)
    return vec / norm


def trajectory_forward(points: np.ndarray, index: int, initial: np.ndarray) -> np.ndarray:
    """Estimate a horizontal camera forward from the trajectory tangent."""
    if index > 0:
        candidate = points[index] - points[index - 1]
        return horizontal_normalize(candidate, initial)
    return horizontal_normalize(initial, np.array([1.0, 0.0, 0.0], dtype=np.float32))


def render_one(trajectory_path: Path, sample_root: Path, output_root: Path) -> dict:
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    points = np.asarray(trajectory["points"], dtype=np.float32).reshape(-1, 3)
    if len(points) == 0:
        raise ValueError(f"trajectory has no points: {trajectory_path}")

    task_name = trajectory_path.stem.split("__", 1)[-1]
    sample_dir = sample_root / task_name
    meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    frames = json.loads((sample_dir / "frames.json").read_text(encoding="utf-8"))
    if frames:
        initial_forward = np.asarray(frames[0].get("camera_forward_std", [1, 0, 0]), dtype=np.float32)
    else:
        initial_forward = np.asarray(meta.get("goal_std", [1, 0, 0]), dtype=np.float32) - points[0]

    size = meta.get("lidar_range_image_size", meta.get("radar_size", [40, 500]))
    height, width = int(size[0]), int(size[1])
    fov = meta.get("lidar_vertical_fov_deg", meta.get("fov_deg", [-5.0, 35.0]))
    pitch_min, pitch_max = float(fov[0]), float(fov[1])
    r_max = float(meta.get("lidar_max_depth_m", meta.get("r_max", 10.0)))

    scene_glb = Path(render_scene_glb(str(sample_dir)))
    if not scene_glb.exists():
        raise FileNotFoundError(f"missing radar scene: {scene_glb}")
    ray_mesh = load_scene_mesh(str(scene_glb))

    task_output = output_root / trajectory_path.stem
    image_output = task_output / "images"
    array_output = task_output / "arrays"
    image_output.mkdir(parents=True, exist_ok=True)
    array_output.mkdir(parents=True, exist_ok=True)
    for old in (*image_output.glob("*.png"), *array_output.glob("*.npy")):
        old.unlink()

    previous_forward = horizontal_normalize(initial_forward, np.array([1, 0, 0], dtype=np.float32))
    for index, position in enumerate(points):
        forward = trajectory_forward(points, index, previous_forward)
        previous_forward = forward
        raw_range, _, valid_mask, _, _ = sample_lidar_raycast(
            ray_mesh=ray_mesh,
            eye_std=position,
            forward_std=forward,
            lidar_height=height,
            lidar_width=width,
            lidar_pitch_min=pitch_min,
            lidar_pitch_max=pitch_max,
            lidar_max_depth=r_max,
        )
        stem = f"{index:04d}"
        np.save(array_output / f"{stem}.npy", raw_range.astype(np.float32))
        np.save(array_output / f"{stem}_valid.npy", valid_mask.astype(np.bool_))
        normalized = np.clip(raw_range / r_max, 0.0, 1.0)
        # Inverted grayscale: no-hit/far range is white, near returns are dark.
        image = (255.0 * normalized).astype(np.uint8)
        Image.fromarray(image, mode="L").save(image_output / f"{stem}.png")

    return {
        "trajectory": str(trajectory_path),
        "sample_dir": str(sample_dir),
        "output_dir": str(task_output),
        "image_dir": str(image_output),
        "array_dir": str(array_output),
        "frames": int(len(points)),
        "radar_size": [height, width],
        "fov_deg": [pitch_min, pitch_max],
        "r_max": r_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, default=DEFAULT_ROLLOUT)
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    rollout_dir = args.rollout_dir.resolve()
    output_dir = (args.output_dir or rollout_dir / "radar_frames").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = rollout_dir / "trajectories"
    trajectories = sorted(trajectory_dir.glob("*.json"))
    if not trajectories:
        raise FileNotFoundError(f"no trajectory JSON files in {trajectory_dir}")

    summaries = [render_one(path, args.sample_root.resolve(), output_dir) for path in trajectories]
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    for item in summaries:
        print(f"{Path(item['output_dir']).name}: {item['frames']} frames -> {item['output_dir']}")


if __name__ == "__main__":
    main()
