#!/usr/bin/env python3
"""Compute per-frame downward ray ranges for AeroVerbs trajectories.

Each sample gets a ``downward_range.npy`` with shape ``[T, 2]``:

- column 0: distance in meters along the world-down ray
- column 1: valid hit mask, 1.0 if the ray hit scene geometry, otherwise 0.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aeroverbs.env.radar import load_scene_mesh, render_scene_glb


DEFAULT_SKILLS = ("around_left", "around_right", "through", "over", "under")


def compute_downward_ranges(ray_mesh, origins: np.ndarray, max_depth: float) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest world-down ray distance and validity for each origin."""
    origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    directions = np.tile(np.array([[0.0, 0.0, -1.0]], dtype=np.float64), (origins.shape[0], 1))
    max_depth = float(max_depth)

    ranges = np.full(origins.shape[0], max_depth, dtype=np.float32)
    valid = np.zeros(origins.shape[0], dtype=np.bool_)
    if origins.shape[0] == 0:
        return ranges, valid

    locations, index_ray, _ = ray_mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=False,
    )
    if locations is None or len(locations) == 0:
        return ranges, valid

    locations = np.asarray(locations, dtype=np.float64)
    index_ray = np.asarray(index_ray, dtype=np.int64)
    hit_dist = np.linalg.norm(locations - origins[index_ray], axis=1)
    for hit_idx, ray_idx in enumerate(index_ray):
        dist = float(hit_dist[hit_idx])
        if 0.0 < dist <= max_depth and (not valid[ray_idx] or dist < float(ranges[ray_idx])):
            ranges[ray_idx] = dist
            valid[ray_idx] = True
    return ranges, valid


def compute_sample_downward_range(
    sample_dir: Path,
    *,
    max_depth: float | None = None,
    output_name: str = "downward_range.npy",
    overwrite: bool = False,
) -> Path | None:
    """Compute and save one sample's downward range file."""
    sample_dir = Path(sample_dir)
    output_path = sample_dir / output_name
    if output_path.exists() and not overwrite:
        return output_path

    meta_path = sample_dir / "meta.json"
    frames_path = sample_dir / "frames.json"
    if not meta_path.exists() or not frames_path.exists():
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frames = json.loads(frames_path.read_text(encoding="utf-8"))
    origins = np.asarray([frame["camera_position_std"] for frame in frames], dtype=np.float32)
    depth_limit = float(max_depth if max_depth is not None else meta.get("lidar_max_depth_m", 10.0))

    scene_glb = Path(render_scene_glb(str(sample_dir)))
    if not scene_glb.exists():
        raise FileNotFoundError(f"missing render scene for downward raycast: {scene_glb}")
    ray_mesh = load_scene_mesh(str(scene_glb))

    ranges, valid = compute_downward_ranges(ray_mesh, origins=origins, max_depth=depth_limit)
    payload = np.stack([ranges, valid.astype(np.float32)], axis=1).astype(np.float32)
    np.save(output_path, payload)
    return output_path


def find_sample_dirs(data_root: Path, skills: Sequence[str]) -> Iterable[Path]:
    data_root = Path(data_root)
    for skill in skills:
        skill_dir = data_root / skill
        if not skill_dir.is_dir():
            continue
        for frames_path in sorted(skill_dir.glob("*/frames.json")):
            yield frames_path.parent


def parse_skills(raw: str | None) -> tuple[str, ...]:
    if raw is None or raw.strip().lower() in {"", "all"}:
        return DEFAULT_SKILLS
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def progress_iterable(items: Sequence[Path], *, enabled: bool = True, desc: str = "samples"):
    if not enabled:
        return items
    try:
        from tqdm import tqdm
    except Exception:
        return items
    return tqdm(items, desc=desc, unit="sample")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-frame downward range rays for AeroVerbs samples.")
    parser.add_argument("--data-root", default="/root/autodl-tmp/data/trajctorys")
    parser.add_argument("--sample-dir", default=None, help="Process one sample directory instead of scanning data-root.")
    parser.add_argument("--skills", default="all", help="Comma-separated skills or 'all'.")
    parser.add_argument("--output-name", default="downward_range.npy")
    parser.add_argument("--max-depth", type=float, default=None, help="Override meta.json lidar_max_depth_m.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable the tqdm progress bar.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.sample_dir:
        sample_dirs = [Path(args.sample_dir)]
    else:
        sample_dirs = list(find_sample_dirs(Path(args.data_root), parse_skills(args.skills)))
        if args.limit is not None:
            sample_dirs = sample_dirs[: int(args.limit)]

    written = 0
    skipped = 0
    failed = 0
    for sample_dir in progress_iterable(sample_dirs, enabled=not args.no_progress, desc="downward"):
        try:
            existed = (Path(sample_dir) / args.output_name).exists()
            output_path = compute_sample_downward_range(
                Path(sample_dir),
                max_depth=args.max_depth,
                output_name=args.output_name,
                overwrite=args.overwrite,
            )
            if output_path is None:
                skipped += 1
                print(f"skip {sample_dir}")
                continue
            if existed and not args.overwrite:
                skipped += 1
                print(f"exists {output_path}")
            else:
                written += 1
                arr = np.load(output_path)
                valid_ratio = float(arr[:, 1].mean()) if arr.size else 0.0
                print(
                    f"wrote {output_path} shape={tuple(arr.shape)} "
                    f"range=({arr[:, 0].min():.3f},{arr[:, 0].max():.3f}) "
                    f"valid={valid_ratio:.2f}"
                )
        except Exception as exc:
            failed += 1
            print(f"failed {sample_dir}: {exc}", file=sys.stderr)

    print(f"done: written={written} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
