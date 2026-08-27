#!/usr/bin/env python
"""Precompute per-scene 3D SDF grids for the differentiable clearance loss.

The grid stores the signed distance from each voxel corner to the nearest
object surface (negative inside the object). Object-only meshes are used (not
the radar's ``_with_ground`` variant) so the ground plane is ignored.

Output layout::

    <out_root>/<skill>/<scene_id>.npz
        grid:       [D, H, W] float32 signed distance (D=z, H=y, W=x)
        origin:     [3] float32 world coordinate of grid[0, 0, 0]
        voxel_size: float

The mesh SDF is built by voxelizing the object mesh with ``trimesh`` and then
running ``scipy.ndimage.distance_transform_edt`` on the binary voxel grid.
This is orders of magnitude faster than per-point trimesh proximity queries and
gives a smooth signed distance field for the clearance-loss hinge.

Scenes are processed in parallel with ``ProcessPoolExecutor`` (Linux fork
start method); the grid is computed and written inside each worker so large
arrays never travel back to the parent process.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from aeroverbs.data.completion import load_object_mesh
from aeroverbs.data.dataset import build_sample_index


def parse_args():
    ap = argparse.ArgumentParser(description="Build per-scene SDF grids for clearance loss.")
    ap.add_argument("--data-root", default="/root/autodl-tmp/data/trajctorys")
    ap.add_argument("--skills", default="around_left,around_right,through,over,under")
    ap.add_argument("--voxel-size", type=float, default=0.05)
    ap.add_argument("--pad", type=float, default=1.0, help="Meters of padding around the object.")
    ap.add_argument("--out-root", default="", help="Defaults to <data_root>/_sdf.")
    ap.add_argument("--max-dim", type=int, default=256, help="Maximum grid dimension per axis.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing grids.")
    ap.add_argument("--workers", type=int, default=0, help="Number of parallel workers (0 = auto).")
    return ap.parse_args()


def _object_bboxes(meta):
    bboxes = []
    for obj in meta.get("objects", []):
        bmin = obj.get("bbox_min")
        bmax = obj.get("bbox_max")
        if bmin is not None and bmax is not None:
            bboxes.append((np.asarray(bmin, dtype=float), np.asarray(bmax, dtype=float)))
    return bboxes


def _bbox_sdf(points, bmin, bmax):
    closest = np.clip(points, bmin, bmax)
    d = np.linalg.norm(points - closest, axis=1)
    inside = np.all((points >= bmin) & (points <= bmax), axis=1)
    return np.where(inside, -d, d)


def _grid_points_xyz(bmin, bmax, shape_xyz):
    axes = [np.linspace(bmin[i], bmax[i], shape_xyz[i], dtype=np.float32) for i in range(3)]
    X, Y, Z = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    return np.stack([X, Y, Z], axis=-1).reshape(-1, 3)


def _mesh_sdf_grid(mesh, pad: float, voxel: float, max_dim: int):
    """Return ``(grid [D,H,W], origin [3], voxel)`` for a trimesh mesh.

    The returned grid is axis-transposed to the storage convention
    ``D=z, H=y, W=x`` used by ``aeroverbs.data.dataset._load_sdf`` and the
    differentiable ``grid_sample`` query.
    """
    span = mesh.bounds[1] - mesh.bounds[0] + 2.0 * pad
    if float(np.max(span)) / voxel > max_dim:
        voxel = float(np.max(span)) / max_dim

    vg = mesh.voxelized(voxel)
    vg = vg.fill()  # close small holes so watertight-ish interiors get signed
    base_origin = np.asarray(vg.origin, dtype=float)  # [x, y, z]
    base_shape = np.asarray(vg.matrix.shape)  # (nx, ny, nz) trimesh order

    pad_vox = int(np.ceil(pad / voxel))
    shape_xyz = tuple(int(s) for s in base_shape + 2 * pad_vox)
    origin = base_origin - pad_vox * voxel

    big_xyz = np.zeros(shape_xyz, dtype=bool)
    slices = tuple(
        slice(pad_vox, pad_vox + int(base_shape[i])) for i in range(3)
    )
    big_xyz[slices] = vg.matrix

    dt = ndimage.distance_transform_edt(~big_xyz, sampling=voxel)
    sdf_xyz = np.where(big_xyz, -dt, dt).astype(np.float32)

    # trimesh matrix is (x, y, z); storage wants (D=z, H=y, W=x).
    grid = np.transpose(sdf_xyz, (2, 1, 0))
    return grid, origin, voxel


def _bbox_sdf_grid(bboxes, pad: float, voxel_size: float, max_dim: int):
    """Return ``(grid [D,H,W], origin [3], voxel)`` from object bboxes."""
    bmin = np.min([b[0] for b in bboxes], axis=0)
    bmax = np.max([b[1] for b in bboxes], axis=0)
    bmin = bmin - pad
    bmax = bmax + pad
    span = bmax - bmin
    voxel = float(voxel_size)
    if float(np.max(span)) / voxel > max_dim:
        voxel = float(np.max(span)) / max_dim
    shape_xyz = tuple(int(np.ceil(s / voxel)) for s in span)
    shape_xyz = tuple(max(s, 2) for s in shape_xyz)

    points = _grid_points_xyz(bmin, bmax, shape_xyz)
    dists = np.full(len(points), 1e9, dtype=float)
    for bmin_b, bmax_b in bboxes:
        dists = np.minimum(dists, _bbox_sdf(points, bmin_b, bmax_b))
    sdf_xyz = dists.reshape(shape_xyz).astype(np.float32)
    grid = np.transpose(sdf_xyz, (2, 1, 0))
    return grid, bmin.astype(np.float32), voxel


def _build_one(
    sample_dir: str,
    data_root: str,
    out_root: str,
    voxel_size: float,
    pad: float,
    max_dim: int,
    force: bool,
) -> dict:
    """Compute and save one scene's SDF grid. Returns a small status dict."""
    meta_path = os.path.join(sample_dir, "meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    skill = meta.get("skill") or os.path.basename(os.path.dirname(sample_dir))
    scene_id = meta.get("scene_id") or os.path.basename(sample_dir)
    out_path = os.path.join(out_root, skill, f"{scene_id}.npz")

    if os.path.exists(out_path) and not force:
        return {
            "skill": skill,
            "scene_id": scene_id,
            "status": "skipped",
            "shape": None,
            "voxel": None,
            "reason": "exists",
        }

    mesh = load_object_mesh(sample_dir, data_root, meta)
    bboxes = _object_bboxes(meta)

    if mesh is not None:
        try:
            grid, origin, voxel = _mesh_sdf_grid(mesh, pad, voxel_size, max_dim)
        except Exception as exc:
            grid, origin, voxel = None, None, None
            if not bboxes:
                return {
                    "skill": skill,
                    "scene_id": scene_id,
                    "status": "failed",
                    "shape": None,
                    "voxel": None,
                    "reason": f"mesh sdf failed: {exc!r}",
                }
            grid, origin, voxel = _bbox_sdf_grid(bboxes, pad, voxel_size, max_dim)
    elif bboxes:
        grid, origin, voxel = _bbox_sdf_grid(bboxes, pad, voxel_size, max_dim)
    else:
        return {
            "skill": skill,
            "scene_id": scene_id,
            "status": "failed",
            "shape": None,
            "voxel": None,
            "reason": "no mesh or bbox",
        }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        grid=np.ascontiguousarray(grid),
        origin=np.asarray(origin, dtype=np.float32),
        voxel_size=np.float32(voxel),
    )
    return {
        "skill": skill,
        "scene_id": scene_id,
        "status": "built",
        "shape": tuple(grid.shape),
        "voxel": float(voxel),
        "reason": out_path,
    }


def _format_progress(done: int, total: int, elapsed: float, statuses: dict, last: dict) -> str:
    pct = 100.0 * done / max(total, 1)
    eta = elapsed / max(done, 1) * (total - done)
    last_str = ""
    if last:
        last_str = f" | last={last['skill']}/{last['scene_id']}"
        if last.get("shape"):
            last_str += f" shape={last['shape']}"
    return (
        f"[{done}/{total}] {pct:5.1f}% | elapsed={elapsed:.1f}s | eta={eta:.1f}s | "
        f"built={statuses.get('built', 0)} skipped={statuses.get('skipped', 0)} "
        f"failed={statuses.get('failed', 0)}{last_str}"
    )


def main():
    args = parse_args()
    skills = [s.strip() for s in args.skills.split(",") if s.strip()]
    samples = build_sample_index(args.data_root, skills)
    if not samples:
        raise SystemExit(f"No samples found under {args.data_root} for skills {skills}")

    out_root = args.out_root or os.path.join(args.data_root, "_sdf")
    workers = int(args.workers) if args.workers > 0 else min(8, os.cpu_count() or 1)
    workers = max(1, workers)

    print(
        f"build_sdf: data_root={args.data_root} skills={skills} samples={len(samples)} "
        f"voxel={args.voxel_size}m pad={args.pad}m max_dim={args.max_dim} "
        f"workers={workers} out_root={out_root} force={args.force}"
    )

    statuses = {"built": 0, "skipped": 0, "failed": 0}
    t_start = time.time()
    done = 0

    def handle_result(res: dict) -> None:
        nonlocal done
        statuses[res["status"]] += 1
        done += 1
        if res["status"] == "failed":
            print(f"  FAILED {res['skill']}/{res['scene_id']}: {res['reason']}", flush=True)
        print(
            "\r" + _format_progress(done, len(samples), time.time() - t_start, statuses, res),
            end="",
            flush=True,
        )

    if workers <= 1:
        for sample_dir in samples:
            res = _build_one(
                sample_dir, args.data_root, out_root, args.voxel_size,
                args.pad, args.max_dim, args.force,
            )
            handle_result(res)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(
                    _build_one,
                    sample_dir,
                    args.data_root,
                    out_root,
                    args.voxel_size,
                    args.pad,
                    args.max_dim,
                    args.force,
                )
                for sample_dir in samples
            ]
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                except Exception as exc:  # worker crashed
                    res = {
                        "skill": "?",
                        "scene_id": "?",
                        "status": "failed",
                        "shape": None,
                        "voxel": None,
                        "reason": repr(exc),
                    }
                handle_result(res)

    print()  # finish the progress line
    print(
        f"done: built={statuses['built']} skipped={statuses['skipped']} "
        f"failed={statuses['failed']} total_elapsed={time.time() - t_start:.1f}s"
    )


if __name__ == "__main__":
    main()
