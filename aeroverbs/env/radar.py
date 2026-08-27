"""Self-contained radar rendering via trimesh raycast (no Habitat).

These functions are copied verbatim from the collection script
``scripts/collect_single_trajectory_sample.py`` (the ``lidar_source=raycast``
path) so that rollout radar is bit-identical to the training data. Only
``trimesh`` + ``numpy`` are required.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np


def render_scene_glb(sample_dir: str) -> str:
    """Return the "with ground" scene glb used for radar raycasting.

    The collection pipeline renders radar against a scene variant with a ground
    plane added, stored at ``<sample_dir>/_render_scene/<scene_id>_with_ground.glb``.
    The ``_scenes`` glb recorded in meta.json contains only the object (no
    ground), so it must NOT be used for radar or collision.
    """
    scene_id = os.path.basename(sample_dir.rstrip("/"))
    return os.path.join(sample_dir, "_render_scene", scene_id + "_with_ground.glb")


def _normalize(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return vector / norm


def load_scene_mesh(scene_glb: str):
    """Load a scene ``.glb`` into a single trimesh mesh for raycasting."""
    import trimesh

    loaded_scene = trimesh.load(str(scene_glb), force="scene")
    if isinstance(loaded_scene, trimesh.Scene):
        ray_mesh = (
            loaded_scene.to_geometry()
            if hasattr(loaded_scene, "to_geometry")
            else loaded_scene.dump(concatenate=True)
        )
    else:
        ray_mesh = loaded_scene
    if not isinstance(ray_mesh, trimesh.Trimesh) or ray_mesh.is_empty:
        raise ValueError(f"raycast requires a single non-empty trimesh mesh from {scene_glb}")
    return ray_mesh


def sample_lidar_raycast(
    ray_mesh,
    eye_std,
    forward_std,
    lidar_height,
    lidar_width,
    lidar_pitch_min,
    lidar_pitch_max,
    lidar_max_depth,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the radar range image via trimesh ray-mesh intersection.

    The angular grid matches the panorama path exactly (same pitch sweep and
    same 360-degree yaw ordering). Each cell stores the true Euclidean hit
    distance along its ray, or ``lidar_max_depth`` when nothing is hit.
    Rays are cast in the scene's native z-up std frame.
    """
    pitch_deg = np.linspace(lidar_pitch_max, lidar_pitch_min, lidar_height, dtype=np.float32)
    yaw_deg = np.linspace(180.0, -180.0, lidar_width, endpoint=False, dtype=np.float32)
    pitch_grid, yaw_grid = np.meshgrid(pitch_deg, yaw_deg, indexing="ij")

    pitch_rad = np.deg2rad(pitch_grid)
    yaw_rad = np.deg2rad(yaw_grid)

    # Body frame (std z-up): x=forward, y=left, z=up.
    forward = _normalize(forward_std)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    up = world_up - float(np.dot(world_up, forward)) * forward
    up_norm = float(np.linalg.norm(up))
    up = up / up_norm if up_norm > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
    left = np.cross(up, forward)
    left_norm = float(np.linalg.norm(left))
    left = left / left_norm if left_norm > 1e-8 else np.array([0.0, 1.0, 0.0], dtype=np.float32)

    cos_pitch = np.cos(pitch_rad)
    dir_std = np.stack(
        [
            cos_pitch * np.cos(yaw_rad),
            cos_pitch * np.sin(yaw_rad),
            np.sin(pitch_rad),
        ],
        axis=-1,
    )
    world_dir_std = (
        dir_std[..., 0:1] * forward
        + dir_std[..., 1:2] * left
        + dir_std[..., 2:3] * up
    ).reshape(-1, 3).astype(np.float32)
    world_dir_std = world_dir_std / np.maximum(
        np.linalg.norm(world_dir_std, axis=1, keepdims=True), 1e-8
    )

    origins = np.tile(
        np.asarray(eye_std, dtype=np.float32).reshape(1, 3), (world_dir_std.shape[0], 1)
    )

    range_flat, valid_flat = _ray_mesh_hits(
        ray_mesh=ray_mesh,
        origins=origins,
        directions=world_dir_std,
        max_depth=float(lidar_max_depth),
    )

    range_image = range_flat.reshape(lidar_height, lidar_width)
    valid_mask = valid_flat.reshape(lidar_height, lidar_width)

    body = np.stack(
        [
            range_image * cos_pitch * np.cos(yaw_rad),
            range_image * cos_pitch * np.sin(yaw_rad),
            range_image * np.sin(pitch_rad),
        ],
        axis=-1,
    ).reshape(-1, 3).astype(np.float32)
    points_body = body[valid_flat]
    return range_image.astype(np.float32), points_body, valid_mask.astype(np.bool_), pitch_deg, yaw_deg


def _ray_mesh_hits(ray_mesh, origins, directions, max_depth):
    """Ray/mesh intersection with an accelerated engine when available."""
    try:
        locations, index_ray, _ = ray_mesh.ray.intersects_location(
            ray_origins=np.asarray(origins, dtype=np.float64),
            ray_directions=np.asarray(directions, dtype=np.float64),
            multiple_hits=False,
        )
        range_flat = np.full(len(origins), float(max_depth), dtype=np.float32)
        valid_flat = np.zeros(len(origins), dtype=np.bool_)
        if locations is not None and len(locations) > 0:
            locations = np.asarray(locations, dtype=np.float64)
            origins64 = np.asarray(origins, dtype=np.float64)
            hit_dist = np.linalg.norm(locations - origins64[index_ray], axis=1)
            for k in range(hit_dist.shape[0]):
                dist = float(hit_dist[k])
                if 0.0 < dist <= float(max_depth):
                    ray_index = int(index_ray[k])
                    if (not valid_flat[ray_index]) or dist < float(range_flat[ray_index]):
                        range_flat[ray_index] = dist
                        valid_flat[ray_index] = True
        return range_flat, valid_flat
    except Exception:
        return _ray_mesh_distances(origins, directions, ray_mesh.triangles, max_depth)


def _ray_mesh_distances(origins, directions, triangles, max_depth):
    """Nearest ray/triangle intersection distance per ray (Möller-Trumbore)."""
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.float64)

    num_rays = origins.shape[0]
    num_tris = triangles.shape[0]
    if num_tris == 0:
        return (
            np.full(num_rays, float(max_depth), dtype=np.float32),
            np.zeros(num_rays, dtype=np.bool_),
        )

    v0 = triangles[:, 0, :]
    v1 = triangles[:, 1, :]
    v2 = triangles[:, 2, :]
    edge1 = v1 - v0
    edge2 = v2 - v0

    best_dist = np.full(num_rays, float(max_depth), dtype=np.float32)
    best_valid = np.zeros(num_rays, dtype=np.bool_)
    eps = 1e-9

    chunk = max(1, int(2_000_000 // max(num_tris, 1)))
    for start in range(0, num_rays, chunk):
        stop = min(start + chunk, num_rays)
        o = origins[start:stop]
        d = directions[start:stop]

        pvec = np.cross(d[:, None, :], edge2[None, :, :])
        det = np.sum(edge1[None, :, :] * pvec, axis=2)
        parallel = np.abs(det) <= eps

        inv_det = np.zeros_like(det)
        np.divide(1.0, det, out=inv_det, where=~parallel)

        tvec = o[:, None, :] - v0[None, :, :]
        u = np.sum(tvec * pvec, axis=2) * inv_det
        qvec = np.cross(tvec, edge1[None, :, :])
        v = np.sum(d[:, None, :] * qvec, axis=2) * inv_det
        tdist = np.sum(edge2[None, :, :] * qvec, axis=2) * inv_det

        hit = (
            (~parallel)
            & (u >= 0.0)
            & (u <= 1.0)
            & (v >= 0.0)
            & ((u + v) <= 1.0)
            & (tdist > eps)
            & (tdist <= float(max_depth))
        )
        ray_best = np.where(hit, tdist, np.inf).min(axis=1)
        has_hit = np.isfinite(ray_best)
        best_dist[start:stop] = np.where(has_hit, ray_best, float(max_depth)).astype(np.float32)
        best_valid[start:stop] = has_hit

    return best_dist, best_valid
