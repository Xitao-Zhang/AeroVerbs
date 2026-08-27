import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from .interfaces import Vector3
from .registry import SemanticObject


@dataclass
class VoxelOccupancy:
    voxel_size: float
    origin: Vector3
    occupied_indices: np.ndarray
    bbox_min: Vector3
    bbox_max: Vector3
    _index_set: set = field(init=False, repr=False)

    def __post_init__(self):
        self.occupied_indices = np.asarray(self.occupied_indices, dtype=np.int32)
        self._index_set = {tuple(int(v) for v in index) for index in self.occupied_indices}

    @property
    def occupied_count(self) -> int:
        return int(len(self.occupied_indices))

    def contains_point(self, point: Vector3, margin: float = 0.0) -> bool:
        if not self._point_in_expanded_bbox(point, margin):
            return False

        center_index = self.point_to_index(point)
        radius = max(0, math.ceil(margin / self.voxel_size))
        if radius == 0:
            return center_index in self._index_set

        radius_sq = (margin / self.voxel_size + 0.5) ** 2
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if dx * dx + dy * dy + dz * dz > radius_sq:
                        continue
                    if (center_index[0] + dx, center_index[1] + dy, center_index[2] + dz) in self._index_set:
                        return True
        return False

    def collides_segment(
        self,
        start: Vector3,
        end: Vector3,
        margin: float = 0.0,
        step: Optional[float] = None,
    ) -> bool:
        step = step or self.voxel_size * 0.5
        start_arr = np.asarray(start, dtype=float)
        end_arr = np.asarray(end, dtype=float)
        distance = float(np.linalg.norm(end_arr - start_arr))
        samples = max(1, math.ceil(distance / step))
        for i in range(samples + 1):
            point = start_arr * (1.0 - i / samples) + end_arr * (i / samples)
            if self.contains_point(tuple(float(v) for v in point), margin=margin):
                return True
        return False

    def point_to_index(self, point: Vector3) -> Tuple[int, int, int]:
        rel = (np.asarray(point, dtype=float) - np.asarray(self.origin, dtype=float)) / self.voxel_size
        index = np.floor(rel + 1e-9).astype(np.int32)
        return int(index[0]), int(index[1]), int(index[2])

    def save(self, path: Path, metadata: Optional[Dict] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            voxel_size=np.array([self.voxel_size], dtype=np.float32),
            origin=np.asarray(self.origin, dtype=np.float32),
            occupied_indices=self.occupied_indices.astype(np.int32),
            bbox_min=np.asarray(self.bbox_min, dtype=np.float32),
            bbox_max=np.asarray(self.bbox_max, dtype=np.float32),
            metadata=np.array([json.dumps(metadata or {}, ensure_ascii=False)]),
        )

    @classmethod
    def load(cls, path: Path) -> "VoxelOccupancy":
        payload = np.load(path, allow_pickle=False)
        return cls(
            voxel_size=float(payload["voxel_size"][0]),
            origin=tuple(float(v) for v in payload["origin"]),
            occupied_indices=payload["occupied_indices"],
            bbox_min=tuple(float(v) for v in payload["bbox_min"]),
            bbox_max=tuple(float(v) for v in payload["bbox_max"]),
        )

    def _point_in_expanded_bbox(self, point: Vector3, margin: float) -> bool:
        return all(self.bbox_min[i] - margin <= point[i] <= self.bbox_max[i] + margin for i in range(3))


_OCCUPANCY_CACHE: Dict[str, VoxelOccupancy] = {}


def occupancy_for_object(obj: SemanticObject) -> Optional[VoxelOccupancy]:
    if obj.voxel_occupancy is not None:
        return obj.voxel_occupancy
    if obj.voxel_occupancy_path:
        path = Path(obj.voxel_occupancy_path)
        key = str(path.resolve())
        if key not in _OCCUPANCY_CACHE:
            _OCCUPANCY_CACHE[key] = VoxelOccupancy.load(path)
        return _OCCUPANCY_CACHE[key]
    if obj.asset_path is None or obj.voxel_size is None:
        return None

    key = _object_cache_key(obj)
    if key not in _OCCUPANCY_CACHE:
        from .habitat_assets import _load_asset_mesh, _transform_asset_mesh, resolve_habitat_asset

        mesh = _transform_asset_mesh(_load_asset_mesh(resolve_habitat_asset(obj.asset_path)), obj)
        _OCCUPANCY_CACHE[key] = voxel_occupancy_from_mesh(mesh, obj.voxel_size)
    return _OCCUPANCY_CACHE[key]


def voxel_occupancy_from_mesh(
    mesh: Any,
    voxel_size: float,
    samples_per_voxel_area: float = 6.0,
) -> VoxelOccupancy:
    from trimesh.sample import sample_surface

    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")

    bbox_min = tuple(float(v) for v in mesh.bounds[0])
    bbox_max = tuple(float(v) for v in mesh.bounds[1])
    origin = tuple(float(v - voxel_size * 0.5) for v in mesh.bounds[0])

    sample_count = max(
        len(mesh.vertices),
        int(max(mesh.area, voxel_size * voxel_size) / (voxel_size * voxel_size) * samples_per_voxel_area),
    )
    surface_points, _ = sample_surface(mesh, sample_count)
    points = np.vstack([mesh.vertices, surface_points])
    indices = np.floor((points - np.asarray(origin, dtype=float)) / voxel_size + 1e-9).astype(np.int32)
    occupied_indices = np.unique(indices, axis=0)
    return VoxelOccupancy(
        voxel_size=voxel_size,
        origin=origin,
        occupied_indices=occupied_indices,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )


def save_object_occupancy(
    obj: SemanticObject,
    mesh: Any,
    output_path: Path,
) -> Optional[VoxelOccupancy]:
    if obj.voxel_size is None:
        return None
    occupancy = voxel_occupancy_from_mesh(mesh, obj.voxel_size)
    occupancy.save(
        output_path,
        metadata={
            "object_id": obj.object_id,
            "object_type": obj.object_type,
            "asset_path": obj.asset_path,
            "voxel_size": obj.voxel_size,
        },
    )
    return occupancy


def _object_cache_key(obj: SemanticObject) -> str:
    payload = {
        "object_id": obj.object_id,
        "asset_path": obj.asset_path,
        "center": obj.center,
        "scale": obj.scale,
        "yaw_degrees": obj.yaw_degrees,
        "voxel_size": obj.voxel_size,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
