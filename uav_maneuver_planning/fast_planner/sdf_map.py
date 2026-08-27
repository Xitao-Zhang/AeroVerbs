import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
from scipy import ndimage

from ..interfaces import SemanticScene, Vector3
from ..registry import SemanticObject


@dataclass(frozen=True)
class SDFMapParams:
    resolution: float = 0.2
    obstacles_inflation: float = 0.3
    bounds_margin: float = 1.5
    virtual_ceil_height: Optional[float] = None


class SDFMap:
    """Static non-ROS equivalent of Fast-Planner's SDFMap.

    The map stores raw occupancy, inflated occupancy, and a signed Euclidean
    distance field over the inflated occupancy grid.
    """

    def __init__(self, origin: Vector3, voxel_num: Tuple[int, int, int], params: SDFMapParams):
        if params.resolution <= 0.0:
            raise ValueError("resolution must be positive")
        self.params = params
        self.origin = np.asarray(origin, dtype=float)
        self.voxel_num = np.asarray(voxel_num, dtype=np.int32)
        self.map_size = self.voxel_num.astype(float) * params.resolution
        self.map_min_boundary = self.origin
        self.map_max_boundary = self.origin + self.map_size
        shape = tuple(int(v) for v in self.voxel_num)
        self.occupancy = np.zeros(shape, dtype=bool)
        self.occupancy_inflate = np.zeros(shape, dtype=bool)
        self.distance = np.full(shape, 10000.0, dtype=float)

    @classmethod
    def from_scene(
        cls,
        scene: SemanticScene,
        params: SDFMapParams,
        obstacles: Optional[Iterable[SemanticObject]] = None,
    ) -> "SDFMap":
        obstacles = tuple(obstacles if obstacles is not None else _solid_obstacles(scene.registry))
        origin, voxel_num = _scene_grid_bounds(scene, obstacles, params)
        sdf_map = cls(origin=origin, voxel_num=voxel_num, params=params)
        sdf_map.rasterize_objects(obstacles)
        sdf_map.update_inflated_occupancy()
        sdf_map.update_esdf_3d()
        return sdf_map

    @classmethod
    def from_precomputed(
        cls,
        grid_zyx: np.ndarray,
        origin: Vector3,
        voxel_size: float,
        inflation: float,
        bounds_points: Iterable[Vector3] = (),
        bounds_margin: float = 0.0,
    ) -> "SDFMap":
        """Build a planner map by embedding an existing signed-distance grid.

        ``grid_zyx`` follows AeroVerbs' on-disk ``[z, y, x]`` convention.
        No voxelization or distance transform is performed here.
        """
        grid_zyx = np.asarray(grid_zyx, dtype=float)
        if grid_zyx.ndim != 3 or any(size < 1 for size in grid_zyx.shape):
            raise ValueError("grid_zyx must be a non-empty 3D array")
        voxel_size = float(voxel_size)
        if voxel_size <= 0.0:
            raise ValueError("voxel_size must be positive")
        inflation = float(inflation)
        source_origin = np.asarray(origin, dtype=float).reshape(3)
        source_xyz = np.transpose(grid_zyx, (2, 1, 0))
        source_shape = np.asarray(source_xyz.shape, dtype=np.int32)
        source_max = source_origin + source_shape.astype(float) * voxel_size

        points = [np.asarray(point, dtype=float).reshape(3) for point in bounds_points]
        required_min = source_origin.copy()
        required_max = source_max.copy()
        if points:
            stacked = np.vstack(points)
            required_min = np.minimum(required_min, stacked.min(axis=0) - float(bounds_margin))
            required_max = np.maximum(required_max, stacked.max(axis=0) + float(bounds_margin))

        map_origin = source_origin + np.floor(
            (required_min - source_origin) / voxel_size
        ) * voxel_size
        voxel_num = np.ceil((required_max - map_origin) / voxel_size).astype(np.int32) + 1
        params = SDFMapParams(
            resolution=voxel_size,
            obstacles_inflation=inflation,
            bounds_margin=float(bounds_margin),
        )
        sdf_map = cls(tuple(map_origin), tuple(int(v) for v in voxel_num), params)

        source_offset = np.rint((source_origin - map_origin) / voxel_size).astype(np.int32)
        slices = tuple(
            slice(int(source_offset[axis]), int(source_offset[axis] + source_shape[axis]))
            for axis in range(3)
        )
        raw_distance = np.full(tuple(int(v) for v in voxel_num), 10000.0, dtype=float)
        raw_distance[slices] = source_xyz
        sdf_map.occupancy = raw_distance <= 0.0
        sdf_map.occupancy_inflate = raw_distance <= inflation
        sdf_map.distance = raw_distance - inflation
        return sdf_map

    def rasterize_objects(self, obstacles: Iterable[SemanticObject]) -> None:
        for obj in obstacles:
            occupancy = _occupancy_for_object(obj)
            if occupancy is not None:
                self._rasterize_voxel_occupancy(occupancy)
                continue
            bbox = _object_bbox(obj)
            if bbox is not None:
                self._rasterize_bbox(bbox[0], bbox[1])
                continue
            if obj.inner_radius is not None and obj.thickness is not None:
                self._rasterize_ring(obj)
                continue
            if obj.radius is not None:
                self._rasterize_cylinder_or_sphere(obj)

    def update_inflated_occupancy(self) -> None:
        if not np.any(self.occupancy):
            self.occupancy_inflate = self.occupancy.copy()
            return
        inflation_steps = max(0, math.ceil(self.params.obstacles_inflation / self.params.resolution))
        if inflation_steps == 0:
            self.occupancy_inflate = self.occupancy.copy()
            return
        structure = np.ones((2 * inflation_steps + 1,) * 3, dtype=bool)
        self.occupancy_inflate = ndimage.binary_dilation(self.occupancy, structure=structure)

    def update_esdf_3d(self) -> None:
        if not np.any(self.occupancy_inflate):
            self.distance = np.full_like(self.distance, 10000.0, dtype=float)
            return
        outside = ndimage.distance_transform_edt(~self.occupancy_inflate) * self.params.resolution
        inside = ndimage.distance_transform_edt(self.occupancy_inflate) * self.params.resolution
        self.distance = outside.copy()
        inside_mask = self.occupancy_inflate
        self.distance[inside_mask] = -inside[inside_mask] + self.params.resolution

    def pos_to_index(self, pos) -> np.ndarray:
        return np.floor((np.asarray(pos, dtype=float) - self.origin) / self.params.resolution).astype(np.int32)

    def index_to_pos(self, idx) -> np.ndarray:
        return (np.asarray(idx, dtype=float) + 0.5) * self.params.resolution + self.origin

    def bound_index(self, idx) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int32)
        return np.minimum(np.maximum(idx, 0), self.voxel_num - 1)

    def is_in_map(self, pos) -> bool:
        pos = np.asarray(pos, dtype=float)
        return bool(np.all(pos >= self.map_min_boundary + 1e-4) and np.all(pos <= self.map_max_boundary - 1e-4))

    def is_index_in_map(self, idx) -> bool:
        idx = np.asarray(idx, dtype=np.int32)
        return bool(np.all(idx >= 0) and np.all(idx < self.voxel_num))

    def get_inflate_occupancy(self, pos) -> int:
        if not self.is_in_map(pos):
            return -1
        idx = self.pos_to_index(pos)
        return int(self.occupancy_inflate[tuple(idx)])

    def get_distance(self, pos_or_idx) -> float:
        arr = np.asarray(pos_or_idx)
        if np.issubdtype(arr.dtype, np.integer):
            idx = self.bound_index(arr)
        else:
            idx = self.bound_index(self.pos_to_index(arr))
        return float(self.distance[tuple(idx)])

    def get_surround_pts(self, pos):
        pos = np.asarray(pos, dtype=float)
        pos_m = pos - 0.5 * self.params.resolution
        base_idx = self.pos_to_index(pos_m)
        base_pos = self.index_to_pos(base_idx)
        diff = (pos - base_pos) / self.params.resolution
        pts = np.zeros((2, 2, 2, 3), dtype=float)
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    pts[x, y, z] = self.index_to_pos(base_idx + np.array([x, y, z], dtype=np.int32))
        return pts, diff

    def get_dist_with_grad_trilinear(self, pos):
        if not self.is_in_map(pos):
            return 0.0, np.zeros(3, dtype=float)
        pts, diff = self.get_surround_pts(pos)
        values = np.zeros((2, 2, 2), dtype=float)
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    values[x, y, z] = self.get_distance(pts[x, y, z])
        return _interpolate_trilinear(values, diff, self.params.resolution)

    def get_region(self):
        return self.origin.copy(), self.map_size.copy()

    def get_resolution(self) -> float:
        return self.params.resolution

    def _rasterize_voxel_occupancy(self, occupancy) -> None:
        points = np.asarray(occupancy.occupied_indices, dtype=float) + 0.5
        world_points = points * occupancy.voxel_size + np.asarray(occupancy.origin, dtype=float)
        indices = np.floor((world_points - self.origin) / self.params.resolution).astype(np.int32)
        valid = np.all((indices >= 0) & (indices < self.voxel_num), axis=1)
        indices = indices[valid]
        if len(indices):
            self.occupancy[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    def _rasterize_bbox(self, bbox_min: Vector3, bbox_max: Vector3) -> None:
        min_idx = self.bound_index(self.pos_to_index(bbox_min))
        max_idx = self.bound_index(self.pos_to_index(bbox_max))
        self.occupancy[min_idx[0] : max_idx[0] + 1, min_idx[1] : max_idx[1] + 1, min_idx[2] : max_idx[2] + 1] = True

    def _rasterize_cylinder_or_sphere(self, obj: SemanticObject) -> None:
        radius = float(obj.radius or 0.0)
        height = obj.height
        center = np.asarray(obj.center, dtype=float)
        min_pt = center - np.array([radius, radius, radius if height is None else height / 2.0])
        max_pt = center + np.array([radius, radius, radius if height is None else height / 2.0])
        min_idx = self.bound_index(self.pos_to_index(min_pt))
        max_idx = self.bound_index(self.pos_to_index(max_pt))
        xs = np.arange(min_idx[0], max_idx[0] + 1)
        ys = np.arange(min_idx[1], max_idx[1] + 1)
        zs = np.arange(min_idx[2], max_idx[2] + 1)
        xw = self.origin[0] + (xs + 0.5) * self.params.resolution
        yw = self.origin[1] + (ys + 0.5) * self.params.resolution
        zw = self.origin[2] + (zs + 0.5) * self.params.resolution
        xy_mask = (xw[:, None] - center[0]) ** 2 + (yw[None, :] - center[1]) ** 2 <= radius * radius
        if height is None:
            z_mask = (zw - center[2]) ** 2 <= radius * radius
        else:
            z_mask = (zw >= center[2] - height / 2.0) & (zw <= center[2] + height / 2.0)
        self.occupancy[np.ix_(xs, ys, zs)] |= xy_mask[:, :, None] & z_mask[None, None, :]

    def _rasterize_ring(self, obj: SemanticObject) -> None:
        inner = float(obj.inner_radius)
        outer = float(obj.radius if obj.radius is not None else inner + 0.1)
        thickness = float(obj.thickness)
        center = np.asarray(obj.center, dtype=float)
        min_pt = center - np.array([thickness / 2.0, outer, outer])
        max_pt = center + np.array([thickness / 2.0, outer, outer])
        min_idx = self.bound_index(self.pos_to_index(min_pt))
        max_idx = self.bound_index(self.pos_to_index(max_pt))
        xs = np.arange(min_idx[0], max_idx[0] + 1)
        ys = np.arange(min_idx[1], max_idx[1] + 1)
        zs = np.arange(min_idx[2], max_idx[2] + 1)
        xw = self.origin[0] + (xs + 0.5) * self.params.resolution
        yw = self.origin[1] + (ys + 0.5) * self.params.resolution
        zw = self.origin[2] + (zs + 0.5) * self.params.resolution
        radial2 = (yw[:, None] - center[1]) ** 2 + (zw[None, :] - center[2]) ** 2
        ring_mask = (radial2 >= inner * inner) & (radial2 <= outer * outer)
        x_mask = np.abs(xw - center[0]) <= thickness / 2.0
        self.occupancy[np.ix_(xs, ys, zs)] |= x_mask[:, None, None] & ring_mask[None, :, :]


def _interpolate_trilinear(values: np.ndarray, diff: np.ndarray, resolution: float):
    diff = np.asarray(diff, dtype=float)
    v00 = (1 - diff[0]) * values[0, 0, 0] + diff[0] * values[1, 0, 0]
    v01 = (1 - diff[0]) * values[0, 0, 1] + diff[0] * values[1, 0, 1]
    v10 = (1 - diff[0]) * values[0, 1, 0] + diff[0] * values[1, 1, 0]
    v11 = (1 - diff[0]) * values[0, 1, 1] + diff[0] * values[1, 1, 1]
    v0 = (1 - diff[1]) * v00 + diff[1] * v10
    v1 = (1 - diff[1]) * v01 + diff[1] * v11
    value = (1 - diff[2]) * v0 + diff[2] * v1
    grad = np.zeros(3, dtype=float)
    inv_res = 1.0 / resolution
    grad[2] = (v1 - v0) * inv_res
    grad[1] = ((1 - diff[2]) * (v10 - v00) + diff[2] * (v11 - v01)) * inv_res
    grad[0] = (1 - diff[2]) * (1 - diff[1]) * (values[1, 0, 0] - values[0, 0, 0])
    grad[0] += (1 - diff[2]) * diff[1] * (values[1, 1, 0] - values[0, 1, 0])
    grad[0] += diff[2] * (1 - diff[1]) * (values[1, 0, 1] - values[0, 0, 1])
    grad[0] += diff[2] * diff[1] * (values[1, 1, 1] - values[0, 1, 1])
    grad[0] *= inv_res
    return float(value), grad


def _scene_grid_bounds(scene: SemanticScene, obstacles: Tuple[SemanticObject, ...], params: SDFMapParams):
    points = [np.asarray(scene.start, dtype=float), np.asarray(scene.goal, dtype=float)]
    for obj in obstacles:
        bbox = _object_bbox(obj)
        if bbox is not None:
            points.append(np.asarray(bbox[0], dtype=float))
            points.append(np.asarray(bbox[1], dtype=float))
        else:
            footprint = _object_footprint_radius(obj)
            height = _object_height(obj) or footprint * 2.0
            center = np.asarray(obj.center, dtype=float)
            points.append(center - np.array([footprint, footprint, height / 2.0]))
            points.append(center + np.array([footprint, footprint, height / 2.0]))
    stacked = np.vstack(points)
    margin = params.bounds_margin
    min_pt = stacked.min(axis=0) - margin
    max_pt = stacked.max(axis=0) + margin
    if params.virtual_ceil_height is not None:
        max_pt[2] = max(max_pt[2], params.virtual_ceil_height)
    min_pt[2] = min(min_pt[2], 0.0)
    size = np.maximum(max_pt - min_pt, params.resolution * 4.0)
    voxel_num = np.ceil(size / params.resolution).astype(np.int32) + 1
    return tuple(float(v) for v in min_pt), tuple(int(v) for v in voxel_num)


def _solid_obstacles(registry) -> Tuple[SemanticObject, ...]:
    passthrough_types = {"door", "window", "opening"}
    return tuple(obj for obj in registry if obj.object_type not in passthrough_types)


def _object_bbox(obj: SemanticObject) -> Optional[Tuple[Vector3, Vector3]]:
    if obj.bbox_min is not None and obj.bbox_max is not None:
        return obj.bbox_min, obj.bbox_max
    return None


def _object_footprint_radius(obj: SemanticObject) -> float:
    if obj.radius is not None:
        return float(obj.radius)
    bbox = _object_bbox(obj)
    if bbox is not None:
        return math.hypot(bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1]) / 2.0
    if obj.width is not None:
        return float(obj.width) / 2.0
    return 0.5


def _object_height(obj: SemanticObject) -> Optional[float]:
    if obj.height is not None:
        return float(obj.height)
    bbox = _object_bbox(obj)
    if bbox is not None:
        return bbox[1][2] - bbox[0][2]
    return None


def _occupancy_for_object(obj: SemanticObject):
    if obj.voxel_occupancy is None and obj.voxel_occupancy_path is None and obj.asset_path is None:
        return None
    try:
        from ..voxel_occupancy import occupancy_for_object
    except ImportError:
        return None
    return occupancy_for_object(obj)
