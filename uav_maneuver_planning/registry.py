from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .interfaces import Vector3


@dataclass(frozen=True)
class SemanticObject:
    object_id: str
    object_type: str
    center: Vector3
    asset_path: Optional[str] = None
    scale: Vector3 = (1.0, 1.0, 1.0)
    yaw_degrees: float = 0.0
    voxel_size: Optional[float] = 0.08
    voxel_occupancy_path: Optional[str] = None
    voxel_occupancy: Optional[Any] = field(default=None, compare=False, repr=False)
    radius: Optional[float] = None
    height: Optional[float] = None
    inner_radius: Optional[float] = None
    thickness: Optional[float] = None
    opening_center: Optional[Vector3] = None
    normal: Optional[Vector3] = None
    width: Optional[float] = None
    bottom_z: Optional[float] = None
    clearance_height: Optional[float] = None
    top_z: Optional[float] = None
    bbox_min: Optional[Vector3] = None
    bbox_max: Optional[Vector3] = None


@dataclass(frozen=True)
class ObjectLocalFrame:
    origin: Vector3
    x_axis: Vector3 = (1.0, 0.0, 0.0)
    y_axis: Vector3 = (0.0, 1.0, 0.0)
    z_axis: Vector3 = (0.0, 0.0, 1.0)


class ObjectRegistry:
    def __init__(self, objects: Iterable[SemanticObject] = ()):
        self._objects: Dict[str, SemanticObject] = {}
        for obj in objects:
            if obj.object_id in self._objects:
                raise ValueError(f"Duplicate object_id: {obj.object_id}")
            self._objects[obj.object_id] = obj

    def get(self, object_id: str) -> SemanticObject:
        try:
            return self._objects[object_id]
        except KeyError as exc:
            raise KeyError(f"Unknown object_id: {object_id}") from exc

    def list_by_type(self, object_type: str) -> List[SemanticObject]:
        return [obj for obj in self._objects.values() if obj.object_type == object_type]

    def get_local_frame(self, object_id: str) -> ObjectLocalFrame:
        obj = self.get(object_id)
        return ObjectLocalFrame(origin=obj.center)

    def __len__(self) -> int:
        return len(self._objects)

    def __iter__(self):
        return iter(self._objects.values())
