from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from .interfaces import SemanticScene, Vector3
from .registry import SemanticObject


PathLike = Union[str, Path]
DEFAULT_HABITAT_DATA_ROOT = Path("/home/zxt/habitat-sim/data")
SCENE_X_LIMITS = (0.0, 10.0)
SCENE_Y_LIMITS = (-5.0, 5.0)
SCENE_SIZE_STD = (
    SCENE_X_LIMITS[1] - SCENE_X_LIMITS[0],
    SCENE_Y_LIMITS[1] - SCENE_Y_LIMITS[0],
)


@dataclass(frozen=True)
class HabitatSceneAsset:
    scene_glb: Path
    metadata_json: Path


def create_render_scene_with_ground(
    scene_glb: PathLike,
    output_glb: PathLike,
    habitat_data_root: PathLike = DEFAULT_HABITAT_DATA_ROOT,
) -> Path:
    import trimesh

    del habitat_data_root
    scene_glb = Path(scene_glb)
    output_glb = Path(output_glb)
    output_glb.parent.mkdir(parents=True, exist_ok=True)

    loaded = trimesh.load(scene_glb, force="scene")
    if isinstance(loaded, trimesh.Scene):
        render_scene = loaded.copy()
    elif isinstance(loaded, trimesh.Trimesh):
        render_scene = trimesh.Scene()
        render_scene.add_geometry(loaded)
    else:
        raise TypeError(f"Unsupported scene type for {scene_glb}: {type(loaded)!r}")

    render_scene.add_geometry(_make_render_ground(), node_name="ground")
    render_scene.export(output_glb)
    return output_glb


def create_habitat_scene(
    scene: SemanticScene,
    output_dir: PathLike = "generated_scenes/semantic_scene",
    habitat_data_root: PathLike = DEFAULT_HABITAT_DATA_ROOT,
) -> HabitatSceneAsset:
    import trimesh

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    habitat_data_root = Path(habitat_data_root)

    objects = tuple(scene.registry)
    if not objects:
        raise ValueError("scene must contain at least one semantic object")

    habitat_scene = trimesh.Scene()
    occupancy_metadata = {}
    for obj in objects:
        mesh = _make_object_mesh(obj, habitat_data_root)
        habitat_scene.add_geometry(mesh, node_name=obj.object_id)
        occupancy_metadata[obj.object_id] = _write_object_occupancy(obj, mesh, output_dir)

    scene_glb = output_dir / f"{scene.scene_id}.glb"
    habitat_scene.export(scene_glb)

    metadata_json = output_dir / "metadata.json"
    metadata = {
        "scene_id": scene.scene_id,
        "coordinate_system": "std_z_up_scene",
        "asset_coordinate_note": "Generated GLB is authored in the project z-up scene frame. Habitat/Y-up object assets are converted into this z-up frame before placement.",
        "environment": {
            "type": "open_empty_scene",
            "plane_size_std": list(SCENE_SIZE_STD),
            "bounds_std": {
                "x": list(SCENE_X_LIMITS),
                "y": list(SCENE_Y_LIMITS),
            },
            "floor_color_rgba": [120, 230, 255, 255],
            "note": "Scene GLB contains placed semantic objects only; no ground plane is written into the GLB. Background is set by the renderer clear color.",
        },
        "start_std": list(scene.start) if scene.start is not None else None,
        "goal_std": list(scene.goal) if scene.goal is not None else None,
        "start_habitat": list(_std_to_habitat(scene.start)) if scene.start is not None else None,
        "goal_habitat": list(_std_to_habitat(scene.goal)) if scene.goal is not None else None,
        "objects": [
            {
                "id": obj.object_id,
                "type": obj.object_type,
                "center_std": list(obj.center),
                "center_habitat": list(_std_to_habitat(obj.center)),
                "asset_path": obj.asset_path,
                "scale": list(obj.scale),
                "yaw_degrees": obj.yaw_degrees,
                "voxel_size": obj.voxel_size,
                "voxel_occupancy_path": (
                    occupancy_metadata[obj.object_id]["path"]
                    if occupancy_metadata.get(obj.object_id)
                    else obj.voxel_occupancy_path
                ),
                "voxel_occupied_count": (
                    occupancy_metadata[obj.object_id]["occupied_count"]
                    if occupancy_metadata.get(obj.object_id)
                    else None
                ),
                "radius": obj.radius,
                "height": obj.height,
                "bbox_min": list(obj.bbox_min) if obj.bbox_min else None,
                "bbox_max": list(obj.bbox_max) if obj.bbox_max else None,
            }
            for obj in objects
        ],
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return HabitatSceneAsset(scene_glb=scene_glb, metadata_json=metadata_json)


def _write_object_occupancy(obj: SemanticObject, mesh: trimesh.Trimesh, output_dir: Path):
    if obj.asset_path is None or obj.voxel_size is None:
        return None
    from .voxel_occupancy import save_object_occupancy

    occupancy_path = output_dir / "occupancy" / f"{obj.object_id}_voxels.npz"
    occupancy = save_object_occupancy(obj, mesh, occupancy_path)
    if occupancy is None:
        return None
    return {
        "path": str(occupancy_path),
        "voxel_size": occupancy.voxel_size,
        "occupied_count": occupancy.occupied_count,
    }


def resolve_habitat_asset(asset: str, habitat_data_root: PathLike = DEFAULT_HABITAT_DATA_ROOT) -> Path:
    habitat_data_root = Path(habitat_data_root)
    asset_path = Path(asset)
    if asset_path.is_absolute():
        if asset_path.exists():
            return asset_path
        raise FileNotFoundError(asset_path)

    direct_path = habitat_data_root / asset_path
    if direct_path.exists():
        return direct_path

    candidates = []
    for suffix in ("", ".glb", ".gltf", ".obj", ".ply"):
        pattern = f"{asset}{suffix}" if suffix and not asset.endswith(suffix) else asset
        candidates.extend(habitat_data_root.rglob(pattern))
    files = [path for path in candidates if path.is_file()]
    if not files:
        raise FileNotFoundError(f"Could not find Habitat asset '{asset}' under {habitat_data_root}")
    return sorted(files, key=lambda path: (len(path.parts), str(path)))[0]


def placed_asset_bbox_std(
    obj: SemanticObject,
    habitat_data_root: PathLike = DEFAULT_HABITAT_DATA_ROOT,
) -> Optional[Tuple[Vector3, Vector3]]:
    if obj.asset_path is None:
        return None
    mesh = _load_asset_mesh(resolve_habitat_asset(obj.asset_path, habitat_data_root))
    mesh = _transform_asset_mesh(mesh, obj)
    return _habitat_bounds_to_std_bbox(mesh.bounds)


def _std_to_habitat(point: Vector3) -> Vector3:
    x, y, z = point
    return (x, z, y)


def _habitat_to_std(point: Vector3) -> Vector3:
    x, y, z = point
    return (x, z, y)


def _make_object_mesh(obj: SemanticObject, habitat_data_root: Path) -> trimesh.Trimesh:
    if obj.asset_path is not None:
        asset_path = resolve_habitat_asset(obj.asset_path, habitat_data_root)
        return _transform_asset_mesh(_load_asset_mesh(asset_path), obj)

    if obj.object_type == "door" and obj.width and obj.height and obj.thickness:
        return _make_door(obj)

    if obj.object_type == "table" and obj.bbox_min is not None and obj.bbox_max is not None:
        return _make_table(obj)

    if obj.object_type in {"cabinet", "chair", "sofa", "bed", "shelf"}:
        raise ValueError(f"{obj.object_type} object {obj.object_id} must provide asset_path")

    if obj.object_type == "ring" and obj.inner_radius is not None and obj.thickness is not None:
        return _make_ring(
            center=obj.center,
            inner_radius=obj.inner_radius,
            outer_radius=obj.radius or (obj.inner_radius + 0.1),
            thickness=obj.thickness,
            color=_object_color(obj),
        )

    if obj.bbox_min is not None and obj.bbox_max is not None:
        min_corner = np.array(obj.bbox_min, dtype=float)
        max_corner = np.array(obj.bbox_max, dtype=float)
        extents = np.abs(max_corner - min_corner)
        translation = (min_corner + max_corner) / 2.0
        return _make_box(extents=tuple(extents), translation=tuple(translation), color=_object_color(obj))

    if obj.radius is not None and obj.height is not None:
        return _make_z_up_cylinder(
            center=obj.center,
            radius=obj.radius,
            height=obj.height,
            color=_object_color(obj),
        )

    return _make_box(
        extents=(0.6, 0.6, 0.6),
        translation=_std_to_habitat(obj.center),
        color=_object_color(obj),
    )


def _load_asset_mesh(asset_path: Path) -> trimesh.Trimesh:
    import trimesh

    loaded = trimesh.load(asset_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_geometry() if hasattr(loaded, "to_geometry") else loaded.dump(concatenate=True)
        if mesh is None or not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
            raise ValueError(f"No mesh geometry found in asset: {asset_path}")
        return mesh.copy()
    if isinstance(loaded, trimesh.Trimesh):
        return loaded.copy()
    raise TypeError(f"Unsupported asset type for {asset_path}: {type(loaded)!r}")


def _transform_asset_mesh(mesh: trimesh.Trimesh, obj: SemanticObject) -> trimesh.Trimesh:
    from trimesh import transformations

    mesh = mesh.copy()
    _convert_y_up_asset_to_z_up_scene(mesh)
    mesh.apply_transform(transformations.scale_matrix(obj.scale[0], direction=[1.0, 0.0, 0.0]))
    mesh.apply_transform(transformations.scale_matrix(obj.scale[1], direction=[0.0, 1.0, 0.0]))
    mesh.apply_transform(transformations.scale_matrix(obj.scale[2], direction=[0.0, 0.0, 1.0]))
    if obj.yaw_degrees:
        mesh.apply_transform(transformations.rotation_matrix(np.deg2rad(obj.yaw_degrees), [0.0, 0.0, 1.0]))

    bounds = mesh.bounds
    horizontal_center = np.array(
        [
            (bounds[0][0] + bounds[1][0]) / 2.0,
            (bounds[0][1] + bounds[1][1]) / 2.0,
            bounds[0][2],
        ],
        dtype=float,
    )
    target = np.array(obj.center, dtype=float)
    mesh.apply_translation(target - horizontal_center)
    return mesh


def _convert_y_up_asset_to_z_up_scene(mesh: trimesh.Trimesh) -> None:
    vertices = mesh.vertices.copy()
    mesh.vertices = np.column_stack((vertices[:, 0], vertices[:, 2], vertices[:, 1]))


def _habitat_bounds_to_std_bbox(bounds) -> Tuple[Vector3, Vector3]:
    return tuple(float(v) for v in bounds[0]), tuple(float(v) for v in bounds[1])


def _object_color(obj: SemanticObject):
    colors = {
        "pillar": (245, 104, 35, 255),
        "table": (155, 103, 60, 255),
        "cabinet": (95, 125, 150, 255),
        "door": (80, 170, 120, 255),
        "window": (90, 160, 220, 180),
        "ring": (230, 90, 90, 255),
    }
    return colors.get(obj.object_type, (190, 190, 190, 255))


def _make_box(extents, translation, color) -> trimesh.Trimesh:
    import trimesh

    box = trimesh.creation.box(extents=extents)
    box.apply_translation(translation)
    _set_vertex_color(box, color)
    return box


def _make_door(obj: SemanticObject) -> trimesh.Trimesh:
    """Make a door frame in the x-z plane, leaving its centered x opening empty."""
    import trimesh

    width = float(obj.width)
    height = float(obj.height)
    thickness = float(obj.thickness)
    center = np.asarray(obj.center, dtype=float)
    post_width = thickness
    side_height = height - post_width
    pieces = [
        _make_box((post_width, thickness, side_height),
                  (center[0] - width / 2.0 - post_width / 2.0, center[1], side_height / 2.0),
                  _object_color(obj)),
        _make_box((post_width, thickness, side_height),
                  (center[0] + width / 2.0 + post_width / 2.0, center[1], side_height / 2.0),
                  _object_color(obj)),
        _make_box((width + 2.0 * post_width, thickness, post_width),
                  (center[0], center[1], height - post_width / 2.0),
                  _object_color(obj)),
    ]
    return trimesh.util.concatenate(pieces)


def _make_table(obj: SemanticObject) -> trimesh.Trimesh:
    """Make a simple four-legged table from its axis-aligned bounding box."""
    import trimesh

    bmin = np.asarray(obj.bbox_min, dtype=float)
    bmax = np.asarray(obj.bbox_max, dtype=float)
    width, depth, height = bmax - bmin
    top_thickness = min(0.12, height * 0.16)
    leg_width = min(0.12, width * 0.12, depth * 0.12)
    color = _object_color(obj)
    pieces = [_make_box((width, depth, top_thickness),
                        ((bmin[0] + bmax[0]) / 2.0, (bmin[1] + bmax[1]) / 2.0,
                         bmax[2] - top_thickness / 2.0), color)]
    for x in (bmin[0] + leg_width / 2.0, bmax[0] - leg_width / 2.0):
        for y in (bmin[1] + leg_width / 2.0, bmax[1] - leg_width / 2.0):
            pieces.append(_make_box((leg_width, leg_width, height - top_thickness),
                                    (x, y, (height - top_thickness) / 2.0), color))
    return trimesh.util.concatenate(pieces)


def _make_render_ground() -> trimesh.Trimesh:
    return _make_box(
        extents=(SCENE_SIZE_STD[0], SCENE_SIZE_STD[1], 0.04),
        translation=((SCENE_X_LIMITS[0] + SCENE_X_LIMITS[1]) / 2.0, (SCENE_Y_LIMITS[0] + SCENE_Y_LIMITS[1]) / 2.0, -0.02),
        color=(120, 230, 255, 255),
    )


def _make_z_up_cylinder(center: Vector3, radius: float, height: float, color) -> trimesh.Trimesh:
    import trimesh

    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=64)
    cylinder.apply_translation(center)
    _set_vertex_color(cylinder, color)
    return cylinder


def _make_ring(center: Vector3, inner_radius: float, outer_radius: float, thickness: float, color) -> trimesh.Trimesh:
    """A flat annular ring standing in the y-z plane (opening along the x axis)."""
    import trimesh
    from trimesh import transformations

    ring = trimesh.creation.annulus(r_min=inner_radius, r_max=outer_radius, height=thickness, sections=64)
    # annulus() is in the x-y plane extruded along z; rotate 90 deg about y so the
    # ring lies in the y-z plane with its thickness along x (the fly-through axis).
    ring.apply_transform(transformations.rotation_matrix(np.deg2rad(90.0), [0.0, 1.0, 0.0]))
    ring.apply_translation(np.asarray(center, dtype=float))
    _set_vertex_color(ring, color)
    return ring


def _set_vertex_color(mesh: trimesh.Trimesh, rgba):
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals

    color = np.array(rgba, dtype=np.float32) / 255.0
    mesh.visual = TextureVisuals(
        material=PBRMaterial(
            baseColorFactor=color.tolist(),
            metallicFactor=0.0,
            roughnessFactor=1.0,
            doubleSided=True,
        )
    )
