#!/usr/bin/env python3
"""View voxel occupancy and a planned trajectory in 3D.

This script is self-contained on purpose: it only needs numpy + matplotlib +
trimesh. The small ``VoxelOccupancy`` / ``SemanticFreeRegion`` classes it used
to import from ``uav_maneuver_planning`` are embedded here so the server does
not need that package.

The convenient server mode only needs the task and scene id::

    python scripts/view_voxel_trajectory.py --task around_left --scene-id auto_around_left_asset_frl_apartment_bike_01_00000

It auto-completes the sample's ``_render_scene/<id>_with_ground.glb``,
``_planned_trajectory.json``, ``semantic_region_<task>.json`` and the output PNG
path under ``outputs/view_voxel_trajectory/``.
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

Vector3 = Tuple[float, float, float]

DEFAULT_DATA_ROOT = os.environ.get(
    "AEROVERBS_DATA_ROOT", "/root/autodl-tmp/data/trajctorys"
)
DEFAULT_AUTO_OUTPUT_DIR = "outputs/view_voxel_trajectory"


class Skill(str, Enum):
    AROUND_LEFT = "around_left"
    AROUND_RIGHT = "around_right"
    THROUGH = "through"
    OVER = "over"
    UNDER = "under"


@dataclass(frozen=True)
class SemanticFreeRegion:
    skill: Skill
    object_id: str
    clearance: float
    points: Tuple[Vector3, ...]
    description: str

    @classmethod
    def from_json_dict(cls, payload: dict) -> "SemanticFreeRegion":
        return cls(
            skill=Skill(payload["skill"]),
            object_id=str(payload["object_id"]),
            clearance=float(payload["clearance"]),
            description=str(payload.get("description", "")),
            points=tuple(
                tuple(float(value) for value in point[:3])
                for point in payload.get("points", [])
            ),
        )

    @classmethod
    def load(cls, path: Path) -> "SemanticFreeRegion":
        return cls.from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class HeatmapGate:
    skill: str
    object_id: str
    center: Vector3
    normal: Vector3
    left_axis: Vector3
    up_axis: Vector3
    width: float
    height: float
    confidence: float

    @classmethod
    def from_json_dict(cls, payload: dict) -> "HeatmapGate":
        return cls(
            skill=str(payload.get("skill", "")),
            object_id=str(payload.get("object_id", "")),
            center=tuple(float(v) for v in payload["center_std"][:3]),
            normal=tuple(float(v) for v in payload["normal_std"][:3]),
            left_axis=tuple(float(v) for v in payload["left_axis_std"][:3]),
            up_axis=tuple(float(v) for v in payload["up_axis_std"][:3]),
            width=float(payload["width_m"]),
            height=float(payload["height_m"]),
            confidence=float(payload.get("confidence", 1.0)),
        )

    @classmethod
    def load(cls, path: Path) -> "HeatmapGate":
        return cls.from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def corners(self) -> np.ndarray:
        center = np.asarray(self.center, dtype=float)
        left = np.asarray(self.left_axis, dtype=float)
        up = np.asarray(self.up_axis, dtype=float)
        hw = 0.5 * self.width
        hh = 0.5 * self.height
        return np.asarray(
            [
                center - hw * left - hh * up,
                center + hw * left - hh * up,
                center + hw * left + hh * up,
                center - hw * left + hh * up,
                center - hw * left - hh * up,
            ],
            dtype=float,
        )


@dataclass
class VoxelOccupancy:
    voxel_size: float
    origin: Vector3
    occupied_indices: np.ndarray
    bbox_min: Vector3
    bbox_max: Vector3

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
        import math

        radius = max(0, math.ceil(margin / self.voxel_size))
        if radius == 0:
            return center_index in self._index_set
        radius_sq = (margin / self.voxel_size + 0.5) ** 2
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if dx * dx + dy * dy + dz * dz > radius_sq:
                        continue
                    if (
                        center_index[0] + dx,
                        center_index[1] + dy,
                        center_index[2] + dz,
                    ) in self._index_set:
                        return True
        return False

    def point_to_index(self, point: Vector3) -> Tuple[int, int, int]:
        rel = (
            np.asarray(point, dtype=float) - np.asarray(self.origin, dtype=float)
        ) / self.voxel_size
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
        return all(
            self.bbox_min[i] - margin <= point[i] <= self.bbox_max[i] + margin
            for i in range(3)
        )


def voxel_occupancy_from_mesh(
    mesh: Any,
    voxel_size: float,
    samples_per_voxel_area: float = 6.0,
) -> VoxelOccupancy:
    """Voxelize mesh surface into occupied voxels (only needs trimesh)."""
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
    indices = np.floor(
        (points - np.asarray(origin, dtype=float)) / voxel_size + 1e-9
    ).astype(np.int32)
    occupied_indices = np.unique(indices, axis=0)
    return VoxelOccupancy(
        voxel_size=voxel_size,
        origin=origin,
        occupied_indices=occupied_indices,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="View voxel occupancy and a planned trajectory in 3D."
    )
    parser.add_argument("--plan", default=None, help="Planned trajectory JSON.")
    parser.add_argument("--region", default=None, help="Semantic free-flight region JSON.")
    parser.add_argument("--gate", default=None, help="Optional heatmap gate summary JSON.")
    parser.add_argument("--occupancy", default=None, help="Optional voxel occupancy .npz file.")
    parser.add_argument("--glb", default=None, help="Optional GLB mesh to voxelize when --occupancy is not provided.")
    parser.add_argument("--task", default=None, help="Skill/task name, e.g. around_left. Auto-completes paths with --scene-id.")
    parser.add_argument("--scene-id", default=None, help="Sample scene id under <data-root>/<task>/<scene-id>.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Trajectory data root for auto-complete mode.")
    parser.add_argument("--voxel-size", type=float, default=0.08, help="Voxel size for --glb voxelization.")
    parser.add_argument("--save-occupancy", default=None, help="Optional path to save occupancy generated from --glb.")
    parser.add_argument("--occupancy-min-z", type=float, help="Hide occupied voxel centers below this std z value.")
    parser.add_argument("--occupancy-max-z", type=float, help="Hide occupied voxel centers above this std z value.")
    parser.add_argument("--output", default=None, help="Optional PNG output path.")
    parser.add_argument("--show", action="store_true", help="Open an interactive matplotlib 3D window.")
    parser.add_argument("--voxel-stride", type=int, default=1, help="Render every Nth occupied voxel.")
    parser.add_argument("--region-stride", type=int, default=1, help="Render every Nth semantic region point.")
    parser.add_argument("--no-path", action="store_true", help="Do not draw the planned trajectory.")
    parser.add_argument("--no-region", action="store_true", help="Do not draw the semantic free-flight region.")
    parser.add_argument("--no-gate", action="store_true", help="Do not draw the heatmap gate.")
    parser.add_argument("--completion", default=None,
                        help="Optional completion.npy (0/1 per trajectory frame); the completed "
                             "tail segment is drawn in a different color.")
    return parser.parse_args()


def _resolve_auto_paths(args):
    """Auto-complete glb/plan/region/output from --task + --scene-id."""
    if not args.task or not args.scene_id:
        return

    sample_dir = Path(args.data_root) / args.task / args.scene_id
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"sample dir not found: {sample_dir}")

    scene_id = args.scene_id
    if args.glb is None:
        candidate = sample_dir / "_render_scene" / f"{scene_id}_with_ground.glb"
        if candidate.exists():
            args.glb = str(candidate)
        else:
            raise FileNotFoundError(f"with-ground glb not found: {candidate}")
    if args.plan is None:
        candidate = sample_dir / "_planned_trajectory.json"
        if candidate.exists():
            args.plan = str(candidate)
        else:
            raise FileNotFoundError(f"planned trajectory not found: {candidate}")
    if args.region is None:
        candidate = sample_dir / f"semantic_region_{args.task}.json"
        if candidate.exists():
            args.region = str(candidate)
        # Region is optional; render without it if missing.
    if args.gate is None:
        candidate = sample_dir / "heatmap_gate_summary.json"
        if candidate.exists():
            args.gate = str(candidate)
        # Gate is optional; render without it if missing.
    if args.completion is None:
        candidate = sample_dir / "completion.npy"
        if candidate.exists():
            args.completion = str(candidate)
        # Completion overlay is optional; render without it if missing.
    if args.output is None:
        args.output = str(
            Path(DEFAULT_AUTO_OUTPUT_DIR) / f"{args.task}__{args.scene_id}.png"
        )


def main():
    args = parse_args()
    _resolve_auto_paths(args)

    if args.output is None:
        args.output = "outputs/view_voxel_trajectory.png"

    plan = None
    if args.plan and not args.no_path:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))

    completion = None
    if args.completion and plan is not None:
        completion = np.load(args.completion).astype(bool)

    region = None
    if args.region and not args.no_region:
        region = SemanticFreeRegion.load(Path(args.region))

    gate = None
    if args.gate and not args.no_gate:
        gate = HeatmapGate.load(Path(args.gate))

    occupancy = _load_or_build_occupancy(args)

    if plan is None and region is None and gate is None and occupancy is None:
        raise ValueError("Provide --plan, --region, --gate, or --glb/--occupancy to visualize.")
    fig = make_figure(
        plan=plan,
        occupancy=occupancy,
        region=region,
        gate=gate,
        voxel_stride=max(1, args.voxel_stride),
        region_stride=max(1, args.region_stride),
        occupancy_min_z=args.occupancy_min_z,
        occupancy_max_z=args.occupancy_max_z,
        completion=completion,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"Wrote 3D voxel trajectory view to {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


def _load_or_build_occupancy(args):
    if args.occupancy:
        occupancy = VoxelOccupancy.load(Path(args.occupancy))
        return _drop_ground_voxels(occupancy)
    if not args.glb:
        return None
    occupancy = _occupancy_from_glb(Path(args.glb), voxel_size=args.voxel_size)
    occupancy = _drop_ground_voxels(occupancy)
    if args.save_occupancy:
        occupancy.save(
            Path(args.save_occupancy),
            metadata={"source_glb": str(Path(args.glb)), "voxel_size": args.voxel_size},
        )
        print(f"Wrote voxel occupancy to {args.save_occupancy}")
    return occupancy


def _drop_ground_voxels(occupancy: VoxelOccupancy, min_z: float = 0.05) -> VoxelOccupancy:
    """Remove occupied voxels that belong to the ground plane.

    The radar renderer uses a ``<scene_id>_with_ground.glb`` mesh, but the 3D
    trajectory view should only show the object / scene structure, not a huge
    flat ground slab. Voxel centers below ``min_z`` are dropped and the lower
    bbox bound is lifted accordingly.
    """
    if occupancy is None or len(occupancy.occupied_indices) == 0:
        return occupancy
    origin = np.asarray(occupancy.origin, dtype=float)
    indices = occupancy.occupied_indices.astype(float)
    centers = origin + (indices + 0.5) * occupancy.voxel_size
    keep = centers[:, 2] >= min_z
    new_indices = occupancy.occupied_indices[keep]
    bbox_min = list(occupancy.bbox_min)
    bbox_min[2] = max(bbox_min[2], min_z - occupancy.voxel_size)
    return VoxelOccupancy(
        voxel_size=occupancy.voxel_size,
        origin=occupancy.origin,
        occupied_indices=new_indices,
        bbox_min=tuple(bbox_min),
        bbox_max=occupancy.bbox_max,
    )


def _occupancy_from_glb(glb_path: Path, voxel_size: float) -> VoxelOccupancy:
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            "Voxelizing --glb requires trimesh. Install it with `pip install trimesh`."
        ) from exc
    loaded = trimesh.load(glb_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded
    if mesh is None or not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"No mesh geometry found in {glb_path}")
    return voxel_occupancy_from_mesh(mesh, voxel_size=voxel_size)


def _filter_centers_by_z(centers: np.ndarray, min_z: float = None, max_z: float = None) -> np.ndarray:
    if len(centers) == 0:
        return centers
    mask = np.ones(len(centers), dtype=bool)
    if min_z is not None:
        mask &= centers[:, 2] >= min_z
    if max_z is not None:
        mask &= centers[:, 2] <= max_z
    return centers[mask]


_ELEV = 24
_AZIM = -58

DEFAULT_VIEWS = (
    (_ELEV, _AZIM, "default"),
    (0.0, 0.0, "front (x+ to x-)"),
    (0.0, -90.0, "side (y- to y+)"),
    (90.0, -180.0, "top (z+ to z-)"),
)
FOUR_VIEWS = (
    (_ELEV, _AZIM, "default"),
    (0.0, 0.0, "front (x+ to x-)"),
    (0.0, -90.0, "side (y- to y+)"),
    (90.0, -180.0, "top (z+ to z-)"),
)


def _scene_points(
    plan=None,
    occupancy: VoxelOccupancy = None,
    region: SemanticFreeRegion = None,
    gate: HeatmapGate = None,
    voxel_stride: int = 1,
    region_stride: int = 1,
    occupancy_min_z: float = None,
    occupancy_max_z: float = None,
):
    points = []
    if occupancy is not None:
        indices = occupancy.occupied_indices[::voxel_stride].astype(float)
        origin = np.asarray(occupancy.origin, dtype=float)
        centers = origin + (indices + 0.5) * occupancy.voxel_size
        centers = _filter_centers_by_z(centers, occupancy_min_z, occupancy_max_z)
        if len(centers):
            points.append(centers)
        points.append(
            np.asarray([occupancy.bbox_min, occupancy.bbox_max], dtype=float)
        )
    if region is not None and region.points:
        points.append(np.asarray(region.points[::region_stride], dtype=float))
    if gate is not None:
        points.append(gate.corners)
    if plan is not None:
        points.append(np.asarray(plan["points"], dtype=float))
    return points


def _draw_scene(
    ax,
    plan=None,
    occupancy: VoxelOccupancy = None,
    region: SemanticFreeRegion = None,
    gate: HeatmapGate = None,
    voxel_stride: int = 1,
    region_stride: int = 1,
    occupancy_min_z: float = None,
    occupancy_max_z: float = None,
    elev: float = _ELEV,
    azim: float = _AZIM,
    view_label: Optional[str] = None,
    show_legend: bool = True,
    completion=None,
):
    # matplotlib's `computed_zorder` only depth-sorts Collections/Patches, not
    # Line3D, so the trajectory line would always paint on top of the voxels.
    # Disable it and do a manual painter's-algorithm depth sort instead.
    ax.computed_zorder = False
    ax.view_init(elev=elev, azim=azim)
    view_dir = _camera_view_dir(elev, azim)
    zorder_base = 100.0

    legend_handles = []
    title_parts = []

    if occupancy is not None:
        indices = occupancy.occupied_indices[::voxel_stride].astype(float)
        origin = np.asarray(occupancy.origin, dtype=float)
        centers = origin + (indices + 0.5) * occupancy.voxel_size
        centers = _filter_centers_by_z(centers, occupancy_min_z, occupancy_max_z)
        bbox_min = np.asarray(occupancy.bbox_min, dtype=float)
        bbox_max = np.asarray(occupancy.bbox_max, dtype=float)
        if len(centers):
            _scatter_depth_sorted(
                ax, centers, view_dir, zorder_base,
                s=10, c="#8b6f47", alpha=0.5, depthshade=False,
            )
            legend_handles.append(
                Line2D([0], [0], marker="o", linestyle="", color="#8b6f47", markersize=6, alpha=0.5, label="occupied voxels")
            )
        _plot_bbox(ax, bbox_min, bbox_max, view_dir, zorder_base)
        title_parts.append("voxel occupancy")

    if region is not None and region.points:
        region_points = np.asarray(region.points[::region_stride], dtype=float)
        _scatter_depth_sorted(
            ax, region_points, view_dir, zorder_base,
            s=14, c="#2ca02c", alpha=0.35, depthshade=False,
        )
        legend_handles.append(
            Line2D([0], [0], marker="o", linestyle="", color="#2ca02c", markersize=7, alpha=0.35,
                   label=f"{region.skill.value} free region ({len(region.points)})")
        )
        title_parts.append("semantic free region")

    if gate is not None:
        corners = gate.corners
        face_z = zorder_base + float(np.mean(corners[:4], axis=0) @ view_dir)
        face = Poly3DCollection(
            [corners[:4]],
            facecolor="#17becf",
            edgecolor="#006d77",
            linewidths=2.0,
            alpha=0.24,
            zorder=face_z,
        )
        ax.add_collection3d(face)
        _plot_depth_sorted(
            ax, corners, view_dir, zorder_base,
            color="#006d77", linewidth=2.5,
        )
        center = np.asarray(gate.center, dtype=float)
        normal = np.asarray(gate.normal, dtype=float)
        normal_len = max(0.4, min(float(gate.width), float(gate.height)))
        ax.scatter(
            [center[0]], [center[1]], [center[2]],
            c="#005f73", s=80, marker="x",
            zorder=zorder_base + float(center @ view_dir),
        )
        ax.quiver(
            center[0], center[1], center[2],
            normal[0], normal[1], normal[2],
            length=normal_len,
            color="#005f73",
            linewidth=2.0,
            normalize=True,
        )
        legend_handles.append(
            Line2D([0], [0], color="#006d77", linewidth=2.5,
                   label=f"heatmap gate ({gate.confidence:.2f})")
        )
        title_parts.append("heatmap gate")

    if plan is not None:
        trajectory = np.asarray(plan["points"], dtype=float)
        _plot_depth_sorted(
            ax, trajectory, view_dir, zorder_base,
            color="#d62728", linewidth=3.0,
        )
        for point, color in ((trajectory[0], "#1f77b4"), (trajectory[-1], "#2ca02c")):
            ax.scatter(
                [point[0]], [point[1]], [point[2]],
                c=color, s=60, zorder=zorder_base + float(point @ view_dir),
            )

        if completion is not None and len(trajectory) > 0:
            mask = np.asarray(completion, dtype=bool).reshape(-1)
            n = min(len(trajectory), len(mask))
            if n > 0 and mask[:n].any():
                comp_points = trajectory[:n][mask[:n]]
                if len(comp_points) >= 2:
                    _plot_depth_sorted(
                        ax, comp_points, view_dir, zorder_base,
                        color="#ff7f0e", linewidth=4.5,
                    )
                    legend_handles.append(
                        Line2D([0], [0], color="#ff7f0e", linewidth=4.5,
                               label="completion segment")
                    )
                    title_parts.append("completion")

        legend_handles.append(Line2D([0], [0], color="#d62728", linewidth=3.0, label=f"trajectory ({plan['semantic_label']})"))
        legend_handles.append(Line2D([0], [0], marker="o", linestyle="", color="#1f77b4", markersize=6, label="start"))
        legend_handles.append(Line2D([0], [0], marker="o", linestyle="", color="#2ca02c", markersize=6, label="goal"))
        title_parts.append("trajectory")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if view_label is not None:
        ax.set_title(view_label)
    else:
        ax.set_title(" + ".join(title_parts))
    if show_legend and legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")


def make_figure(
    plan=None,
    occupancy: VoxelOccupancy = None,
    region: SemanticFreeRegion = None,
    gate: HeatmapGate = None,
    voxel_stride: int = 1,
    region_stride: int = 1,
    occupancy_min_z: float = None,
    occupancy_max_z: float = None,
    views=None,
    completion=None,
):
    if views is None:
        views = DEFAULT_VIEWS
    views = list(views)
    if len(views) == 0:
        views = DEFAULT_VIEWS

    n = len(views)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    # Keep the combined image small: a 2x2 figure at dpi=100 is 1200x900 px.
    fig = plt.figure(figsize=(9 * cols, 7 * rows), dpi=100)

    scene_points = _scene_points(
        plan, occupancy, region, gate, voxel_stride, region_stride,
        occupancy_min_z, occupancy_max_z,
    )
    all_points = np.vstack(scene_points) if scene_points else None

    for i, (elev, azim, label) in enumerate(views):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        _draw_scene(
            ax,
            plan=plan,
            occupancy=occupancy,
            region=region,
            gate=gate,
            voxel_stride=voxel_stride,
            region_stride=region_stride,
            occupancy_min_z=occupancy_min_z,
            occupancy_max_z=occupancy_max_z,
            elev=float(elev),
            azim=float(azim),
            view_label=label,
            show_legend=(i == 0),
            completion=completion,
        )
        if all_points is not None and len(all_points):
            _set_equal_axes(ax, all_points)

    fig.tight_layout()
    return fig


def _camera_view_dir(elev_deg, azim_deg):
    """Unit vector pointing from the scene center toward the camera."""
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    return np.array(
        [
            np.cos(elev) * np.cos(azim),
            np.cos(elev) * np.sin(azim),
            np.sin(elev),
        ]
    )


def _depth(points, view_dir):
    """Projection onto the view direction; larger means closer to the camera."""
    return np.asarray(points, dtype=float) @ view_dir


def _scatter_depth_sorted(ax, points, view_dir, zorder_base, max_buckets=64, **kwargs):
    """Draw a point cloud split into depth buckets so nearer points occlude farther ones."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return
    depths = _depth(points, view_dir)
    order = np.argsort(depths)
    n_buckets = max(1, min(max_buckets, len(points)))
    for bucket_pts, bucket_depths in zip(
        np.array_split(points[order], n_buckets),
        np.array_split(depths[order], n_buckets),
    ):
        if len(bucket_pts) == 0:
            continue
        ax.scatter(
            bucket_pts[:, 0],
            bucket_pts[:, 1],
            bucket_pts[:, 2],
            zorder=zorder_base + float(bucket_depths.mean()),
            **kwargs,
        )


def _plot_depth_sorted(ax, points, view_dir, zorder_base, **kwargs):
    """Draw a polyline segment-by-segment with per-segment depth zorder."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return
    for i in range(len(points) - 1):
        segment = points[i : i + 2]
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            zorder=zorder_base + float(segment.mean(axis=0) @ view_dir),
            **kwargs,
        )


def _plot_edges_depth_sorted(ax, segments, view_dir, zorder_base, **kwargs):
    """Draw independent 2-point segments with per-segment depth zorder."""
    segments = np.asarray(segments, dtype=float)
    for segment in segments:
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            zorder=zorder_base + float(segment.mean(axis=0) @ view_dir),
            **kwargs,
        )


def _plot_bbox(ax, bbox_min, bbox_max, view_dir, zorder_base):
    corners = _bbox_corners(bbox_min, bbox_max)
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    segments = np.stack([corners[np.array([a, b])] for a, b in edges])
    _plot_edges_depth_sorted(ax, segments, view_dir, zorder_base, color="0.25", linestyle=":", linewidth=1.0)


def _bbox_corners(bbox_min, bbox_max):
    return np.array(
        [
            [bbox_min[0], bbox_min[1], bbox_min[2]],
            [bbox_max[0], bbox_min[1], bbox_min[2]],
            [bbox_max[0], bbox_max[1], bbox_min[2]],
            [bbox_min[0], bbox_max[1], bbox_min[2]],
            [bbox_min[0], bbox_min[1], bbox_max[2]],
            [bbox_max[0], bbox_min[1], bbox_max[2]],
            [bbox_max[0], bbox_max[1], bbox_max[2]],
            [bbox_min[0], bbox_max[1], bbox_max[2]],
        ],
        dtype=float,
    )


def _set_equal_axes(ax, points):
    ranges = points.max(axis=0) - points.min(axis=0)
    mid = (points.max(axis=0) + points.min(axis=0)) / 2.0
    radius = max(ranges.max() / 2.0, 1.0)
    ax.set_xlim(mid[0] - radius, mid[0] + radius)
    ax.set_ylim(mid[1] - radius, mid[1] + radius)
    ax.set_zlim(max(0.0, mid[2] - radius), mid[2] + radius)


if __name__ == "__main__":
    main()
