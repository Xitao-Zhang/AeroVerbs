#!/usr/bin/env python3
"""Plot procedural rollout trajectories in combined 3-D and orthographic views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.collections import PolyCollection


COLORS = {
    "around_left": "tab:blue",
    "through": "tab:green",
    "under": "tab:orange",
}


def load_mesh(path: Path):
    loaded = trimesh.load(str(path), force="scene")
    if isinstance(loaded, trimesh.Scene):
        return loaded.dump(concatenate=True)
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollout-dir",
        type=Path,
        default=Path("outputs/rollout/procedural_three_tasks_dynamic_through"),
    )
    parser.add_argument(
        "--task-root",
        type=Path,
        default=Path("generated_scenes/procedural_test_tasks"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: <rollout-dir>/combined_trajectories.png).",
    )
    parser.add_argument(
        "--views-output",
        type=Path,
        default=None,
        help="Output PNG for XY/XZ/YZ views (default: <rollout-dir>/combined_trajectories_views.png).",
    )
    args = parser.parse_args()
    output = args.output or args.rollout_dir / "combined_trajectories.png"
    views_output = args.views_output or args.rollout_dir / "combined_trajectories_views.png"

    tasks = [
        ("around_left", "around_left_pillar", "pillar_0"),
        ("through", "through_door", "door_0"),
        ("under", "under_table", "table_0"),
    ]
    all_points = []
    loaded_tasks = []
    for skill, scene_id, object_id in tasks:
        trajectory_path = args.rollout_dir / "trajectories" / f"{skill}__{scene_id}.json"
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        points = np.asarray(trajectory["points"], dtype=float)
        all_points.append(points)
        meta = json.loads((args.task_root / scene_id / "meta.json").read_text(encoding="utf-8"))
        mesh_path = Path(meta["scene_glb"])
        if not mesh_path.is_absolute():
            mesh_path = (args.task_root / scene_id / mesh_path).resolve()
        mesh = load_mesh(mesh_path)
        loaded_tasks.append((skill, scene_id, object_id, points, meta, mesh))

    points = np.concatenate(all_points, axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    # One image: 3-D view plus XY/XZ/YZ orthographic views.
    fig = plt.figure(figsize=(16, 14), dpi=160)
    ax = fig.add_subplot(221, projection="3d")
    for skill, scene_id, object_id, points, meta, mesh in loaded_tasks:
        color = COLORS[skill]
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=2.5, label=skill)
        ax.scatter(*points[0], color=color, marker="o", s=35)
        ax.scatter(*points[-1], color=color, marker="X", s=55)
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        # A translucent obstacle overlay keeps the trajectory readable.
        ax.plot_trisurf(
            vertices[:, 0], vertices[:, 1], vertices[:, 2],
            triangles=faces, color=color, alpha=0.18, linewidth=0.05,
        )
        center = next(obj["center"] for obj in meta.get("objects", []) if obj.get("id") == object_id)
        ax.text(*center, object_id, fontsize=8, color=color)

    spans = np.maximum(maxs - mins, 1.0)
    centers = (mins + maxs) / 2.0
    radius = 0.55 * float(spans.max())
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(max(0.0, centers[2] - radius), centers[2] + radius)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Combined Procedural Rollout Trajectories")
    ax.legend(loc="upper left")
    ax.view_init(elev=25.0, azim=-58.0)
    ax.grid(True, alpha=0.25)
    projections = [(0, 1, "X", "Y", "Top (XY)"), (0, 2, "X", "Z", "Front (XZ)"), (1, 2, "Y", "Z", "Side (YZ)")]
    axes = [fig.add_subplot(222), fig.add_subplot(223), fig.add_subplot(224)]
    # Fixed 10 x 10 framing for all orthographic panels.
    limits = [(0.0, 10.0), (-5.0, 5.0), (0.0, 10.0)]
    for ax, (ia, ib, xlabel, ylabel, title) in zip(axes, projections):
        for skill, scene_id, object_id, points, meta, mesh in loaded_tasks:
            color = COLORS[skill]
            vertices = np.asarray(mesh.vertices, dtype=float)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            # Project each mesh triangle into this view so the obstacle keeps
            # its solid structure instead of appearing as isolated vertices.
            polygons = [vertices[face][:, [ia, ib]] for face in faces]
            ax.add_collection(
                PolyCollection(
                    polygons,
                    facecolors=color,
                    edgecolors=color,
                    linewidths=0.15,
                    alpha=0.18,
                )
            )
            ax.plot(points[:, ia], points[:, ib], color=color, linewidth=2.2, label=skill)
            ax.scatter(points[0, ia], points[0, ib], color=color, marker="o", s=32)
            ax.scatter(points[-1, ia], points[-1, ib], color=color, marker="X", s=48)
            center = next(obj["center"] for obj in meta.get("objects", []) if obj.get("id") == object_id)
            ax.text(center[ia], center[ib], object_id, fontsize=8, color=color)
        ax.set_xlim(*limits[ia])
        ax.set_ylim(*limits[ib])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.975))
    fig.suptitle("Combined Procedural Rollout Trajectories", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
