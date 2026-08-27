#!/usr/bin/env python3
"""Build voxel occupancy files for procedural through scenes.

Through scenes are procedural GLBs under ``<data-root>/_scenes/through``. This
script voxelizes those object-only GLBs and writes occupancy npz files in the
same format used by under-table scenes:

``occupancy/<object_id>_voxels.npz``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from view_voxel_trajectory import voxel_occupancy_from_mesh


def load_glb_mesh(glb_path: Path):
    """Load a GLB as one trimesh mesh."""
    import trimesh

    loaded = trimesh.load(str(glb_path), force="scene")
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded
    if mesh is None or not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"failed to load a non-empty mesh from {glb_path}")
    return mesh


def object_id_from_metadata(scene_dir: Path) -> str:
    meta_path = scene_dir / "metadata.json"
    if not meta_path.exists():
        return "through_0"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    objects = meta.get("objects") or []
    if not objects:
        return "through_0"
    obj = objects[0]
    return str(obj.get("id") or obj.get("object_id") or "through_0")


def build_scene_occupancy(
    scene_dir: Path,
    *,
    voxel_size: float = 0.08,
    samples_per_voxel_area: float = 6.0,
    overwrite: bool = False,
) -> Path:
    """Voxelize one ``_scenes/through/<scene_id>`` directory."""
    scene_dir = Path(scene_dir)
    scene_id = scene_dir.name
    glb_path = scene_dir / f"{scene_id}.glb"
    if not glb_path.exists():
        raise FileNotFoundError(f"missing scene glb: {glb_path}")

    object_id = object_id_from_metadata(scene_dir)
    output_path = scene_dir / "occupancy" / f"{object_id}_voxels.npz"
    if output_path.exists() and not overwrite:
        return output_path

    mesh = load_glb_mesh(glb_path)
    occupancy = voxel_occupancy_from_mesh(
        mesh,
        voxel_size=voxel_size,
        samples_per_voxel_area=samples_per_voxel_area,
    )
    occupancy.save(
        output_path,
        metadata={
            "object_id": object_id,
            "object_type": "through",
            "source_glb": str(glb_path),
            "voxel_size": voxel_size,
        },
    )
    _update_scene_metadata(scene_dir, object_id, output_path, voxel_size, occupancy.occupied_count)
    return output_path


def _update_scene_metadata(
    scene_dir: Path,
    object_id: str,
    output_path: Path,
    voxel_size: float,
    occupied_count: int,
) -> None:
    meta_path = scene_dir / "metadata.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    objects = meta.get("objects") or []
    for obj in objects:
        if str(obj.get("id") or obj.get("object_id") or "") == object_id:
            obj["voxel_size"] = float(voxel_size)
            obj["voxel_occupancy_path"] = str(output_path)
            obj["voxel_occupied_count"] = int(occupied_count)
            break
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def iter_scene_dirs(data_root: Path, scene_ids: Sequence[str] | None = None) -> Iterable[Path]:
    through_root = Path(data_root) / "_scenes" / "through"
    if scene_ids:
        for scene_id in scene_ids:
            yield through_root / scene_id
        return
    yield from sorted(path for path in through_root.iterdir() if path.is_dir())


def progress_iterable(items: Sequence[Path], *, enabled: bool = True):
    if not enabled:
        return items
    try:
        from tqdm import tqdm
    except Exception:
        return items
    return tqdm(items, desc="through occupancy", unit="scene")


def parse_scene_ids(raw: str | None) -> tuple[str, ...] | None:
    if raw is None or raw.strip() == "":
        return None
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build occupancy npz files for through procedural GLBs.")
    parser.add_argument("--data-root", default="/root/autodl-tmp/data/trajctorys")
    parser.add_argument("--scene-ids", default=None, help="Optional comma-separated scene ids. Default: all through scenes.")
    parser.add_argument("--voxel-size", type=float, default=0.08)
    parser.add_argument("--samples-per-voxel-area", type=float, default=6.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_dirs = list(iter_scene_dirs(Path(args.data_root), parse_scene_ids(args.scene_ids)))
    built = 0
    skipped = 0
    failed = 0
    for scene_dir in progress_iterable(scene_dirs, enabled=not args.no_progress):
        try:
            existed = (scene_dir / "occupancy").exists() and any((scene_dir / "occupancy").glob("*.npz"))
            output = build_scene_occupancy(
                scene_dir,
                voxel_size=args.voxel_size,
                samples_per_voxel_area=args.samples_per_voxel_area,
                overwrite=args.overwrite,
            )
            if existed and not args.overwrite:
                skipped += 1
            else:
                built += 1
            print(f"{scene_dir.name}: {output}")
        except Exception as exc:
            failed += 1
            print(f"{scene_dir.name}: ERROR {exc}", file=sys.stderr)
    print(f"done: built={built} skipped={skipped} failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
