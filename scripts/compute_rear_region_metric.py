#!/usr/bin/env python3
"""Measure how closely rollout endpoints reach an obstacle's +x rear region.

The target obstacle is voxelized from each scene's ``*_with_ground.glb`` mesh,
with ground voxels removed.  Its yz projection is extruded toward +x to form a
half-infinite rear region.  Gate data and expert trajectories are not used.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from view_voxel_trajectory import _drop_ground_voxels, _occupancy_from_glb


MODEL_NAMES = ("bc", "dagger", "openvla", "aerialclaw", "spf")
CSV_FIELDS = (
    "model",
    "batch",
    "skill",
    "scene_id",
    "status",
    "endpoint_x_m",
    "endpoint_y_m",
    "endpoint_z_m",
    "rear_region_reached",
    "rear_region_distance_m",
    "rear_region_score",
    "rear_region_depth_m",
    "rear_region_lateral_distance_m",
    "rear_region_axial_gap_m",
    "obstacle_back_x_m",
    "closest_shadow_y_m",
    "closest_shadow_z_m",
)


@dataclass(frozen=True)
class RearColumns:
    """One +x-facing back surface value for each occupied yz voxel column."""

    yz_centers: np.ndarray
    back_x: np.ndarray

    def __post_init__(self) -> None:
        yz = np.asarray(self.yz_centers, dtype=np.float64)
        back = np.asarray(self.back_x, dtype=np.float64)
        if yz.ndim != 2 or yz.shape[1] != 2 or len(yz) == 0:
            raise ValueError("yz_centers must be a non-empty [N, 2] array")
        if back.shape != (len(yz),):
            raise ValueError("back_x must have one value per yz column")
        if not np.isfinite(yz).all() or not np.isfinite(back).all():
            raise ValueError("rear columns must contain finite values")
        object.__setattr__(self, "yz_centers", yz)
        object.__setattr__(self, "back_x", back)


def build_rear_columns(occupied_centers: Sequence[Sequence[float]], voxel_size: float) -> RearColumns:
    """Collapse occupied voxel centers into yz columns using their +x surface."""
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")
    centers = np.asarray(occupied_centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) == 0:
        raise ValueError("occupied_centers must be a non-empty [N, 3] array")
    if not np.isfinite(centers).all():
        raise ValueError("occupied_centers contains non-finite coordinates")

    yz_centers, inverse = np.unique(centers[:, 1:3], axis=0, return_inverse=True)
    back_x = np.full(len(yz_centers), -np.inf, dtype=np.float64)
    np.maximum.at(back_x, inverse, centers[:, 0] + voxel_size * 0.5)
    return RearColumns(yz_centers=yz_centers, back_x=back_x)


def compute_rear_region_metric(
    endpoint: Sequence[float],
    columns: RearColumns,
    voxel_size: float,
    lateral_margin: float,
    rear_clearance: float,
    score_scale: float,
) -> Dict[str, object]:
    """Return endpoint distance and penetration into the +x rear-region union."""
    point = np.asarray(endpoint, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("endpoint must contain three finite coordinates")
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")
    if lateral_margin < 0.0 or rear_clearance < 0.0:
        raise ValueError("margins must be non-negative")
    if score_scale <= 0.0:
        raise ValueError("score_scale must be positive")

    half_extent = voxel_size * 0.5 + lateral_margin
    offset_yz = np.abs(columns.yz_centers - point[1:3])
    outside_yz = np.maximum(offset_yz - half_extent, 0.0)
    lateral_distances = np.linalg.norm(outside_yz, axis=1)
    shadow_start_x = columns.back_x + rear_clearance
    axial_gaps = np.maximum(shadow_start_x - point[0], 0.0)
    distances = np.hypot(lateral_distances, axial_gaps)
    closest_index = int(np.argmin(distances))
    distance = float(distances[closest_index])
    tolerance = 1e-9
    reached_mask = (lateral_distances <= tolerance) & (axial_gaps <= tolerance)
    reached = bool(np.any(reached_mask))
    depth = 0.0
    if reached:
        depth = float(np.max(point[0] - shadow_start_x[reached_mask]))

    return {
        "rear_region_reached": reached,
        "rear_region_distance_m": distance,
        "rear_region_score": float(math.exp(-distance / score_scale)),
        "rear_region_depth_m": max(0.0, depth),
        "rear_region_lateral_distance_m": float(lateral_distances[closest_index]),
        "rear_region_axial_gap_m": float(axial_gaps[closest_index]),
        "obstacle_back_x_m": float(columns.back_x[closest_index]),
        "closest_shadow_y_m": float(columns.yz_centers[closest_index, 0]),
        "closest_shadow_z_m": float(columns.yz_centers[closest_index, 1]),
    }


def _trajectory_identity(path: Path) -> Tuple[str, str]:
    if "__" not in path.stem:
        raise ValueError(f"trajectory filename must be skill__scene_id.json: {path}")
    skill, scene_id = path.stem.split("__", 1)
    return skill, scene_id


def _batch_number(path: Path) -> int:
    try:
        return int(path.name.split("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid batch directory name: {path.name}") from exc


def _find_model_runs(exp_root: Path, model: str) -> List[Tuple[str, Path]]:
    """Find runs in legacy roots or the organized ``rollouts/`` subtree."""
    discovered: Dict[str, Path] = {}
    run_names = (model, "bc_dagger") if model == "dagger" else (model,)
    search_roots = [exp_root]
    rollout_root = exp_root / "rollouts"
    if rollout_root.is_dir():
        search_roots.insert(0, rollout_root)
    for root in search_roots:
        for run_name in run_names:
            for batch_dir in (root / run_name).glob("batch_*"):
                discovered.setdefault(batch_dir.name, batch_dir)
        for batch_dir in root.glob("batch_*"):
            for run_name in run_names:
                run_dir = batch_dir / run_name
                if run_dir.is_dir():
                    discovered.setdefault(batch_dir.name, run_dir)
    return sorted(discovered.items(), key=lambda item: _batch_number(Path(item[0])))


def _load_results(path: Path) -> Dict[Tuple[str, str], Mapping]:
    latest: Dict[Tuple[str, str], Mapping] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            latest[(str(record["skill"]), str(record["scene_id"]))] = record
    return latest


def _stable_scene_seed(skill: str, scene_id: str) -> int:
    digest = hashlib.sha256(f"{skill}/{scene_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def load_scene_columns(
    data_root: Path, skill: str, scene_id: str, voxel_size: float
) -> RearColumns:
    glb_path = (
        data_root
        / skill
        / scene_id
        / "_render_scene"
        / f"{scene_id}_with_ground.glb"
    )
    if not glb_path.is_file():
        raise FileNotFoundError(f"missing scene mesh: {glb_path}")

    random_state = np.random.get_state()
    np.random.seed(_stable_scene_seed(skill, scene_id))
    try:
        occupancy = _drop_ground_voxels(
            _occupancy_from_glb(glb_path, voxel_size=voxel_size)
        )
    finally:
        np.random.set_state(random_state)
    indices = occupancy.occupied_indices.astype(np.float64)
    origin = np.asarray(occupancy.origin, dtype=np.float64)
    centers = origin + (indices + 0.5) * occupancy.voxel_size
    return build_rear_columns(centers, voxel_size=occupancy.voxel_size)


def _metric_statistics(records: Sequence[Mapping]) -> Dict[str, object]:
    reached_count = sum(bool(record["rear_region_reached"]) for record in records)
    output: Dict[str, object] = {
        "count": len(records),
        "reached_count": reached_count,
        "reached_rate": reached_count / len(records) if records else 0.0,
    }
    for field in (
        "rear_region_distance_m",
        "rear_region_score",
        "rear_region_depth_m",
        "rear_region_lateral_distance_m",
        "rear_region_axial_gap_m",
    ):
        values = [float(record[field]) for record in records]
        output[field] = {
            "mean": float(statistics.fmean(values)),
            "median": float(statistics.median(values)),
        }
    return output


def _summarize(records: Sequence[Mapping], configuration: Mapping) -> Dict[str, object]:
    overall: Dict[str, object] = {}
    per_skill: Dict[str, object] = {}
    for model in sorted({str(record["model"]) for record in records}):
        model_records = [record for record in records if record["model"] == model]
        overall[model] = _metric_statistics(model_records)
        per_skill[model] = {}
        for skill in sorted({str(record["skill"]) for record in model_records}):
            skill_records = [record for record in model_records if record["skill"] == skill]
            per_skill[model][skill] = _metric_statistics(skill_records)
    return {
        "schema_version": 1,
        "configuration": dict(configuration),
        "record_count": len(records),
        "overall": overall,
        "per_skill": per_skill,
    }


def _write_outputs(output_dir: Path, records: Sequence[Mapping], summary: Mapping) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_scene.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    with (output_dir / "per_scene.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def evaluate_tree(
    exp_root: str,
    data_root: str,
    output_dir: str,
    models: Sequence[str],
    voxel_size: float,
    lateral_margin: float,
    rear_clearance: float,
    score_scale: float,
    column_loader: Optional[Callable[[str, str], RearColumns]] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    exp_path = Path(exp_root).resolve()
    data_path = Path(data_root).resolve()
    if column_loader is None:
        column_loader = lambda skill, scene_id: load_scene_columns(
            data_path, skill, scene_id, voxel_size
        )
    column_cache: Dict[Tuple[str, str], RearColumns] = {}
    records: List[Dict[str, object]] = []

    for model in models:
        if model not in MODEL_NAMES:
            raise ValueError(f"unsupported model {model!r}; choose from {MODEL_NAMES}")
        model_runs = _find_model_runs(exp_path, model)
        if not model_runs:
            raise FileNotFoundError(f"no batch directories found for {model} under {exp_path}")
        for batch_name, run_dir in model_runs:
            results = _load_results(run_dir / "results.jsonl")
            trajectory_paths = sorted((run_dir / "trajectories").glob("*.json"))
            if not trajectory_paths:
                raise FileNotFoundError(f"no trajectories found under {run_dir}")
            for trajectory_path in trajectory_paths:
                skill, scene_id = _trajectory_identity(trajectory_path)
                result_record = results.get((skill, scene_id))
                if result_record is None:
                    raise KeyError(
                        f"missing result for {skill}/{scene_id} in "
                        f"{run_dir / 'results.jsonl'}"
                    )
                trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
                points = np.asarray(trajectory["points"], dtype=np.float64)
                if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
                    raise ValueError(f"invalid trajectory points: {trajectory_path}")
                endpoint = points[-1, :3]
                cache_key = (skill, scene_id)
                if cache_key not in column_cache:
                    column_cache[cache_key] = column_loader(skill, scene_id)
                    print(
                        f"[scene {len(column_cache)}] loaded rear columns for "
                        f"{skill}/{scene_id}",
                        flush=True,
                    )
                metric = compute_rear_region_metric(
                    endpoint=endpoint,
                    columns=column_cache[cache_key],
                    voxel_size=voxel_size,
                    lateral_margin=lateral_margin,
                    rear_clearance=rear_clearance,
                    score_scale=score_scale,
                )
                records.append(
                    {
                        "model": model,
                        "batch": batch_name,
                        "skill": skill,
                        "scene_id": scene_id,
                        "status": str(result_record.get("status", "unknown")),
                        "endpoint_x_m": float(endpoint[0]),
                        "endpoint_y_m": float(endpoint[1]),
                        "endpoint_z_m": float(endpoint[2]),
                        **metric,
                    }
                )
            print(
                f"[{model}/{batch_name}] processed {len(trajectory_paths)} trajectories",
                flush=True,
            )

    records.sort(
        key=lambda record: (
            MODEL_NAMES.index(str(record["model"])),
            _batch_number(Path(str(record["batch"]))),
            str(record["skill"]),
            str(record["scene_id"]),
        )
    )
    configuration = {
        "exp_root": str(exp_path),
        "data_root": str(data_path),
        "models": list(models),
        "rear_direction": "+x",
        "voxel_size_m": float(voxel_size),
        "lateral_margin_m": float(lateral_margin),
        "rear_clearance_m": float(rear_clearance),
        "score_scale_m": float(score_scale),
        "score_definition": "exp(-rear_region_distance_m / score_scale_m)",
        "gate_used": False,
        "expert_trajectory_used": False,
    }
    summary = _summarize(records, configuration)
    _write_outputs(Path(output_dir), records, summary)
    return records, summary


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-root", default="exp")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES)
    )
    parser.add_argument("--voxel-size", type=float, default=0.08)
    parser.add_argument("--drone-radius", type=float, default=0.15)
    parser.add_argument("--safety-margin", type=float, default=0.20)
    parser.add_argument("--rear-clearance", type=float, default=0.15)
    parser.add_argument("--score-scale", type=float, default=0.50)
    return parser.parse_args(argv)


def main(argv: Sequence[str] = None) -> int:
    args = parse_args(argv)
    if args.voxel_size <= 0.0:
        raise ValueError("--voxel-size must be positive")
    if args.drone_radius < 0.0 or args.safety_margin < 0.0:
        raise ValueError("--drone-radius and --safety-margin must be non-negative")
    if args.rear_clearance < 0.0:
        raise ValueError("--rear-clearance must be non-negative")
    if args.score_scale <= 0.0:
        raise ValueError("--score-scale must be positive")
    output_dir = args.output_dir or os.path.join(args.exp_root, "metrics", "rear_region")
    records, _ = evaluate_tree(
        exp_root=args.exp_root,
        data_root=args.data_root,
        output_dir=output_dir,
        models=args.models,
        voxel_size=args.voxel_size,
        lateral_margin=args.drone_radius + args.safety_margin,
        rear_clearance=args.rear_clearance,
        score_scale=args.score_scale,
    )
    print(f"Wrote {len(records)} records to {Path(output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
