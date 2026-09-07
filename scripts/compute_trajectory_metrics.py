#!/usr/bin/env python3
"""Compute goal-free trajectory fidelity metrics for rollout results.

The expert path is cropped to the rollout's traveled arc length before both
paths are spatially resampled.  The evaluator does not read or use gate data.
"""

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


MODEL_NAMES = ("bc", "dagger", "openvla", "aerialclaw", "spf")
CSV_FIELDS = (
    "model",
    "batch",
    "skill",
    "scene_id",
    "status",
    "crop_rule",
    "rollout_points",
    "expert_full_points",
    "expert_cropped_points",
    "rollout_resampled_points",
    "expert_resampled_points",
    "rollout_length_m",
    "expert_full_length_m",
    "expert_cropped_length_m",
    "expert_crop_fraction",
    "rollout_max_x_m",
    "expert_crop_end_x_m",
    "ndtw",
    "path_coverage",
    "length_score",
    "cls",
)


def _path_array(points: Sequence[Sequence[float]], name: str = "path") -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 3 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty [N, >=3] array")
    array = array[:, :3]
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    return array


def path_length(path: Sequence[Sequence[float]]) -> float:
    points = _path_array(path)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _deduplicate_consecutive(path: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    if len(path) < 2:
        return path.copy()
    keep = np.concatenate(
        ([True], np.linalg.norm(np.diff(path, axis=0), axis=1) > tolerance)
    )
    return path[keep]


def crop_path_to_length(
    path: Sequence[Sequence[float]], target_length: float
) -> np.ndarray:
    """Return the path prefix ending at exactly ``target_length`` meters."""
    if target_length < 0.0:
        raise ValueError("target_length must be non-negative")
    points = _deduplicate_consecutive(_path_array(path))
    if len(points) == 1 or target_length == 0.0:
        return points[:1].copy()

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_length = float(segment_lengths.sum())
    if target_length >= total_length:
        return points.copy()

    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    segment_index = int(np.searchsorted(cumulative, target_length, side="right") - 1)
    segment_index = min(segment_index, len(segment_lengths) - 1)
    local_length = target_length - cumulative[segment_index]
    fraction = local_length / segment_lengths[segment_index]
    endpoint = points[segment_index] + fraction * (
        points[segment_index + 1] - points[segment_index]
    )
    return np.vstack((points[: segment_index + 1], endpoint))


def crop_expert_to_rollout_max_x(
    expert: Sequence[Sequence[float]], rollout: Sequence[Sequence[float]]
) -> Tuple[np.ndarray, float]:
    """Crop expert at its first crossing of the rollout's farthest x position."""
    expert_points = _deduplicate_consecutive(_path_array(expert, "expert path"))
    rollout_points = _path_array(rollout, "rollout path")
    target_x = float(np.max(rollout_points[:, 0]))
    expert_max_x = float(np.max(expert_points[:, 0]))
    tolerance = 1e-12

    if target_x <= float(expert_points[0, 0]) + tolerance:
        return expert_points[:1].copy(), target_x
    if target_x >= expert_max_x - tolerance:
        return expert_points.copy(), target_x

    for index in range(len(expert_points) - 1):
        start = expert_points[index]
        end = expert_points[index + 1]
        start_x = float(start[0])
        end_x = float(end[0])
        if math.isclose(start_x, target_x, rel_tol=0.0, abs_tol=tolerance):
            return expert_points[: index + 1].copy(), target_x
        if (start_x - target_x) * (end_x - target_x) <= 0.0 and not math.isclose(
            start_x, end_x, rel_tol=0.0, abs_tol=tolerance
        ):
            fraction = (target_x - start_x) / (end_x - start_x)
            endpoint = start + fraction * (end - start)
            return np.vstack((expert_points[: index + 1], endpoint)), target_x

    raise ValueError(
        f"expert path never crosses rollout maximum x={target_x:.6f} despite "
        f"expert maximum x={expert_max_x:.6f}"
    )


def resample_path(path: Sequence[Sequence[float]], step: float) -> np.ndarray:
    """Resample a polyline at fixed arc-length intervals, retaining its endpoint."""
    if step <= 0.0:
        raise ValueError("resample step must be positive")
    points = _deduplicate_consecutive(_path_array(path))
    if len(points) == 1:
        return points.copy()

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    targets = np.arange(0.0, total_length, step, dtype=np.float64)
    if len(targets) == 0 or not math.isclose(
        float(targets[-1]), total_length, rel_tol=0.0, abs_tol=1e-10
    ):
        targets = np.append(targets, total_length)

    output = np.empty((len(targets), 3), dtype=np.float64)
    for axis in range(3):
        output[:, axis] = np.interp(targets, cumulative, points[:, axis])
    return output


def compute_ndtw(
    predicted: Sequence[Sequence[float]],
    reference: Sequence[Sequence[float]],
    distance_threshold: float,
) -> float:
    """Compute normalized Dynamic Time Warping with 3-D Euclidean distance."""
    if distance_threshold <= 0.0:
        raise ValueError("distance_threshold must be positive")
    predicted_array = _path_array(predicted, "predicted path")
    reference_array = _path_array(reference, "reference path")

    previous = np.full(len(reference_array) + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0
    for predicted_point in predicted_array:
        current = np.full(len(reference_array) + 1, np.inf, dtype=np.float64)
        for j, reference_point in enumerate(reference_array, start=1):
            distance = float(np.linalg.norm(predicted_point - reference_point))
            current[j] = distance + min(previous[j], current[j - 1], previous[j - 1])
        previous = current
    dtw_distance = float(previous[-1])
    return float(
        math.exp(-dtw_distance / (distance_threshold * len(reference_array)))
    )


def point_to_polyline_distances(
    points: Sequence[Sequence[float]], polyline: Sequence[Sequence[float]]
) -> np.ndarray:
    """Return each point's exact minimum Euclidean distance to a 3-D polyline."""
    query = _path_array(points, "query points")
    line = _deduplicate_consecutive(_path_array(polyline, "polyline"))
    if len(line) == 1:
        return np.linalg.norm(query - line[0], axis=1)

    starts = line[:-1]
    vectors = line[1:] - line[:-1]
    squared_lengths = np.sum(vectors * vectors, axis=1)
    relative = query[:, None, :] - starts[None, :, :]
    parameters = np.sum(relative * vectors[None, :, :], axis=2) / squared_lengths[None, :]
    parameters = np.clip(parameters, 0.0, 1.0)
    projections = starts[None, :, :] + parameters[:, :, None] * vectors[None, :, :]
    return np.linalg.norm(query[:, None, :] - projections, axis=2).min(axis=1)


def compute_cls(
    predicted: Sequence[Sequence[float]],
    reference: Sequence[Sequence[float]],
    distance_threshold: float,
) -> Dict[str, float]:
    """Compute path coverage, length score, and Coverage weighted Length Score."""
    if distance_threshold <= 0.0:
        raise ValueError("distance_threshold must be positive")
    predicted_array = _path_array(predicted, "predicted path")
    reference_array = _path_array(reference, "reference path")
    distances = point_to_polyline_distances(reference_array, predicted_array)
    path_coverage = float(np.exp(-distances / distance_threshold).mean())

    predicted_length = path_length(predicted_array)
    reference_length = path_length(reference_array)
    expected_length = path_coverage * reference_length
    denominator = expected_length + abs(expected_length - predicted_length)
    if denominator > 0.0:
        length_score = expected_length / denominator
    else:
        length_score = 1.0
    cls_score = path_coverage * length_score
    return {
        "path_coverage": float(path_coverage),
        "length_score": float(length_score),
        "cls": float(cls_score),
    }


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
            key = (str(record["skill"]), str(record["scene_id"]))
            latest[key] = record
    return latest


def _load_expert_path(expert_root: Path, skill: str, scene_id: str) -> np.ndarray:
    frames_path = expert_root / skill / scene_id / "frames.json"
    if not frames_path.is_file():
        raise FileNotFoundError(f"missing expert frames: {frames_path}")
    frames = json.loads(frames_path.read_text(encoding="utf-8"))
    try:
        points = [frame["camera_position_std"] for frame in frames]
    except KeyError as exc:
        raise ValueError(f"expert frame missing camera_position_std: {frames_path}") from exc
    return _path_array(points, f"expert path {skill}/{scene_id}")


def _trajectory_identity(path: Path) -> Tuple[str, str]:
    stem = path.stem
    if "__" not in stem:
        raise ValueError(f"trajectory filename must be skill__scene_id.json: {path}")
    return tuple(stem.split("__", 1))  # type: ignore[return-value]


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


def _evaluate_trajectory(
    model: str,
    batch: str,
    trajectory_path: Path,
    result: Mapping,
    expert_root: Path,
    distance_threshold: float,
    resample_step: float,
    crop_rule: str,
) -> Dict:
    skill, scene_id = _trajectory_identity(trajectory_path)
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    rollout = _path_array(trajectory["points"], f"rollout {model}/{batch}/{scene_id}")
    expert_full = _load_expert_path(expert_root, skill, scene_id)

    rollout_length = path_length(rollout)
    expert_full_length = path_length(expert_full)
    rollout_max_x = float(np.max(rollout[:, 0]))
    if crop_rule == "max_x":
        expert_cropped, _ = crop_expert_to_rollout_max_x(expert_full, rollout)
    elif crop_rule == "arc_length":
        expert_cropped = crop_path_to_length(expert_full, rollout_length)
    else:
        raise ValueError(f"unsupported crop rule: {crop_rule}")
    expert_cropped_length = path_length(expert_cropped)
    crop_fraction = (
        expert_cropped_length / expert_full_length if expert_full_length > 0.0 else 1.0
    )

    rollout_resampled = resample_path(rollout, resample_step)
    expert_resampled = resample_path(expert_cropped, resample_step)
    ndtw = compute_ndtw(rollout_resampled, expert_resampled, distance_threshold)
    cls_values = compute_cls(rollout_resampled, expert_resampled, distance_threshold)

    return {
        "model": model,
        "batch": batch,
        "skill": skill,
        "scene_id": scene_id,
        "status": str(result.get("status", "unknown")),
        "crop_rule": crop_rule,
        "rollout_points": int(len(rollout)),
        "expert_full_points": int(len(expert_full)),
        "expert_cropped_points": int(len(expert_cropped)),
        "rollout_resampled_points": int(len(rollout_resampled)),
        "expert_resampled_points": int(len(expert_resampled)),
        "rollout_length_m": float(rollout_length),
        "expert_full_length_m": float(expert_full_length),
        "expert_cropped_length_m": float(expert_cropped_length),
        "expert_crop_fraction": float(crop_fraction),
        "rollout_max_x_m": rollout_max_x,
        "expert_crop_end_x_m": float(expert_cropped[-1, 0]),
        "ndtw": float(ndtw),
        **cls_values,
    }


def _metric_statistics(records: Sequence[Mapping]) -> Dict:
    output: Dict[str, object] = {"count": len(records)}
    for field in (
        "ndtw",
        "path_coverage",
        "length_score",
        "cls",
        "expert_crop_fraction",
        "rollout_length_m",
    ):
        values = [float(record[field]) for record in records]
        output[field] = {
            "mean": float(statistics.fmean(values)),
            "median": float(statistics.median(values)),
        }
    statuses: Dict[str, int] = {}
    for record in records:
        status = str(record["status"])
        statuses[status] = statuses.get(status, 0) + 1
    output["status_counts"] = dict(sorted(statuses.items()))
    return output


def _summarize(records: Sequence[Mapping], configuration: Mapping) -> Dict:
    overall = {}
    per_skill = {}
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
    jsonl_path = output_dir / "per_scene.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    csv_path = output_dir / "per_scene.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def evaluate_tree(
    exp_root: str,
    expert_root: str,
    output_dir: str,
    models: Sequence[str],
    distance_threshold: float,
    resample_step: float,
    crop_rule: str = "max_x",
) -> Tuple[List[Dict], Dict]:
    exp_path = Path(exp_root).resolve()
    expert_path = Path(expert_root).resolve()
    records: List[Dict] = []
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
                result = results.get((skill, scene_id))
                if result is None:
                    raise KeyError(
                        f"missing result for {skill}/{scene_id} in {run_dir / 'results.jsonl'}"
                    )
                records.append(
                    _evaluate_trajectory(
                        model=model,
                        batch=batch_name,
                        trajectory_path=trajectory_path,
                        result=result,
                        expert_root=expert_path,
                        distance_threshold=distance_threshold,
                        resample_step=resample_step,
                        crop_rule=crop_rule,
                    )
                )
            print(f"[{model}/{batch_name}] processed {len(trajectory_paths)} trajectories")

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
        "expert_root": str(expert_path),
        "models": list(models),
        "distance_threshold_m": float(distance_threshold),
        "resample_step_m": float(resample_step),
        "expert_crop_rule": crop_rule,
        "expert_crop_rule_description": (
            "crop expert path at its first interpolated crossing of the rollout maximum x; "
            "use the full expert path when rollout maximum x reaches or exceeds expert maximum x"
            if crop_rule == "max_x"
            else "crop expert path to rollout 3-D arc length; cap at full expert length"
        ),
        "expert_path_source": "frames.json:camera_position_std",
        "gate_used": False,
    }
    summary = _summarize(records, configuration)
    _write_outputs(Path(output_dir), records, summary)
    return records, summary


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-root", default="exp")
    parser.add_argument("--expert-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--distance-threshold", type=float, default=0.5)
    parser.add_argument("--resample-step", type=float, default=0.1)
    parser.add_argument(
        "--crop-rule",
        choices=("max_x", "arc_length"),
        default="max_x",
        help="expert trajectory cropping rule (default: max_x)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] = None) -> int:
    args = parse_args(argv)
    if args.distance_threshold <= 0.0:
        raise ValueError("--distance-threshold must be positive")
    if args.resample_step <= 0.0:
        raise ValueError("--resample-step must be positive")
    output_dir = args.output_dir or os.path.join(
        args.exp_root, "metrics", "trajectory_fidelity", args.crop_rule
    )
    records, _ = evaluate_tree(
        exp_root=args.exp_root,
        expert_root=args.expert_root,
        output_dir=output_dir,
        models=args.models,
        distance_threshold=args.distance_threshold,
        resample_step=args.resample_step,
        crop_rule=args.crop_rule,
    )
    print(f"Wrote {len(records)} records to {Path(output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
