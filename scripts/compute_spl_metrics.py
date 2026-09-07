#!/usr/bin/env python3
"""Compute Success weighted by Path Length (SPL) for rollout batches."""

import argparse
import csv
import json
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from compute_trajectory_metrics import MODEL_NAMES, _find_model_runs, path_length


CSV_FIELDS = (
    "model",
    "batch",
    "skill",
    "scene_id",
    "status",
    "success",
    "reference_path_length_m",
    "rollout_path_length_m",
    "path_efficiency",
    "spl",
    "trajectory_available",
)


def _load_latest_results(path: Path) -> Dict[Tuple[str, str], Mapping]:
    latest: Dict[Tuple[str, str], Mapping] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = (str(record["skill"]), str(record["scene_id"]))
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"invalid result at {path}:{line_number}: {exc}") from exc
            latest[key] = record
    return latest


def _reference_length(reference_root: Path, skill: str, scene_id: str) -> float:
    metadata_path = reference_root / skill / scene_id / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("planner_success") is not True:
        raise ValueError(f"reference planner did not succeed: {metadata_path}")
    length = float(metadata["reference_path_length_m"])
    if length <= 0.0:
        raise ValueError(f"reference path length must be positive: {metadata_path}")
    return length


def _evaluate_result(
    model: str,
    batch: str,
    run_dir: Path,
    result: Mapping,
    reference_root: Path,
) -> Dict[str, object]:
    skill = str(result["skill"])
    scene_id = str(result["scene_id"])
    status = str(result.get("status", "unknown"))
    reference_length = _reference_length(reference_root, skill, scene_id)
    trajectory_path = run_dir / "trajectories" / f"{skill}__{scene_id}.json"
    trajectory_available = trajectory_path.is_file()
    rollout_length = None
    if trajectory_available:
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        rollout_length = path_length(trajectory["points"])

    success = status == "success"
    if success and rollout_length is None:
        raise FileNotFoundError(f"successful result has no trajectory: {trajectory_path}")
    efficiency = (
        reference_length / max(float(rollout_length), reference_length)
        if rollout_length is not None
        else 0.0
    )
    return {
        "model": model,
        "batch": batch,
        "skill": skill,
        "scene_id": scene_id,
        "status": status,
        "success": int(success),
        "reference_path_length_m": reference_length,
        "rollout_path_length_m": rollout_length,
        "path_efficiency": efficiency,
        "spl": efficiency if success else 0.0,
        "trajectory_available": int(trajectory_available),
    }


def _statistics(records: Sequence[Mapping]) -> Dict[str, object]:
    spl_values = [float(record["spl"]) for record in records]
    successful = [record for record in records if bool(record["success"])]
    return {
        "count": len(records),
        "success_count": len(successful),
        "success_rate": len(successful) / len(records) if records else 0.0,
        "spl": statistics.fmean(spl_values) if spl_values else 0.0,
        "successful_path_efficiency_mean": (
            statistics.fmean(float(record["path_efficiency"]) for record in successful)
            if successful
            else 0.0
        ),
        "missing_trajectory_count": sum(
            not bool(record["trajectory_available"]) for record in records
        ),
        "status_counts": dict(sorted(Counter(str(record["status"]) for record in records).items())),
    }


def evaluate_tree(
    exp_root: str,
    reference_root: str,
    output_dir: str,
    models: Sequence[str],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    exp_path = Path(exp_root).resolve()
    reference_path = Path(reference_root).resolve()
    records: List[Dict[str, object]] = []
    for model in models:
        model_runs = _find_model_runs(exp_path, model)
        if not model_runs:
            raise FileNotFoundError(f"no batch directories found for {model} under {exp_path}")
        for batch, run_dir in model_runs:
            results = _load_latest_results(run_dir / "results.jsonl")
            for result in results.values():
                records.append(
                    _evaluate_result(model, batch, run_dir, result, reference_path)
                )
            print(f"[{model}/{batch}] processed {len(results)} results")

    records.sort(
        key=lambda record: (
            MODEL_NAMES.index(str(record["model"])),
            int(str(record["batch"]).split("_", 1)[1]),
            str(record["skill"]),
            str(record["scene_id"]),
        )
    )
    overall = {}
    per_skill = {}
    for model in models:
        model_records = [record for record in records if record["model"] == model]
        overall[model] = _statistics(model_records)
        per_skill[model] = {}
        for skill in sorted({str(record["skill"]) for record in model_records}):
            per_skill[model][skill] = _statistics(
                [record for record in model_records if record["skill"] == skill]
            )
    summary = {
        "schema_version": 1,
        "configuration": {
            "exp_root": str(exp_path),
            "reference_root": str(reference_path),
            "models": list(models),
            "success_definition": "status == success",
            "spl_definition": "mean_i[S_i * L_i / max(P_i, L_i)]",
            "reference_path_source": "frozen Fast-Planner reference paths",
        },
        "record_count": len(records),
        "overall": overall,
        "per_skill": per_skill,
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "per_scene.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    with (destination / "per_scene.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return records, summary


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-root", default="exp")
    parser.add_argument("--reference-root", default="eval_reference_paths")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    return parser.parse_args(argv)


def main(argv: Sequence[str] = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or os.path.join(args.exp_root, "metrics", "spl")
    records, _ = evaluate_tree(
        args.exp_root, args.reference_root, output_dir, args.models
    )
    print(f"Wrote {len(records)} records to {Path(output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
