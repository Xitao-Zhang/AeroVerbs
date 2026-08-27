#!/usr/bin/env python
"""Precompute Fast-Planner semantic reference paths for SPL evaluation."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
import traceback
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from aeroverbs.config import CompletionConfig, load_config
from aeroverbs.data.completion import (
    CompletionJudge,
    load_object_mesh,
    make_clearance_provider,
)
from aeroverbs.data.fast_planner_teacher import (
    FastPlannerTeacherConfig,
    FastPlannerTeacherLabeler,
)


def path_length_3d(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def truncate_at_first_completion(
    points: np.ndarray, completion: Sequence[float]
) -> Tuple[np.ndarray, int]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    completion = np.asarray(completion).reshape(-1)
    if len(completion) != len(points):
        raise ValueError("completion labels must match trajectory length")
    indices = np.flatnonzero(completion > 0.5)
    if not len(indices):
        raise ValueError("trajectory never reaches completion")
    index = int(indices[0])
    return points[: index + 1].copy(), index


def validate_collision_clearance(labels: Dict[str, np.ndarray], collision_radius: float) -> float:
    clearance = np.asarray(labels["clearance"], dtype=np.float64).reshape(-1)
    if not len(clearance) or not np.all(np.isfinite(clearance)):
        raise RuntimeError("trajectory clearance labels are invalid")
    minimum = float(clearance.min())
    if minimum + 1e-6 < float(collision_radius):
        raise RuntimeError(
            f"trajectory violates {collision_radius:.3f}m collision radius "
            f"(minimum clearance {minimum:.3f}m)"
        )
    return minimum


def _atomic_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(temporary, path)


def save_reference(output_dir: str, metadata: dict, points: np.ndarray) -> str:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    sample_output = os.path.join(
        output_dir, str(metadata["skill"]), str(metadata["scene_id"])
    )
    os.makedirs(sample_output, exist_ok=True)

    npz_path = os.path.join(sample_output, "reference_path.npz")
    temporary_npz = f"{npz_path}.tmp.npz"
    np.savez_compressed(temporary_npz, points_std=points)
    os.replace(temporary_npz, npz_path)

    _atomic_json(
        os.path.join(sample_output, "trajectory.json"),
        {"points_std": points.tolist()},
    )
    _atomic_json(os.path.join(sample_output, "metadata.json"), metadata)
    return sample_output


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_manifest_index(index_path: str) -> Tuple[str, List[dict], dict]:
    index_path = os.path.abspath(index_path)
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    directory = os.path.dirname(index_path)
    samples: List[dict] = []
    seen = set()
    for entry in index.get("manifests", []):
        path = os.path.join(directory, entry["file"])
        if entry.get("sha256") and _sha256(path) != entry["sha256"]:
            raise ValueError(f"manifest checksum mismatch: {path}")
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
        for sample in manifest.get("samples", []):
            identifier = (sample["skill"], sample["scene_id"])
            if identifier in seen:
                raise ValueError(f"duplicate manifest sample: {identifier}")
            seen.add(identifier)
            item = dict(sample)
            item["batch_id"] = int(manifest.get("batch_id", entry.get("batch_id", 0)))
            samples.append(item)
    expected = index.get("total_samples")
    if expected is not None and len(samples) != int(expected):
        raise ValueError(
            f"manifest index expected {expected} samples, loaded {len(samples)}"
        )
    return str(index["data_root"]), samples, index


def _load_expert_positions(sample_dir: str) -> np.ndarray:
    with open(os.path.join(sample_dir, "frames.json"), encoding="utf-8") as f:
        frames = json.load(f)
    points = np.asarray(
        [frame["camera_position_std"] for frame in frames], dtype=np.float32
    ).reshape(-1, 3)
    if len(points) < 2:
        raise ValueError("expert trajectory contains fewer than two points")
    if not np.all(np.isfinite(points)):
        raise ValueError("expert trajectory contains non-finite points")
    return points


def _completion_labels(
    sample_dir: str,
    data_root: str,
    points: np.ndarray,
    completion_cfg: CompletionConfig,
) -> Dict[str, np.ndarray]:
    with open(os.path.join(sample_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    mesh = load_object_mesh(sample_dir, data_root, meta)
    provider = make_clearance_provider(sample_dir, data_root, meta, mesh=mesh)
    judge = CompletionJudge(
        meta,
        clearance_provider=provider,
        cfg=completion_cfg,
        mesh=mesh,
    )
    return judge.label_sequence(points)


def plan_until_endpoint(
    labeler,
    start: np.ndarray,
    endpoint: np.ndarray,
    *,
    timeout_s: float,
    max_replans: int = 32,
    endpoint_tolerance: float = 1e-3,
    route_points: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[np.ndarray, int]:
    """Replan partial Fast-Planner horizons until the endpoint is reached."""
    current = np.asarray(start, dtype=np.float32).reshape(3)
    endpoint = np.asarray(endpoint, dtype=np.float32).reshape(3)
    joined = [current.copy()]
    started = time.perf_counter()
    requested_route = labeler._route_for(current) if route_points is None else route_points
    route = [np.asarray(point, dtype=np.float32).reshape(3) for point in requested_route]
    if float(np.linalg.norm(route[0] - current)) > endpoint_tolerance:
        route.insert(0, current.copy())
    if float(np.linalg.norm(route[-1] - endpoint)) > endpoint_tolerance:
        route.append(endpoint.copy())
    replan_count = 0
    for target in route[1:]:
        while float(np.linalg.norm(current - target)) > endpoint_tolerance:
            replan_count += 1
            if replan_count > max_replans:
                remaining = float(np.linalg.norm(current - target))
                raise RuntimeError(
                    f"Fast-Planner did not reach route waypoint after {max_replans} replans "
                    f"(remaining distance {remaining:.3f}m)"
                )
            segment = np.asarray(
                labeler.planner.plan_waypoints(
                    [tuple(current), tuple(target)],
                    start_vel=np.zeros(3, dtype=np.float32),
                    start_acc=np.zeros(3, dtype=np.float32),
                ),
                dtype=np.float32,
            ).reshape(-1, 3)
            elapsed = time.perf_counter() - started
            if elapsed > timeout_s:
                raise TimeoutError(
                    f"Fast-Planner took {elapsed:.3f}s, budget is {timeout_s:.3f}s"
                )
            if len(segment) < 2 or not np.all(np.isfinite(segment)):
                raise RuntimeError("Fast-Planner returned an invalid partial path")
            if float(np.linalg.norm(segment[0] - current)) > 0.25:
                raise RuntimeError("Fast-Planner partial path is disconnected")
            joined.extend(point.copy() for point in segment[1:])
            current = segment[-1].copy()
        joined[-1] = target.copy()
        current = target.copy()
    if float(np.linalg.norm(current - endpoint)) <= endpoint_tolerance:
        joined[-1] = endpoint.copy()
        return np.asarray(joined, dtype=np.float32), replan_count
    remaining = float(np.linalg.norm(current - endpoint))
    raise RuntimeError(
        f"Fast-Planner did not reach endpoint after {max_replans} replans "
        f"(remaining distance {remaining:.3f}m)"
    )


def _plan_to_endpoint(
    sample_dir: str,
    data_root: str,
    endpoint: np.ndarray,
    planner_cfg: FastPlannerTeacherConfig,
    route_override: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[np.ndarray, dict]:
    labeler = FastPlannerTeacherLabeler(
        sample_dir,
        data_root=data_root,
        config=planner_cfg,
    )
    start = labeler.expert_positions[0].copy()
    labeler.goal = np.asarray(endpoint, dtype=np.float32).reshape(3)
    route = labeler._route_for(start) if route_override is None else route_override
    started = time.perf_counter()
    path, replan_count = plan_until_endpoint(
        labeler,
        start,
        endpoint,
        timeout_s=planner_cfg.timeout_s,
        route_points=route,
    )
    elapsed = time.perf_counter() - started
    sdf_distances = []
    for point in path:
        if not labeler.sdf_map.is_in_map(point):
            raise RuntimeError("Fast-Planner path leaves the precomputed SDF map")
        distance, _ = labeler.sdf_map.get_dist_with_grad_trilinear(point)
        if not np.isfinite(distance):
            raise RuntimeError("Fast-Planner path has invalid SDF clearance")
        sdf_distances.append(float(distance))
    diagnostics = {
        "route_points": [list(map(float, point)) for point in route],
        "guide": None if labeler.guide is None else labeler.guide.tolist(),
        "guide_source": labeler.guide_source,
        "preferred_clearance_m": float(labeler.preferred_clearance),
        "planning_time_s": float(elapsed),
        "replan_count": int(replan_count),
        "sdf_path": labeler.sdf_path,
        "minimum_inflated_sdf_m": float(min(sdf_distances)),
        "route_override": route_override is not None,
    }
    return path, diagnostics


def build_reference_for_sample(
    sample: dict,
    *,
    data_root: str,
    output_dir: str,
    completion_cfg: CompletionConfig,
    planner_cfg: FastPlannerTeacherConfig,
    planner_config_hash: str,
    completion_config_hash: str,
) -> dict:
    skill = str(sample["skill"])
    scene_id = str(sample["scene_id"])
    sample_dir = os.path.join(data_root, sample.get("relative_path", f"{skill}/{scene_id}"))
    started = time.perf_counter()
    expert = _load_expert_positions(sample_dir)
    expert_labels = _completion_labels(sample_dir, data_root, expert, completion_cfg)
    expert_prefix, expert_completion_index = truncate_at_first_completion(
        expert, expert_labels["completion"]
    )
    endpoint = expert_prefix[-1]
    semantic_indices = np.flatnonzero(expert_labels["semantic_done"] > 0.5)
    if not len(semantic_indices):
        raise ValueError("expert trajectory never reaches semantic completion")
    semantic_guide_index = int(semantic_indices[0])
    semantic_guide = expert[semantic_guide_index]
    hard_route = [expert[0], semantic_guide, endpoint]
    conservative_radius = max(float(planner_cfg.collision_radius) + 0.03, 0.18)
    conservative_cfg = dataclasses.replace(
        planner_cfg,
        collision_radius=conservative_radius,
        preferred_clearance=max(0.5, conservative_radius),
    )
    candidates = [
        ("default", planner_cfg, None),
        ("hard_semantic_guide", planner_cfg, hard_route),
        ("conservative_clearance", conservative_cfg, None),
        ("conservative_clearance_hard_semantic_guide", conservative_cfg, hard_route),
    ]
    attempt_errors = []
    for attempt_name, effective_cfg, route_override in candidates:
        try:
            planned, planner_diagnostics = _plan_to_endpoint(
                sample_dir,
                data_root,
                endpoint,
                effective_cfg,
                route_override=route_override,
            )
            planned_labels = _completion_labels(
                sample_dir, data_root, planned, completion_cfg
            )
            minimum_clearance = validate_collision_clearance(
                planned_labels, planner_cfg.collision_radius
            )
            reference, planner_completion_index = truncate_at_first_completion(
                planned, planned_labels["completion"]
            )
        except (RuntimeError, ValueError, TimeoutError) as exc:
            attempt_errors.append({"attempt": attempt_name, "error": repr(exc)})
            continue
        planner_diagnostics["selected_attempt"] = attempt_name
        planner_diagnostics["attempt_errors"] = attempt_errors
        planner_diagnostics["effective_planner_config"] = dataclasses.asdict(
            effective_cfg
        )
        if route_override is not None:
            planner_diagnostics["semantic_guide_fallback"] = {
                "source": "expert_first_semantic_done",
                "expert_index": semantic_guide_index,
                "point": semantic_guide.astype(float).tolist(),
            }
        break
    else:
        details = "; ".join(
            f"{item['attempt']}: {item['error']}" for item in attempt_errors
        )
        raise RuntimeError(f"all Fast-Planner reference attempts failed: {details}")
    length = path_length_3d(reference)
    metadata = {
        "schema_version": 1,
        "status": "valid",
        "batch_id": int(sample.get("batch_id", 0)),
        "skill": skill,
        "scene_id": scene_id,
        "category": sample.get("category"),
        "sample_relative_path": sample.get("relative_path", f"{skill}/{scene_id}"),
        "reference_endpoint_source": "expert_first_completion",
        "reference_endpoint": endpoint.astype(float).tolist(),
        "expert_completion_index": int(expert_completion_index),
        "planner_completion_index": int(planner_completion_index),
        "reference_num_points": int(len(reference)),
        "reference_path_length_m": float(length),
        "planner_success": True,
        "completion_validated": True,
        "collision_radius_m": float(planner_cfg.collision_radius),
        "minimum_authoritative_clearance_m": float(minimum_clearance),
        "planner_config": dataclasses.asdict(planner_cfg),
        "planner_config_hash": planner_config_hash,
        "completion_config": dataclasses.asdict(completion_cfg),
        "completion_config_hash": completion_config_hash,
        "planner_diagnostics": planner_diagnostics,
        "elapsed_s": float(time.perf_counter() - started),
    }
    output_path = save_reference(output_dir, metadata, reference)
    metadata["output_relative_path"] = os.path.relpath(output_path, output_dir)
    _atomic_json(os.path.join(output_path, "metadata.json"), metadata)
    return metadata


def _save_failure(output_dir: str, sample: dict, error: BaseException) -> dict:
    metadata = {
        "schema_version": 1,
        "status": "reference_invalid",
        "batch_id": int(sample.get("batch_id", 0)),
        "skill": str(sample["skill"]),
        "scene_id": str(sample["scene_id"]),
        "category": sample.get("category"),
        "sample_relative_path": sample.get("relative_path"),
        "planner_success": False,
        "completion_validated": False,
        "error": repr(error),
        "traceback": traceback.format_exc(),
    }
    sample_output = os.path.join(output_dir, metadata["skill"], metadata["scene_id"])
    _atomic_json(os.path.join(sample_output, "metadata.json"), metadata)
    return metadata


def _existing_metadata(output_dir: str, sample: dict) -> Optional[dict]:
    path = os.path.join(
        output_dir, sample["skill"], sample["scene_id"], "metadata.json"
    )
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Fast-Planner semantic reference trajectories for SPL."
    )
    parser.add_argument(
        "--manifest-index", default="eval_manifests/manifest_index.json"
    )
    parser.add_argument("--output-dir", default="eval_reference_paths")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--collision-radius", type=float, default=0.15)
    parser.add_argument("--planner-timeout", type=float, default=10.0)
    parser.add_argument("--planner-max-iter", type=int, default=2000)
    parser.add_argument("--optimization-max-iter", type=int, default=80)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-invalid", action="store_true")
    parser.add_argument("--fail-on-invalid", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root, samples, manifest_index = load_manifest_index(args.manifest_index)
    if args.scene_id:
        requested = {value.strip() for value in args.scene_id.split(",") if value.strip()}
        samples = [sample for sample in samples if sample["scene_id"] in requested]
    if args.limit is not None:
        samples = samples[: max(int(args.limit), 0)]

    cfg = load_config(args.config)
    completion_cfg = cfg.completion
    planner_cfg = FastPlannerTeacherConfig(
        collision_radius=float(args.collision_radius),
        preferred_clearance=None,
        replan_interval=1,
        timeout_s=float(args.planner_timeout),
        max_iter=int(args.planner_max_iter),
        optimization_max_iter=int(args.optimization_max_iter),
        optimize=True,
    )
    completion_hash = _config_hash(dataclasses.asdict(completion_cfg))
    planner_hash = _config_hash(dataclasses.asdict(planner_cfg))
    os.makedirs(args.output_dir, exist_ok=True)

    records: List[dict] = []
    for index, sample in enumerate(samples, start=1):
        existing = _existing_metadata(args.output_dir, sample)
        if existing is not None and not args.overwrite:
            if existing.get("status") == "valid" or not args.retry_invalid:
                records.append(existing)
                print(
                    f"[{index}/{len(samples)}] skip {sample['skill']}/{sample['scene_id']} "
                    f"status={existing.get('status')}"
                )
                continue
        print(f"[{index}/{len(samples)}] plan {sample['skill']}/{sample['scene_id']}")
        try:
            record = build_reference_for_sample(
                sample,
                data_root=data_root,
                output_dir=args.output_dir,
                completion_cfg=completion_cfg,
                planner_cfg=planner_cfg,
                planner_config_hash=planner_hash,
                completion_config_hash=completion_hash,
            )
            print(
                f"  valid length={record['reference_path_length_m']:.3f}m "
                f"points={record['reference_num_points']} "
                f"time={record['elapsed_s']:.3f}s"
            )
        except Exception as exc:
            record = _save_failure(args.output_dir, sample, exc)
            print(f"  reference_invalid: {exc}")
        records.append(record)
        _atomic_json(
            os.path.join(args.output_dir, "progress.json"),
            {
                "processed": len(records),
                "requested": len(samples),
                "status_counts": dict(Counter(r["status"] for r in records)),
            },
        )

    reference_index = {
        "schema_version": 1,
        "source_manifest_index": os.path.abspath(args.manifest_index),
        "source_manifest_sha256": _sha256(args.manifest_index),
        "source_total_samples": manifest_index.get("total_samples"),
        "processed_samples": len(records),
        "status_counts": dict(Counter(record["status"] for record in records)),
        "planner_config": dataclasses.asdict(planner_cfg),
        "planner_config_hash": planner_hash,
        "completion_config": dataclasses.asdict(completion_cfg),
        "completion_config_hash": completion_hash,
        "records": records,
    }
    _atomic_json(os.path.join(args.output_dir, "reference_index.json"), reference_index)
    print(f"summary: {reference_index['status_counts']}")
    if args.fail_on_invalid and reference_index["status_counts"].get(
        "reference_invalid", 0
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
