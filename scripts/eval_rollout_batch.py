#!/usr/bin/env python
"""Evaluate one supported baseline on a frozen AeroVerbs rollout batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from collections import Counter
from types import SimpleNamespace
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from rollout import _load_model, _run_one_sample


DEFAULT_BC_CHECKPOINT = "/root/autodl-tmp/outputs/bc_baseline/20260826-074509/final.pt"
DEFAULT_DAGGER_CHECKPOINT = "outputs/dagger_next_v2/round2/20260826-205424/final.pt"
DEFAULT_OPENVLA_SERVER_URL = "http://127.0.0.1:5007/predict"
MODEL_CHOICES = ("bc", "bc_dagger", "openvla", "aerialclaw", "worldvln")
UNIMPLEMENTED_MODELS = {"worldvln"}


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: str, payload: Mapping) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def load_manifest(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("manifest samples must be a non-empty list")
    expected = manifest.get("num_samples")
    if expected is not None and int(expected) != len(samples):
        raise ValueError(
            f"manifest expected {expected} samples but contains {len(samples)}"
        )
    seen = set()
    for sample in samples:
        for key in ("skill", "scene_id", "relative_path"):
            if not sample.get(key):
                raise ValueError(f"manifest sample missing {key}: {sample}")
        identifier = (str(sample["skill"]), str(sample["scene_id"]))
        if identifier in seen:
            raise ValueError(f"duplicate manifest sample: {identifier}")
        seen.add(identifier)
    return manifest


def load_latest_results(path: str) -> Dict[Tuple[str, str], dict]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    lines = raw.splitlines()
    latest: Dict[Tuple[str, str], dict] = {}
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            is_unterminated_tail = index == len(lines) - 1 and not raw.endswith("\n")
            if is_unterminated_tail:
                continue
            raise ValueError(f"invalid JSONL record at line {index + 1}: {path}")
        key = (str(record["skill"]), str(record["scene_id"]))
        latest[key] = record
    return latest


def repair_truncated_jsonl(path: str) -> bool:
    """Remove only an invalid unterminated tail before appending new records."""
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        raw = f.read()
    if not raw or raw.endswith(b"\n"):
        return False
    last_newline = raw.rfind(b"\n")
    prefix = raw[: last_newline + 1] if last_newline >= 0 else b""
    tail = raw[last_newline + 1 :]
    try:
        json.loads(tail.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        repaired = prefix
    else:
        repaired = raw + b"\n"
    with open(path, "wb") as f:
        f.write(repaired)
        f.flush()
        os.fsync(f.fileno())
    return True


def pending_samples(
    samples: Sequence[dict],
    latest_results: Mapping[Tuple[str, str], dict],
    *,
    retry_infrastructure_errors: bool = False,
) -> List[dict]:
    pending = []
    for sample in samples:
        key = (str(sample["skill"]), str(sample["scene_id"]))
        existing = latest_results.get(key)
        if existing is None:
            pending.append(sample)
        elif retry_infrastructure_errors and existing.get("status") == "infrastructure_error":
            pending.append(sample)
    return pending


def summarize_results(manifest: Mapping, latest_results: Mapping[Tuple[str, str], dict]) -> dict:
    samples = list(manifest["samples"])
    expected_by_skill = Counter(str(sample["skill"]) for sample in samples)
    selected = []
    for sample in samples:
        key = (str(sample["skill"]), str(sample["scene_id"]))
        if key in latest_results:
            selected.append(latest_results[key])
    status_counts = Counter(str(record.get("status", "unknown")) for record in selected)
    success = int(status_counts.get("success", 0))
    expected = len(samples)
    processed = len(selected)
    per_skill = {}
    for skill in expected_by_skill:
        records = [record for record in selected if record.get("skill") == skill]
        counts = Counter(str(record.get("status", "unknown")) for record in records)
        skill_success = int(counts.get("success", 0))
        skill_expected = int(expected_by_skill[skill])
        per_skill[skill] = {
            "expected": skill_expected,
            "processed": len(records),
            "missing": skill_expected - len(records),
            "success": skill_success,
            "success_rate": skill_success / skill_expected,
            "status_counts": dict(sorted(counts.items())),
        }
    return {
        "batch_id": manifest.get("batch_id"),
        "expected_samples": expected,
        "processed_samples": processed,
        "missing_samples": expected - processed,
        "complete": processed == expected,
        "success_count": success,
        "success_rate": success / expected,
        "success_rate_processed": success / processed if processed else 0.0,
        "status_counts": dict(sorted(status_counts.items())),
        "per_skill": per_skill,
        "rollout_elapsed_s": float(
            sum(float(record.get("elapsed_s", 0.0)) for record in selected)
        ),
    }


def build_rollout_namespace(
    *,
    data_root: str,
    output_dir: str,
    max_steps: int = 200,
    step_size: float = 0.1,
    waypoint_index: int = 2,
    turn_rate: float = 0.45,
    drone_radius: float = 0.15,
    gate_correction_start_m: float = 1.2,
    gate_correction_blend: float = 0.25,
) -> SimpleNamespace:
    return SimpleNamespace(
        data_root=data_root,
        output_dir=output_dir,
        max_steps=int(max_steps),
        step_size=float(step_size),
        waypoint_index=int(waypoint_index),
        turn_rate=float(turn_rate),
        drone_radius=float(drone_radius),
        no_collision_check=False,
        goal_radius=1.0,
        gate_correction_start_m=float(gate_correction_start_m),
        gate_correction_blend=float(gate_correction_blend),
        render_voxel_size=0.2,
        render_voxel_stride=2,
        render_region_stride=2,
        no_render=True,
        seed=0,
    )


def build_openvla_rollout_namespace(
    *,
    data_root: str,
    output_dir: str,
    server_url: str,
    model_id: str,
    max_steps: int = 200,
    step_size: float = 0.1,
    turn_rate: float = 0.45,
    drone_radius: float = 0.15,
    minimum_target_distance_m: float = 1e-4,
    request_timeout_s: float = 300.0,
    rgb_width: int = 640,
    rgb_height: int = 480,
    rgb_gpu_device_id: int = -1,
    show_step_progress: bool = False,
    render_trajectory: bool = False,
    render_voxel_size: float = 0.2,
    render_voxel_stride: int = 2,
    render_region_stride: int = 2,
    render_region: bool = False,
    render_gate: bool = False,
    render_ground: bool = False,
) -> SimpleNamespace:
    """Build the argument namespace consumed by OpenVLA's single-scene runner."""

    return SimpleNamespace(
        data_root=data_root,
        output_dir=output_dir,
        server_url=server_url,
        model_id=str(model_id),
        instruction=None,
        max_steps=int(max_steps),
        step_size=float(step_size),
        turn_rate=float(turn_rate),
        drone_radius=float(drone_radius),
        minimum_target_distance_m=float(minimum_target_distance_m),
        request_timeout_s=float(request_timeout_s),
        rgb_width=int(rgb_width),
        rgb_height=int(rgb_height),
        rgb_gpu_device_id=int(rgb_gpu_device_id),
        show_step_progress=bool(show_step_progress),
        no_collision_check=False,
        render_trajectory=bool(render_trajectory),
        render_voxel_size=float(render_voxel_size),
        render_voxel_stride=int(render_voxel_stride),
        render_region_stride=int(render_region_stride),
        render_region=bool(render_region),
        render_gate=bool(render_gate),
        render_ground=bool(render_ground),
    )


def build_aerialclaw_rollout_namespace(
    *,
    data_root: str,
    output_dir: str,
    max_steps: int = 200,
    step_size: float = 0.1,
    turn_rate: float = 0.45,
    drone_radius: float = 0.15,
    rgb_width: int = 640,
    rgb_height: int = 480,
    rgb_gpu_device_id: int = -1,
    aerialclaw_root: str = "/home/zxt/AerialClaw",
    aerialclaw_python: str = "/home/zxt/miniconda3/envs/aerialclaw-eval/bin/python",
    agent_max_iterations: int = 40,
    request_timeout_s: float = 15.0,
    subprocess_timeout_s: float = 1800.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        data_root=data_root,
        output_dir=output_dir,
        instruction=None,
        max_steps=int(max_steps),
        step_size=float(step_size),
        turn_rate=float(turn_rate),
        drone_radius=float(drone_radius),
        rgb_width=int(rgb_width),
        rgb_height=int(rgb_height),
        rgb_gpu_device_id=int(rgb_gpu_device_id),
        aerialclaw_root=os.path.abspath(aerialclaw_root),
        aerialclaw_python=os.path.abspath(aerialclaw_python),
        agent_max_iterations=int(agent_max_iterations),
        request_timeout_s=float(request_timeout_s),
        subprocess_timeout_s=float(subprocess_timeout_s),
    )


def ensure_model_implemented(model_name: str) -> None:
    if model_name in UNIMPLEMENTED_MODELS:
        raise NotImplementedError(
            f"batch evaluation for {model_name!r} is reserved but not implemented"
        )


def _validate_or_write_run_config(path: str, config: dict) -> None:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        identity_keys = [
            "manifest_sha256",
            "checkpoint_sha256",
            "model",
            "data_root",
            "expected_samples",
            "rollout",
        ]
        if config.get("model") == "openvla":
            identity_keys.extend(("server_url", "model_id"))
        for key in identity_keys:
            if existing.get(key) != config.get(key):
                raise RuntimeError(
                    f"existing run_config mismatch for {key}: "
                    f"{existing.get(key)!r} != {config.get(key)!r}"
                )
        return
    _atomic_json(path, config)


def _require_run_config_for_existing_results(config_path: str, results_path: str) -> None:
    if (
        os.path.isfile(results_path)
        and os.path.getsize(results_path) > 0
        and not os.path.isfile(config_path)
    ):
        raise RuntimeError(
            f"existing results require their original run_config: {config_path}"
        )


def run_model_batch(
    *,
    model_name: str,
    checkpoint: str,
    manifest_path: str,
    manifest: dict,
    output_dir: str,
    rollout_args: SimpleNamespace,
    retry_infrastructure_errors: bool,
) -> dict:
    checkpoint = os.path.abspath(checkpoint)
    manifest_path = os.path.abspath(manifest_path)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    os.makedirs(os.path.join(output_dir, "trajectories"), exist_ok=True)
    results_path = os.path.join(output_dir, "results.jsonl")
    repair_truncated_jsonl(results_path)
    latest = load_latest_results(results_path)
    pending = pending_samples(
        manifest["samples"],
        latest,
        retry_infrastructure_errors=retry_infrastructure_errors,
    )
    run_config = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname(),
        "model": model_name,
        "batch_id": manifest.get("batch_id"),
        "manifest": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "checkpoint": checkpoint,
        "checkpoint_sha256": _sha256(checkpoint),
        "data_root": rollout_args.data_root,
        "expected_samples": len(manifest["samples"]),
        "rollout": {
            "max_steps": rollout_args.max_steps,
            "step_size": rollout_args.step_size,
            "waypoint_index": rollout_args.waypoint_index,
            "turn_rate": rollout_args.turn_rate,
            "drone_radius": rollout_args.drone_radius,
            "gate_correction_start_m": rollout_args.gate_correction_start_m,
            "gate_correction_blend": rollout_args.gate_correction_blend,
            "collision_check": True,
            "render": False,
        },
    }
    config_path = os.path.join(output_dir, "run_config.json")
    _require_run_config_for_existing_results(config_path, results_path)
    _validate_or_write_run_config(config_path, run_config)

    if pending:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{model_name}] device={device} checkpoint={checkpoint}")
        model, cfg = _load_model(checkpoint, device)
        sample_order = {
            (str(sample["skill"]), str(sample["scene_id"])): index
            for index, sample in enumerate(manifest["samples"], start=1)
        }
        with open(results_path, "a", encoding="utf-8") as f:
            for pending_index, sample in enumerate(pending, start=1):
                key = (str(sample["skill"]), str(sample["scene_id"]))
                sample_dir = os.path.join(rollout_args.data_root, sample["relative_path"])
                print(
                    f"[{model_name} {pending_index}/{len(pending)}] "
                    f"{sample['skill']}/{sample['scene_id']}"
                )
                try:
                    result = _run_one_sample(sample_dir, model, cfg, device, rollout_args)
                except Exception as exc:
                    result = {
                        "skill": sample["skill"],
                        "scene_id": sample["scene_id"],
                        "status": "infrastructure_error",
                        "reason": repr(exc),
                    }
                    print(f"  infrastructure_error: {exc!r}")
                result.update(
                    {
                        "skill": sample["skill"],
                        "scene_id": sample["scene_id"],
                        "model": model_name,
                        "batch_id": manifest.get("batch_id"),
                        "manifest_sample_index": sample_order[key],
                        "category": sample.get("category"),
                    }
                )
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
                latest[key] = result
                summary = summarize_results(manifest, latest)
                summary.update(
                    {
                        "model": model_name,
                        "checkpoint": checkpoint,
                        "checkpoint_sha256": run_config["checkpoint_sha256"],
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    }
                )
                _atomic_json(os.path.join(output_dir, "summary.json"), summary)
                print(
                    f"  status={result.get('status')} "
                    f"success={summary['success_count']}/{summary['processed_samples']}"
                )

    summary = summarize_results(manifest, latest)
    summary.update(
        {
            "model": model_name,
            "checkpoint": checkpoint,
            "checkpoint_sha256": run_config["checkpoint_sha256"],
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    _atomic_json(os.path.join(output_dir, "summary.json"), summary)
    return summary


def run_openvla_batch(
    *,
    manifest_path: str,
    manifest: dict,
    output_dir: str,
    rollout_args: SimpleNamespace,
    retry_infrastructure_errors: bool,
    sample_runner: Optional[Callable[[str, SimpleNamespace], dict]] = None,
) -> dict:
    """Evaluate the remote OpenVLA service with resumable per-scene writes."""

    if sample_runner is None:
        from rollout_openvla import run_one_sample as sample_runner

    manifest_path = os.path.abspath(manifest_path)
    os.makedirs(os.path.join(output_dir, "trajectories"), exist_ok=True)
    results_path = os.path.join(output_dir, "results.jsonl")
    repair_truncated_jsonl(results_path)
    latest = load_latest_results(results_path)
    pending = pending_samples(
        manifest["samples"],
        latest,
        retry_infrastructure_errors=retry_infrastructure_errors,
    )
    run_config = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname(),
        "model": "openvla",
        "batch_id": manifest.get("batch_id"),
        "manifest": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "checkpoint": None,
        "checkpoint_sha256": None,
        "server_url": rollout_args.server_url,
        "model_id": rollout_args.model_id,
        "data_root": rollout_args.data_root,
        "expected_samples": len(manifest["samples"]),
        "rollout": {
            "max_steps": rollout_args.max_steps,
            "step_size": rollout_args.step_size,
            "turn_rate": rollout_args.turn_rate,
            "drone_radius": rollout_args.drone_radius,
            "minimum_target_distance_m": rollout_args.minimum_target_distance_m,
            "request_timeout_s": rollout_args.request_timeout_s,
            "rgb_width": rollout_args.rgb_width,
            "rgb_height": rollout_args.rgb_height,
            "rgb_gpu_device_id": rollout_args.rgb_gpu_device_id,
            "collision_check": True,
            "render": rollout_args.render_trajectory,
            "render_voxel_size": rollout_args.render_voxel_size,
            "render_voxel_stride": rollout_args.render_voxel_stride,
            "render_region_stride": rollout_args.render_region_stride,
            "render_region": rollout_args.render_region,
            "render_gate": rollout_args.render_gate,
            "render_ground": rollout_args.render_ground,
            "uses_gate_correction": False,
        },
    }
    config_path = os.path.join(output_dir, "run_config.json")
    _require_run_config_for_existing_results(config_path, results_path)
    _validate_or_write_run_config(config_path, run_config)

    sample_order = {
        (str(sample["skill"]), str(sample["scene_id"])): index
        for index, sample in enumerate(manifest["samples"], start=1)
    }
    if pending:
        with open(results_path, "a", encoding="utf-8") as f:
            for pending_index, sample in enumerate(pending, start=1):
                key = (str(sample["skill"]), str(sample["scene_id"]))
                sample_dir = os.path.join(rollout_args.data_root, sample["relative_path"])
                print(
                    f"[openvla {pending_index}/{len(pending)}] "
                    f"{sample['skill']}/{sample['scene_id']}"
                )
                try:
                    result = sample_runner(sample_dir, rollout_args)
                except Exception as exc:
                    result = {
                        "skill": sample["skill"],
                        "scene_id": sample["scene_id"],
                        "status": "infrastructure_error",
                        "reason": repr(exc),
                    }
                    print(f"  infrastructure_error: {exc!r}")
                result.update(
                    {
                        "skill": sample["skill"],
                        "scene_id": sample["scene_id"],
                        "model": "openvla",
                        "batch_id": manifest.get("batch_id"),
                        "manifest_sample_index": sample_order[key],
                        "category": sample.get("category"),
                    }
                )
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
                latest[key] = result
                summary = summarize_results(manifest, latest)
                summary.update(
                    {
                        "model": "openvla",
                        "server_url": rollout_args.server_url,
                        "model_id": rollout_args.model_id,
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    }
                )
                _atomic_json(os.path.join(output_dir, "summary.json"), summary)
                print(
                    f"  status={result.get('status')} "
                    f"success={summary['success_count']}/{summary['processed_samples']}"
                )

    summary = summarize_results(manifest, latest)
    summary.update(
        {
            "model": "openvla",
            "server_url": rollout_args.server_url,
            "model_id": rollout_args.model_id,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    _atomic_json(os.path.join(output_dir, "summary.json"), summary)
    return summary


def run_aerialclaw_batch(
    *,
    manifest_path: str,
    manifest: dict,
    output_dir: str,
    rollout_args: SimpleNamespace,
    retry_infrastructure_errors: bool,
    sample_runner: Optional[Callable[[str, SimpleNamespace], dict]] = None,
) -> dict:
    """Evaluate AerialClaw with resumable, authoritative per-scene writes."""
    if sample_runner is None:
        from rollout_aerialclaw import run_one_sample as sample_runner

    manifest_path = os.path.abspath(manifest_path)
    os.makedirs(os.path.join(output_dir, "trajectories"), exist_ok=True)
    results_path = os.path.join(output_dir, "results.jsonl")
    repair_truncated_jsonl(results_path)
    latest = load_latest_results(results_path)
    pending = pending_samples(
        manifest["samples"],
        latest,
        retry_infrastructure_errors=retry_infrastructure_errors,
    )
    run_config = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname(),
        "model": "aerialclaw",
        "batch_id": manifest.get("batch_id"),
        "manifest": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "checkpoint": None,
        "checkpoint_sha256": None,
        "data_root": rollout_args.data_root,
        "expected_samples": len(manifest["samples"]),
        "rollout": {
            "max_steps": rollout_args.max_steps,
            "step_size": rollout_args.step_size,
            "turn_rate": rollout_args.turn_rate,
            "drone_radius": rollout_args.drone_radius,
            "rgb_width": rollout_args.rgb_width,
            "rgb_height": rollout_args.rgb_height,
            "rgb_gpu_device_id": rollout_args.rgb_gpu_device_id,
            "agent_max_iterations": rollout_args.agent_max_iterations,
            "request_timeout_s": rollout_args.request_timeout_s,
            "subprocess_timeout_s": rollout_args.subprocess_timeout_s,
            "aerialclaw_root": rollout_args.aerialclaw_root,
            "aerialclaw_python": rollout_args.aerialclaw_python,
            "default_flight_speed_mps": 1.0,
            "minimum_flight_speed_mps": 0.1,
            "minimum_safe_distance_m": 0.30,
            "collision_check": True,
            "success_authority": "CompletionJudge",
            "uses_gate_correction": False,
        },
    }
    config_path = os.path.join(output_dir, "run_config.json")
    _require_run_config_for_existing_results(config_path, results_path)
    _validate_or_write_run_config(config_path, run_config)

    sample_order = {
        (str(sample["skill"]), str(sample["scene_id"])): index
        for index, sample in enumerate(manifest["samples"], start=1)
    }
    if pending:
        with open(results_path, "a", encoding="utf-8") as stream:
            for pending_index, sample in enumerate(pending, start=1):
                key = (str(sample["skill"]), str(sample["scene_id"]))
                sample_dir = os.path.join(rollout_args.data_root, sample["relative_path"])
                print(
                    f"[aerialclaw {pending_index}/{len(pending)}] "
                    f"{sample['skill']}/{sample['scene_id']}"
                )
                try:
                    result = sample_runner(sample_dir, rollout_args)
                except Exception as exc:
                    result = {
                        "skill": sample["skill"],
                        "scene_id": sample["scene_id"],
                        "status": "infrastructure_error",
                        "reason": repr(exc),
                    }
                    print(f"  infrastructure_error: {exc!r}")
                result.update(
                    {
                        "skill": sample["skill"],
                        "scene_id": sample["scene_id"],
                        "model": "aerialclaw",
                        "batch_id": manifest.get("batch_id"),
                        "manifest_sample_index": sample_order[key],
                        "category": sample.get("category"),
                    }
                )
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                latest[key] = result
                summary = summarize_results(manifest, latest)
                summary.update(
                    {
                        "model": "aerialclaw",
                        "aerialclaw_root": rollout_args.aerialclaw_root,
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    }
                )
                _atomic_json(os.path.join(output_dir, "summary.json"), summary)

    summary = summarize_results(manifest, latest)
    summary.update(
        {
            "model": "aerialclaw",
            "aerialclaw_root": rollout_args.aerialclaw_root,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    _atomic_json(os.path.join(output_dir, "summary.json"), summary)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one baseline's closed-loop rollout on one frozen batch."
    )
    parser.add_argument("--manifest", default="eval_manifests/batch_1.json")
    parser.add_argument("--output-root", default="exp/rollouts")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="bc")
    parser.add_argument("--bc-checkpoint", default=DEFAULT_BC_CHECKPOINT)
    parser.add_argument(
        "--bc-dagger-checkpoint",
        "--dagger-checkpoint",
        dest="bc_dagger_checkpoint",
        default=DEFAULT_DAGGER_CHECKPOINT,
    )
    parser.add_argument("--openvla-server-url", default=DEFAULT_OPENVLA_SERVER_URL)
    parser.add_argument(
        "--openvla-model-id",
        default=None,
        help="Stable identifier for the checkpoint served at --openvla-server-url.",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--aerialclaw-root", default="/home/zxt/AerialClaw")
    parser.add_argument(
        "--aerialclaw-python",
        default="/home/zxt/miniconda3/envs/aerialclaw-eval/bin/python",
    )
    parser.add_argument("--agent-max-iterations", type=int, default=40)
    parser.add_argument("--subprocess-timeout-s", type=float, default=1800.0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--waypoint-index", type=int, default=2)
    parser.add_argument("--turn-rate", type=float, default=0.45)
    parser.add_argument("--drone-radius", type=float, default=0.15)
    parser.add_argument("--gate-correction-start-m", type=float, default=1.2)
    parser.add_argument("--gate-correction-blend", type=float, default=0.25)
    parser.add_argument("--minimum-target-distance-m", type=float, default=1e-4)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--rgb-width", type=int, default=640)
    parser.add_argument("--rgb-height", type=int, default=480)
    parser.add_argument("--rgb-gpu-device-id", type=int, default=-1)
    parser.add_argument(
        "--show-step-progress",
        action="store_true",
        help="Print one progress line after each completed OpenVLA inference step.",
    )
    parser.add_argument("--render-trajectory", action="store_true")
    parser.add_argument("--render-voxel-size", type=float, default=0.2)
    parser.add_argument("--render-voxel-stride", type=int, default=2)
    parser.add_argument("--render-region-stride", type=int, default=2)
    parser.add_argument("--render-region", action="store_true")
    parser.add_argument("--render-gate", action="store_true")
    parser.add_argument("--render-ground", action="store_true")
    parser.add_argument("--retry-infrastructure-errors", action="store_true")
    args = parser.parse_args(argv)
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.step_size <= 0.0:
        parser.error("--step-size must be positive")
    if args.turn_rate < 0.0:
        parser.error("--turn-rate must be non-negative")
    if args.drone_radius <= 0.0:
        parser.error("--drone-radius must be positive")
    if args.minimum_target_distance_m < 0.0:
        parser.error("--minimum-target-distance-m must be non-negative")
    if args.request_timeout_s <= 0.0:
        parser.error("--request-timeout-s must be positive")
    if args.agent_max_iterations <= 0:
        parser.error("--agent-max-iterations must be positive")
    if args.subprocess_timeout_s <= 0.0:
        parser.error("--subprocess-timeout-s must be positive")
    if args.rgb_width <= 0 or args.rgb_height <= 0:
        parser.error("--rgb-width and --rgb-height must be positive")
    if args.render_voxel_size <= 0.0:
        parser.error("--render-voxel-size must be positive")
    if args.render_voxel_stride <= 0 or args.render_region_stride <= 0:
        parser.error("render strides must be positive")
    if args.model == "openvla":
        args.openvla_model_id = str(args.openvla_model_id or "").strip()
        if not args.openvla_model_id:
            parser.error("--openvla-model-id is required when --model openvla")
        if not args.openvla_server_url.startswith(("http://", "https://")):
            parser.error("--openvla-server-url must start with http:// or https://")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    ensure_model_implemented(args.model)
    manifest_path = os.path.abspath(args.manifest)
    manifest = load_manifest(manifest_path)
    data_root = os.path.abspath(args.data_root or manifest["data_root"])
    batch_name = f"batch_{manifest.get('batch_id')}"
    batch_output = os.path.join(os.path.abspath(args.output_root), batch_name)
    model_name = args.model
    model_output = os.path.join(batch_output, model_name)

    if model_name == "openvla":
        rollout_args = build_openvla_rollout_namespace(
            data_root=data_root,
            output_dir=model_output,
            server_url=args.openvla_server_url,
            model_id=args.openvla_model_id,
            max_steps=args.max_steps,
            step_size=args.step_size,
            turn_rate=args.turn_rate,
            drone_radius=args.drone_radius,
            minimum_target_distance_m=args.minimum_target_distance_m,
            request_timeout_s=args.request_timeout_s,
            rgb_width=args.rgb_width,
            rgb_height=args.rgb_height,
            rgb_gpu_device_id=args.rgb_gpu_device_id,
            show_step_progress=args.show_step_progress,
            render_trajectory=args.render_trajectory,
            render_voxel_size=args.render_voxel_size,
            render_voxel_stride=args.render_voxel_stride,
            render_region_stride=args.render_region_stride,
            render_region=args.render_region,
            render_gate=args.render_gate,
            render_ground=args.render_ground,
        )
        summary = run_openvla_batch(
            manifest_path=manifest_path,
            manifest=manifest,
            output_dir=model_output,
            rollout_args=rollout_args,
            retry_infrastructure_errors=args.retry_infrastructure_errors,
        )
    elif model_name == "aerialclaw":
        rollout_args = build_aerialclaw_rollout_namespace(
            data_root=data_root,
            output_dir=model_output,
            max_steps=args.max_steps,
            step_size=args.step_size,
            turn_rate=args.turn_rate,
            drone_radius=args.drone_radius,
            rgb_width=args.rgb_width,
            rgb_height=args.rgb_height,
            rgb_gpu_device_id=args.rgb_gpu_device_id,
            aerialclaw_root=args.aerialclaw_root,
            aerialclaw_python=args.aerialclaw_python,
            agent_max_iterations=args.agent_max_iterations,
            request_timeout_s=args.request_timeout_s,
            subprocess_timeout_s=args.subprocess_timeout_s,
        )
        summary = run_aerialclaw_batch(
            manifest_path=manifest_path,
            manifest=manifest,
            output_dir=model_output,
            rollout_args=rollout_args,
            retry_infrastructure_errors=args.retry_infrastructure_errors,
        )
    else:
        checkpoints = {
            "bc": args.bc_checkpoint,
            "bc_dagger": args.bc_dagger_checkpoint,
        }
        rollout_args = build_rollout_namespace(
            data_root=data_root,
            output_dir=model_output,
            max_steps=args.max_steps,
            step_size=args.step_size,
            waypoint_index=args.waypoint_index,
            turn_rate=args.turn_rate,
            drone_radius=args.drone_radius,
            gate_correction_start_m=args.gate_correction_start_m,
            gate_correction_blend=args.gate_correction_blend,
        )
        summary = run_model_batch(
            model_name=model_name,
            checkpoint=checkpoints[model_name],
            manifest_path=manifest_path,
            manifest=manifest,
            output_dir=model_output,
            rollout_args=rollout_args,
            retry_infrastructure_errors=args.retry_infrastructure_errors,
        )
    print(
        f"[{model_name}] success={summary['success_count']}/"
        f"{summary['expected_samples']} ({summary['success_rate']:.1%}) "
        f"complete={summary['complete']}"
    )


if __name__ == "__main__":
    main()
