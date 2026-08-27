#!/usr/bin/env python
"""Evaluate BC and/or DAgger checkpoints on one frozen rollout batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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


def _validate_or_write_run_config(path: str, config: dict) -> None:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        for key in ("manifest_sha256", "checkpoint_sha256", "model"):
            if existing.get(key) != config.get(key):
                raise RuntimeError(
                    f"existing run_config mismatch for {key}: "
                    f"{existing.get(key)!r} != {config.get(key)!r}"
                )
        return
    _atomic_json(path, config)


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
    _validate_or_write_run_config(os.path.join(output_dir, "run_config.json"), run_config)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BC and/or DAgger closed-loop rollout on one frozen batch."
    )
    parser.add_argument("--manifest", default="eval_manifests/batch_1.json")
    parser.add_argument("--output-root", default="exp")
    parser.add_argument("--model", choices=("bc", "dagger", "both"), default="both")
    parser.add_argument("--bc-checkpoint", default=DEFAULT_BC_CHECKPOINT)
    parser.add_argument("--dagger-checkpoint", default=DEFAULT_DAGGER_CHECKPOINT)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--waypoint-index", type=int, default=2)
    parser.add_argument("--turn-rate", type=float, default=0.45)
    parser.add_argument("--drone-radius", type=float, default=0.15)
    parser.add_argument("--gate-correction-start-m", type=float, default=1.2)
    parser.add_argument("--gate-correction-blend", type=float, default=0.25)
    parser.add_argument("--retry-infrastructure-errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = os.path.abspath(args.manifest)
    manifest = load_manifest(manifest_path)
    data_root = os.path.abspath(args.data_root or manifest["data_root"])
    batch_name = f"batch_{manifest.get('batch_id')}"
    batch_output = os.path.join(os.path.abspath(args.output_root), batch_name)
    selected = ("bc", "dagger") if args.model == "both" else (args.model,)
    checkpoints = {"bc": args.bc_checkpoint, "dagger": args.dagger_checkpoint}
    for model_name in selected:
        model_output = os.path.join(batch_output, model_name)
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
