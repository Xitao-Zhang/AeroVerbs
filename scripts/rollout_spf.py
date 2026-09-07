#!/usr/bin/env python3
"""Run SPF closed-loop on one AeroVerbs RGB scene."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from aeroverbs.baselines.spf import SPFClient
from aeroverbs.env.rgb_renderer import HabitatRGBRenderer, load_sample_frame

INSTRUCTION_FALLBACKS = {
    "around_left": "Fly around the target on its left side.",
    "around_right": "Fly around the target on its right side.",
    "through": "Fly through the opening.",
    "over": "Fly over the target.",
    "under": "Fly under the target.",
}


def _object_name(meta: dict) -> str:
    """Return the semantic object type stored by AeroVerbs collection."""
    objects = meta.get("objects")
    if isinstance(objects, list) and objects:
        obj_type = str(objects[0].get("type", "")).strip().replace("_", " ")
        if obj_type:
            return obj_type
    object_id = str(meta.get("object_id", "")).strip()
    if object_id:
        return object_id.rsplit("_", 1)[0].replace("_", " ")
    return "target"


def _instruction(meta: dict, override: str | None = None) -> str:
    if override:
        return str(override).strip()
    skill = str(meta.get("skill") or "")
    name = _object_name(meta)
    if skill == "around_left":
        return f"Fly around the {name} on its left side."
    if skill == "around_right":
        return f"Fly around the {name} on its right side."
    if skill == "through":
        return f"Fly through the {name} opening."
    if skill == "over":
        return f"Fly over the {name}."
    if skill == "under":
        return f"Fly under the {name}."
    return INSTRUCTION_FALLBACKS[skill]


def horizontal_forward(value: Sequence[float]) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    norm = float(np.hypot(vector[0], vector[1]))
    if vector.shape != (3,) or not np.all(np.isfinite(vector)) or norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return np.array([vector[0] / norm, vector[1] / norm, 0.0], dtype=np.float32)


def apply_spf_action(
    position_std: Sequence[float],
    forward_std: Sequence[float],
    action_aeroverbs: Sequence[float],
    *,
    step_size: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply a fixed-length step in the direction returned by SPF."""
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    position = np.asarray(position_std, dtype=np.float32)
    action = np.asarray(action_aeroverbs, dtype=np.float32)
    if position.shape != (3,) or action.shape != (3,) or not np.all(np.isfinite(action)):
        raise ValueError("position and action must be finite 3-vectors")
    forward = horizontal_forward(forward_std)
    left = np.array([-forward[1], forward[0], 0.0], dtype=np.float32)
    world_delta = action[0] * forward + action[1] * left + np.array([0.0, 0.0, action[2]])
    distance = float(np.linalg.norm(world_delta))
    if distance < 1e-8:
        return position.copy(), forward, distance
    next_position = position + world_delta * (float(step_size) / distance)
    next_forward = horizontal_forward(world_delta) if np.hypot(world_delta[0], world_delta[1]) > 1e-8 else forward
    return next_position.astype(np.float32), next_forward, distance


def _runtime_components():
    from aeroverbs.config import Config
    from aeroverbs.data.completion import CompletionJudge, make_clearance_provider
    from aeroverbs.env.radar import load_scene_mesh, render_scene_glb
    from rollout import CollisionChecker, _load_collision_mesh, _render_trajectory
    return SimpleNamespace(
        Config=Config, CompletionJudge=CompletionJudge,
        make_clearance_provider=make_clearance_provider,
        load_scene_mesh=load_scene_mesh, render_scene_glb=render_scene_glb,
        CollisionChecker=CollisionChecker, load_collision_mesh=_load_collision_mesh,
        render_trajectory=_render_trajectory,
    )


def path_length(points: Sequence[np.ndarray]) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1).sum())


def _print_step_progress(
    *, step: int, total: int, raw_distance: float, executed_distance: float,
    latency: float, width: int = 28,
) -> None:
    """Refresh one in-place progress bar for the current sample."""
    completed = min(width, int(width * step / max(total, 1)))
    bar = "=" * completed + ">" + " " * max(0, width - completed - 1)
    text = (
        f"\r    [{bar}] {step:>3}/{total:<3} "
        f"raw={raw_distance:.3f}m exec={executed_distance:.3f}m "
        f"latency={latency:.2f}s"
    )
    # Clear remnants of a longer previous line before rewriting it.
    sys.stdout.write(text.ljust(120))
    sys.stdout.flush()


def run_one_sample(sample_dir: Path, args: argparse.Namespace) -> dict:
    runtime = _runtime_components()
    meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    skill = str(meta.get("skill") or sample_dir.parent.name)
    instruction = _instruction(meta, args.instruction)
    frame0 = load_sample_frame(sample_dir, 0)
    position = frame0.position_std.copy()
    forward = horizontal_forward(frame0.forward_std)
    goal = np.asarray(meta["goal_std"], dtype=np.float32)

    radar_scene_path = runtime.render_scene_glb(str(sample_dir))
    radar_mesh = runtime.load_scene_mesh(radar_scene_path)
    collision_mesh = runtime.load_collision_mesh(str(sample_dir), args.data_root, meta)
    collision_checker = runtime.CollisionChecker(collision_mesh, args.drone_radius, meta)
    completion_judge = runtime.CompletionJudge(
        meta, clearance_provider=runtime.make_clearance_provider(str(sample_dir), args.data_root, meta),
        cfg=runtime.Config().completion, mesh=collision_mesh,
    )
    points = [position.copy()]
    records = []
    status, reason = "timeout", ""
    semantic_ok = False
    minimum_clearance = float("inf")
    completion_detail = ""
    collision_point = None
    started = time.time()

    with HabitatRGBRenderer(frame0.scene_path, width=args.rgb_width, height=args.rgb_height,
                            gpu_device_id=args.rgb_gpu_device_id) as renderer, SPFClient(
                                args.spf_url, timeout_s=args.request_timeout_s) as client:
        for step in range(args.max_steps):
            rgb = renderer.render(position, forward)
            prediction = client.predict(rgb, instruction, prompt_mode=args.prompt_mode)
            next_position, next_forward, distance = apply_spf_action(
                position, forward, prediction.action_aeroverbs,
                step_size=args.step_size,
            )
            record = {
                "step": step, "action_spf": prediction.action_spf.tolist(),
                "action_aeroverbs": prediction.action_aeroverbs.tolist(),
                "screen_point": list(prediction.screen_point),
                "raw_distance_m": distance,
                "executed_distance_m": min(distance, args.step_size),
                "latency_s": prediction.latency_s, "position_before": position.tolist(),
                "position_after": next_position.tolist(),
            }
            records.append(record)
            if args.show_step_progress:
                _print_step_progress(
                    step=step + 1,
                    total=args.max_steps,
                    raw_distance=distance,
                    executed_distance=min(distance, args.step_size),
                    latency=prediction.latency_s,
                )
            if distance < args.minimum_action_distance_m:
                status, reason = "stalled", f"action distance {distance:.6f}m below minimum"
                break
            if not args.no_collision_check and collision_checker.collides(next_position):
                collision_point = next_position.copy()
                points.append(collision_point)
                status, reason = "collision", f"collision at step {step + 1}"
                break
            position, forward = next_position, next_forward
            points.append(position.copy())
            result = completion_judge.update(points, position)
            semantic_ok = bool(result.semantic_done)
            minimum_clearance = min(minimum_clearance, float(result.clearance))
            completion_detail = str(result)
            if result.completion:
                status, reason = "success", "completion judge reported completion"
                break

    if args.show_step_progress:
        print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = output_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = trajectory_dir / f"{skill}__{sample_dir.name}.json"
    trajectory_path.write_text(
        json.dumps({
            "points": [p.tolist() for p in points],
            "semantic_label": skill,
            "instruction": instruction,
            "spf_steps": records,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rendered = None
    if args.render_trajectory:
        render_dir = output_dir / "renders"
        render_dir.mkdir(parents=True, exist_ok=True)
        render_path = render_dir / f"{skill}__{sample_dir.name}.png"
        render_args = SimpleNamespace(
            no_render=False,
            render_voxel_size=args.render_voxel_size,
            render_voxel_stride=args.render_voxel_stride,
            render_region_stride=args.render_region_stride,
            render_region=getattr(args, "render_region", False),
            render_gate=getattr(args, "render_gate", False),
            render_ground=getattr(args, "render_ground", False),
        )
        rendered_path = runtime.render_trajectory(
            str(sample_dir), points, skill, str(render_path), radar_mesh, render_args
        )
        if rendered_path is not None:
            rendered = str(Path("renders") / render_path.name)
    return {
        "model": "spf", "execution_mode": "fixed_step_body_displacement",
        "prompt_mode": args.prompt_mode, "skill": skill,
        "scene_id": sample_dir.name, "instruction": instruction, "status": status,
        "reason": reason, "steps": len(points) - 1, "trajectory_points": len(points),
        "actual_path_length_m": path_length(points), "start_position": points[0].tolist(),
        "final_position": position.tolist(), "goal": goal.tolist(),
        "goal_distance": float(np.linalg.norm(position - goal)), "semantic_ok": semantic_ok,
        "completion_detail": completion_detail,
        "minimum_clearance_m": None if not np.isfinite(minimum_clearance) else minimum_clearance,
        "collision_point": None if collision_point is None else collision_point.tolist(),
        "request_count": len(records),
        "mean_request_latency_s": None if not records else float(np.mean([r["latency_s"] for r in records])),
        "max_request_latency_s": None if not records else float(np.max([r["latency_s"] for r in records])),
        "elapsed_s": round(time.time() - started, 3),
        "trajectory": str(Path("trajectories") / trajectory_path.name),
        "render": rendered,
        "gate_fallback_steps": 0,
        "uses_gate_correction": False,
        "inference_records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", choices=tuple(INSTRUCTION_FALLBACKS))
    parser.add_argument("--scene-id")
    parser.add_argument("--manifest", type=Path,
                        help="Evaluation manifest, for example eval_manifests/batch_1.json")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "trajctorys"))
    parser.add_argument("--spf-url", default="http://127.0.0.1:5010")
    parser.add_argument("--instruction")
    parser.add_argument("--prompt-mode", choices=("original", "relation"), default="original")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--minimum-action-distance-m", type=float, default=1e-3)
    parser.add_argument("--step-size", type=float, default=0.2,
                        help="Fixed executed displacement in metres (default: 0.2).")
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--drone-radius", type=float, default=0.15)
    parser.add_argument("--rgb-width", type=int, default=640)
    parser.add_argument("--rgb-height", type=int, default=480)
    parser.add_argument("--rgb-gpu-device-id", type=int, default=-1)
    parser.add_argument("--show-step-progress", action="store_true")
    parser.add_argument("--no-collision-check", action="store_true")
    parser.add_argument("--render-trajectory", action="store_true")
    parser.add_argument("--render-voxel-size", type=float, default=0.2)
    parser.add_argument("--render-voxel-stride", type=int, default=2)
    parser.add_argument("--render-region-stride", type=int, default=2)
    parser.add_argument("--render-region", action="store_true")
    parser.add_argument("--render-gate", action="store_true")
    parser.add_argument("--render-ground", action="store_true")
    args = parser.parse_args()
    args.data_root = str(Path(args.data_root).resolve())
    if args.step_size <= 0.0:
        parser.error("--step-size must be positive")
    if bool(args.manifest) == bool(args.scene_id):
        parser.error("provide exactly one of --manifest or --scene-id")
    if args.manifest:
        manifest_path = args.manifest.resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        samples = manifest.get("samples")
        if not isinstance(samples, list) or not samples:
            parser.error(f"manifest has no samples: {manifest_path}")
        batch_id = manifest.get("batch_id", manifest_path.stem.replace("batch_", ""))
        sample_specs = [(str(item["skill"]), str(item["scene_id"])) for item in samples]
        args.output_dir = args.output_dir or str(PROJECT_ROOT / "exp" / "spf" / f"batch_{batch_id}")
    else:
        if not args.skill:
            parser.error("--skill is required with --scene-id")
        sample_specs = [(args.skill, args.scene_id)]
        args.output_dir = args.output_dir or str(PROJECT_ROOT / "exp" / "spf" / f"batch_manual_{time.strftime('%Y%m%d-%H%M%S')}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)
    print(f"spf_url={args.spf_url}\nprompt_mode={args.prompt_mode}\nstep_size={args.step_size}\noutput={output_dir}", flush=True)
    run_config = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    run_config.update({
        "model": "spf",
        "execution_mode": "fixed_step_body_displacement",
        "batch_manifest": str(args.manifest.resolve()) if args.manifest else None,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "schema_version": 1,
        "rollout": {
            "collision_check": not args.no_collision_check,
            "drone_radius": args.drone_radius,
            "max_steps": args.max_steps,
            "minimum_action_distance_m": args.minimum_action_distance_m,
            "render": False,
            "request_timeout_s": args.request_timeout_s,
            "rgb_width": args.rgb_width,
            "rgb_height": args.rgb_height,
            "rgb_gpu_device_id": args.rgb_gpu_device_id,
            "prompt_mode": args.prompt_mode,
            "direct_returned_distance": False,
            "step_size": args.step_size,
        },
    })
    (output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    results = []
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as stream:
        for index, (skill, scene_id) in enumerate(sample_specs, 1):
            args.skill = skill
            sample_dir = Path(args.data_root) / skill / scene_id
            print(f"[{index}/{len(sample_specs)}] {skill}/{scene_id}", flush=True)
            try:
                if not sample_dir.is_dir():
                    raise FileNotFoundError(sample_dir)
                result = run_one_sample(sample_dir, args)
            except Exception as exc:
                result = {"model": "spf", "skill": skill, "scene_id": scene_id,
                          "status": "infrastructure_error", "reason": repr(exc)}
                print(f"infrastructure_error: {exc!r}", flush=True)
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    status_counts = {}
    for result in results:
        status = result.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "model": "spf",
        "batch_manifest": str(args.manifest.resolve()) if args.manifest else None,
        "num_samples": len(results),
        "status_counts": status_counts,
        "results": "results.jsonl",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
