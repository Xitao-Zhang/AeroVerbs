#!/usr/bin/env python3
"""Run OpenVLA-UAV closed-loop on one AeroVerbs scene.

Each step renders RGB, sends RGB plus the first-frame-local UAV pose to the
remote service, converts its centimetre-scale absolute local target back to
AeroVerbs world metres, and applies the same fixed-step kinematic controller
used by ``scripts/rollout.py``. OpenVLA receives no radar, heatmap, gate
features, or gate correction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from aeroverbs.baselines.openvla import OpenVLAClient
from aeroverbs.env.rgb_renderer import HabitatRGBRenderer, load_sample_frame


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "trajctorys"
INSTRUCTION_TEMPLATES = {
    "around_left": "Fly around the object on its left side.",
    "around_right": "Fly around the object on its right side.",
    "through": "Fly through the opening.",
    "over": "Fly over the object.",
    "under": "Fly under the object.",
}


def _horizontal_forward(value: Sequence[float]) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"forward must be a finite shape-(3,) vector, got {vector}")
    norm = float(np.hypot(vector[0], vector[1]))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return np.array([vector[0] / norm, vector[1] / norm, 0.0], dtype=np.float32)


def _signed_angle_2d(source: np.ndarray, target: np.ndarray) -> float:
    source = _horizontal_forward(source)
    target = _horizontal_forward(target)
    dot = float(np.clip(np.dot(source[:2], target[:2]), -1.0, 1.0))
    cross = float(source[0] * target[1] - source[1] * target[0])
    return float(math.atan2(cross, dot))


@dataclass(frozen=True)
class OpenVLALocalFrame:
    """First-frame local frame used by the released UAV-Flow service.

    Local x is initial forward, local y is initial left, and local z is world
    up. Positions use centimetres. Proprio yaw uses degrees; returned action
    yaw uses radians.
    """

    origin_std: np.ndarray
    initial_forward_std: np.ndarray

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin_std, dtype=np.float32)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError(f"origin_std must be a finite shape-(3,) vector, got {origin}")
        forward = _horizontal_forward(self.initial_forward_std)
        left = np.array([-forward[1], forward[0], 0.0], dtype=np.float32)
        object.__setattr__(self, "origin_std", origin)
        object.__setattr__(self, "initial_forward_std", forward)
        object.__setattr__(self, "_initial_left_std", left)

    def world_to_proprio(
        self,
        position_std: Sequence[float],
        forward_std: Sequence[float],
    ) -> np.ndarray:
        position = np.asarray(position_std, dtype=np.float32)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(f"position_std must be a finite shape-(3,) vector, got {position}")
        delta = position - self.origin_std
        yaw_deg = math.degrees(
            _signed_angle_2d(self.initial_forward_std, np.asarray(forward_std))
        )
        return np.array(
            [
                100.0 * float(np.dot(delta, self.initial_forward_std)),
                100.0 * float(np.dot(delta, self._initial_left_std)),
                100.0 * float(delta[2]),
                yaw_deg,
            ],
            dtype=np.float32,
        )

    def action_to_world_target(
        self,
        absolute_action: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        action = np.asarray(absolute_action, dtype=np.float32)
        if action.shape != (4,) or not np.all(np.isfinite(action)):
            raise ValueError(f"absolute_action must be a finite shape-(4,) vector, got {action}")
        position = (
            self.origin_std
            + (float(action[0]) / 100.0) * self.initial_forward_std
            + (float(action[1]) / 100.0) * self._initial_left_std
            + np.array([0.0, 0.0, float(action[2]) / 100.0], dtype=np.float32)
        )
        yaw = float(action[3])
        forward = (
            math.cos(yaw) * self.initial_forward_std
            + math.sin(yaw) * self._initial_left_std
        )
        return position.astype(np.float32), _horizontal_forward(forward)


def kinematic_step_toward(
    position_std: Sequence[float],
    forward_std: Sequence[float],
    target_position_std: Sequence[float],
    *,
    step_size: float,
    turn_rate: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Advance toward a target using AeroVerbs' fixed-step turn-limited rule."""

    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if turn_rate < 0.0:
        raise ValueError("turn_rate must be non-negative")
    position = np.asarray(position_std, dtype=np.float32)
    target = np.asarray(target_position_std, dtype=np.float32)
    if position.shape != (3,) or target.shape != (3,):
        raise ValueError("position_std and target_position_std must have shape (3,)")
    delta = target - position
    distance = float(np.linalg.norm(delta))
    if distance < 1e-8:
        raise ValueError("OpenVLA target coincides with current position")
    step_direction = delta / distance
    current_forward = _horizontal_forward(forward_std)

    if turn_rate == 0.0:
        next_position = position + step_direction * float(step_size)
        return next_position.astype(np.float32), _horizontal_forward(step_direction)

    current_heading = math.atan2(current_forward[1], current_forward[0])
    if float(np.hypot(step_direction[0], step_direction[1])) < 1e-8:
        target_heading = current_heading
    else:
        target_heading = math.atan2(step_direction[1], step_direction[0])
    heading_delta = (target_heading - current_heading + math.pi) % (2.0 * math.pi) - math.pi
    heading_delta = float(np.clip(heading_delta, -turn_rate, turn_rate))
    new_heading = current_heading + heading_delta
    next_forward = np.array(
        [math.cos(new_heading), math.sin(new_heading), 0.0], dtype=np.float32
    )
    next_position = position + next_forward * float(step_size)
    next_position[2] = position[2] + step_direction[2] * float(step_size)
    return next_position.astype(np.float32), next_forward


def _runtime_components():
    """Load existing evaluation components only for a real rollout."""

    try:
        from aeroverbs.config import Config
        from aeroverbs.data.completion import CompletionJudge, make_clearance_provider
        from aeroverbs.env.radar import load_scene_mesh, render_scene_glb
        from rollout import CollisionChecker, _load_collision_mesh, _render_trajectory
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The local checkout is missing an AeroVerbs evaluation module "
            f"({exc.name}). Restore the server's aeroverbs/data source directory before "
            "running rollout_openvla.py; the current 'data/' ignore rule also matches "
            "aeroverbs/data/."
        ) from exc
    return SimpleNamespace(
        Config=Config,
        CompletionJudge=CompletionJudge,
        make_clearance_provider=make_clearance_provider,
        load_scene_mesh=load_scene_mesh,
        render_scene_glb=render_scene_glb,
        CollisionChecker=CollisionChecker,
        load_collision_mesh=_load_collision_mesh,
        render_trajectory=_render_trajectory,
    )


def _completion_text(result) -> str:
    return (
        f"semantic_done={result.semantic_done}, far_side={result.far_side}, "
        f"clearance={result.clearance:.3f}, safe_clearance={result.safe_clearance}, "
        f"ground_ok={result.ground_ok}, sustained={result.sustained}"
    )


def _path_length(points: Sequence[np.ndarray]) -> float:
    if len(points) < 2:
        return 0.0
    array = np.asarray(points, dtype=np.float32)
    return float(np.linalg.norm(np.diff(array, axis=0), axis=1).sum())


def run_one_sample(sample_dir: Path, args: argparse.Namespace) -> dict:
    runtime = _runtime_components()
    sample_dir = Path(sample_dir)
    meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    skill = str(meta.get("skill") or sample_dir.parent.name)
    if skill not in INSTRUCTION_TEMPLATES:
        raise ValueError(f"unsupported skill: {skill}")
    instruction = args.instruction or INSTRUCTION_TEMPLATES[skill]

    first_frame = load_sample_frame(sample_dir, frame_index=0)
    position = first_frame.position_std.copy()
    forward = _horizontal_forward(first_frame.forward_std)
    start_position = position.copy()
    local_frame = OpenVLALocalFrame(start_position, forward)
    goal = np.asarray(meta["goal_std"], dtype=np.float32)

    radar_scene_path = runtime.render_scene_glb(str(sample_dir))
    radar_mesh = runtime.load_scene_mesh(radar_scene_path)
    collision_mesh = runtime.load_collision_mesh(str(sample_dir), args.data_root, meta)
    collision_checker = runtime.CollisionChecker(collision_mesh, args.drone_radius, meta)
    completion_provider = runtime.make_clearance_provider(str(sample_dir), args.data_root, meta)
    completion_judge = runtime.CompletionJudge(
        meta,
        clearance_provider=completion_provider,
        cfg=runtime.Config().completion,
        mesh=collision_mesh,
    )

    points: List[np.ndarray] = [position.copy()]
    inference_records = []
    collision_point = None
    semantic_ok = False
    completion_detail = ""
    minimum_clearance = float("inf")
    status = "timeout"
    reason = ""
    started = time.time()

    with HabitatRGBRenderer(
        first_frame.scene_path,
        width=args.rgb_width,
        height=args.rgb_height,
        gpu_device_id=args.rgb_gpu_device_id,
    ) as renderer, OpenVLAClient(
        args.server_url,
        timeout_s=args.request_timeout_s,
    ) as client:
        for step in range(args.max_steps):
            rgb = renderer.render(position, forward)
            proprio = local_frame.world_to_proprio(position, forward)
            prediction = client.predict(rgb, proprio, instruction)
            target_position, predicted_forward = local_frame.action_to_world_target(prediction.action)
            target_distance = float(np.linalg.norm(target_position - position))
            inference_records.append(
                {
                    "step": step,
                    "proprio_cm_deg": proprio.tolist(),
                    "action": prediction.action.tolist(),
                    "action_ori": prediction.action_ori.tolist(),
                    "target_position_std": target_position.tolist(),
                    "predicted_forward_std": predicted_forward.tolist(),
                    "target_distance_m": target_distance,
                    "request_latency_s": prediction.request_latency_s,
                }
            )
            if target_distance < args.minimum_target_distance_m:
                reason = (
                    f"OpenVLA target displacement {target_distance:.6f}m is below "
                    f"minimum {args.minimum_target_distance_m:.6f}m at step {step + 1}"
                )
                break

            next_position, next_forward = kinematic_step_toward(
                position,
                forward,
                target_position,
                step_size=args.step_size,
                turn_rate=args.turn_rate,
            )
            if not args.no_collision_check and collision_checker.collides(next_position):
                collision_point = next_position.copy()
                points.append(collision_point)
                status = "collision"
                reason = f"collision at step {step + 1}, point={next_position.tolist()}"
                break

            position = next_position
            forward = next_forward
            points.append(position.copy())

            completion_result = completion_judge.update(points, position)
            semantic_ok = bool(completion_result.semantic_done)
            minimum_clearance = min(minimum_clearance, float(completion_result.clearance))
            completion_detail = _completion_text(completion_result)
            if completion_result.completion:
                status = "success"
                reason = (
                    f"completion_judge: {completion_detail}; "
                    f"goal_distance={np.linalg.norm(position - goal):.3f}"
                )
                break

    elapsed = time.time() - started
    if status == "timeout" and not reason:
        reason = (
            f"max_steps={args.max_steps} reached; "
            f"goal_distance={np.linalg.norm(position - goal):.3f}, "
            f"semantic_ok={semantic_ok} ({completion_detail or 'n/a'})"
        )

    trajectory = {
        "points": [[float(value) for value in point] for point in points],
        "semantic_label": skill,
        "instruction": instruction,
        "openvla_steps": inference_records,
    }
    trajectory_dir = Path(args.output_dir) / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    trajectory_name = f"{skill}__{sample_dir.name}.json"
    trajectory_path = trajectory_dir / trajectory_name
    trajectory_path.write_text(
        json.dumps(trajectory, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rendered = None
    if args.render_trajectory:
        render_dir = Path(args.output_dir) / "renders"
        render_dir.mkdir(parents=True, exist_ok=True)
        render_path = render_dir / f"{skill}__{sample_dir.name}.png"
        render_args = SimpleNamespace(
            no_render=False,
            render_voxel_size=args.render_voxel_size,
            render_voxel_stride=args.render_voxel_stride,
            render_region_stride=args.render_region_stride,
        )
        rendered_path = runtime.render_trajectory(
            str(sample_dir), points, skill, str(render_path), radar_mesh, render_args
        )
        if rendered_path is not None:
            rendered = str(Path("renders") / render_path.name)

    latencies = [record["request_latency_s"] for record in inference_records]
    return {
        "model": "openvla",
        "skill": skill,
        "scene_id": sample_dir.name,
        "instruction": instruction,
        "status": status,
        "steps": len(points) - 1,
        "trajectory_points": len(points),
        "actual_path_length_m": _path_length(points),
        "goal_distance": float(np.linalg.norm(position - goal)),
        "start_position": start_position.tolist(),
        "final_position": position.tolist(),
        "goal": goal.tolist(),
        "semantic_ok": semantic_ok,
        "completion_detail": completion_detail,
        "minimum_clearance_m": None if not np.isfinite(minimum_clearance) else minimum_clearance,
        "collision_point": None if collision_point is None else collision_point.tolist(),
        "gate_fallback_steps": 0,
        "uses_gate_correction": False,
        "request_count": len(inference_records),
        "mean_request_latency_s": None if not latencies else float(np.mean(latencies)),
        "max_request_latency_s": None if not latencies else float(np.max(latencies)),
        "elapsed_s": round(elapsed, 3),
        "reason": reason,
        "trajectory": str(Path("trajectories") / trajectory_name),
        "render": rendered,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one OpenVLA-UAV closed-loop rollout on an AeroVerbs sample."
    )
    parser.add_argument("--skill", required=True, choices=tuple(INSTRUCTION_TEMPLATES))
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--server-url", default="http://127.0.0.1:5007/predict")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--turn-rate", type=float, default=0.45)
    parser.add_argument("--drone-radius", type=float, default=0.15)
    parser.add_argument("--minimum-target-distance-m", type=float, default=1e-4)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--rgb-width", type=int, default=640)
    parser.add_argument("--rgb-height", type=int, default=480)
    parser.add_argument("--rgb-gpu-device-id", type=int, default=-1)
    parser.add_argument("--no-collision-check", action="store_true")
    parser.add_argument("--render-trajectory", action="store_true")
    parser.add_argument("--render-voxel-size", type=float, default=0.2)
    parser.add_argument("--render-voxel-stride", type=int, default=2)
    parser.add_argument("--render-region-stride", type=int, default=2)
    args = parser.parse_args(argv)
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.step_size <= 0.0:
        parser.error("--step-size must be positive")
    if args.turn_rate < 0.0:
        parser.error("--turn-rate must be non-negative")
    if args.drone_radius <= 0.0:
        parser.error("--drone-radius must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    data_root = Path(args.data_root).resolve()
    args.data_root = str(data_root)
    sample_dir = data_root / args.skill / args.scene_id
    if not sample_dir.is_dir():
        raise FileNotFoundError(sample_dir)
    if args.output_dir is None:
        args.output_dir = str(
            PROJECT_ROOT / "outputs" / "rollout" / "openvla" / time.strftime("%Y%m%d-%H%M%S")
        )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)

    run_config = dict(vars(args))
    run_config.update(
        {
            "model": "openvla",
            "sample_dir": str(sample_dir),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"sample: {sample_dir}")
    print(f"server: {args.server_url}")
    print(f"output: {output_dir}")
    try:
        result = run_one_sample(sample_dir, args)
    except Exception as exc:
        result = {
            "model": "openvla",
            "skill": args.skill,
            "scene_id": args.scene_id,
            "status": "infrastructure_error",
            "reason": repr(exc),
        }
        print(f"infrastructure_error: {exc!r}")

    with (output_dir / "results.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
