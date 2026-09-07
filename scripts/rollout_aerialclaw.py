#!/usr/bin/env python3
"""Run AerialClaw closed-loop on one AeroVerbs scene."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Optional, Sequence, Union

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aeroverbs.baselines.aerialclaw_bridge import (
    AerialClawBridgeServer,
    AeroVerbsEpisode,
    EpisodeConfig,
)


INSTRUCTION_TEMPLATES = {
    "around_left": "Fly around the object on its left side.",
    "around_right": "Fly around the object on its right side.",
    "through": "Fly through the opening.",
    "over": "Fly over the object.",
    "under": "Fly under the object.",
}


AERIALCLAW_OUTPUT_PROTOCOL = """\
Evaluation protocol (follow exactly on every iteration):
1. Return exactly one JSON object. Do not use Markdown or add text outside JSON.
2. Use exactly these top-level fields and types:
   {"thinking":"string","decision":"act","action":{"skill":"string","robot":"UAV_1","parameters":{}},"reflection":null,"goal_progress":"string"}
3. `action` MUST be a JSON object and `parameters` MUST be a JSON object. Never emit shorthand such as `"action":"observe"`.
4. Keep decision=`act` until the external benchmark terminates the episode. Do not declare `done` or `stuck`.
5. Your FIRST action MUST be:
   {"skill":"observe","robot":"UAV_1","parameters":{"direction":"front","focus":"Locate the target obstacle or opening and identify free space for the requested maneuver."}}
6. Use only these exact parameter schemas:
   - observe: {"direction":"front|left|right|rear|down","focus":"string"}
   - perceive: {"direction":"front|left|right|rear|down","focus":"string"}
   - get_position: {}
   - get_sensor_data: {"sensor_types":["lidar"],"vehicle_id":"UAV_1"}
   - fly_relative: {"forward":number,"right":number,"up":number,"speed":number}
   - fly_to: {"target_position":[north,east,down],"speed":number}
   - hover: {"duration":number}
7. For fly_relative, NEVER use `direction` or `distance`; those keys are invalid and cause zero motion. Positive right means right; negative right means left. Positive up means up.
8. Move cautiously in increments no larger than 1.0 metre per action. Never command a blind 5-metre move. Observe again after changing position, and use the visual result plus LiDAR before choosing the next move.
9. The vehicle is already airborne. Do not take off, land, return to launch, report, update a map, ask the user, or write files.
10. Complete the requested spatial relation, not merely approach the object. Continue until the external CompletionJudge accepts the trajectory or the simulator terminates it.
"""


def build_instruction(skill: str, custom_instruction: Optional[str] = None) -> str:
    """Combine the maneuver request with the benchmark output/action contract."""
    task_instruction = custom_instruction or INSTRUCTION_TEMPLATES[skill]
    return f"{task_instruction}\n\n{AERIALCLAW_OUTPUT_PROTOCOL}"


def _path_length(points) -> float:
    values = np.asarray(points, dtype=float)
    if len(values) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(values, axis=0), axis=1).sum())


def run_one_sample(sample_dir: Union[str, Path], args: argparse.Namespace) -> dict:
    sample_path = Path(sample_dir)
    meta = json.loads((sample_path / "meta.json").read_text(encoding="utf-8"))
    skill = str(meta.get("skill") or sample_path.parent.name)
    if skill not in INSTRUCTION_TEMPLATES:
        raise ValueError(f"unsupported skill: {skill}")
    instruction = build_instruction(skill, args.instruction)
    started = time.time()

    episode = AeroVerbsEpisode.from_sample(
        sample_path,
        data_root=args.data_root,
        config=EpisodeConfig(
            max_steps=args.max_steps,
            step_size=args.step_size,
            turn_rate=args.turn_rate,
            drone_radius=args.drone_radius,
        ),
        rgb_width=args.rgb_width,
        rgb_height=args.rgb_height,
        rgb_gpu_device_id=args.rgb_gpu_device_id,
    )

    with tempfile.TemporaryDirectory(prefix="aerialclaw-aeroverbs-") as temp_dir:
        agent_output_path = Path(temp_dir) / "agent_result.json"
        with AerialClawBridgeServer(episode, host="127.0.0.1", port=0) as server:
            command = [
                str(args.aerialclaw_python),
                str(Path(args.aerialclaw_root) / "scripts" / "run_aeroverbs_episode.py"),
                "--bridge-url",
                server.base_url,
                "--instruction",
                instruction,
                "--output-json",
                str(agent_output_path),
                "--max-iterations",
                str(args.agent_max_iterations),
                "--timeout",
                str(args.request_timeout_s),
            ]
            environment = os.environ.copy()
            environment.setdefault("AERIALCLAW_DEFAULT_FLIGHT_SPEED", "1.0")
            environment.setdefault("AERIALCLAW_MIN_FLIGHT_SPEED", "0.1")
            environment.setdefault("AERIALCLAW_MIN_SAFE_DIST", "0.30")
            process = subprocess.run(
                command,
                cwd=str(args.aerialclaw_root),
                env=environment,
                text=True,
                capture_output=True,
                timeout=args.subprocess_timeout_s,
                check=False,
            )
            if process.returncode != 0:
                detail = (process.stderr or process.stdout or "").strip()[-1000:]
                raise RuntimeError(
                    f"AerialClaw runner exited with code {process.returncode}: {detail}"
                )
            if not agent_output_path.is_file():
                raise RuntimeError("AerialClaw runner did not write its output JSON")
            agent_metadata = json.loads(agent_output_path.read_text(encoding="utf-8"))
            if agent_metadata.get("infrastructure_error"):
                raise RuntimeError(str(agent_metadata["infrastructure_error"]))
            if not episode.termination_state().get("terminal"):
                episode.declare_agent_done()

    authoritative = episode.result()
    points = authoritative["points"]
    goal = np.asarray(meta.get("goal_std", points[-1]), dtype=float)
    final_position = np.asarray(points[-1], dtype=float)
    start_position = np.asarray(points[0], dtype=float)
    status = str(authoritative.get("status", "timeout"))
    completion_detail = (
        f"semantic_done={bool(authoritative.get('semantic_done', False))}, "
        f"far_side={bool(authoritative.get('far_side', False))}, "
        f"completion={bool(authoritative.get('completion', False))}"
    )

    trajectory = {
        "points": points,
        "semantic_label": skill,
        "instruction": instruction,
        "aerialclaw_actions": authoritative.get("aerialclaw_actions", []),
        "agent_action_events": agent_metadata.get("agent_action_events", []),
    }
    trajectory_dir = Path(args.output_dir) / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    trajectory_name = f"{skill}__{sample_path.name}.json"
    (trajectory_dir / trajectory_name).write_text(
        json.dumps(trajectory, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "model": "aerialclaw",
        "execution_mode": "native_sim_adapter_http",
        "skill": skill,
        "scene_id": sample_path.name,
        "instruction": instruction,
        "status": status,
        "steps": int(authoritative.get("steps", max(0, len(points) - 1))),
        "trajectory_points": len(points),
        "actual_path_length_m": _path_length(points),
        "goal_distance": float(np.linalg.norm(final_position - goal)),
        "start_position": start_position.tolist(),
        "final_position": final_position.tolist(),
        "goal": goal.tolist(),
        "semantic_ok": bool(authoritative.get("semantic_done", False)),
        "completion_detail": completion_detail,
        "minimum_clearance_m": authoritative.get("min_clearance"),
        "collision_point": points[-1] if status == "collision" else None,
        "gate_fallback_steps": 0,
        "uses_gate_correction": False,
        "request_count": len(agent_metadata.get("agent_action_events", [])),
        "agent_iterations": int(agent_metadata.get("agent_iterations", 0)),
        "agent_completion_events": agent_metadata.get("agent_completion_events", []),
        "elapsed_s": round(time.time() - started, 3),
        "reason": str(authoritative.get("reason", "")),
        "trajectory": str(Path("trajectories") / trajectory_name),
        "render": None,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", required=True, choices=tuple(INSTRUCTION_TEMPLATES))
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "trajctorys"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--instruction")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--turn-rate", type=float, default=0.45)
    parser.add_argument("--drone-radius", type=float, default=0.15)
    parser.add_argument("--rgb-width", type=int, default=640)
    parser.add_argument("--rgb-height", type=int, default=480)
    parser.add_argument("--rgb-gpu-device-id", type=int, default=-1)
    parser.add_argument("--aerialclaw-root", default="/home/zxt/AerialClaw")
    parser.add_argument(
        "--aerialclaw-python",
        default="/home/zxt/miniconda3/envs/aerialclaw-eval/bin/python",
    )
    parser.add_argument("--agent-max-iterations", type=int, default=40)
    parser.add_argument("--request-timeout-s", type=float, default=15.0)
    parser.add_argument("--subprocess-timeout-s", type=float, default=1800.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    sample_dir = Path(args.data_root) / args.skill / args.scene_id
    print(json.dumps(run_one_sample(sample_dir, args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
