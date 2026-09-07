#!/usr/bin/env python3
"""Closed-loop WorldVLN evaluation on one AeroVerbs sample."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from aeroverbs.baselines.worldvln import WorldVLNClient, apply_delta_body
from aeroverbs.env.rgb_renderer import HabitatRGBRenderer

INSTRUCTIONS = {
    "around_left": "Fly around the object on its left side.",
    "around_right": "Fly around the object on its right side.",
    "through": "Fly through the opening.",
    "over": "Fly over the object.",
    "under": "Fly under the object.",
}


def _runtime():
    from aeroverbs.config import Config
    from aeroverbs.data.completion import CompletionJudge, make_clearance_provider
    from aeroverbs.env.radar import load_scene_mesh, render_scene_glb
    from rollout import CollisionChecker, _load_collision_mesh
    return SimpleNamespace(
        Config=Config, CompletionJudge=CompletionJudge,
        make_clearance_provider=make_clearance_provider,
        load_scene_mesh=load_scene_mesh, render_scene_glb=render_scene_glb,
        CollisionChecker=CollisionChecker, load_collision_mesh=_load_collision_mesh,
    )


def _instruction(meta, override=None):
    return str(override or meta.get("instruction") or INSTRUCTIONS[meta["skill"]])


def run_one_sample(sample_dir: str | Path, args) -> dict:
    runtime = _runtime()
    sample_dir = Path(sample_dir)
    meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    skill = str(meta.get("skill") or sample_dir.parent.name)
    if skill not in INSTRUCTIONS:
        raise ValueError(f"unsupported skill: {skill}")
    frame0 = json.loads((sample_dir / "frames.json").read_text(encoding="utf-8"))[0]
    position = np.asarray(frame0.get("camera_position_std", frame0.get("position_std", frame0.get("position"))), dtype=np.float32)
    forward = np.asarray(frame0.get("camera_forward_std", frame0.get("forward_std", frame0.get("forward", [1, 0, 0]))), dtype=np.float32)
    forward[:2] /= max(float(np.linalg.norm(forward[:2])), 1e-8)
    pose = np.array([position[0], position[1], position[2], 0.0, math.atan2(forward[1], forward[0]), 0.0], dtype=np.float32)
    start = pose.copy()
    goal = np.asarray(meta["goal_std"], dtype=np.float32)
    collision_mesh = runtime.load_collision_mesh(str(sample_dir), args.data_root, meta)
    checker = runtime.CollisionChecker(collision_mesh, args.drone_radius, meta)
    provider = runtime.make_clearance_provider(str(sample_dir), args.data_root, meta)
    judge = runtime.CompletionJudge(meta, clearance_provider=provider, cfg=runtime.Config().completion, mesh=collision_mesh)
    scene_path = runtime.render_scene_glb(str(sample_dir))
    points = [position.copy()]
    records = []
    status, reason = "timeout", ""
    collision_point = None
    semantic_ok = False
    min_clearance = float("inf")
    session_id = f"{skill}__{sample_dir.name}__{args.run_id}"
    instruction = _instruction(meta, args.instruction)
    started = time.time()
    with HabitatRGBRenderer(scene_path, width=args.rgb_width, height=args.rgb_height, gpu_device_id=args.rgb_gpu_device_id) as renderer:
        client = WorldVLNClient(args.server_url, timeout_s=args.request_timeout_s)
        warm = renderer.render(position, forward)
        # In WorldVLN's autoregressive protocol, the reset/warmup call may
        # already emit segment 0 when allow_future_segments is enabled.  Keep
        # that prediction instead of silently discarding the first action
        # segment.
        prediction = client.predict(
            [warm], instruction=instruction, session_id=session_id, reset=True,
            action_head_mode=args.action_head_mode,
        )
        pending_prediction = bool(prediction.actions_cm_deg)
        for segment in range(args.max_segments):
            # If warmup emitted actions, execute them first.  Otherwise collect
            # the next real observation block before requesting a segment.
            if not pending_prediction:
                observations = [renderer.render(position, forward) for _ in range(args.step)]
                prediction = client.predict(observations, session_id=session_id, action_head_mode=args.action_head_mode)
            pending_prediction = False
            actions = list(prediction.actions_cm_deg)
            if not actions:
                reason = "server returned no actions"
                break
            repeat = max(1, args.step // len(actions)) if args.action_head_mode == "tsformer_latent" else 1
            executed = []
            for action in actions:
                for _ in range(repeat):
                    prev = pose.copy()
                    pose = apply_delta_body(pose, action, fraction=1.0 / repeat)
                    next_position = pose[:3].copy()
                    next_forward = np.array([math.cos(float(pose[4])), math.sin(float(pose[4])), 0], dtype=np.float32)
                    if not args.no_collision_check and checker.collides(next_position):
                        collision_point = next_position.copy(); points.append(next_position); status = "collision"
                        reason = f"collision at segment={segment}"
                        break
                    points.append(next_position)
                    executed.append(list(action))
                    result = judge.update(points, next_position)
                    semantic_ok = bool(result.semantic_done)
                    min_clearance = min(min_clearance, float(result.clearance))
                    if result.completion:
                        status = "success"; reason = "completion_judge"; break
                if status in ("collision", "success"):
                    break
            records.append({"segment": segment, "predicted_actions_cm_deg": [list(a) for a in actions], "executed_actions_cm_deg": executed, "request_latency_s": prediction.latency_s, "server": dict(prediction.raw)})
            if status in ("collision", "success"):
                break
            # ``done`` terminates the server-side action stream.  Do not issue
            # a follow-up request that can only return an empty action list.
            if bool(prediction.raw.get("done", False)):
                reason = "server_done_before_completion"
                break
    elapsed = time.time() - started
    if status == "timeout" and not reason:
        reason = f"max_segments={args.max_segments} reached"
    return {
        "model": "worldvln", "skill": skill, "scene_id": sample_dir.name, "instruction": instruction,
        "status": status, "reason": reason, "steps": len(points)-1, "trajectory_points": len(points),
        "actual_path_length_m": float(np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1).sum()) if len(points)>1 else 0.0,
        "start_position": start[:3].tolist(), "final_position": points[-1].tolist(), "goal": goal.tolist(),
        "trajectory": [p.tolist() for p in points],
        "goal_distance": float(np.linalg.norm(points[-1] - goal)), "semantic_ok": semantic_ok,
        "minimum_clearance_m": None if not np.isfinite(min_clearance) else min_clearance,
        "collision_point": None if collision_point is None else collision_point.tolist(),
        "request_count": len(records)+1, "mean_request_latency_s": float(np.mean([r["request_latency_s"] for r in records])) if records else None,
        "elapsed_s": round(elapsed, 3), "session_id": session_id, "worldvln_steps": records,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sample-dir", required=True); p.add_argument("--data-root", required=True)
    p.add_argument("--server-url", default="http://127.0.0.1:8001"); p.add_argument("--instruction")
    p.add_argument("--output-dir", required=True); p.add_argument("--run-id", default="single")
    p.add_argument("--action-head-mode", choices=["tsformer_latent", "actionhead_ref_vit"], default="tsformer_latent")
    p.add_argument("--max-segments", type=int, default=12); p.add_argument("--step", type=int, default=16)
    p.add_argument("--request-timeout-s", type=float, default=600); p.add_argument("--drone-radius", type=float, default=0.15)
    p.add_argument("--rgb-width", type=int, default=640); p.add_argument("--rgb-height", type=int, default=480); p.add_argument("--rgb-gpu-device-id", type=int, default=-1)
    p.add_argument("--no-collision-check", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(); Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    try: result = run_one_sample(args.sample_dir, args)
    except Exception as exc: result = {"model":"worldvln", "status":"infrastructure_error", "reason":repr(exc)}
    with (Path(args.output_dir)/"results.jsonl").open("a", encoding="utf-8") as f: f.write(json.dumps(result, ensure_ascii=False)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
