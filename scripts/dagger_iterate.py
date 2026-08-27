#!/usr/bin/env python
"""Iterate the DAgger loop: collect -> train -> (optional) evaluate.

Every round:
1. collect DAgger samples with the current student checkpoint;
2. symlink the new samples into a shared ``all_dagger`` root so training sees
   the whole DAgger history;
3. fine-tune the student on expert data + all DAgger samples;
4. optionally run rollout evaluation.

Usage::

    python scripts/dagger_iterate.py \
        --start-checkpoint outputs/bc_baseline/20260821-155734/final.pt \
        --data-root /root/autodl-tmp/data/trajctorys \
        --skills around_left,around_right \
        --num-rounds 3 --num-per-skill 5 --train-epochs 5 \
        --output-root outputs/dagger
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def parse_args():
    ap = argparse.ArgumentParser(description="Run DAgger iterations.")
    ap.add_argument("--start-checkpoint", required=True, help="Initial student checkpoint.")
    ap.add_argument("--data-root", default="/root/autodl-tmp/data/trajctorys", help="Expert data root used for collecting samples.")
    ap.add_argument("--expert-data-root", default=None, help="Expert data root used for training (defaults to --data-root).")
    ap.add_argument("--skills", default="around_left,around_right", help="Comma-separated skills.")
    ap.add_argument("--num-rounds", type=int, default=3)
    ap.add_argument("--start-round", type=int, default=1,
                    help="Number assigned to the first round in this run.")
    ap.add_argument("--seed-dagger-root", default=None,
                    help="Optional historical DAgger sample root linked into all_dagger before collection.")
    ap.add_argument("--num-per-skill", type=int, default=20, help="Max DAgger samples collected per skill per round (20-30 recommended).")
    ap.add_argument("--train-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--mix-ratio", type=float, default=0.25)
    ap.add_argument("--max-dagger-repeats", type=int, default=4,
                    help="Cap repeated use of a small DAgger set during training; <=0 disables the cap.")
    ap.add_argument("--priority-results", default=None,
                    help="Deprecated compatibility option; Fast-Planner collection currently ignores it.")
    ap.add_argument("--danger-threshold", type=float, default=0.45,
                    help="Deprecated compatibility option; ignored by the Fast-Planner teacher.")
    ap.add_argument("--push-distance", type=float, default=0.5,
                    help="Deprecated compatibility option; ignored by the Fast-Planner teacher.")
    ap.add_argument("--push-waypoint-count", type=int, default=3,
                    help="Deprecated compatibility option; ignored by the Fast-Planner teacher.")
    ap.add_argument("--push-blend-width", type=float, default=0.15,
                    help="Deprecated compatibility option; ignored by the Fast-Planner teacher.")
    ap.add_argument("--danger-weight", type=float, default=1.5,
                    help="Waypoint-loss multiplier for danger frames during DAgger training.")
    ap.add_argument("--lambda-completion", type=float, default=None,
                    help="Optional DAgger completion-loss override; omitted inherits the checkpoint.")
    ap.add_argument("--lambda-clearance", type=float, default=None,
                    help="Optional DAgger clearance-loss override; omitted inherits the checkpoint.")
    ap.add_argument("--lambda-temporal-smooth", type=float, default=None,
                    help="Optional DAgger temporal-smoothness override; omitted inherits the checkpoint.")
    ap.add_argument("--completion-pos-weight", type=float, default=None,
                    help="Optional fixed completion pos_weight for DAgger training.")
    ap.add_argument("--max-steps", type=int, default=200, help="Max rollout steps per collected sample.")
    ap.add_argument("--drone-radius", type=float, default=0.15,
                    help="Hard collision and Fast-Planner inflation radius in meters.")
    ap.add_argument("--teacher-preferred-clearance", type=float, default=None,
                    help="Optional global Teacher preferred-clearance override; omitted uses per-skill defaults.")
    ap.add_argument("--sample-seed", type=int, default=42,
                    help="Fixed seed for category-stratified DAgger sample ordering.")
    ap.add_argument("--teacher-replan-interval", type=int, default=5,
                    help="Run Fast-Planner every N collected frames.")
    ap.add_argument("--teacher-timeout", type=float, default=2.0,
                    help="Post-call Fast-Planner time budget in seconds.")
    ap.add_argument("--teacher-max-iter", type=int, default=2000,
                    help="Maximum kinodynamic A* expansions per search.")
    ap.add_argument("--output-root", default="outputs/dagger", help="Where round outputs and all_dagger live.")
    ap.add_argument("--eval-skill", default=None, help="Optional: skill used for post-round rollout eval.")
    ap.add_argument("--eval-scene-ids", default=None, help="Optional: comma-separated scene ids for post-round eval.")
    return ap.parse_args()


def _run(cmd: list, verbose: bool = True) -> int:
    print("  CMD:", " ".join(cmd))
    t0 = time.time()
    rc = subprocess.call(cmd)
    print(f"  -> exit={rc} elapsed={time.time() - t0:.1f}s")
    return rc


def _build_collect_cmd(args, checkpoint: str, sample_root: str) -> list:
    """Build a collector command containing only options it supports."""
    command = [
        sys.executable,
        os.path.join(ROOT, "scripts", "dagger_collect.py"),
        "--checkpoint", checkpoint,
        "--data-root", args.data_root,
        "--skills", args.skills,
        "--num-per-skill", str(args.num_per_skill),
        "--max-steps", str(args.max_steps),
        "--output-dir", sample_root,
        "--drone-radius", str(args.drone_radius),
        "--sample-seed", str(args.sample_seed),
        "--teacher-replan-interval", str(args.teacher_replan_interval),
        "--teacher-timeout", str(args.teacher_timeout),
        "--teacher-max-iter", str(args.teacher_max_iter),
    ]
    if args.teacher_preferred_clearance is not None:
        command.extend(
            ["--teacher-preferred-clearance", str(args.teacher_preferred_clearance)]
        )
    return command


def _build_train_cmd(
    args,
    checkpoint: str,
    expert_data_root: str,
    all_dagger_root: str,
    train_experiment: str,
    train_name: str,
) -> list:
    command = [
        sys.executable,
        os.path.join(ROOT, "scripts", "train_dagger.py"),
        "--checkpoint", checkpoint,
        "--expert-data-root", expert_data_root,
        "--dagger-data-root", all_dagger_root,
        "--skills", args.skills,
        "--mix-ratio", str(args.mix_ratio),
        "--max-dagger-repeats", str(args.max_dagger_repeats),
        "--epochs", str(args.train_epochs),
        "--lr", str(args.lr),
        "--danger-weight", str(args.danger_weight),
        "--output-dir", args.output_root,
        "--experiment", train_experiment,
        "--name", train_name,
    ]
    for option in ("lambda_completion", "lambda_clearance", "lambda_temporal_smooth"):
        value = getattr(args, option)
        if value is not None:
            command.extend(["--" + option.replace("_", "-"), str(value)])
    if args.completion_pos_weight is not None:
        command.extend(["--completion-pos-weight", str(args.completion_pos_weight)])
    return command


def _round_training_paths(
    output_root: str,
    round_idx: int,
    timestamp: str | None = None,
) -> tuple[str, str, str]:
    """Return the BC-style timestamped run location for one DAgger round."""
    train_experiment = f"round{round_idx}"
    train_name = timestamp or time.strftime("%Y%m%d-%H%M%S")
    checkpoint = os.path.join(
        output_root, train_experiment, train_name, "final.pt"
    )
    return train_experiment, train_name, checkpoint


def _link_into_all_dagger(sample_root: str, all_root: str, skills: list) -> int:
    linked = 0
    for skill in skills:
        src_skill = os.path.join(sample_root, skill)
        if not os.path.isdir(src_skill):
            continue
        dst_skill = os.path.join(all_root, skill)
        os.makedirs(dst_skill, exist_ok=True)
        for name in sorted(os.listdir(src_skill)):
            src = os.path.abspath(os.path.join(src_skill, name))
            if not os.path.isdir(src):
                continue
            dst_name = name
            dst = os.path.join(dst_skill, dst_name)
            if os.path.islink(dst) and os.path.realpath(dst) == os.path.realpath(src):
                continue
            if os.path.lexists(dst):
                round_name = os.path.basename(sample_root.rstrip(os.sep))
                dst_name = f"{round_name}__{name}"
                dst = os.path.join(dst_skill, dst_name)
                suffix = 1
                while os.path.lexists(dst):
                    dst_name = f"{round_name}__{name}__{suffix}"
                    dst = os.path.join(dst_skill, dst_name)
                    suffix += 1
            os.symlink(src, dst)
            linked += 1
    return linked


def _seed_all_dagger(seed_root: str, all_root: str, skills: list) -> int:
    if not os.path.isdir(seed_root):
        raise ValueError(f"seed DAgger root does not exist: {seed_root}")
    return _link_into_all_dagger(seed_root, all_root, skills)


def _round_indices(args) -> range:
    return range(args.start_round, args.start_round + args.num_rounds)


def _run_eval(checkpoint: str, data_root: str, skill: str, scene_ids: str, output_dir: str):
    if not skill or not scene_ids:
        return
    cmd = [
        sys.executable, os.path.join(ROOT, "scripts", "rollout.py"),
        "--checkpoint", checkpoint,
        "--data-root", data_root,
        "--skill", skill,
        "--scene-id", scene_ids,
        "--output-dir", output_dir,
    ]
    _run(cmd)
    results_path = os.path.join(output_dir, "results.jsonl")
    if os.path.exists(results_path):
        statuses = {}
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    statuses[row.get("status", "?")] = statuses.get(row.get("status", "?"), 0) + 1
        print(f"  eval statuses: {statuses}")


def main():
    args = parse_args()
    skills = [s.strip() for s in args.skills.split(",") if s.strip()]
    expert_data_root = args.expert_data_root or args.data_root
    os.makedirs(args.output_root, exist_ok=True)
    all_dagger_root = os.path.join(args.output_root, "all_dagger")
    os.makedirs(all_dagger_root, exist_ok=True)

    if args.seed_dagger_root:
        seeded = _seed_all_dagger(args.seed_dagger_root, all_dagger_root, skills)
        print(
            f"  seeded {seeded} historical DAgger samples from "
            f"{args.seed_dagger_root} into {all_dagger_root}"
        )

    checkpoint = args.start_checkpoint
    for round_position, round_idx in enumerate(_round_indices(args), start=1):
        print(
            f"\n===== DAgger round {round_idx} "
            f"({round_position}/{args.num_rounds} in this run) ====="
        )
        sample_root = os.path.join(args.output_root, f"round{round_idx}_samples")
        train_experiment, train_name, round_checkpoint = _round_training_paths(
            args.output_root, round_idx
        )

        # 1) collect
        collect_cmd = _build_collect_cmd(args, checkpoint, sample_root)
        rc = _run(collect_cmd)
        if rc != 0:
            print("collect failed, stopping")
            return rc

        # 2) merge new samples into the shared DAgger root
        linked = _link_into_all_dagger(sample_root, all_dagger_root, skills)
        print(f"  linked {linked} new DAgger samples into {all_dagger_root}")
        if linked == 0:
            print("no new samples, stopping")
            return 1

        # 3) train
        rc = _run(
            _build_train_cmd(
                args,
                checkpoint=checkpoint,
                expert_data_root=expert_data_root,
                all_dagger_root=all_dagger_root,
                train_experiment=train_experiment,
                train_name=train_name,
            )
        )
        if rc != 0:
            print("train failed, stopping")
            return rc

        checkpoint = round_checkpoint
        if not os.path.exists(checkpoint):
            print(f"expected checkpoint not found: {checkpoint}")
            return 1
        print(f"  round {round_idx} checkpoint: {checkpoint}")

        # 4) optional eval
        if args.eval_skill:
            _run_eval(
                checkpoint,
                args.data_root,
                args.eval_skill,
                args.eval_scene_ids,
                os.path.join(args.output_root, train_experiment, train_name, "eval"),
            )

    print(f"\nDAgger iterations complete. Final checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
