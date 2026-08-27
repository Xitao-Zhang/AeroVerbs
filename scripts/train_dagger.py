#!/usr/bin/env python
"""Train (or fine-tune) the policy on a mix of expert and DAgger samples.

The DAgger sample dirs contain ``teacher_waypoints.npy`` /
``teacher_waypoint_valid.npy`` / ``teacher_completion.npy`` and are loaded by
the regular :class:`TrajectoryDataset` without any special casing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from aeroverbs.data.dataset import (
    TrajectoryDataset,
    build_sample_index,
    split_samples,
    trajectory_collate_fn,
)
from aeroverbs.training import trainer as T
from rollout import _load_model


def parse_args():
    ap = argparse.ArgumentParser(description="Fine-tune a policy on expert + DAgger data.")
    ap.add_argument("--checkpoint", required=True, help="Student checkpoint to fine-tune.")
    ap.add_argument("--expert-data-root", default="/root/autodl-tmp/data/trajctorys")
    ap.add_argument("--dagger-data-root", required=True, help="Root of collected DAgger samples.")
    ap.add_argument("--skills", default="around_left,around_right", help="Comma-separated skills.")
    ap.add_argument("--mix-ratio", type=float, default=0.25, help="Target fraction of DAgger samples in training.")
    ap.add_argument("--max-dagger-repeats", type=int, default=4,
                    help="Cap repeated use of a small DAgger set; <=0 disables the cap.")
    ap.add_argument("--danger-weight", type=float, default=1.5,
                    help="Waypoint-loss multiplier for danger frames (danger_mask == 1).")
    ap.add_argument("--lambda-completion", type=float, default=None,
                    help="Optional completion-loss override; omitted inherits the checkpoint value.")
    ap.add_argument("--lambda-clearance", type=float, default=None,
                    help="Optional clearance-loss override; omitted inherits the checkpoint value.")
    ap.add_argument("--lambda-temporal-smooth", type=float, default=None,
                    help="Optional temporal-smoothness override; omitted inherits the checkpoint value.")
    ap.add_argument("--temporal-smooth-waypoint-index", type=int, default=2,
                    help="Waypoint index regularized by temporal smoothness.")
    ap.add_argument("--completion-pos-weight", type=float, default=None,
                    help="Optional fixed completion pos_weight; None keeps config value.")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--output-dir", default="outputs/dagger")
    ap.add_argument("--experiment", default="round1")
    ap.add_argument("--name", default=None)
    ap.add_argument("--log-interval", type=int, default=5)
    ap.add_argument("--val-interval", type=int, default=1)
    return ap.parse_args()


def _mix_samples(expert_samples, dagger_samples, mix_ratio: float, max_repeats: int = 4):
    if not dagger_samples:
        return list(expert_samples)
    if mix_ratio <= 0.0:
        return list(expert_samples)
    if mix_ratio >= 1.0:
        return list(dagger_samples)
    n_expert = len(expert_samples)
    n_dagger = len(dagger_samples)
    # repeat dagger samples so dagger fraction ~= mix_ratio
    if n_expert == 0:
        return list(dagger_samples)
    target = int(n_expert * mix_ratio / max(1.0 - mix_ratio, 1e-6))
    repeats = max(1, int(round(target / max(n_dagger, 1))))
    if max_repeats and max_repeats > 0:
        repeats = min(repeats, int(max_repeats))
    return list(expert_samples) + list(dagger_samples) * repeats


def _make_trajectory_datasets(train_samples, val_samples, test_samples, data_cfg):
    """Construct datasets using the current TrajectoryDataset API."""
    return (
        TrajectoryDataset(train_samples, data_cfg),
        TrajectoryDataset(val_samples, data_cfg),
        TrajectoryDataset(test_samples, data_cfg),
    )


def _apply_loss_overrides(cfg, args) -> None:
    """Preserve BC loss settings, except student-action imitation is disabled."""
    for name in ("lambda_completion", "lambda_clearance", "lambda_temporal_smooth"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg.train, name, value)
    if args.completion_pos_weight is not None:
        cfg.train.completion_pos_weight = args.completion_pos_weight
    cfg.train.lambda_action_consistency = 0.0
    if float(getattr(cfg.train, "lambda_clearance", 0.0)) <= 0.0:
        cfg.data.use_sdf = False


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model, cfg = _load_model(args.checkpoint, device)
    cfg.train.lr = args.lr
    cfg.train.epochs = args.epochs
    cfg.train.log_interval = args.log_interval
    cfg.train.val_interval = args.val_interval
    cfg.train.danger_weight = args.danger_weight
    _apply_loss_overrides(cfg, args)
    cfg.train.temporal_smooth_waypoint_index = args.temporal_smooth_waypoint_index
    cfg.train.output_dir = args.output_dir
    cfg.train.experiment = args.experiment
    cfg.train.run_name = args.name

    skills = [s.strip() for s in args.skills.split(",") if s.strip()]
    expert_samples = build_sample_index(args.expert_data_root, skills)
    dagger_samples = build_sample_index(args.dagger_data_root, skills)
    if not dagger_samples:
        raise SystemExit(f"No DAgger samples found under {args.dagger_data_root}")

    train_expert, val_expert, test_expert = split_samples(
        expert_samples, cfg.data.train_frac, cfg.data.val_frac, cfg.data.seed
    )
    train_samples = _mix_samples(
        train_expert, dagger_samples, args.mix_ratio, args.max_dagger_repeats
    )
    print(
        f"samples -> expert_train={len(train_expert)} dagger={len(dagger_samples)} "
        f"mixed_train={len(train_samples)} val={len(val_expert)} test={len(test_expert)}"
    )

    train_ds, val_ds, test_ds = _make_trajectory_datasets(
        train_samples, val_expert, test_expert, cfg.data
    )
    batch_size = max(1, int(getattr(cfg.train, "trajectory_batch_size", 8)))
    common = dict(collate_fn=trajectory_collate_fn, num_workers=cfg.data.num_workers)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=T._BucketBatchSampler(train_ds.lengths, batch_size, shuffle=True, seed=cfg.data.seed),
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=T._BucketBatchSampler(val_ds.lengths, batch_size, shuffle=False),
        **common,
    )
    test_loader = DataLoader(
        test_ds,
        batch_sampler=T._BucketBatchSampler(test_ds.lengths, batch_size, shuffle=False),
        **common,
    )

    run_dir = T._make_run_dir(cfg)
    T._dump_config(cfg, run_dir)
    print(f"run dir: {run_dir}")

    optimizer = T._make_optimizer(model, cfg)
    scheduler = T._make_scheduler(optimizer, cfg, max(len(train_loader), 1))

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=os.path.join(run_dir, "logs"))
    except Exception:
        writer = None

    chunk_len = max(1, int(getattr(cfg.train, "chunk_len", cfg.data.seq_len)))
    global_step = 0
    for epoch in range(cfg.train.epochs):
        model.train()
        epoch_metrics = {}
        n_chunks = 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"dagger epoch {epoch + 1}/{cfg.train.epochs}")
        for batch in pbar:
            heatmap = batch["heatmap"].to(device)
            skill_id = batch["skill_id"].to(device)
            hidden = None
            T_len = int(batch["radar"].shape[1])
            for start in range(0, T_len, chunk_len):
                end = min(start + chunk_len, T_len)
                comps, hidden = T._run_trajectory_chunk(
                    model, batch, cfg, device, optimizer,
                    heatmap, skill_id, hidden, start, end,
                )
                for k, v in comps.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
                n_chunks += 1
                global_step += 1
                pbar.set_postfix({k: f"{v:.3f}" for k, v in comps.items()})
                if cfg.train.log_interval > 0 and global_step % cfg.train.log_interval == 0:
                    T._log_metrics(run_dir, "train", comps, global_step, writer)
            if scheduler is not None:
                scheduler.step()

        epoch_metrics = {k: v / max(n_chunks, 1) for k, v in epoch_metrics.items()}
        T._log_metrics(run_dir, "train/epoch", epoch_metrics, epoch, writer)
        print(f"epoch {epoch + 1} elapsed {time.time() - t0:.1f}s")

        if (epoch + 1) % cfg.train.val_interval == 0:
            val_metrics = T._evaluate_trajectory(model, val_loader, cfg, device)
            T._log_metrics(run_dir, "val", val_metrics, epoch, writer)

        ckpt_path = os.path.join(run_dir, "checkpoints", f"epoch_{epoch + 1:03d}.pt")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "global_step": global_step,
                "config": cfg.to_dict(),
            },
            ckpt_path,
        )

    test_metrics = T._evaluate_trajectory(model, test_loader, cfg, device)
    T._log_metrics(run_dir, "test", test_metrics, cfg.train.epochs - 1, writer)

    final_path = os.path.join(run_dir, "final.pt")
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, final_path)
    print(f"saved final model to {final_path}")

    summary = {
        "run_dir": run_dir,
        "experiment": cfg.train.experiment,
        "test_metrics": test_metrics,
        "mix_ratio": args.mix_ratio,
        "max_dagger_repeats": args.max_dagger_repeats,
        "expert_train": len(train_expert),
        "dagger_train": len(dagger_samples),
        "danger_weight": args.danger_weight,
        "lambda_completion": float(getattr(cfg.train, "lambda_completion", 0.0)),
        "lambda_clearance": float(getattr(cfg.train, "lambda_clearance", 0.0)),
        "lambda_temporal_smooth": float(getattr(cfg.train, "lambda_temporal_smooth", 0.0)),
        "lambda_action_consistency": float(
            getattr(cfg.train, "lambda_action_consistency", 0.0)
        ),
        "use_sdf": bool(getattr(cfg.data, "use_sdf", True)),
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"test metrics: {test_metrics}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
