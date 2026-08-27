"""Training loop for AeroVerbs.

Outputs are organized by experiment for easy ablation comparison::

    outputs/<experiment>/<run_name>/
        config.yaml
        metrics.jsonl
        summary.json
        checkpoints/epoch_XXX.pt
        logs/            # tensorboard
        final.pt

Training is trajectory-level: each trajectory initializes the GRU once and is
unrolled in temporal order. Chunks are used only for TBPTT when the sequence is
longer than ``chunk_len``; the hidden state is carried across chunks and
detached between chunks.
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import Config
from ..data.dataset import (
    ManeuverDataset,
    TrajectoryDataset,
    build_sample_index,
    collate_fn,
    split_samples,
    trajectory_collate_fn,
)
from ..model import AeroVerbsPolicy
from .losses import dynamic_pos_weight, total_loss


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log_print(message: str) -> None:
    print(f"[{_now()}] {message}", flush=True)


def make_model(cfg: Config) -> AeroVerbsPolicy:
    m = cfg.model
    d = cfg.data
    return AeroVerbsPolicy(
        num_skills=m.num_skills,
        radar_in_channels=m.radar_in_channels,
        hidden=m.hidden,
        intent_dim=m.intent_dim,
        skill_dim=m.skill_dim,
        radar_dim=m.radar_dim,
        gate_feature_dim=m.gate_feature_dim,
        gate_dim=m.gate_dim,
        downward_dim=m.downward_dim,
        action_feature_dim=m.action_feature_dim,
        use_heatmap_latent=m.use_heatmap_latent,
        K=m.K,
        waypoint_dim=m.waypoint_dim,
        action_history_len=d.action_history_len,
        action_dim=d.action_dim,
    )


def make_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Windowed dataloaders (kept for compatibility and quick smoke tests)."""
    d = cfg.data
    samples = build_sample_index(d.data_root, d.skills)
    if len(samples) == 0:
        raise RuntimeError(f"No samples found under {d.data_root} for skills {d.skills}")
    train_samples, val_samples, test_samples = split_samples(
        samples, d.train_frac, d.val_frac, d.seed
    )
    train_ds = ManeuverDataset(train_samples, d)
    val_ds = ManeuverDataset(val_samples, d)
    test_ds = ManeuverDataset(test_samples, d)
    common = dict(batch_size=cfg.train.batch_size, collate_fn=collate_fn, num_workers=d.num_workers)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=d.drop_last, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, test_loader


class _BucketBatchSampler:
    """Batch complete trajectories with similar lengths, then shuffle batches."""

    def __init__(self, lengths, batch_size: int, shuffle: bool = True, seed: int = 0):
        self.batch_size = int(batch_size)
        import random

        indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.batches = [
            indices[i : i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]
        if shuffle:
            random.Random(seed).shuffle(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def make_trajectory_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Trajectory-level dataloaders with length bucketing.

    One sample is one complete trajectory. Trajectories are grouped into
    batches of similar length and padded to the batch's max length; a
    ``time_mask`` marks valid frames.
    """
    d = cfg.data
    samples = build_sample_index(d.data_root, d.skills)
    if len(samples) == 0:
        raise RuntimeError(f"No samples found under {d.data_root} for skills {d.skills}")
    train_samples, val_samples, test_samples = split_samples(
        samples, d.train_frac, d.val_frac, d.seed
    )
    train_ds = TrajectoryDataset(train_samples, d)
    val_ds = TrajectoryDataset(val_samples, d)
    test_ds = TrajectoryDataset(test_samples, d)

    batch_size = max(1, int(getattr(cfg.train, "trajectory_batch_size", 8)))
    common = dict(collate_fn=trajectory_collate_fn, num_workers=d.num_workers)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=_BucketBatchSampler(
            train_ds.lengths, batch_size, shuffle=True, seed=d.seed
        ),
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=_BucketBatchSampler(val_ds.lengths, batch_size, shuffle=False),
        **common,
    )
    test_loader = DataLoader(
        test_ds,
        batch_sampler=_BucketBatchSampler(test_ds.lengths, batch_size, shuffle=False),
        **common,
    )
    return train_loader, val_loader, test_loader


def _make_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )


def _make_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, steps_per_epoch: int):
    total_epochs = cfg.train.epochs
    warmup_epochs = cfg.train.warmup_epochs
    if cfg.train.lr_scheduler == "cosine":
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = total_epochs * steps_per_epoch

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if cfg.train.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(total_epochs // 3, 1), gamma=0.5)
    return None


def _loss_components(
    waypoints: torch.Tensor,
    completion: torch.Tensor,
    progress_logits: torch.Tensor,
    wp_target: torch.Tensor,
    wp_valid: torch.Tensor,
    comp_target: torch.Tensor,
    cfg: Config,
    device: torch.device,
    time_mask: Optional[torch.Tensor] = None,
    progress_target: Optional[torch.Tensor] = None,
    gate_features: Optional[torch.Tensor] = None,
    action_target: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
    forwards: Optional[torch.Tensor] = None,
    skill_id: Optional[torch.Tensor] = None,
    object_center: Optional[torch.Tensor] = None,
    object_radius: Optional[torch.Tensor] = None,
    occupied_points: Optional[torch.Tensor] = None,
    occupied_mask: Optional[torch.Tensor] = None,
    free_region_points: Optional[torch.Tensor] = None,
    free_region_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    front_weights = torch.tensor(cfg.train.front_weights, dtype=torch.float32, device=device)
    axis_weights = None
    if getattr(cfg.train, "waypoint_axis_weights", None):
        axis_weights = torch.tensor(cfg.train.waypoint_axis_weights, dtype=torch.float32, device=device)

    pos_weight = None
    if cfg.train.completion_pos_weight is not None:
        pos_weight = torch.tensor(cfg.train.completion_pos_weight, device=device)
    else:
        if time_mask is not None:
            tm = time_mask.to(comp_target.device).float()
            pos_target = comp_target * tm.unsqueeze(-1)
        else:
            pos_target = comp_target
        pos_weight = dynamic_pos_weight(pos_target)

    return total_loss(
        waypoints,
        completion,
        progress_logits,
        wp_target,
        wp_valid,
        comp_target,
        progress_target=progress_target,
        gate_features=gate_features,
        action_target=action_target,
        positions=positions,
        forwards=forwards,
        skill_id=skill_id,
        object_center=object_center,
        object_radius=object_radius,
        occupied_points=occupied_points,
        occupied_mask=occupied_mask,
        free_region_points=free_region_points,
        free_region_mask=free_region_mask,
        front_weights=front_weights,
        axis_weights=axis_weights,
        lambda_completion=cfg.train.lambda_completion,
        lambda_smooth=cfg.train.lambda_smooth,
        lambda_temporal_direction=cfg.train.lambda_temporal_direction,
        lambda_action_consistency=cfg.train.lambda_action_consistency,
        lambda_gate_cross=cfg.train.lambda_gate_cross,
        lambda_gate_direction=cfg.train.lambda_gate_direction,
        lambda_progress=cfg.train.lambda_progress,
        lambda_clearance=cfg.train.lambda_clearance,
        clearance_drone_radius=cfg.train.clearance_drone_radius,
        clearance_margin=cfg.train.clearance_margin,
        clearance_region_radius=cfg.train.clearance_region_radius,
        gate_waypoint_index=cfg.train.gate_waypoint_index,
        gate_active_far_m=cfg.train.gate_active_far_m,
        gate_active_post_m=cfg.train.gate_active_post_m,
        pos_weight=pos_weight,
        time_mask=time_mask,
    )


def _run_batch(
    model: AeroVerbsPolicy,
    batch: Dict[str, torch.Tensor],
    cfg: Config,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    """Windowed training step (compatibility path)."""
    heatmap = batch["heatmap"].to(device)
    skill_id = batch["skill_id"].to(device)
    radar = batch["radar"].to(device)
    action_history = batch["action_history"].to(device)
    action_history_mask = batch["action_history_mask"].to(device)
    downward_range = batch["downward_range"].to(device)
    gate_features = batch["gate_features"].to(device)
    progress_target = batch["progress"].to(device)
    action_target = batch["actions"].to(device)
    positions = batch["positions"].to(device)
    forwards = batch["forwards"].to(device)
    object_center = batch["object_center"].to(device)
    object_radius = batch["object_radius"].to(device)
    occupied_points = batch["occupied_points"].to(device)
    occupied_mask = batch["occupied_mask"].to(device)
    free_region_points = batch["free_region_points"].to(device)
    free_region_mask = batch["free_region_mask"].to(device)
    wp_target = batch["waypoints"].to(device)
    wp_valid = batch["waypoint_valid"].to(device)
    comp_target = batch["completion"].to(device)

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    waypoints, completion, progress = model(
        heatmap, skill_id, radar, action_history, action_history_mask,
        downward_range, gate_features
    )
    loss, components = _loss_components(
        waypoints, completion, progress, wp_target, wp_valid, comp_target, cfg, device,
        progress_target=progress_target,
        gate_features=gate_features,
        action_target=action_target,
        positions=positions,
        forwards=forwards,
        skill_id=skill_id,
        object_center=object_center,
        object_radius=object_radius,
        occupied_points=occupied_points,
        occupied_mask=occupied_mask,
        free_region_points=free_region_points,
        free_region_mask=free_region_mask,
    )

    if optimizer is not None:
        loss.backward()
        if cfg.train.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()

    return {k: v.item() for k, v in components.items()}


@torch.no_grad()
def _evaluate(
    model: AeroVerbsPolicy,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    agg: Dict[str, float] = {}
    count = 0
    for batch in loader:
        comps = _run_batch(model, batch, cfg, device, optimizer=None)
        for k, v in comps.items():
            agg[k] = agg.get(k, 0.0) + v
        count += 1
    model.train()
    return {k: v / max(count, 1) for k, v in agg.items()}


@torch.no_grad()
def _evaluate_trajectory(
    model: AeroVerbsPolicy,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> Dict[str, float]:
    """Trajectory-level evaluation: initialize once, unroll to the end."""
    model.eval()
    agg: Dict[str, float] = {}
    count = 0
    for batch in loader:
        heatmap = batch["heatmap"].to(device)
        skill_id = batch["skill_id"].to(device)
        radar = batch["radar"].to(device)
        action_history = batch["action_history"].to(device)
        action_history_mask = batch["action_history_mask"].to(device)
        downward_range = batch["downward_range"].to(device)
        gate_features = batch["gate_features"].to(device)
        progress_target = batch["progress"].to(device)
        action_target = batch["actions"].to(device)
        positions = batch["positions"].to(device)
        forwards = batch["forwards"].to(device)
        object_center = batch["object_center"].to(device)
        object_radius = batch["object_radius"].to(device)
        occupied_points = batch["occupied_points"].to(device)
        occupied_mask = batch["occupied_mask"].to(device)
        free_region_points = batch["free_region_points"].to(device)
        free_region_mask = batch["free_region_mask"].to(device)
        wp_target = batch["waypoints"].to(device)
        wp_valid = batch["waypoint_valid"].to(device)
        comp_target = batch["completion"].to(device)

        waypoints, completion, progress = model(
            heatmap, skill_id, radar, action_history, action_history_mask,
            downward_range, gate_features
        )
        time_mask = batch.get("time_mask")
        if time_mask is not None:
            time_mask = time_mask.to(device)
        _, components = _loss_components(
            waypoints, completion, progress, wp_target, wp_valid, comp_target, cfg, device,
            time_mask=time_mask,
            progress_target=progress_target,
            gate_features=gate_features,
            action_target=action_target,
            positions=positions,
            forwards=forwards,
            skill_id=skill_id,
            object_center=object_center,
            object_radius=object_radius,
            occupied_points=occupied_points,
            occupied_mask=occupied_mask,
            free_region_points=free_region_points,
            free_region_mask=free_region_mask,
        )
        for k, v in components.items():
            agg[k] = agg.get(k, 0.0) + float(v.item())
        count += 1
    model.train()
    return {k: v / max(count, 1) for k, v in agg.items()}


def _run_trajectory_chunk(
    model: AeroVerbsPolicy,
    batch: Dict[str, torch.Tensor],
    cfg: Config,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    heatmap: torch.Tensor,
    skill_id: torch.Tensor,
    hidden: torch.Tensor,
    start: int,
    end: int,
) -> Tuple[Dict[str, float], torch.Tensor]:
    """Run one TBPTT chunk.

    The intent/skill encoders are re-run for every chunk so each chunk owns an
    independent graph (standard TBPTT practice for shared activations); the GRU
    hidden is carried across chunks and detached after this chunk.
    """
    radar = batch["radar"].to(device)
    action_history = batch["action_history"].to(device)
    action_history_mask = batch["action_history_mask"].to(device)
    downward_range = batch["downward_range"].to(device)
    gate_features = batch["gate_features"].to(device)
    progress_target = batch["progress"].to(device)
    action_target = batch["actions"].to(device)
    positions = batch["positions"].to(device)
    forwards = batch["forwards"].to(device)
    object_center = batch["object_center"].to(device)
    object_radius = batch["object_radius"].to(device)
    occupied_points = batch["occupied_points"].to(device)
    occupied_mask = batch["occupied_mask"].to(device)
    free_region_points = batch["free_region_points"].to(device)
    free_region_mask = batch["free_region_mask"].to(device)
    wp_target = batch["waypoints"].to(device)
    wp_valid = batch["waypoint_valid"].to(device)
    comp_target = batch["completion"].to(device)
    time_mask = batch.get("time_mask")
    if time_mask is not None:
        time_mask = time_mask.to(device)

    z_intent, z_skill, init_hidden = model.initialize(heatmap, skill_id)
    if hidden is None:
        hidden = init_hidden

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    wp_list, comp_list, progress_list = [], [], []
    for t in range(start, end):
        wp, comp, progress, hidden = model.step(
            radar[:, t],
            action_history[:, t],
            action_history_mask[:, t],
            downward_range[:, t],
            gate_features[:, t],
            z_intent,
            z_skill,
            hidden,
        )
        wp_list.append(wp)
        comp_list.append(comp)
        progress_list.append(progress)

    waypoints = torch.stack(wp_list, dim=1)  # [B, chunk, K, 3]
    completion = torch.stack(comp_list, dim=1)  # [B, chunk, 1]
    progress = torch.stack(progress_list, dim=1)  # [B, chunk, 1]

    chunk_time_mask = None
    if time_mask is not None:
        chunk_time_mask = time_mask[:, start:end]  # [B, chunk]

    loss, components = _loss_components(
        waypoints,
        completion,
        progress,
        wp_target[:, start:end],
        wp_valid[:, start:end],
        comp_target[:, start:end],
        cfg,
        device,
        time_mask=chunk_time_mask,
        progress_target=progress_target[:, start:end],
        gate_features=gate_features[:, start:end],
        action_target=action_target[:, start:end],
        positions=positions[:, start:end],
        forwards=forwards[:, start:end],
        skill_id=skill_id,
        object_center=object_center,
        object_radius=object_radius,
        occupied_points=occupied_points,
        occupied_mask=occupied_mask,
        free_region_points=free_region_points,
        free_region_mask=free_region_mask,
    )

    if optimizer is not None:
        loss.backward()
        if cfg.train.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()

    return {k: v.item() for k, v in components.items()}, hidden.detach()


def _record_metrics(run_dir: str, phase: str, step: int, metrics: Dict[str, float]) -> None:
    """Append one metrics row to ``metrics.jsonl`` (for ablation comparison)."""
    path = os.path.join(run_dir, "metrics.jsonl")
    row = {"timestamp": _now(), "phase": phase, "step": step, **metrics}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _log_metrics(
    run_dir: str,
    phase: str,
    metrics: Dict[str, float],
    step: int,
    writer,
) -> None:
    line = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    _log_print(f"[{phase}] step {step} | {line}")
    _record_metrics(run_dir, phase, step, metrics)
    if writer is not None:
        for k, v in metrics.items():
            writer.add_scalar(f"{phase}/{k}", v, step)


def _make_run_dir(cfg: Config) -> str:
    run = cfg.train.run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(cfg.train.output_dir, cfg.train.experiment, run)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    return run_dir


def _dump_config(cfg: Config, run_dir: str) -> None:
    import yaml

    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)


def train(cfg: Config, resume_from: Optional[str] = None) -> AeroVerbsPolicy:
    device = torch.device(cfg.resolve_device())
    _log_print(f"device: {device}")

    model = make_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    _log_print(f"model params: {n_params:,}")

    train_loader, val_loader, test_loader = make_trajectory_dataloaders(cfg)
    _log_print(
        f"trajectories -> train: {len(train_loader.dataset)}, "
        f"val: {len(val_loader.dataset)}, test: {len(test_loader.dataset)}"
    )

    run_dir = _make_run_dir(cfg)
    _dump_config(cfg, run_dir)
    _log_print(f"run dir: {run_dir}")

    optimizer = _make_optimizer(model, cfg)
    scheduler = _make_scheduler(optimizer, cfg, max(len(train_loader), 1))

    ckpt_dir = os.path.join(run_dir, "checkpoints")

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=os.path.join(run_dir, "logs"))
    except Exception:  # tensorboard optional
        writer = None

    start_epoch = 0
    global_step = 0
    if resume_from and os.path.exists(resume_from):
        state = torch.load(resume_from, map_location=device)
        model.load_state_dict(state["model"])
        if "optimizer" in state and "epoch" in state:
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = state["epoch"] + 1
            # Do not restore the scheduler: the new dataloader has a different
            # number of batches per epoch (bucketed batches), so the old scheduler
            # step counter would push the learning rate to its minimum.
            _log_print(
                f"resumed from {resume_from} at epoch {start_epoch} "
                "(optimizer restored, scheduler reset)"
            )
        else:
            _log_print(f"loaded model weights from {resume_from} (fresh optimizer, epoch starts at 1)")

    chunk_len = max(1, int(getattr(cfg.train, "chunk_len", cfg.data.seq_len)))
    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        epoch_metrics: Dict[str, float] = {}
        n_chunks = 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"{_now()} epoch {epoch + 1}/{cfg.train.epochs}")
        for batch in pbar:
            heatmap = batch["heatmap"].to(device)
            skill_id = batch["skill_id"].to(device)
            hidden = None
            T = int(batch["radar"].shape[1])
            for start in range(0, T, chunk_len):
                end = min(start + chunk_len, T)
                comps, hidden = _run_trajectory_chunk(
                    model, batch, cfg, device, optimizer,
                    heatmap, skill_id, hidden, start, end,
                )
                for k, v in comps.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
                n_chunks += 1
                global_step += 1
                pbar.set_postfix({k: f"{v:.3f}" for k, v in comps.items()})
                if cfg.train.log_interval > 0 and global_step % cfg.train.log_interval == 0:
                    _log_metrics(run_dir, "train", comps, global_step, writer)
            if scheduler is not None:
                scheduler.step()

        epoch_metrics = {k: v / max(n_chunks, 1) for k, v in epoch_metrics.items()}
        elapsed = time.time() - t0
        _log_metrics(run_dir, "train/epoch", epoch_metrics, epoch, writer)
        _log_print(f"epoch {epoch + 1} elapsed {elapsed:.1f}s")

        if (epoch + 1) % cfg.train.val_interval == 0:
            val_metrics = _evaluate_trajectory(model, val_loader, cfg, device)
            _log_metrics(run_dir, "val", val_metrics, epoch, writer)

        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch + 1:03d}.pt")
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

    test_metrics = _evaluate_trajectory(model, test_loader, cfg, device)
    _log_metrics(run_dir, "test", test_metrics, cfg.train.epochs - 1, writer)

    final_path = os.path.join(run_dir, "final.pt")
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, final_path)
    _log_print(f"saved final model to {final_path}")

    summary = {
        "run_dir": run_dir,
        "experiment": cfg.train.experiment,
        "best_epoch": None,
        "test_metrics": test_metrics,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _log_print(f"test metrics: {test_metrics}")

    if writer is not None:
        writer.close()
    return model
