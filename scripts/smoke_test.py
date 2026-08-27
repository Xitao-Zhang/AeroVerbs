#!/usr/bin/env python
"""Smoke test: verify model shapes and one real data sample without training.

Usage::

    python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aeroverbs.config import Config
from aeroverbs.data.dataset import ManeuverDataset, build_sample_index
from aeroverbs.model import AeroVerbsPolicy
from aeroverbs.training.losses import total_loss


def test_model_shapes() -> None:
    B, T, K = 2, 6, 5
    model = AeroVerbsPolicy(num_skills=5, K=K)
    heatmap = torch.randn(B, 1, 120, 160)
    skill_id = torch.randint(0, 5, (B,))
    radar = torch.randn(B, T, 3, 40, 500)
    action_history = torch.randn(B, T, 10, 4)
    action_history_mask = torch.ones(B, T, 10)
    downward_range = torch.randn(B, T, 2)
    gate_features = torch.randn(B, T, 12)

    z_intent, z_skill, hidden = model.initialize(heatmap, skill_id)
    assert z_intent.shape == (B, 0), z_intent.shape
    assert z_skill.shape == (B, 32), z_skill.shape
    assert hidden.shape == (1, B, 256), hidden.shape

    wp, comp, progress, hidden2 = model.step(
        radar[:, 0],
        action_history[:, 0],
        action_history_mask[:, 0],
        downward_range[:, 0],
        gate_features[:, 0],
        z_intent,
        z_skill,
        hidden,
    )
    assert wp.shape == (B, K, 3), wp.shape
    assert comp.shape == (B, 1), comp.shape
    assert progress.shape == (B, 1), progress.shape

    wp_seq, comp_seq, progress_seq = model(
        heatmap, skill_id, radar, action_history, action_history_mask,
        downward_range, gate_features
    )
    assert wp_seq.shape == (B, T, K, 3), wp_seq.shape
    assert comp_seq.shape == (B, T, 1), comp_seq.shape
    assert progress_seq.shape == (B, T, 1), progress_seq.shape

    loss, comps = total_loss(
        wp_seq,
        comp_seq,
        progress_seq,
        wp_seq.detach().clone(),
        torch.ones(B, T, K),
        torch.zeros(B, T, 1),
        progress_target=torch.zeros(B, T, 1),
        gate_features=gate_features,
        action_target=torch.zeros(B, T, 4),
    )
    assert torch.isfinite(loss)
    print(f"[model] OK  loss={loss.item():.4f} components={ {k: round(v.item(), 4) for k, v in comps.items()} }")


def test_data_sample() -> None:
    cfg = Config()
    cfg.data.data_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "trajctorys"
    )
    samples = build_sample_index(cfg.data.data_root, ["around_left"])
    if not samples:
        print("[data] no samples found, skipping")
        return
    ds = ManeuverDataset(samples[:4], cfg.data)
    item = ds[0]
    print(
        "[data] OK "
        f"heatmap={tuple(item['heatmap'].shape)} skill_id={item['skill_id'].item()} "
        f"radar={tuple(item['radar'].shape)} waypoints={tuple(item['waypoints'].shape)} "
        f"valid_frac={item['waypoint_valid'].float().mean().item():.3f} "
        f"completion_pos_frac={item['completion'].float().mean().item():.3f}"
    )


if __name__ == "__main__":
    test_model_shapes()
    test_data_sample()
    print("all smoke tests passed")
