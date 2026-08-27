import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aeroverbs.training.losses import (
    action_consistency_loss,
    gate_cross_loss,
    temporal_direction_loss,
)


def test_gate_cross_loss_uses_gate_center_lateral_offset():
    gate_features = torch.tensor(
        [[[[1.0, 2.0, 1.0, 1.0, 0.0, 0.0, -1.0, 3.0, 2.0, 2.0, 1.0, 1.0]]]],
        dtype=torch.float32,
    ).squeeze(2)
    pred_inside_centered_gate = torch.tensor(
        [[[[0.5, 2.0, 1.0], [1.0, 2.0, 1.0], [1.5, 2.0, 1.0]]]],
        dtype=torch.float32,
    )
    pred_near_body_origin = torch.tensor(
        [[[[0.5, 0.0, 1.0], [1.0, 0.0, 1.0], [1.5, 0.0, 1.0]]]],
        dtype=torch.float32,
    )

    good = gate_cross_loss(pred_inside_centered_gate, gate_features)
    bad = gate_cross_loss(pred_near_body_origin, gate_features)

    assert good < 1e-5
    assert bad > good + 0.5


def test_action_consistency_aligns_prediction_with_next_action():
    pred = torch.zeros((1, 3, 1, 3), dtype=torch.float32)
    pred[:, :, 0, 0] = 1.0
    actions = torch.tensor(
        [[
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]],
        dtype=torch.float32,
    )

    loss = action_consistency_loss(pred, actions, waypoint_index=0)

    assert loss < 1e-5


def test_temporal_direction_compares_world_directions_when_forward_changes():
    pred = torch.tensor(
        [[
            [[1.0, 0.0, 0.0]],
            [[0.0, -1.0, 0.0]],
        ]],
        dtype=torch.float32,
    )
    forwards = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        dtype=torch.float32,
    )

    loss = temporal_direction_loss(pred, waypoint_index=0, forwards=forwards)

    assert loss < 1e-5


def test_temporal_direction_accepts_xy_forward_vectors():
    pred = torch.tensor(
        [[
            [[1.0, 0.0, 0.0]],
            [[0.0, -1.0, 0.0]],
        ]],
        dtype=torch.float32,
    )
    forwards = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)

    loss = temporal_direction_loss(pred, waypoint_index=0, forwards=forwards)

    assert loss < 1e-5
