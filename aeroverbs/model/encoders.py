"""Feature encoders for AeroVerbs.

Layers follow `maneuver_policy_network_v1.md`:
- IntentEncoder   (§2)  one-shot maneuver heatmap -> z_intent
- SkillEncoder    (§3)  skill_id -> z_skill
- CircularRadarEncoder (§4) 3-channel radar -> z_radar
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .circular_conv import CircularConv2d, ResidualCircularBlock


class ActionEncoder(nn.Module):
    """Encode a sliding window of recently executed kinematic actions.

    Input : ``history [B, H, 4]``, ``mask [B, H]``
    Output: ``[B, 64]``

    The four action channels are ``[dx_forward, dy_left, dz_up, delta_yaw]``.
    Padding entries are zeroed via the mask before flattening.
    """

    def __init__(self, history_len: int = 10, action_dim: int = 4, out_dim: int = 64) -> None:
        super().__init__()
        self.history_len = int(history_len)
        self.action_dim = int(action_dim)
        self.net = nn.Sequential(
            nn.Linear(history_len * action_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = history * mask.unsqueeze(-1).float()
        x = x.flatten(1)
        return self.net(x)


class DownwardRangeEncoder(nn.Module):
    """Encode downward ray distance and valid mask.

    Input: ``[B, 2]`` containing ``[distance_m, valid]``.
    """

    def __init__(self, input_dim: int = 2, out_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, out_dim),
            nn.SiLU(),
        )

    def forward(self, downward_range: torch.Tensor) -> torch.Tensor:
        return self.net(downward_range.float())


class GateEncoder(nn.Module):
    """Encode body-frame heatmap-derived gate geometry features."""

    def __init__(self, input_dim: int = 12, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, out_dim),
            nn.SiLU(),
        )

    def forward(self, gate_features: torch.Tensor) -> torch.Tensor:
        return self.net(gate_features.float())


class IntentEncoder(nn.Module):
    """Encode a one-shot maneuver heatmap into an intent latent vector.

    Input : ``[B, 1, 120, 160]``
    Output: ``[B, 64]``
    """

    def __init__(self, in_channels: int = 1, out_dim: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((3, 4))
        self.head = nn.Sequential(
            nn.Linear(64 * 3 * 4, 128),
            nn.SiLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, heatmap: torch.Tensor) -> torch.Tensor:
        x = self.stem(heatmap)  # [B, 64, 15, 20]
        x = self.pool(x)  # [B, 64, 3, 4]
        return self.head(x.flatten(1))  # [B, out_dim]


class SkillEncoder(nn.Module):
    """Map a discrete skill id to a conditioning vector.

    Input : ``[B]`` long tensor
    Output: ``[B, 32]``
    """

    def __init__(self, num_skills: int = 5, embed_dim: int = 16, out_dim: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_skills, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.SiLU(),
        )

    def forward(self, skill_id: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(skill_id))


class CircularRadarEncoder(nn.Module):
    """Encode a 3-channel 360-degree radar range image.

    Input : ``[B, 3, 40, 500]``
    Output: ``[B, 256]``

    Spatial aggregation keeps a 5x16 layout and concatenates average and max
    pooling so sparse strong echoes are not diluted by empty space.
    """

    def __init__(self, in_channels: int = 3, out_dim: int = 256) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            CircularConv2d(in_channels, 32, kernel_size=(3, 7), stride=(1, 2), padding=(1, 3)),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            ResidualCircularBlock(32, kernel_size=3, padding=1),
            CircularConv2d(32, 64, kernel_size=(3, 5), stride=(2, 2), padding=(1, 2)),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            ResidualCircularBlock(64, kernel_size=3, padding=1),
            CircularConv2d(64, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            ResidualCircularBlock(96, kernel_size=3, padding=1),
            nn.Conv2d(96, 32, kernel_size=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((5, 16))
        self.max_pool = nn.AdaptiveMaxPool2d((5, 16))
        self.head = nn.Sequential(
            nn.Linear(32 * 5 * 16 * 2, 512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, out_dim),
            nn.SiLU(),
        )

    def forward(self, radar: torch.Tensor) -> torch.Tensor:
        x = self.stem(radar)  # [B, 32, 10, 63]
        avg = self.avg_pool(x)  # [B, 32, 5, 16]
        mx = self.max_pool(x)  # [B, 32, 5, 16]
        cat = torch.cat([avg, mx], dim=1)  # [B, 64, 5, 16]
        return self.head(cat.flatten(1))  # [B, out_dim]
