"""The full AeroVerbs policy network.

Current gate-based interface::

    z_intent, z_skill, hidden = model.initialize(None, skill_id)
    waypoints, completion, hidden = model.step(radar, z_intent, z_skill, hidden)

`forward` is the batched training convenience: it initializes once from the
skill and unrolls the GRU over a radar sequence. ``IntentEncoder`` and the
optional heatmap argument remain only for loading historical checkpoints.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .encoders import (
    ActionEncoder,
    CircularRadarEncoder,
    DownwardRangeEncoder,
    GateEncoder,
    IntentEncoder,
    SkillEncoder,
)


class FusionMLP(nn.Module):
    def __init__(self, in_dim: int = 416, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GRUInitHead(nn.Module):
    """Initialize the GRU hidden state from intent + skill."""

    def __init__(self, in_dim: int = 96, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class WaypointHead(nn.Module):
    """Predict K future waypoints in body frame: ``[B, K, 3]``."""

    def __init__(self, hidden: int = 256, K: int = 5, waypoint_dim: int = 3) -> None:
        super().__init__()
        self.K = K
        self.waypoint_dim = waypoint_dim
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 128),
            nn.SiLU(),
            nn.Linear(128, K * waypoint_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).view(h.shape[0], self.K, self.waypoint_dim)


class CompletionHead(nn.Module):
    """Predict maneuver-completion logit: ``[B, 1]``."""

    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class ProgressHead(nn.Module):
    """Predict gate-relative progress in ``[0, 1]`` after sigmoid."""

    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class AeroVerbsPolicy(nn.Module):
    def __init__(
        self,
        num_skills: int = 5,
        radar_in_channels: int = 3,
        hidden: int = 256,
        intent_dim: int = 64,
        skill_dim: int = 32,
        radar_dim: int = 256,
        gate_feature_dim: int = 12,
        gate_dim: int = 64,
        downward_dim: int = 32,
        action_feature_dim: int = 64,
        use_heatmap_latent: bool = False,
        K: int = 5,
        waypoint_dim: int = 3,
        action_history_len: int = 10,
        action_dim: int = 4,
    ) -> None:
        super().__init__()
        self.K = K
        self.hidden = hidden
        self.use_heatmap_latent = bool(use_heatmap_latent)
        self.gate_feature_dim = int(gate_feature_dim)
        self.intent_encoder = IntentEncoder(in_channels=1, out_dim=intent_dim)
        self.skill_encoder = SkillEncoder(num_skills=num_skills, out_dim=skill_dim)
        self.radar_encoder = CircularRadarEncoder(in_channels=radar_in_channels, out_dim=radar_dim)
        self.gate_encoder = GateEncoder(input_dim=gate_feature_dim, out_dim=gate_dim)
        self.downward_encoder = DownwardRangeEncoder(input_dim=2, out_dim=downward_dim)
        self.action_encoder = ActionEncoder(
            history_len=action_history_len, action_dim=action_dim, out_dim=action_feature_dim
        )
        fusion_in = radar_dim + gate_dim + downward_dim + skill_dim + action_feature_dim
        init_in = skill_dim
        if self.use_heatmap_latent:
            fusion_in += intent_dim
            init_in += intent_dim
        self.fusion = FusionMLP(in_dim=fusion_in, hidden=hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        self.gru_init = GRUInitHead(in_dim=init_in, hidden=hidden)
        self.waypoint_head = WaypointHead(hidden=hidden, K=K, waypoint_dim=waypoint_dim)
        self.completion_head = CompletionHead(hidden=hidden)
        self.progress_head = ProgressHead(hidden=hidden)

    def initialize(
        self, heatmap: torch.Tensor | None, skill_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode one-shot inputs and produce the initial GRU hidden state.

        Args:
            heatmap: optional ``[B, 1, H, W]`` legacy intention heatmap.
                It may be ``None`` when ``use_heatmap_latent`` is false.
            skill_id: ``[B]`` long tensor.

        Returns:
            z_intent ``[B, D_i]``, z_skill ``[B, D_s]``, hidden ``[1, B, H]``.
        """
        z_skill = self.skill_encoder(skill_id)
        if self.use_heatmap_latent:
            if heatmap is None:
                raise ValueError("heatmap is required when use_heatmap_latent=True")
            z_intent = self.intent_encoder(heatmap)
            init = torch.cat([z_intent, z_skill], dim=-1)
        else:
            z_intent = z_skill.new_zeros((z_skill.shape[0], 0))
            init = z_skill
        h0 = self.gru_init(init)
        hidden = h0.unsqueeze(0)  # [num_layers=1, B, H]
        return z_intent, z_skill, hidden

    def step(
        self,
        radar: torch.Tensor,
        action_history: torch.Tensor,
        action_history_mask: torch.Tensor,
        downward_range: torch.Tensor | None,
        gate_features: torch.Tensor | None,
        z_intent: torch.Tensor,
        z_skill: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one control step.

        Args:
            radar: ``[B, 3, H, W]`` preprocessed radar range image.
            action_history: ``[B, H, 4]`` recent executed kinematic actions.
            action_history_mask: ``[B, H]`` valid mask for the history.
            z_intent, z_skill: cached vectors from :meth:`initialize`.
            hidden: ``[1, B, H]`` GRU hidden state.

        Returns:
            waypoints ``[B, K, 3]``, completion logits ``[B, 1]``,
            progress logits ``[B, 1]``, next hidden.
        """
        z_radar = self.radar_encoder(radar)
        z_action = self.action_encoder(action_history, action_history_mask)
        if downward_range is None:
            downward_range = radar.new_zeros((radar.shape[0], 2))
        if gate_features is None:
            gate_features = radar.new_zeros((radar.shape[0], self.gate_feature_dim))
        z_downward = self.downward_encoder(downward_range)
        z_gate = self.gate_encoder(gate_features)
        parts = [z_radar, z_gate, z_downward, z_skill, z_action]
        if self.use_heatmap_latent:
            parts.insert(1, z_intent)
        f = self.fusion(torch.cat(parts, dim=-1))
        out, hidden = self.gru(f.unsqueeze(1), hidden)
        h = out[:, -1]
        waypoints = self.waypoint_head(h)
        completion = self.completion_head(h)
        progress = self.progress_head(h)
        return waypoints, completion, progress, hidden

    def forward(
        self,
        heatmap: torch.Tensor | None,
        skill_id: torch.Tensor,
        radar_seq: torch.Tensor,
        action_history_seq: torch.Tensor,
        action_history_mask_seq: torch.Tensor,
        downward_range_seq: torch.Tensor | None = None,
        gate_features_seq: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched training forward over a radar sequence.

        Args:
            heatmap: optional legacy input. Current gate-based models pass
                ``None`` because ``use_heatmap_latent`` is false.
            skill_id: ``[B]``.
            radar_seq: ``[B, T, 3, H, W]``.
            action_history_seq: ``[B, T, H, 4]``.
            action_history_mask_seq: ``[B, T, H]``.

        Returns:
            waypoints ``[B, T, K, 3]``, completion logits ``[B, T, 1]``,
            progress logits ``[B, T, 1]``.
        """
        z_intent, z_skill, hidden = self.initialize(heatmap, skill_id)
        T = radar_seq.shape[1]
        wp_list, comp_list, progress_list = [], [], []
        for t in range(T):
            down_t = None if downward_range_seq is None else downward_range_seq[:, t]
            gate_t = None if gate_features_seq is None else gate_features_seq[:, t]
            wp, comp, progress, hidden = self.step(
                radar_seq[:, t],
                action_history_seq[:, t],
                action_history_mask_seq[:, t],
                down_t,
                gate_t,
                z_intent,
                z_skill,
                hidden,
            )
            wp_list.append(wp)
            comp_list.append(comp)
            progress_list.append(progress)
        waypoints = torch.stack(wp_list, dim=1)  # [B, T, K, 3]
        completion = torch.stack(comp_list, dim=1)  # [B, T, 1]
        progress = torch.stack(progress_list, dim=1)  # [B, T, 1]
        return waypoints, completion, progress

    @torch.no_grad()
    def roll(
        self,
        heatmap: torch.Tensor | None,
        skill_id: torch.Tensor,
        radar_seq: torch.Tensor,
        action_history_seq: torch.Tensor,
        action_history_mask_seq: torch.Tensor,
        downward_range_seq: torch.Tensor | None = None,
        gate_features_seq: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference helper returning waypoints and completion probabilities."""
        waypoints, completion, progress = self.forward(
            heatmap,
            skill_id,
            radar_seq,
            action_history_seq,
            action_history_mask_seq,
            downward_range_seq,
            gate_features_seq,
        )
        return waypoints, torch.sigmoid(completion), torch.sigmoid(progress)
