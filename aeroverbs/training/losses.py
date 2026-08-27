"""Loss functions for AeroVerbs (maneuver_policy_network_v1.md §10).

Total loss: ``L = L_wp + 0.2 * L_completion + 0.05 * L_smooth``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ..constants import SKILL_TO_ID


def waypoint_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    front_weights: Optional[torch.Tensor] = None,
    axis_weights: Optional[torch.Tensor] = None,
    time_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Masked, front-weighted, axis-weighted SmoothL1 loss.

    Args:
        pred: ``[B, T, K, 3]`` (body frame: forward, left, up).
        target: ``[B, T, K, 3]``.
        valid: ``[B, T, K]`` float mask.
        front_weights: ``[K]`` per-waypoint weights.
        axis_weights: ``[3]`` per-axis weights. Because trajectories are mostly
            straight forward flight, the forward (x) error is cheap to reduce
            and dominates the loss; weighting lateral/vertical higher forces
            the policy to actually learn the maneuver (around/over/under).
        time_mask: ``[B, T]`` optional float mask, 1 for valid frames and 0 for
            padded frames.
    """
    diff = pred - target
    per_axis = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none")  # [B,T,K,3]
    if axis_weights is not None:
        aw = axis_weights.to(diff.device).view(1, 1, 1, -1)
        per_axis = per_axis * aw
        loss = per_axis.sum(dim=-1) / aw.sum()  # weighted mean over xyz
    else:
        loss = per_axis.mean(dim=-1)  # [B,T,K]
    mask = valid.float()
    if time_mask is not None:
        mask = mask * time_mask.to(mask.device).float().unsqueeze(-1)  # [B,T,1]
    if front_weights is not None:
        mask = mask * front_weights.to(mask.device).view(1, 1, -1)
    num = (loss * mask).sum()
    den = mask.sum() + 1e-8
    return num / den


def completion_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    time_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if time_mask is None:
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    per_frame = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none"
    )  # [B, T, 1]
    mask = time_mask.to(per_frame.device).float().unsqueeze(-1)  # [B, T, 1]
    return (per_frame * mask).sum() / (mask.sum() + 1e-8)


def dynamic_pos_weight(target: torch.Tensor) -> torch.Tensor:
    """Compute ``pos_weight = negatives / positives`` for completion BCE."""
    pos = target.sum()
    neg = target.numel() - pos
    pos = pos.clamp(min=1.0)
    return (neg / pos).clamp(min=1.0, max=100.0)


def smoothness_loss(pred: torch.Tensor, time_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Penalize second differences of the K predicted waypoints.

    ``L_smooth = mean_i || w_{i+1} - 2 w_i + w_{i-1} ||_1`` over i=1..K-2.
    """
    acc = pred[:, :, 2:] - 2.0 * pred[:, :, 1:-1] + pred[:, :, :-2]  # [B,T,K-2,3]
    if time_mask is None:
        return acc.abs().mean()
    mask = time_mask.to(acc.device).float()[:, :, None, None]  # [B,T,1,1]
    return (acc.abs() * mask).sum() / (mask.sum() * acc.shape[-2] + 1e-8)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.device).float()
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return (values * mask).sum() / (mask.sum() + 1e-8)


def _body_vectors_to_world(vectors: torch.Tensor, forwards: torch.Tensor) -> torch.Tensor:
    """Convert body-frame vectors to world-frame using horizontal forward."""
    body_x = forwards.float()
    if body_x.shape[-1] == 2:
        body_x = torch.cat([body_x, torch.zeros_like(body_x[..., :1])], dim=-1)
    body_x = body_x / body_x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    left = torch.stack([-body_x[..., 1], body_x[..., 0], torch.zeros_like(body_x[..., 0])], dim=-1)
    up = torch.zeros_like(body_x)
    up[..., 2] = 1.0
    while body_x.ndim < vectors.ndim:
        body_x = body_x.unsqueeze(-2)
        left = left.unsqueeze(-2)
        up = up.unsqueeze(-2)
    return vectors[..., 0:1] * body_x + vectors[..., 1:2] * left + vectors[..., 2:3] * up


def _waypoints_to_world(
    waypoints: torch.Tensor,
    positions: torch.Tensor,
    forwards: torch.Tensor,
) -> torch.Tensor:
    world_vec = _body_vectors_to_world(waypoints, forwards)
    return positions.float().unsqueeze(-2) + world_vec


def temporal_direction_loss(
    pred: torch.Tensor,
    *,
    waypoint_index: int = 2,
    forwards: Optional[torch.Tensor] = None,
    time_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Penalize frame-to-frame changes in selected waypoint direction."""
    if pred.shape[1] < 2:
        return pred.new_zeros(())
    idx = min(max(int(waypoint_index), 0), pred.shape[2] - 1)
    vec = pred[:, :, idx, :]
    if forwards is not None:
        vec = _body_vectors_to_world(vec, forwards)
    direction = F.normalize(vec, dim=-1, eps=1e-6)
    diff = direction[:, 1:] - direction[:, :-1]
    per_frame = diff.abs().mean(dim=-1)
    if time_mask is None:
        return per_frame.mean()
    mask = time_mask[:, 1:] * time_mask[:, :-1]
    return _masked_mean(per_frame, mask)


def action_consistency_loss(
    pred: torch.Tensor,
    action_target: torch.Tensor,
    *,
    waypoint_index: int = 2,
    time_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Align the selected predicted waypoint with the next executed action."""
    idx = min(max(int(waypoint_index), 0), pred.shape[2] - 1)
    if pred.shape[1] < 2:
        return pred.new_zeros(())
    vec = pred[:, :-1, idx, :]
    direction = F.normalize(vec, dim=-1, eps=1e-6)
    target = action_target[:, 1:]
    target_xyz = target[..., :3]
    target_speed = torch.linalg.norm(target_xyz, dim=-1, keepdim=True).clamp(min=1e-4)
    pred_xyz = direction * target_speed
    pred_yaw = torch.atan2(direction[..., 1:2], direction[..., 0:1].clamp(min=1e-6))
    pred_action = torch.cat([pred_xyz, pred_yaw], dim=-1)
    per_frame = F.smooth_l1_loss(pred_action, target, reduction="none").mean(dim=-1)
    if time_mask is None:
        return per_frame.mean()
    mask = time_mask[:, :-1] * time_mask[:, 1:]
    return _masked_mean(per_frame, mask)


def gate_active_mask(
    gate_features: torch.Tensor,
    time_mask: Optional[torch.Tensor] = None,
    *,
    far_m: float = 4.0,
    post_m: float = 0.5,
) -> torch.Tensor:
    signed = gate_features[..., 6]
    valid = gate_features[..., 11] > 0.5
    active = valid & (signed >= -float(far_m)) & (signed <= float(post_m))
    mask = active.float()
    if time_mask is not None:
        mask = mask * time_mask.to(mask.device).float()
    return mask


def gate_direction_loss(
    pred: torch.Tensor,
    gate_features: torch.Tensor,
    *,
    waypoint_index: int = -1,
    time_mask: Optional[torch.Tensor] = None,
    far_m: float = 4.0,
    post_m: float = 0.5,
) -> torch.Tensor:
    idx = pred.shape[2] - 1 if int(waypoint_index) < 0 else min(int(waypoint_index), pred.shape[2] - 1)
    pred_vec = pred[:, :, idx, :]
    gate_vec = gate_features[..., :3]
    cosine = F.cosine_similarity(pred_vec, gate_vec, dim=-1, eps=1e-6)
    values = 1.0 - cosine
    mask = gate_active_mask(gate_features, time_mask, far_m=far_m, post_m=post_m)
    return _masked_mean(values, mask)


def gate_cross_loss(
    pred: torch.Tensor,
    gate_features: torch.Tensor,
    *,
    time_mask: Optional[torch.Tensor] = None,
    far_m: float = 4.0,
    post_m: float = 0.5,
) -> torch.Tensor:
    """Softly require predicted body-frame waypoints to cross inside the gate."""
    center = gate_features[..., None, 0:3]  # [B,T,1,3]
    normal = F.normalize(gate_features[..., None, 3:6], dim=-1, eps=1e-6)
    width = gate_features[..., 8].clamp(min=1e-3)
    height = gate_features[..., 9].clamp(min=1e-3)
    current_signed = gate_features[..., 6]

    signed = ((pred - center) * normal).sum(dim=-1)  # [B,T,K]
    all_signed = torch.cat([current_signed.unsqueeze(-1), signed], dim=-1)
    crosses = (all_signed[..., :-1] * all_signed[..., 1:] <= 0.0).any(dim=-1)

    abs_signed = signed.abs()
    best_idx = abs_signed.argmin(dim=-1, keepdim=True)
    best_point = pred.gather(
        2, best_idx.unsqueeze(-1).expand(-1, -1, -1, pred.shape[-1])
    ).squeeze(2)
    center_point = center.squeeze(2)
    lateral = (best_point[..., 1] - center_point[..., 1]).abs()
    vertical = (best_point[..., 2] - center_point[..., 2]).abs()
    inside_penalty = F.relu(lateral - width * 0.5).pow(2) + F.relu(vertical - height * 0.5).pow(2)
    plane_penalty = abs_signed.min(dim=-1).values
    no_cross_penalty = torch.where(crosses, plane_penalty.new_zeros(()), plane_penalty + 0.2)
    values = inside_penalty + no_cross_penalty
    mask = gate_active_mask(gate_features, time_mask, far_m=far_m, post_m=post_m)
    return _masked_mean(values, mask)


def progress_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    time_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    per_frame = F.smooth_l1_loss(torch.sigmoid(logits), target, reduction="none").squeeze(-1)
    if time_mask is None:
        return per_frame.mean()
    return _masked_mean(per_frame, time_mask)


def _skill_mask(skill_id: torch.Tensor, names: Tuple[str, ...]) -> torch.Tensor:
    ids = [SKILL_TO_ID[name] for name in names]
    mask = torch.zeros_like(skill_id, dtype=torch.bool)
    for skill in ids:
        mask = mask | (skill_id == int(skill))
    return mask


def clearance_loss(
    pred: torch.Tensor,
    *,
    positions: torch.Tensor,
    forwards: torch.Tensor,
    skill_id: torch.Tensor,
    object_center: Optional[torch.Tensor] = None,
    object_radius: Optional[torch.Tensor] = None,
    occupied_points: Optional[torch.Tensor] = None,
    occupied_mask: Optional[torch.Tensor] = None,
    free_region_points: Optional[torch.Tensor] = None,
    free_region_mask: Optional[torch.Tensor] = None,
    gate_features: Optional[torch.Tensor] = None,
    time_mask: Optional[torch.Tensor] = None,
    drone_radius: float = 0.35,
    margin: float = 0.15,
    region_radius: float = 0.35,
    gate_active_far_m: float = 4.0,
    gate_active_post_m: float = 0.5,
) -> torch.Tensor:
    """Penalize predicted world-frame waypoints that do not preserve clearance.

    around/over use an object sphere. through/under use occupied voxel samples
    when available, plus a soft pull toward the precomputed passable gate
    region near the gate.
    """
    points = _waypoints_to_world(pred, positions, forwards)  # [B,T,K,3]
    point_mask = torch.ones(points.shape[:-1], dtype=torch.float32, device=points.device)
    if time_mask is not None:
        point_mask = point_mask * time_mask.to(points.device).float().unsqueeze(-1)

    safe_dist = float(drone_radius) + float(margin)
    losses = []

    if object_center is not None and object_radius is not None:
        sphere_skill = _skill_mask(skill_id.to(points.device), ("around_left", "around_right", "over"))
        if sphere_skill.any():
            center = object_center.to(points.device).float().view(points.shape[0], 1, 1, 3)
            radius = object_radius.to(points.device).float().view(points.shape[0], 1, 1)
            clearance = torch.linalg.norm(points - center, dim=-1) - radius
            values = F.relu(safe_dist - clearance).pow(2)
            mask = point_mask * sphere_skill.float().view(-1, 1, 1)
            losses.append(_masked_mean(values, mask))

    tunnel_skill = _skill_mask(skill_id.to(points.device), ("through", "under"))
    tunnel_point_mask = point_mask * tunnel_skill.float().view(-1, 1, 1)

    if occupied_points is not None and occupied_mask is not None and tunnel_skill.any():
        occ = occupied_points.to(points.device).float()
        occ_mask = occupied_mask.to(points.device).bool()
        flat = points.reshape(points.shape[0], -1, 3)
        dists = torch.cdist(flat, occ)
        dists = dists.masked_fill(~occ_mask[:, None, :], 1e6)
        nearest = dists.min(dim=-1).values.reshape(points.shape[:-1])
        values = F.relu(safe_dist - nearest).pow(2)
        has_occ = occ_mask.any(dim=-1).float().view(-1, 1, 1)
        losses.append(_masked_mean(values, tunnel_point_mask * has_occ))

    if free_region_points is not None and free_region_mask is not None and tunnel_skill.any():
        region = free_region_points.to(points.device).float()
        region_mask = free_region_mask.to(points.device).bool()
        flat = points.reshape(points.shape[0], -1, 3)
        dists = torch.cdist(flat, region)
        dists = dists.masked_fill(~region_mask[:, None, :], 1e6)
        nearest = dists.min(dim=-1).values.reshape(points.shape[:-1])
        values = F.relu(nearest - float(region_radius)).pow(2)
        active = tunnel_point_mask
        if gate_features is not None:
            active = active * gate_active_mask(
                gate_features,
                time_mask,
                far_m=gate_active_far_m,
                post_m=gate_active_post_m,
            ).to(points.device).float().unsqueeze(-1)
        has_region = region_mask.any(dim=-1).float().view(-1, 1, 1)
        losses.append(_masked_mean(values, active * has_region))

    if not losses:
        return pred.new_zeros(())
    return torch.stack(losses).mean()


def total_loss(
    waypoints: torch.Tensor,
    completion_logits: torch.Tensor,
    progress_logits: Optional[torch.Tensor],
    wp_target: torch.Tensor,
    wp_valid: torch.Tensor,
    comp_target: torch.Tensor,
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
    front_weights: Optional[torch.Tensor] = None,
    axis_weights: Optional[torch.Tensor] = None,
    lambda_completion: float = 0.2,
    lambda_smooth: float = 0.05,
    lambda_temporal_direction: float = 0.0,
    lambda_action_consistency: float = 0.0,
    lambda_gate_cross: float = 0.0,
    lambda_gate_direction: float = 0.0,
    lambda_progress: float = 0.0,
    lambda_clearance: float = 0.0,
    clearance_drone_radius: float = 0.35,
    clearance_margin: float = 0.15,
    clearance_region_radius: float = 0.35,
    gate_waypoint_index: int = 2,
    gate_active_far_m: float = 4.0,
    gate_active_post_m: float = 0.5,
    pos_weight: Optional[torch.Tensor] = None,
    time_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Return the total loss and a dict of its components.

    ``time_mask`` masks padded frames so batched trajectories with different
    lengths can be trained together.
    """
    l_wp = waypoint_loss(waypoints, wp_target, wp_valid, front_weights, axis_weights, time_mask)
    l_comp = completion_loss(completion_logits, comp_target, pos_weight, time_mask)
    l_smooth = smoothness_loss(waypoints, time_mask)
    l_temporal = temporal_direction_loss(
        waypoints, waypoint_index=gate_waypoint_index, forwards=forwards, time_mask=time_mask
    )
    l_action = waypoints.new_zeros(())
    if action_target is not None:
        l_action = action_consistency_loss(
            waypoints, action_target, waypoint_index=gate_waypoint_index, time_mask=time_mask
        )
    l_gate_cross = waypoints.new_zeros(())
    l_gate_dir = waypoints.new_zeros(())
    if gate_features is not None:
        l_gate_cross = gate_cross_loss(
            waypoints,
            gate_features,
            time_mask=time_mask,
            far_m=gate_active_far_m,
            post_m=gate_active_post_m,
        )
        l_gate_dir = gate_direction_loss(
            waypoints,
            gate_features,
            waypoint_index=-1,
            time_mask=time_mask,
            far_m=gate_active_far_m,
            post_m=gate_active_post_m,
        )
    l_progress = waypoints.new_zeros(())
    if progress_logits is not None and progress_target is not None:
        l_progress = progress_loss(progress_logits, progress_target, time_mask)
    l_clearance = waypoints.new_zeros(())
    if (
        float(lambda_clearance) > 0.0
        and positions is not None
        and forwards is not None
        and skill_id is not None
    ):
        l_clearance = clearance_loss(
            waypoints,
            positions=positions,
            forwards=forwards,
            skill_id=skill_id,
            object_center=object_center,
            object_radius=object_radius,
            occupied_points=occupied_points,
            occupied_mask=occupied_mask,
            free_region_points=free_region_points,
            free_region_mask=free_region_mask,
            gate_features=gate_features,
            time_mask=time_mask,
            drone_radius=clearance_drone_radius,
            margin=clearance_margin,
            region_radius=clearance_region_radius,
            gate_active_far_m=gate_active_far_m,
            gate_active_post_m=gate_active_post_m,
        )
    loss = l_wp + lambda_completion * l_comp + lambda_smooth * l_smooth
    loss = (
        loss
        + float(lambda_temporal_direction) * l_temporal
        + float(lambda_action_consistency) * l_action
        + float(lambda_gate_cross) * l_gate_cross
        + float(lambda_gate_direction) * l_gate_dir
        + float(lambda_progress) * l_progress
        + float(lambda_clearance) * l_clearance
    )
    return loss, {
        "waypoint": l_wp.detach(),
        "completion": l_comp.detach(),
        "smoothness": l_smooth.detach(),
        "temporal_direction": l_temporal.detach(),
        "action_consistency": l_action.detach(),
        "gate_cross": l_gate_cross.detach(),
        "gate_direction": l_gate_dir.detach(),
        "progress": l_progress.detach(),
        "clearance": l_clearance.detach(),
        "total": loss.detach(),
    }
