"""Typed configuration for AeroVerbs.

Config can be built directly or loaded from a YAML file. YAML sections mirror
the nested dataclasses (`model`, `data`, `train`), so a YAML key such as
``train.batch_size`` overrides ``TrainConfig.batch_size``.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ModelConfig:
    num_skills: int = 5
    radar_in_channels: int = 3
    hidden: int = 256
    intent_dim: int = 64
    skill_dim: int = 32
    radar_dim: int = 256
    gate_feature_dim: int = 12
    gate_dim: int = 64
    downward_dim: int = 32
    action_feature_dim: int = 64
    use_heatmap_latent: bool = False
    K: int = 5  # number of predicted future waypoints
    waypoint_dim: int = 3


@dataclass
class DataConfig:
    data_root: str = "data/trajctorys"
    skills: List[str] = field(
        default_factory=lambda: ["around_left", "around_right", "through", "over", "under"]
    )
    heatmap_size: Tuple[int, int] = (120, 160)  # (H, W) fed to IntentEncoder
    radar_size: Tuple[int, int] = (40, 500)  # (rows, cols) of raw range image
    r_max: float = 10.0  # meters; range values at/above this are "no hit"
    fov_deg: Tuple[float, float] = (-5.0, 35.0)  # vertical pitch span
    action_history_len: int = 10  # recent kinematic actions fed to ActionEncoder
    action_dim: int = 4  # [dx_forward, dy_left, dz_up, delta_yaw]
    use_downward_range: bool = True
    downward_range_name: str = "downward_range.npy"
    gate_summary_name: str = "heatmap_gate_summary.json"
    gate_region_name: str = "heatmap_gate_region.json"
    clearance_occupied_points: int = 1024
    clearance_region_points: int = 512
    future_offsets: List[int] = field(default_factory=lambda: [3, 6, 10, 15, 20])
    completion_tail_frac: float = 0.2
    seq_len: int = 20
    stride: int = 10
    train_frac: float = 0.8
    val_frac: float = 0.1
    # test = 1 - train_frac - val_frac
    seed: int = 0
    num_workers: int = 0
    drop_last: bool = True


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    lambda_waypoint: float = 1.0
    lambda_completion: float = 0.05
    lambda_smooth: float = 0.05
    lambda_temporal_direction: float = 0.1
    lambda_action_consistency: float = 0.1
    lambda_gate_cross: float = 0.5
    lambda_gate_direction: float = 0.05
    lambda_progress: float = 0.05
    lambda_clearance: float = 1.0
    clearance_drone_radius: float = 0.15
    clearance_margin: float = 0.20
    clearance_region_radius: float = 0.35
    gate_waypoint_index: int = 2
    gate_active_far_m: float = 4.0
    gate_active_post_m: float = 0.5
    # "dynamic" computes pos_weight per batch; a float fixes it.
    completion_pos_weight: Optional[float] = None
    front_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 0.8, 0.6, 0.5])
    # Body-frame axis weights (forward, left, up). Weight lateral/vertical
    # higher so the policy learns the maneuver instead of just flying forward.
    waypoint_axis_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    log_interval: int = 20
    val_interval: int = 1
    chunk_len: int = 20  # TBPTT chunk length for trajectory-level training
    trajectory_batch_size: int = 8  # batched trajectories with length bucketing
    device: str = "auto"
    output_dir: str = "outputs"
    experiment: str = "default"
    run_name: Optional[str] = None  # None -> auto timestamp
    use_amp: bool = False
    lr_scheduler: str = "cosine"  # none | cosine | step
    warmup_epochs: int = 1


@dataclass
class CompletionConfig:
    d_exit: float = 0.4
    sustain_frames: int = 3
    vertical_margin: float = 0.2
    over_vertical_margin: float = 0.0
    far_eps: float = 0.3
    ground_z_min: float = 0.2


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    completion: CompletionConfig = field(default_factory=CompletionConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def resolve_device(self) -> str:
        dev = self.train.device
        if dev == "auto":
            import torch

            dev = "cuda" if torch.cuda.is_available() else "cpu"
        return dev


def _apply(cfg: Config, raw: Dict[str, Any]) -> None:
    for section in ("model", "data", "train", "completion"):
        sub = raw.get(section)
        if isinstance(sub, dict):
            target = getattr(cfg, section)
            for key, value in sub.items():
                if hasattr(target, key):
                    setattr(target, key, value)


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Build a Config, optionally loading a YAML file and applying overrides."""
    cfg = Config()
    if path:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        _apply(cfg, raw)
    if overrides:
        _apply(cfg, overrides)
    return cfg
