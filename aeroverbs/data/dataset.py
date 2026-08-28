"""Dataset for sequence-level behavior cloning of AeroVerbs.

Each sample directory under ``data_root/<skill>/<scene_id>/`` contains a
``meta.json``, ``frames.json``, ``trajectory.json`` and a one-shot
``intention_heatmap.npy``. Frames are split into overlapping windows of length
``seq_len`` so the GRU can learn maneuver progress from the radar sequence.
"""

from __future__ import annotations

import functools
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import DataConfig
from .heatmap import bbox_to_heatmap
from .labels import (
    build_action_history,
    compute_action_labels,
    compute_completion_labels,
    compute_waypoint_labels,
    skill_to_id,
)
from .gate_features import GateGeometry, gate_features_for_pose, gate_progress_label
from .transforms import load_heatmap_npy, preprocess_radar


def build_sample_index(data_root: str, skills: List[str]) -> List[str]:
    """Return absolute sample directories for the given skills."""
    samples: List[str] = []
    for skill in skills:
        skill_dir = os.path.join(data_root, skill)
        if not os.path.isdir(skill_dir):
            continue
        for name in sorted(os.listdir(skill_dir)):
            d = os.path.join(skill_dir, name)
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "frames.json")):
                samples.append(d)
    return samples


def object_key(sample_dir: str) -> str:
    """Return a stable object identity for a sample directory.

    ``scene_id`` is unique per sample, but its trailing ``_NNNNN`` index
    identifies the object family (e.g. ``auto_around_left_pillar``,
    ``auto_through_door_opening``, ``auto_under_table_frl_apartment_table_01``).
    Splitting on this key keeps every object family represented in train, val
    and test, instead of leaking object geometry across splits.
    """
    meta = _load_meta(sample_dir)
    return re.sub(r"_\d+$", "", meta.get("scene_id", os.path.basename(sample_dir)))


def split_samples(
    samples: List[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 0,
) -> Tuple[List[str], List[str], List[str]]:
    """Stratified split into train / val / test by object family.

    Samples are grouped by :func:`object_key` (scene_id prefix), and each
    group is split by ratio so every object contributes some validation and
    test data. Groups too small to split (``< 3`` samples) go entirely to
    train. Returns ``(train, val, test)``.
    """
    rng = np.random.RandomState(seed)
    groups: Dict[str, List[str]] = defaultdict(list)
    for s in samples:
        groups[object_key(s)].append(s)

    test_frac = 1.0 - train_frac - val_frac
    train: List[str] = []
    val: List[str] = []
    test: List[str] = []
    for key in sorted(groups.keys()):
        group = groups[key]
        rng.shuffle(group)
        n = len(group)
        if n < 3:
            # Not enough samples to hold out any; keep entirely in train.
            train.extend(group)
            continue
        n_val = int(round(n * val_frac))
        n_test = int(round(n * test_frac))
        n_val = max(n_val, 1)
        n_test = max(n_test, 1)
        n_train = n - n_val - n_test
        if n_train < 1:
            # Degenerate guard: prioritize keeping one val/test each.
            n_val = 1
            n_test = 1
            n_train = n - 2
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])
    return train, val, test


@functools.lru_cache(maxsize=None)
def _load_meta(sample_dir: str) -> dict:
    with open(os.path.join(sample_dir, "meta.json"), "r", encoding="utf-8") as f:
        return json.load(f)


@functools.lru_cache(maxsize=None)
def _load_frames(sample_dir: str) -> dict:
    with open(os.path.join(sample_dir, "frames.json"), "r", encoding="utf-8") as f:
        frames = json.load(f)
    positions = np.array([fr["camera_position_std"] for fr in frames], dtype=np.float32)
    forwards = np.array([fr["camera_forward_std"][:2] for fr in frames], dtype=np.float32)
    # frames.json may store a sample-relative path (preferred, relocatable) or
    # an absolute path from the capture machine. Resolve the former against
    # the sample dir; use the latter as-is for backward compatibility.
    lidar_paths = []
    for fr in frames:
        raw = fr["lidar_range_npy"]
        lidar_paths.append(raw if os.path.isabs(raw) else os.path.join(sample_dir, raw))
    return {
        "positions": positions,  # [T, 3]
        "forwards": forwards,  # [T, 2]
        "lidar_paths": lidar_paths,  # [T]
        "num_frames": len(frames),
    }


@functools.lru_cache(maxsize=None)
def _load_skill_id(sample_dir: str) -> int:
    meta = _load_meta(sample_dir)
    skill = meta.get("skill") or os.path.basename(os.path.dirname(sample_dir))
    return skill_to_id(skill)


@functools.lru_cache(maxsize=None)
def _load_heatmap(sample_dir: str, size: Tuple[int, int]) -> torch.Tensor:
    path = os.path.join(sample_dir, "intention_heatmap.npy")
    if os.path.exists(path):
        return load_heatmap_npy(path, size=size)
    # Fallback: rebuild from the stored bbox when the npy is missing.
    meta = _load_meta(sample_dir)
    bbox = meta.get("initial_bbox_xyxy")
    image_size = tuple(meta.get("image_size", [640, 480]))
    skill = meta.get("skill", "through")
    hm = bbox_to_heatmap(bbox, image_size=image_size, skill=skill)
    from .transforms import resize_heatmap

    return resize_heatmap(torch.from_numpy(hm), size=size)


@functools.lru_cache(maxsize=None)
def _load_waypoint_labels(
    sample_dir: str, offsets_tuple: Tuple[int, ...]
) -> Tuple[np.ndarray, np.ndarray]:
    frames = _load_frames(sample_dir)
    return compute_waypoint_labels(
        frames["positions"], frames["forwards"], offsets=list(offsets_tuple)
    )


@functools.lru_cache(maxsize=None)
def _load_completion_labels(sample_dir: str, tail_frac: float) -> np.ndarray:
    frames = _load_frames(sample_dir)
    return compute_completion_labels(frames["positions"], tail_frac=tail_frac)


@functools.lru_cache(maxsize=None)
def _load_action_history(
    sample_dir: str, history_len: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = _load_frames(sample_dir)
    actions, action_valid = compute_action_labels(
        frames["positions"], frames["forwards"]
    )
    history, history_mask = build_action_history(actions, action_valid, history_len)
    return actions, history, history_mask


@functools.lru_cache(maxsize=None)
def _load_downward_range(sample_dir: str, output_name: str) -> np.ndarray:
    path = os.path.join(sample_dir, output_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing downward range file: {path}")
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected downward range shape [T,2], got {arr.shape}: {path}")
    frames = _load_frames(sample_dir)
    if arr.shape[0] != frames["num_frames"]:
        raise ValueError(
            f"downward range length {arr.shape[0]} does not match frames "
            f"{frames['num_frames']}: {path}"
        )
    return arr


@functools.lru_cache(maxsize=None)
def _load_gate_geometry(sample_dir: str, summary_name: str) -> GateGeometry:
    path = os.path.join(sample_dir, summary_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing heatmap gate summary: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return GateGeometry.from_summary(json.load(f))


@functools.lru_cache(maxsize=None)
def _load_gate_features_and_progress(
    sample_dir: str,
    summary_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    frames = _load_frames(sample_dir)
    gate = _load_gate_geometry(sample_dir, summary_name)
    positions = frames["positions"]
    forwards = frames["forwards"]
    features = np.stack(
        [gate_features_for_pose(gate, positions[t], forwards[t]) for t in range(len(positions))],
        axis=0,
    ).astype(np.float32)
    progress = np.asarray(
        [gate_progress_label(gate, positions[t]) for t in range(len(positions))],
        dtype=np.float32,
    )
    return features, progress


def _has_teacher_labels(sample_dir: str) -> bool:
    return os.path.exists(os.path.join(sample_dir, "teacher_waypoints.npy"))


def _sample_or_pad_points(points: np.ndarray, count: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return a deterministic fixed-size point set plus a validity mask."""
    count = int(count)
    out = np.zeros((count, 3), dtype=np.float32)
    mask = np.zeros((count,), dtype=np.float32)
    if count <= 0:
        return out, mask
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0:
        return out, mask
    if len(points) > count:
        idx = np.linspace(0, len(points) - 1, count).round().astype(np.int64)
        points = points[idx]
    out[: len(points)] = points
    mask[: len(points)] = 1.0
    return out, mask


@functools.lru_cache(maxsize=None)
def _load_object_sphere(sample_dir: str) -> Tuple[np.ndarray, np.float32]:
    meta = _load_meta(sample_dir)
    objects = meta.get("objects") or []
    obj = objects[0] if objects else {}
    bbox_min = np.asarray(obj.get("bbox_min", meta.get("bbox_min", [0.0, 0.0, 0.0])), dtype=np.float32)
    bbox_max = np.asarray(obj.get("bbox_max", meta.get("bbox_max", [0.0, 0.0, 0.0])), dtype=np.float32)
    center = np.asarray(obj.get("center", (bbox_min + bbox_max) * 0.5), dtype=np.float32)
    radius = obj.get("radius", None)
    if radius is None:
        radius = 0.5 * float(np.linalg.norm(bbox_max - bbox_min))
    return center.astype(np.float32), np.float32(radius)


def _scene_dir_for_sample(sample_dir: str) -> str:
    meta = _load_meta(sample_dir)
    skill = meta.get("skill") or os.path.basename(os.path.dirname(sample_dir))
    scene_id = meta.get("scene_id") or os.path.basename(sample_dir)
    data_root = os.path.dirname(os.path.dirname(sample_dir))
    return os.path.join(data_root, "_scenes", skill, scene_id)


def _resolve_existing_occupancy_path(sample_dir: str) -> Optional[str]:
    meta = _load_meta(sample_dir)
    for obj in meta.get("objects") or []:
        raw = obj.get("voxel_occupancy_path")
        if raw and os.path.exists(raw):
            return raw
    occ_dir = os.path.join(_scene_dir_for_sample(sample_dir), "occupancy")
    if os.path.isdir(occ_dir):
        for name in sorted(os.listdir(occ_dir)):
            if name.endswith(".npz"):
                return os.path.join(occ_dir, name)
    return None


@functools.lru_cache(maxsize=None)
def _load_occupied_points(sample_dir: str, count: int) -> Tuple[np.ndarray, np.ndarray]:
    path = _resolve_existing_occupancy_path(sample_dir)
    if path is None:
        return _sample_or_pad_points(np.zeros((0, 3), dtype=np.float32), count)
    data = np.load(path)
    if "occupied_indices" in data and "origin" in data and "voxel_size" in data:
        indices = data["occupied_indices"].astype(np.float32)
        origin = data["origin"].astype(np.float32).reshape(1, 3)
        voxel_size = float(np.asarray(data["voxel_size"]).reshape(-1)[0])
        points = origin + (indices + 0.5) * voxel_size
    elif "points" in data:
        points = data["points"].astype(np.float32)
    else:
        points = np.zeros((0, 3), dtype=np.float32)
    return _sample_or_pad_points(points, count)


@functools.lru_cache(maxsize=None)
def _load_region_points(
    sample_dir: str,
    skill: str,
    count: int,
    gate_region_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    candidates = [
        os.path.join(sample_dir, f"semantic_region_{skill}.json"),
        os.path.join(sample_dir, gate_region_name),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        points = raw.get("points", raw if isinstance(raw, list) else [])
        return _sample_or_pad_points(np.asarray(points, dtype=np.float32), count)
    return _sample_or_pad_points(np.zeros((0, 3), dtype=np.float32), count)


def _load_clearance_targets(sample_dir: str, cfg: DataConfig) -> Dict[str, torch.Tensor]:
    meta = _load_meta(sample_dir)
    skill = meta.get("skill") or os.path.basename(os.path.dirname(sample_dir))
    center, radius = _load_object_sphere(sample_dir)
    occupied, occupied_mask = _load_occupied_points(sample_dir, cfg.clearance_occupied_points)
    region, region_mask = _load_region_points(
        sample_dir,
        skill,
        cfg.clearance_region_points,
        cfg.gate_region_name,
    )
    return {
        "object_center": torch.from_numpy(center),
        "object_radius": torch.tensor(radius, dtype=torch.float32),
        "occupied_points": torch.from_numpy(occupied),
        "occupied_mask": torch.from_numpy(occupied_mask),
        "free_region_points": torch.from_numpy(region),
        "free_region_mask": torch.from_numpy(region_mask),
    }


@functools.lru_cache(maxsize=None)
def _load_teacher_labels(
    sample_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    wp = np.load(os.path.join(sample_dir, "teacher_waypoints.npy"))
    wp_valid = np.load(os.path.join(sample_dir, "teacher_waypoint_valid.npy"))
    completion = np.load(os.path.join(sample_dir, "teacher_completion.npy")).reshape(-1)
    return wp, wp_valid, completion


class ManeuverDataset(Dataset):
    def __init__(self, sample_dirs: List[str], cfg: DataConfig) -> None:
        self.sample_dirs = list(sample_dirs)
        self.cfg = cfg
        self.seq_len = cfg.seq_len
        self.stride = cfg.stride
        self.offsets = tuple(cfg.future_offsets)
        self.heatmap_size = tuple(cfg.heatmap_size)

        self.index: List[Tuple[str, int]] = []
        for d in self.sample_dirs:
            T = _load_frames(d)["num_frames"]
            if T < self.seq_len:
                continue
            last = T - self.seq_len
            starts = list(range(0, last + 1, self.stride))
            if starts[-1] != last:
                starts.append(last)
            for s in starts:
                self.index.append((d, s))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_dir, start = self.index[idx]
        frames = _load_frames(sample_dir)
        positions = frames["positions"]
        forwards = frames["forwards"]
        lidar_paths = frames["lidar_paths"]

        end = start + self.seq_len
        radar_frames = []
        for path in lidar_paths[start:end]:
            raw = torch.from_numpy(np.load(path).astype(np.float32))
            radar_frames.append(
                preprocess_radar(
                    raw,
                    r_max=self.cfg.r_max,
                    fov_deg=tuple(self.cfg.fov_deg) if self.cfg.fov_deg else None,
                )
            )
        radar = torch.stack(radar_frames, dim=0)  # [T, 3, H, W]

        if _has_teacher_labels(sample_dir):
            wp, wp_valid, completion = _load_teacher_labels(sample_dir)
        else:
            wp, wp_valid = _load_waypoint_labels(sample_dir, self.offsets)
            completion = _load_completion_labels(sample_dir, self.cfg.completion_tail_frac)
        actions, action_history, action_history_mask = _load_action_history(
            sample_dir, self.cfg.action_history_len
        )
        downward_range = _load_downward_range(sample_dir, self.cfg.downward_range_name)
        gate_features, progress = _load_gate_features_and_progress(
            sample_dir, self.cfg.gate_summary_name
        )

        out = {
            "heatmap": _load_heatmap(sample_dir, self.heatmap_size),  # [1, Hh, Wh]
            "skill_id": torch.tensor(_load_skill_id(sample_dir), dtype=torch.long),
            "radar": radar,
            "action_history": torch.from_numpy(action_history[start:end]),  # [T, H, 4]
            "action_history_mask": torch.from_numpy(action_history_mask[start:end]),  # [T, H]
            "actions": torch.from_numpy(actions[start:end]),  # [T, 4]
            "downward_range": torch.from_numpy(downward_range[start:end]),  # [T, 2]
            "gate_features": torch.from_numpy(gate_features[start:end]),  # [T, Dg]
            "progress": torch.from_numpy(progress[start:end]).unsqueeze(-1),  # [T, 1]
            "positions": torch.from_numpy(positions[start:end]),  # [T, 3]
            "forwards": torch.from_numpy(forwards[start:end]),  # [T, 2]
            "waypoints": torch.from_numpy(wp[start:end]),  # [T, K, 3]
            "waypoint_valid": torch.from_numpy(wp_valid[start:end]),  # [T, K]
            "completion": torch.from_numpy(completion[start:end]).unsqueeze(-1),  # [T, 1]
        }
        out.update(_load_clearance_targets(sample_dir, self.cfg))
        return out


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in batch[0].keys():
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    return out


class TrajectoryDataset(Dataset):
    """One sample = one complete trajectory, for trajectory-level GRU training.

    Returns the same keys as :class:`ManeuverDataset` but with the full time
    axis ``[T, ...]`` instead of a fixed-length window.
    """

    def __init__(self, sample_dirs: List[str], cfg: DataConfig) -> None:
        self.sample_dirs = list(sample_dirs)
        self.cfg = cfg
        self.offsets = tuple(cfg.future_offsets)
        self.heatmap_size = tuple(cfg.heatmap_size)
        self.lengths = [_load_frames(d)["num_frames"] for d in self.sample_dirs]

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_dir = self.sample_dirs[idx]
        frames = _load_frames(sample_dir)
        lidar_paths = frames["lidar_paths"]

        radar_frames = []
        for path in lidar_paths:
            raw = torch.from_numpy(np.load(path).astype(np.float32))
            radar_frames.append(
                preprocess_radar(
                    raw,
                    r_max=self.cfg.r_max,
                    fov_deg=tuple(self.cfg.fov_deg) if self.cfg.fov_deg else None,
                )
            )
        radar = torch.stack(radar_frames, dim=0)  # [T, 3, H, W]

        if _has_teacher_labels(sample_dir):
            wp, wp_valid, completion = _load_teacher_labels(sample_dir)
        else:
            wp, wp_valid = _load_waypoint_labels(sample_dir, self.offsets)
            completion = _load_completion_labels(sample_dir, self.cfg.completion_tail_frac)
        actions, action_history, action_history_mask = _load_action_history(
            sample_dir, self.cfg.action_history_len
        )
        downward_range = _load_downward_range(sample_dir, self.cfg.downward_range_name)
        gate_features, progress = _load_gate_features_and_progress(
            sample_dir, self.cfg.gate_summary_name
        )

        out = {
            "sample_dir": sample_dir,
            "heatmap": _load_heatmap(sample_dir, self.heatmap_size),  # [1, Hh, Wh]
            "skill_id": torch.tensor(_load_skill_id(sample_dir), dtype=torch.long),
            "radar": radar,
            "action_history": torch.from_numpy(action_history),  # [T, H, 4]
            "action_history_mask": torch.from_numpy(action_history_mask),  # [T, H]
            "actions": torch.from_numpy(actions),  # [T, 4]
            "downward_range": torch.from_numpy(downward_range),  # [T, 2]
            "gate_features": torch.from_numpy(gate_features),  # [T, Dg]
            "progress": torch.from_numpy(progress).unsqueeze(-1),  # [T, 1]
            "positions": torch.from_numpy(frames["positions"]),  # [T, 3]
            "forwards": torch.from_numpy(frames["forwards"]),  # [T, 2]
            "waypoints": torch.from_numpy(wp),  # [T, K, 3]
            "waypoint_valid": torch.from_numpy(wp_valid),  # [T, K]
            "completion": torch.from_numpy(completion).unsqueeze(-1),  # [T, 1]
        }
        out.update(_load_clearance_targets(sample_dir, self.cfg))
        return out


def _pad_time_tensor(t: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad a ``[T, ...]`` tensor with zeros along the time axis."""
    if t.shape[0] == target_len:
        return t
    pad_shape = (target_len - t.shape[0], *t.shape[1:])
    return torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype)], dim=0)


def trajectory_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate complete trajectories of different lengths into one batch.

    All time axes are padded to ``T_max`` and a ``time_mask`` marks valid
    frames so padded positions do not contribute to the loss.
    """
    t_max = max(int(item["radar"].shape[0]) for item in batch)

    def stack_time(key):
        return torch.stack([_pad_time_tensor(item[key], t_max) for item in batch], dim=0)

    time_mask = torch.stack(
        [
            torch.cat(
                [
                    torch.ones(item["radar"].shape[0], dtype=torch.float32),
                    torch.zeros(t_max - item["radar"].shape[0], dtype=torch.float32),
                ]
            )
            for item in batch
        ],
        dim=0,
    )  # [B, T_max]

    return {
        "sample_dir": [item["sample_dir"] for item in batch],
        "heatmap": torch.stack([item["heatmap"] for item in batch], dim=0),  # [B,1,Hh,Wh]
        "skill_id": torch.stack([item["skill_id"] for item in batch], dim=0),  # [B]
        "radar": stack_time("radar"),  # [B,Tmax,3,H,W]
        "action_history": stack_time("action_history"),  # [B,Tmax,H,4]
        "action_history_mask": stack_time("action_history_mask"),  # [B,Tmax,H]
        "actions": stack_time("actions"),  # [B,Tmax,4]
        "downward_range": stack_time("downward_range"),  # [B,Tmax,2]
        "gate_features": stack_time("gate_features"),  # [B,Tmax,Dg]
        "progress": stack_time("progress"),  # [B,Tmax,1]
        "positions": stack_time("positions"),  # [B,Tmax,3]
        "forwards": stack_time("forwards"),  # [B,Tmax,2]
        "waypoints": stack_time("waypoints"),  # [B,Tmax,K,3]
        "waypoint_valid": stack_time("waypoint_valid"),  # [B,Tmax,K]
        "completion": stack_time("completion"),  # [B,Tmax,1]
        "object_center": torch.stack([item["object_center"] for item in batch], dim=0),  # [B,3]
        "object_radius": torch.stack([item["object_radius"] for item in batch], dim=0),  # [B]
        "occupied_points": torch.stack([item["occupied_points"] for item in batch], dim=0),  # [B,N,3]
        "occupied_mask": torch.stack([item["occupied_mask"] for item in batch], dim=0),  # [B,N]
        "free_region_points": torch.stack([item["free_region_points"] for item in batch], dim=0),  # [B,M,3]
        "free_region_mask": torch.stack([item["free_region_mask"] for item in batch], dim=0),  # [B,M]
        "time_mask": time_mask,  # [B,Tmax]
    }
