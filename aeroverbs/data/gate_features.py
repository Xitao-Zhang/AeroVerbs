"""Heatmap-gate geometry utilities for training and rollout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .labels import body_frame_basis


GATE_FEATURE_DIM = 12


def _unit(vector: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return arr / norm


@dataclass(frozen=True)
class GateGeometry:
    center: np.ndarray
    normal: np.ndarray
    left_axis: np.ndarray
    up_axis: np.ndarray
    width: float
    height: float
    confidence: float = 1.0

    @classmethod
    def from_summary(cls, payload: Mapping) -> "GateGeometry":
        return cls(
            center=np.asarray(payload["center_std"], dtype=np.float32),
            normal=_unit(payload["normal_std"]),
            left_axis=_unit(payload["left_axis_std"]),
            up_axis=_unit(payload["up_axis_std"]),
            width=float(payload.get("width_m", 0.0)),
            height=float(payload.get("height_m", 0.0)),
            confidence=float(payload.get("confidence", 1.0)),
        )

    @property
    def valid(self) -> float:
        return float(self.confidence > 0.0 and self.width > 0.0 and self.height > 0.0)


def body_basis_3d(forward_xy: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    body_x_xy, body_y_xy = body_frame_basis(np.asarray(forward_xy, dtype=np.float32)[:2])
    body_x = np.array([body_x_xy[0], body_x_xy[1], 0.0], dtype=np.float32)
    body_y = np.array([body_y_xy[0], body_y_xy[1], 0.0], dtype=np.float32)
    body_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return body_x, body_y, body_z


def world_vector_to_body(vector: Sequence[float], forward_xy: Sequence[float]) -> np.ndarray:
    body_x, body_y, body_z = body_basis_3d(forward_xy)
    vec = np.asarray(vector, dtype=np.float32)
    return np.array(
        [float(np.dot(vec, body_x)), float(np.dot(vec, body_y)), float(np.dot(vec, body_z))],
        dtype=np.float32,
    )


def body_waypoints_to_world(
    position: Sequence[float],
    forward_xy: Sequence[float],
    waypoints_body: np.ndarray,
) -> np.ndarray:
    body_x, body_y, body_z = body_basis_3d(forward_xy)
    wp = np.asarray(waypoints_body, dtype=np.float32).reshape(-1, 3)
    pos = np.asarray(position, dtype=np.float32)
    return pos[None, :] + wp[:, 0:1] * body_x + wp[:, 1:2] * body_y + wp[:, 2:3] * body_z


def gate_features_for_pose(
    gate: GateGeometry,
    position: Sequence[float],
    forward_xy: Sequence[float],
) -> np.ndarray:
    pos = np.asarray(position, dtype=np.float32)
    rel = gate.center - pos
    center_body = world_vector_to_body(rel, forward_xy)
    normal_body = world_vector_to_body(gate.normal, forward_xy)
    signed_distance = float(np.dot(pos - gate.center, gate.normal))
    distance = float(np.linalg.norm(rel))
    return np.asarray(
        [
            center_body[0],
            center_body[1],
            center_body[2],
            normal_body[0],
            normal_body[1],
            normal_body[2],
            signed_distance,
            distance,
            gate.width,
            gate.height,
            gate.confidence,
            gate.valid,
        ],
        dtype=np.float32,
    )


def gate_progress_label(
    gate: GateGeometry,
    position: Sequence[float],
    *,
    scale: float = 2.0,
) -> float:
    signed = float(np.dot(np.asarray(position, dtype=np.float32) - gate.center, gate.normal))
    return float(np.clip(0.5 + signed / max(float(scale), 1e-6), 0.0, 1.0))


def waypoints_cross_gate(
    gate: GateGeometry,
    position: Sequence[float],
    forward_xy: Sequence[float],
    waypoints_body: np.ndarray,
    *,
    margin: float = 0.0,
) -> bool:
    points = np.concatenate(
        [
            np.asarray(position, dtype=np.float32).reshape(1, 3),
            body_waypoints_to_world(position, forward_xy, waypoints_body),
        ],
        axis=0,
    )
    signed = (points - gate.center[None, :]) @ gate.normal
    for idx in range(len(points) - 1):
        s0 = float(signed[idx])
        s1 = float(signed[idx + 1])
        if s0 == s1 or s0 * s1 > 0.0:
            continue
        alpha = s0 / (s0 - s1)
        cross = points[idx] + alpha * (points[idx + 1] - points[idx])
        rel = cross - gate.center
        u = float(np.dot(rel, gate.left_axis))
        v = float(np.dot(rel, gate.up_axis))
        if abs(u) <= gate.width * 0.5 + margin and abs(v) <= gate.height * 0.5 + margin:
            return True
    return False
