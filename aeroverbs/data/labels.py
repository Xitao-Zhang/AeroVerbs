"""Label computation for waypoint and completion heads.

Waypoint labels are expressed in the *body frame* at time ``t`` (see
`maneuver_policy_network_v1.md` §7.3):

- x: current forward direction
- y: current left direction (``left = (-forward_y, forward_x)``)
- z: up

Only camera pose / forward direction are used to compute labels offline; these
values never enter the model.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..constants import SKILL_TO_ID


def skill_to_id(skill: str) -> int:
    return SKILL_TO_ID[skill]


def body_frame_basis(forward: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(body_x, body_y)`` unit basis from a 2D forward vector.

    ``body_y`` is the left direction, matching the ``left_xy`` definition used
    by the collection scripts (``left = (-fy, fx)``).
    """
    fx, fy = forward[0], forward[1]
    norm = np.hypot(fx, fy) + 1e-8
    fx, fy = fx / norm, fy / norm
    return np.array([fx, fy], dtype=np.float32), np.array([-fy, fx], dtype=np.float32)


def _to_body(delta_world: np.ndarray, body_x: np.ndarray, body_y: np.ndarray) -> np.ndarray:
    """Project a world-frame 3D delta into the body frame ``(x, y, z)``."""
    return np.array(
        [
            delta_world[0] * body_x[0] + delta_world[1] * body_x[1],
            delta_world[0] * body_y[0] + delta_world[1] * body_y[1],
            delta_world[2],
        ],
        dtype=np.float32,
    )


def _signed_angle_2d(a: np.ndarray, b: np.ndarray) -> float:
    """Signed angle from 2D unit vector ``a`` to ``b`` in radians."""
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    cross = float(a[0] * b[1] - a[1] * b[0])
    return float(np.arctan2(cross, dot))


def compute_action_labels(
    positions_world: np.ndarray,
    forwards: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute executed kinematic actions between adjacent frames.

    Args:
        positions_world: ``[T, 3]`` world positions.
        forwards: ``[T, 2]`` horizontal forward vectors per frame.

    Returns:
        ``actions [T, 4]``, ``action_valid [T]``.

    ``action[t]`` is the already-executed action from frame ``t-1`` to ``t``,
    expressed in the body frame of frame ``t-1``:
    ``[dx_forward, dy_left, dz_up, delta_yaw]``. ``action_valid[0]`` is 0
    because there is no previous frame.
    """
    T = positions_world.shape[0]
    actions = np.zeros((T, 4), dtype=np.float32)
    valid = np.zeros((T,), dtype=np.float32)
    if T < 2:
        return actions, valid

    # Normalize each forward row.
    fwd_norm = np.linalg.norm(forwards, axis=1, keepdims=True)
    fwd_unit = forwards / np.maximum(fwd_norm, 1e-8)

    for t in range(1, T):
        delta_world = positions_world[t] - positions_world[t - 1]
        body_x, body_y = body_frame_basis(fwd_unit[t - 1])
        actions[t, 0] = float(np.dot(delta_world[:2], body_x))
        actions[t, 1] = float(np.dot(delta_world[:2], body_y))
        actions[t, 2] = float(delta_world[2])
        actions[t, 3] = _signed_angle_2d(fwd_unit[t - 1], fwd_unit[t])
        valid[t] = 1.0
    return actions, valid


def build_action_history(
    actions: np.ndarray,
    action_valid: np.ndarray,
    history_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a right-aligned sliding window of past actions.

    Args:
        actions: ``[T, 4]`` from :func:`compute_action_labels`.
        action_valid: ``[T]`` valid mask for each action.
        history_len: number of recent actions to keep.

    Returns:
        ``history [T, H, 4]``, ``history_mask [T, H]``. The most recent action
        sits at index ``H - 1``; earlier entries are zero-padded on the left.
    """
    T = actions.shape[0]
    history = np.zeros((T, history_len, actions.shape[1]), dtype=np.float32)
    mask = np.zeros((T, history_len), dtype=np.float32)
    for t in range(T):
        start = max(0, t - history_len + 1)
        segment = actions[start : t + 1]
        valid_segment = action_valid[start : t + 1]
        # Right-align.
        offset = history_len - len(segment)
        history[t, offset:] = segment
        mask[t, offset:] = valid_segment
    return history, mask


def compute_waypoint_labels(
    positions_world: np.ndarray,
    forwards: np.ndarray,
    offsets: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute body-frame future waypoint labels for every frame.

    Args:
        positions_world: ``[T, 3]`` world positions.
        forwards: ``[T, 2]`` forward direction in world XY per frame.
        offsets: list of future frame offsets (default ``[3, 6, 10, 15, 20]``).

    Returns:
        ``(labels [T, K, 3], valid [T, K])``. ``valid`` is 0 where the future
        index runs past the end of the trajectory.
    """
    if offsets is None:
        offsets = [3, 6, 10, 15, 20]
    offsets = [int(o) for o in offsets]
    T = positions_world.shape[0]
    K = len(offsets)
    labels = np.zeros((T, K, 3), dtype=np.float32)
    valid = np.zeros((T, K), dtype=np.float32)
    for t in range(T):
        body_x, body_y = body_frame_basis(forwards[t])
        for k, off in enumerate(offsets):
            j = t + off
            if j >= T:
                continue
            delta = positions_world[j] - positions_world[t]
            labels[t, k] = _to_body(delta, body_x, body_y)
            valid[t, k] = 1.0
    return labels, valid


def compute_completion_labels(
    positions_world: np.ndarray,
    tail_frac: float = 0.2,
) -> np.ndarray:
    """Heuristic completion labels from arc-length progress.

    Marks frames in the final ``tail_frac`` of the trajectory's arc length as
    complete. This is a first-version placeholder; it should later be replaced
    by the privileged ``TrajectorySemanticJudge`` (which knows when a specific
    maneuver has actually finished).
    """
    T = positions_world.shape[0]
    if T == 1:
        return np.ones((T,), dtype=np.float32)
    seg = np.linalg.norm(np.diff(positions_world, axis=0), axis=1)
    cum = np.concatenate([np.array([0.0]), np.cumsum(seg)])
    total = cum[-1]
    frac = cum / max(total, 1e-6)
    return (frac >= (1.0 - float(tail_frac))).astype(np.float32)
