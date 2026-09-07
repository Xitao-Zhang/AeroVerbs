"""Small client and coordinate adapter for the WorldVLN HTTP service."""

from __future__ import annotations

import base64
import io
import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class WorldVLNPrediction:
    actions_cm_deg: tuple[tuple[float, ...], ...]
    raw: Mapping[str, Any]
    latency_s: float


def encode_rgb(rgb: np.ndarray, quality: int = 90) -> str:
    array = np.asarray(rgb, dtype=np.uint8)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"RGB image must have HxWx3 shape, got {array.shape}")
    image = Image.fromarray(array, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality))
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class WorldVLNClient:
    def __init__(self, server_url: str, *, timeout_s: float = 600.0, session=None):
        self.url = server_url.rstrip("/") + "/v1/predict_delta_actions"
        self.timeout_s = float(timeout_s)
        self._session = session

    def predict(
        self,
        images: Sequence[np.ndarray],
        *,
        instruction: str | None = None,
        session_id: str,
        reset: bool = False,
        action_head_mode: str = "tsformer_latent",
        allow_future_segments: bool = True,
    ) -> WorldVLNPrediction:
        if not images:
            raise ValueError("at least one RGB image is required")
        payload: dict[str, Any] = {
            "session_id": str(session_id),
            "images_base64": [encode_rgb(image) for image in images],
            "allow_future_segments": bool(allow_future_segments),
            "action_head_mode": str(action_head_mode),
        }
        if instruction is not None:
            payload["instruction"] = str(instruction)
        if reset:
            payload["reset_session"] = True
        started = time.time()
        if self._session is not None:
            response = self._session.post(self.url, json=payload, timeout=self.timeout_s)
            response.raise_for_status()
            raw = response.json()
        else:
            import requests
            response = requests.post(self.url, json=payload, timeout=self.timeout_s)
            response.raise_for_status()
            raw = response.json()
        actions = raw.get("actions", [])
        if not isinstance(actions, list):
            raise ValueError(f"WorldVLN response actions must be a list, got {type(actions)}")
        parsed = []
        for action in actions:
            if not isinstance(action, (list, tuple)) or len(action) != 6:
                raise ValueError(f"WorldVLN action must be 6D, got {action!r}")
            values = tuple(float(value) for value in action)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"WorldVLN action contains non-finite values: {action!r}")
            parsed.append(values)
        return WorldVLNPrediction(tuple(parsed), raw, time.time() - started)


def apply_delta_body(
    pose_m_rad: Sequence[float],
    action_cm_deg: Sequence[float],
    *,
    fraction: float = 1.0,
) -> np.ndarray:
    """Apply WorldVLN [dx,dy,dz,droll,dyaw,dpitch] in body coordinates.

    AeroVerbs uses x=forward, y=left, z=up. WorldVLN's dy is treated as left,
    matching the visual-dataset contract. Translation is rotated by the current
    yaw, then yaw/roll/pitch are integrated in radians.
    """
    pose = np.asarray(pose_m_rad, dtype=np.float64)
    action = np.asarray(action_cm_deg, dtype=np.float64)
    if pose.shape != (6,) or action.shape != (6,):
        raise ValueError("pose and action must both have shape (6,)")
    if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(action)):
        raise ValueError("pose and action must be finite")
    scale = float(fraction)
    dx, dy, dz = action[:3] * (0.01 * scale)
    dyaw = math.radians(float(action[4]) * scale)
    pose[0] += dx * math.cos(pose[4]) - dy * math.sin(pose[4])
    pose[1] += dx * math.sin(pose[4]) + dy * math.cos(pose[4])
    pose[2] += dz
    pose[3] += math.radians(float(action[3]) * scale)
    pose[4] += dyaw
    pose[5] += math.radians(float(action[5]) * scale)
    return pose.astype(np.float32)
