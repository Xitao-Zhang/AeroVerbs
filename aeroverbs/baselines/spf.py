"""HTTP client for the SPF inference service."""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import requests
from PIL import Image


ImageInput = Union[Image.Image, np.ndarray, str, Path]


class SPFClientError(RuntimeError):
    pass


class SPFConnectionError(SPFClientError):
    pass


class SPFResponseError(SPFClientError):
    pass


@dataclass(frozen=True)
class SPFPrediction:
    action_spf: np.ndarray
    action_aeroverbs: np.ndarray
    screen_point: tuple[int, int]
    latency_s: float
    prompt_mode: str
    raw: Mapping[str, Any]


def _load_rgb(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        with Image.open(image) as loaded:
            return loaded.convert("RGB")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] not in (3, 4) or array.dtype != np.uint8:
        raise ValueError(f"image must be uint8 HxWx3/4, got {array.shape} {array.dtype}")
    return Image.fromarray(array[..., :3], mode="RGB")


def _encode_image(image: ImageInput) -> str:
    buffer = io.BytesIO()
    _load_rgb(image).save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _vector(body: Mapping[str, Any], key: str) -> np.ndarray:
    value = np.asarray(body.get(key), dtype=np.float32)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise SPFResponseError(f"response {key} must be a finite 3-vector, got {value}")
    return value


class SPFClient:
    def __init__(self, server_url: str = "http://127.0.0.1:5010", *, timeout_s: float = 300.0, session=None):
        if not server_url.startswith(("http://", "https://")):
            raise ValueError("server_url must start with http:// or https://")
        self.url = server_url.rstrip("/") + "/predict"
        self.timeout_s = float(timeout_s)
        self.session = session or requests.Session()

    def predict(self, image: ImageInput, instruction: str, *, prompt_mode: str = "original") -> SPFPrediction:
        instruction = str(instruction).strip()
        if not instruction:
            raise ValueError("instruction must not be empty")
        if prompt_mode not in {"original", "relation"}:
            raise ValueError("prompt_mode must be original or relation")
        payload = {"image": _encode_image(image), "instruction": instruction, "prompt_mode": prompt_mode}
        started = time.perf_counter()
        try:
            response = self.session.post(self.url, json=payload, timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise SPFConnectionError(f"failed to reach SPF at {self.url}: {exc}") from exc
        latency = time.perf_counter() - started
        try:
            body = response.json()
        except ValueError as exc:
            raise SPFResponseError(f"SPF returned non-JSON HTTP {response.status_code}") from exc
        if not response.ok or body.get("status") != "success":
            raise SPFResponseError(str(body.get("message", body)))
        point = body.get("screen_point")
        if not isinstance(point, list) or len(point) != 2:
            raise SPFResponseError(f"invalid screen_point: {point!r}")
        return SPFPrediction(
            action_spf=_vector(body, "action_spf"),
            action_aeroverbs=_vector(body, "action_aeroverbs"),
            screen_point=(int(point[0]), int(point[1])),
            latency_s=latency,
            prompt_mode=str(body.get("prompt_mode", prompt_mode)),
            raw=body,
        )

    def close(self) -> None:
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
