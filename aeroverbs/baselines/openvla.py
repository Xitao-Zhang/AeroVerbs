"""HTTP client for the OpenVLA-UAV Flask inference service.

The released service accepts a base64-encoded RGB image, a four-element UAV
state ``[x, y, z, yaw_degrees]``, and a natural-language instruction at its
``/predict`` endpoint. Positions are expressed in the UAV-Flow simulator's
centimetre-scale local frame; coordinate conversion from AeroVerbs' metre-scale
world frame belongs to the later rollout adapter, not this transport client.
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import requests
from PIL import Image


ImageInput = Union[Image.Image, np.ndarray, str, Path]


class OpenVLAClientError(RuntimeError):
    """Base error raised by the OpenVLA-UAV client."""


class OpenVLAConnectionError(OpenVLAClientError):
    """The inference service could not be reached or timed out."""


class OpenVLAResponseError(OpenVLAClientError):
    """The service returned an HTTP or payload-level error."""


@dataclass(frozen=True)
class OpenVLAPrediction:
    """Validated result returned by the released OpenVLA-UAV service."""

    action: np.ndarray
    action_ori: np.ndarray
    request_latency_s: float
    message: str

    def to_json_dict(self) -> dict:
        return {
            "status": "success",
            "action": self.action.tolist(),
            "action_ori": self.action_ori.tolist(),
            "request_latency_s": self.request_latency_s,
            "message": self.message,
        }


class OpenVLAClient:
    """Small synchronous client for ``openvla_act.py``'s ``/predict`` API."""

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:5007/predict",
        *,
        timeout_s: float = 120.0,
        image_size: tuple[int, int] = (224, 224),
        session: Optional[requests.Session] = None,
    ) -> None:
        if not server_url.startswith(("http://", "https://")):
            raise ValueError("server_url must start with http:// or https://")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if len(image_size) != 2 or min(image_size) <= 0:
            raise ValueError("image_size must contain two positive integers")
        self.server_url = server_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.session = session or requests.Session()

    def predict(
        self,
        image: ImageInput,
        proprio: Sequence[float],
        instruction: str,
    ) -> OpenVLAPrediction:
        """Send one observation and return validated absolute/raw actions.

        ``action`` is the next absolute pose in the local UAV-Flow frame after
        the server applies the current proprio. ``action_ori`` is the raw
        body-frame delta ``[dx, dy, dz, delta_yaw_rad]`` predicted by OpenVLA.
        """

        state = _validate_proprio(proprio)
        instruction = str(instruction).strip()
        if not instruction:
            raise ValueError("instruction must not be empty")
        payload = {
            "image": _encode_image(image, self.image_size),
            "proprio": state.tolist(),
            "instr": instruction,
        }

        started = time.perf_counter()
        try:
            response = self.session.post(
                self.server_url,
                json=payload,
                timeout=self.timeout_s,
            )
        except requests.RequestException as exc:
            raise OpenVLAConnectionError(
                f"failed to reach OpenVLA-UAV at {self.server_url}: {exc}"
            ) from exc
        latency = time.perf_counter() - started

        try:
            body = response.json()
        except ValueError as exc:
            snippet = response.text[:500]
            raise OpenVLAResponseError(
                f"OpenVLA-UAV returned non-JSON HTTP {response.status_code}: {snippet!r}"
            ) from exc
        if not response.ok:
            raise OpenVLAResponseError(
                f"OpenVLA-UAV returned HTTP {response.status_code}: "
                f"{_response_message(body)}"
            )
        return _parse_prediction(body, latency)

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "OpenVLAClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _validate_proprio(proprio: Sequence[float]) -> np.ndarray:
    state = np.asarray(proprio, dtype=np.float32)
    if state.shape != (4,) or not np.all(np.isfinite(state)):
        raise ValueError(
            "proprio must be finite [x, y, z, yaw_degrees] with shape (4,), "
            f"got {state}"
        )
    return state


def _load_rgb(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as loaded:
            return loaded.convert("RGB")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"image array must have shape HxWx3/4, got {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"image array must have dtype uint8, got {array.dtype}")
    return Image.fromarray(array[..., :3], mode="RGB")


def _encode_image(image: ImageInput, image_size: tuple[int, int]) -> str:
    rgb = _load_rgb(image)
    if rgb.size != image_size:
        rgb = rgb.resize(image_size)
    buffer = BytesIO()
    rgb.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _parse_prediction(
    body: Mapping[str, Any],
    request_latency_s: float,
) -> OpenVLAPrediction:
    if not isinstance(body, Mapping):
        raise OpenVLAResponseError(f"expected JSON object, got {type(body).__name__}")
    if body.get("status") != "success":
        raise OpenVLAResponseError(f"OpenVLA-UAV prediction failed: {_response_message(body)}")
    action = _single_action(body.get("action"), "action")
    action_ori = _single_action(body.get("action_ori"), "action_ori")
    return OpenVLAPrediction(
        action=action,
        action_ori=action_ori,
        request_latency_s=float(request_latency_s),
        message=str(body.get("message", "")),
    )


def _single_action(value: Any, field_name: str) -> np.ndarray:
    action = np.asarray(value, dtype=np.float32)
    if action.shape == (1, 4):
        action = action[0]
    if action.shape != (4,) or not np.all(np.isfinite(action)):
        raise OpenVLAResponseError(
            f"response {field_name} must be one finite four-element action, got {action}"
        )
    return action


def _response_message(body: Any) -> str:
    if isinstance(body, Mapping):
        return str(body.get("message", body))
    return str(body)


def _parse_proprio(text: str) -> list[float]:
    try:
        values = [float(value.strip()) for value in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "proprio must be comma-separated numbers: x,y,z,yaw_degrees"
        ) from exc
    if len(values) != 4:
        raise argparse.ArgumentTypeError(
            "proprio must contain exactly four values: x,y,z,yaw_degrees"
        )
    return values


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send one RGB/state/instruction request to OpenVLA-UAV."
    )
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:5007/predict",
    )
    image_source = parser.add_mutually_exclusive_group(required=True)
    image_source.add_argument("--image", help="RGB image path")
    image_source.add_argument(
        "--sample-dir",
        help="AeroVerbs sample directory; uses its initial_rgb.png",
    )
    parser.add_argument("--instruction", required=True)
    parser.add_argument(
        "--proprio",
        type=_parse_proprio,
        default=[0.0, 0.0, 0.0, 0.0],
        help="OpenVLA local state in centimetres/degrees: x,y,z,yaw",
    )
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    image_path = Path(args.image) if args.image else Path(args.sample_dir) / "initial_rgb.png"
    with OpenVLAClient(args.server_url, timeout_s=args.timeout_s) as client:
        prediction = client.predict(image_path, args.proprio, args.instruction)
    print(json.dumps(prediction.to_json_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
