"""AerialClaw baseline bridge for AeroVerbs.

This module keeps AeroVerbs authoritative over geometry, physical stepping,
collision detection, and semantic completion.  Only simulator-style state and
sensors are exposed to AerialClaw.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import base64
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import queue
import threading
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Union
from urllib.parse import parse_qs, urlparse

import numpy as np


def _unit_horizontal(vector: Sequence[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(3).copy()
    value[2] = 0.0
    norm = float(np.linalg.norm(value))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return value / norm


class AerialClawLocalFrame:
    """Episode-local NED frame anchored at the first AeroVerbs pose."""

    def __init__(
        self,
        origin_world: Sequence[float],
        initial_forward_world: Sequence[float],
    ) -> None:
        self.origin_world = np.asarray(origin_world, dtype=float).reshape(3)
        self.north_world = _unit_horizontal(initial_forward_world)
        self.east_world = np.array(
            [self.north_world[1], -self.north_world[0], 0.0], dtype=float
        )

    def world_to_ned(self, point_world: Sequence[float]) -> np.ndarray:
        relative = np.asarray(point_world, dtype=float).reshape(3) - self.origin_world
        return np.array(
            [
                float(np.dot(relative, self.north_world)),
                float(np.dot(relative, self.east_world)),
                -float(relative[2]),
            ],
            dtype=float,
        )

    def ned_to_world(self, point_ned: Sequence[float]) -> np.ndarray:
        north, east, down = np.asarray(point_ned, dtype=float).reshape(3)
        return (
            self.origin_world
            + north * self.north_world
            + east * self.east_world
            + np.array([0.0, 0.0, -down], dtype=float)
        )

    def heading_deg(self, forward_world: Sequence[float]) -> float:
        forward = _unit_horizontal(forward_world)
        north = float(np.dot(forward, self.north_world))
        east = float(np.dot(forward, self.east_world))
        return float(math.degrees(math.atan2(east, north)) % 360.0)


def radar_to_lidar_scan(
    range_image: np.ndarray,
    pitch_deg: Sequence[float],
    yaw_deg: Optional[Sequence[float]] = None,
    *,
    target_count: int = 360,
    range_min: float = 0.1,
    range_max: float = 10.0,
) -> Dict[str, Any]:
    """Convert an AeroVerbs radar frame into increasing-angle planar LiDAR."""
    radar = np.asarray(range_image, dtype=float)
    pitches = np.asarray(pitch_deg, dtype=float).reshape(-1)
    if radar.ndim != 2 or radar.shape[0] != pitches.size:
        raise ValueError("range_image rows must match pitch_deg")
    if target_count <= 0:
        raise ValueError("target_count must be positive")

    if yaw_deg is None:
        yaws = np.linspace(180.0, -180.0, radar.shape[1], endpoint=False)
    else:
        yaws = np.asarray(yaw_deg, dtype=float).reshape(-1)
    if radar.shape[1] != yaws.size:
        raise ValueError("range_image columns must match yaw_deg")

    row = radar[int(np.argmin(np.abs(pitches)))].astype(float)
    source_angles = np.deg2rad(yaws)
    source_angles = (source_angles + np.pi) % (2.0 * np.pi) - np.pi
    order = np.argsort(source_angles)
    source_angles = source_angles[order]
    row = row[order]

    # Duplicate one revolution on either side so interpolation is circular.
    extended_angles = np.concatenate(
        [source_angles[-1:] - 2.0 * np.pi, source_angles, source_angles[:1] + 2.0 * np.pi]
    )
    extended_ranges = np.concatenate([row[-1:], row, row[:1]])
    target_angles = np.linspace(-np.pi, np.pi, target_count, endpoint=False)
    ranges = np.interp(target_angles, extended_angles, extended_ranges)
    ranges = np.nan_to_num(ranges, nan=range_max, posinf=range_max, neginf=range_min)
    ranges = np.clip(ranges, range_min, range_max)

    return {
        "ranges": ranges.astype(float).tolist(),
        "angle_min": -math.pi,
        "angle_max": math.pi,
        "angle_increment": 2.0 * math.pi / target_count,
        "range_min": float(range_min),
        "range_max": float(range_max),
        "count": int(target_count),
        "vertical_count": 1,
    }


@dataclass(frozen=True)
class EpisodeConfig:
    max_steps: int = 200
    step_size: float = 0.1
    turn_rate: float = 0.45
    drone_radius: float = 0.15
    ground_clearance: float = 0.2
    hover_hz: float = 10.0
    target_tolerance: float = 0.05

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        for name in ("step_size", "turn_rate", "drone_radius", "hover_hz"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


class EpisodeController:
    """Expand macro actions into authoritative AeroVerbs physical steps."""

    def __init__(
        self,
        *,
        initial_position_world: Sequence[float],
        initial_forward_world: Sequence[float],
        clearance_fn: Callable[[np.ndarray], float],
        completion_fn: Callable[[np.ndarray, np.ndarray], Any],
        config: Optional[EpisodeConfig] = None,
    ) -> None:
        self.config = config or EpisodeConfig()
        self.position_world = np.asarray(initial_position_world, dtype=float).reshape(3)
        self.forward_world = _unit_horizontal(initial_forward_world)
        self.frame = AerialClawLocalFrame(self.position_world, self.forward_world)
        self.clearance_fn = clearance_fn
        self.completion_fn = completion_fn
        self.points = [self.position_world.astype(float).tolist()]
        self.actions = []
        self.physical_steps = 0
        self.min_clearance = math.inf
        self.last_completion: Optional[Any] = None
        self.termination: Dict[str, Any] = {
            "terminal": False,
            "success": False,
            "status": "running",
            "reason": "",
        }

    def _terminate(self, status: str, reason: str, *, success: bool = False) -> None:
        if self.termination["terminal"]:
            return
        self.termination = {
            "terminal": True,
            "success": bool(success),
            "status": status,
            "reason": reason,
        }

    def _turn_towards(self, horizontal_direction: np.ndarray) -> None:
        desired = _unit_horizontal(horizontal_direction)
        current = self.forward_world
        cross_z = float(current[0] * desired[1] - current[1] * desired[0])
        dot = float(np.clip(np.dot(current, desired), -1.0, 1.0))
        angle = math.atan2(cross_z, dot)
        bounded = float(np.clip(angle, -self.config.turn_rate, self.config.turn_rate))
        cosine, sine = math.cos(bounded), math.sin(bounded)
        self.forward_world = np.array(
            [
                cosine * current[0] - sine * current[1],
                sine * current[0] + cosine * current[1],
                0.0,
            ],
            dtype=float,
        )

    def _record_step(self, candidate: np.ndarray) -> bool:
        self.physical_steps += 1
        self.position_world = np.asarray(candidate, dtype=float).reshape(3)
        self.points.append(self.position_world.astype(float).tolist())

        clearance = float(self.clearance_fn(self.position_world.copy()))
        if math.isfinite(clearance):
            self.min_clearance = min(self.min_clearance, clearance)
        if self.position_world[2] < self.config.ground_clearance:
            self._terminate("collision", "ground_clearance_violation")
            return False
        if clearance < self.config.drone_radius:
            self._terminate("collision", "object_clearance_violation")
            return False

        self.last_completion = self.completion_fn(
            np.asarray(self.points, dtype=float), self.position_world.copy()
        )
        if bool(getattr(self.last_completion, "completion", False)):
            self._terminate("success", "completion_judge", success=True)
            return False
        if self.physical_steps >= self.config.max_steps:
            self._terminate("timeout", "physical_step_budget_exhausted")
            return False
        return True

    def fly_to_ned(
        self, target_ned: Sequence[float], speed: Optional[float] = None
    ) -> Dict[str, Any]:
        if self.termination["terminal"]:
            return self._action_result(False, "episode is already terminal")

        target_ned_array = np.asarray(target_ned, dtype=float).reshape(3)
        target_world = self.frame.ned_to_world(target_ned_array)
        action = {
            "type": "fly_to_ned",
            "target_ned": target_ned_array.astype(float).tolist(),
            "target_world": target_world.astype(float).tolist(),
            "speed": None if speed is None else float(speed),
            "start_physical_step": self.physical_steps,
        }
        self.actions.append(action)

        reached = False
        while not self.termination["terminal"]:
            delta = target_world - self.position_world
            distance = float(np.linalg.norm(delta))
            if distance <= self.config.target_tolerance:
                reached = True
                break
            if self.physical_steps >= self.config.max_steps:
                self._terminate("timeout", "physical_step_budget_exhausted")
                break

            horizontal = delta.copy()
            horizontal[2] = 0.0
            horizontal_distance = float(np.linalg.norm(horizontal))
            if horizontal_distance > 1e-8:
                self._turn_towards(horizontal)

            travel = min(self.config.step_size, distance)
            if horizontal_distance <= 1e-8:
                direction = np.array([0.0, 0.0, math.copysign(1.0, delta[2])])
            else:
                vertical_ratio = float(delta[2] / max(distance, 1e-8))
                horizontal_ratio = math.sqrt(max(0.0, 1.0 - vertical_ratio**2))
                direction = self.forward_world * horizontal_ratio
                direction[2] = vertical_ratio
            if not self._record_step(self.position_world + travel * direction):
                break

        action["end_physical_step"] = self.physical_steps
        action["status"] = self.termination["status"] if self.termination["terminal"] else "reached"
        if self.termination["success"]:
            return self._action_result(True, "CompletionJudge accepted the trajectory")
        if self.termination["terminal"]:
            return self._action_result(False, self.termination["reason"])
        return self._action_result(reached, "target reached" if reached else "target not reached")

    def hover(self, duration: float = 1.0) -> Dict[str, Any]:
        if self.termination["terminal"]:
            return self._action_result(False, "episode is already terminal")
        ticks = max(1, int(math.ceil(max(0.0, float(duration)) * self.config.hover_hz)))
        action = {
            "type": "hover",
            "duration": float(duration),
            "start_physical_step": self.physical_steps,
        }
        self.actions.append(action)
        for _ in range(ticks):
            if self.physical_steps >= self.config.max_steps:
                self._terminate("timeout", "physical_step_budget_exhausted")
                break
            if not self._record_step(self.position_world.copy()):
                break
        action["end_physical_step"] = self.physical_steps
        action["status"] = self.termination["status"] if self.termination["terminal"] else "reached"
        return self._action_result(
            not self.termination["terminal"] or self.termination["success"],
            self.termination["reason"] if self.termination["terminal"] else "hover complete",
        )

    def declare_agent_done(self) -> Dict[str, Any]:
        if not self.termination["terminal"]:
            self._terminate(
                "timeout", "agent_declared_done_before_semantic_completion"
            )
        return dict(self.termination)

    def state(self) -> Dict[str, Any]:
        north, east, down = self.frame.world_to_ned(self.position_world)
        return {
            "position": {"north": north, "east": east, "down": down},
            "heading": self.frame.heading_deg(self.forward_world),
            "battery": 100.0,
            "armed": True,
            "in_air": True,
            "connected": True,
            "physical_steps": self.physical_steps,
        }

    def result(self) -> Dict[str, Any]:
        completion = self.last_completion
        completion_fields: Dict[str, Any] = {}
        if completion is not None:
            if hasattr(completion, "__dataclass_fields__"):
                completion_fields = asdict(completion)
            else:
                for key in (
                    "semantic_done",
                    "far_side",
                    "clearance",
                    "safe_clearance",
                    "ground_ok",
                    "sustained",
                    "completion",
                ):
                    if hasattr(completion, key):
                        value = getattr(completion, key)
                        completion_fields[key] = value.item() if hasattr(value, "item") else value
        return {
            **self.termination,
            **completion_fields,
            "steps": self.physical_steps,
            "min_clearance": None if math.isinf(self.min_clearance) else self.min_clearance,
            "points": list(self.points),
            "aerialclaw_actions": list(self.actions),
        }

    def _action_result(self, success: bool, message: str) -> Dict[str, Any]:
        return {
            "success": bool(success),
            "message": str(message),
            "termination": dict(self.termination),
            "state": self.state(),
        }


class HabitatRGBRenderWorker:
    """Own one Habitat renderer and all of its calls on a single thread.

    Magnum/OpenGL contexts are thread-affine.  The AerialClaw bridge uses a
    ``ThreadingHTTPServer``, so creating or reusing the renderer directly from
    request handlers can move the context between threads and crash the whole
    Python process.  This worker keeps construction, rendering, and teardown
    on one dedicated owner thread while HTTP handlers exchange plain arrays
    through futures.
    """

    _STOP = object()

    def __init__(
        self,
        scene_path: Union[str, Path],
        *,
        width: int,
        height: int,
        gpu_device_id: int,
        renderer_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.scene_path = Path(scene_path)
        self.width = int(width)
        self.height = int(height)
        self.gpu_device_id = int(gpu_device_id)
        self._renderer_factory = renderer_factory
        self._jobs = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._lifecycle_lock = threading.Lock()

    def start(self) -> "HabitatRGBRenderWorker":
        with self._lifecycle_lock:
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run,
                    name="aeroverbs-habitat-rgb-renderer",
                    daemon=True,
                )
                self._thread.start()
        return self

    def render(
        self,
        position_std: Sequence[float],
        forward_std: Sequence[float],
    ) -> np.ndarray:
        self.start()
        future = Future()
        self._jobs.put(
            (
                future,
                np.asarray(position_std, dtype=np.float32).copy(),
                np.asarray(forward_std, dtype=np.float32).copy(),
            )
        )
        return future.result()

    def close(self) -> None:
        with self._lifecycle_lock:
            thread = self._thread
            if thread is None:
                return
            self._jobs.put(self._STOP)
            thread.join(timeout=30.0)
            if thread.is_alive():
                raise RuntimeError("Habitat RGB render worker did not stop")
            self._thread = None

    def _make_renderer(self):
        if self._renderer_factory is None:
            from aeroverbs.env.rgb_renderer import HabitatRGBRenderer

            renderer_factory = HabitatRGBRenderer
        else:
            renderer_factory = self._renderer_factory
        return renderer_factory(
            self.scene_path,
            width=self.width,
            height=self.height,
            gpu_device_id=self.gpu_device_id,
        )

    def _run(self) -> None:
        renderer = None
        try:
            while True:
                job = self._jobs.get()
                if job is self._STOP:
                    break
                future, position_std, forward_std = job
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    if renderer is None:
                        renderer = self._make_renderer()
                        renderer.start()
                    future.set_result(renderer.render(position_std, forward_std))
                except BaseException as exc:
                    future.set_exception(exc)
        finally:
            if renderer is not None:
                renderer.close()


class AeroVerbsEpisode:
    """Scene-backed episode exposing only AerialClaw-safe observations."""

    _CAMERA_DIRECTIONS = {"front", "rear", "left", "right", "down"}

    def __init__(
        self,
        *,
        sample_dir: Path,
        meta: Mapping[str, Any],
        controller: EpisodeController,
        radar_mesh: Any,
        render_scene_path: Path,
        rgb_width: int = 640,
        rgb_height: int = 480,
        rgb_gpu_device_id: int = -1,
        radar_height: int = 40,
        radar_width: int = 500,
        radar_pitch_min: float = -5.0,
        radar_pitch_max: float = 35.0,
        radar_max_depth: float = 10.0,
    ) -> None:
        self.sample_dir = Path(sample_dir)
        self.meta = dict(meta)
        self.controller = controller
        self.radar_mesh = radar_mesh
        self.render_scene_path = Path(render_scene_path)
        self.rgb_width = int(rgb_width)
        self.rgb_height = int(rgb_height)
        self.rgb_gpu_device_id = int(rgb_gpu_device_id)
        self.radar_height = int(radar_height)
        self.radar_width = int(radar_width)
        self.radar_pitch_min = float(radar_pitch_min)
        self.radar_pitch_max = float(radar_pitch_max)
        self.radar_max_depth = float(radar_max_depth)
        self._render_worker: Optional[HabitatRGBRenderWorker] = None
        self._render_worker_lock = threading.Lock()

    @classmethod
    def from_sample(
        cls,
        sample_dir: Union[str, Path],
        *,
        data_root: Optional[str] = None,
        config: Optional[EpisodeConfig] = None,
        rgb_width: int = 640,
        rgb_height: int = 480,
        rgb_gpu_device_id: int = -1,
    ) -> "AeroVerbsEpisode":
        from aeroverbs.config import Config
        from aeroverbs.data.completion import (
            CompletionJudge,
            load_object_mesh,
            make_clearance_provider,
        )
        from aeroverbs.env.radar import load_scene_mesh, render_scene_glb
        from aeroverbs.env.rgb_renderer import load_sample_frame

        sample_path = Path(sample_dir)
        meta_path = sample_path / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        first_frame = load_sample_frame(sample_path, frame_index=0)
        radar_path = Path(render_scene_glb(str(sample_path)))
        radar_mesh = load_scene_mesh(str(radar_path))
        collision_mesh = load_object_mesh(str(sample_path), data_root, meta)
        clearance_provider = make_clearance_provider(
            str(sample_path), data_root, meta, mesh=collision_mesh
        )
        judge = CompletionJudge(
            meta,
            clearance_provider=clearance_provider,
            cfg=Config().completion,
            mesh=collision_mesh,
        )
        controller = EpisodeController(
            initial_position_world=first_frame.position_std,
            initial_forward_world=first_frame.forward_std,
            clearance_fn=lambda point: float(
                clearance_provider.clearance(np.asarray(point).reshape(1, 3))[0]
            ),
            completion_fn=lambda points, position: judge.update(points, position),
            config=config,
        )
        return cls(
            sample_dir=sample_path,
            meta=meta,
            controller=controller,
            radar_mesh=radar_mesh,
            render_scene_path=first_frame.scene_path,
            rgb_width=rgb_width,
            rgb_height=rgb_height,
            rgb_gpu_device_id=rgb_gpu_device_id,
        )

    def state(self) -> Dict[str, Any]:
        return self.controller.state()

    def termination_state(self) -> Dict[str, Any]:
        return dict(self.controller.termination)

    def fly_to_ned(
        self, target: Sequence[float], speed: Optional[float] = None
    ) -> Dict[str, Any]:
        return self.controller.fly_to_ned(target, speed=speed)

    def hover(self, duration: float = 1.0) -> Dict[str, Any]:
        return self.controller.hover(duration)

    def declare_agent_done(self) -> Dict[str, Any]:
        return self.controller.declare_agent_done()

    def lidar_scan(self) -> Dict[str, Any]:
        from aeroverbs.env.radar import sample_lidar_raycast

        ranges, _points, _valid, pitches, yaws = sample_lidar_raycast(
            ray_mesh=self.radar_mesh,
            eye_std=self.controller.position_world,
            forward_std=self.controller.forward_world,
            lidar_height=self.radar_height,
            lidar_width=self.radar_width,
            lidar_pitch_min=self.radar_pitch_min,
            lidar_pitch_max=self.radar_pitch_max,
            lidar_max_depth=self.radar_max_depth,
        )
        return radar_to_lidar_scan(
            ranges,
            pitches,
            yaws,
            target_count=360,
            range_max=self.radar_max_depth,
        )

    def _camera_forward(self, direction: str) -> np.ndarray:
        if direction not in self._CAMERA_DIRECTIONS:
            raise ValueError(f"unsupported camera direction: {direction}")
        front = self.controller.forward_world
        left = np.array([-front[1], front[0], 0.0], dtype=float)
        return {
            "front": front,
            "rear": -front,
            "left": left,
            "right": -left,
            "down": np.array([0.0, 0.0, -1.0], dtype=float),
        }[direction]

    def camera_base64(self, direction: str = "front") -> Dict[str, Any]:
        from PIL import Image

        direction = str(direction).lower()
        forward = self._camera_forward(direction)
        with self._render_worker_lock:
            if self._render_worker is None:
                self._render_worker = HabitatRGBRenderWorker(
                    self.render_scene_path,
                    width=self.rgb_width,
                    height=self.rgb_height,
                    gpu_device_id=self.rgb_gpu_device_id,
                )
            worker = self._render_worker
        rgb = worker.render(self.controller.position_world, forward)
        payload = io.BytesIO()
        Image.fromarray(rgb).save(payload, format="JPEG", quality=90)
        return {
            "image_base64": base64.b64encode(payload.getvalue()).decode("ascii"),
            "direction": direction,
            "mime_type": "image/jpeg",
            "width": self.rgb_width,
            "height": self.rgb_height,
        }

    def result(self) -> Dict[str, Any]:
        result = self.controller.result()
        result.update(
            {
                "skill": self.meta.get("skill") or self.sample_dir.parent.name,
                "scene_id": self.meta.get("scene_id") or self.sample_dir.name,
            }
        )
        return result

    def close(self) -> None:
        with self._render_worker_lock:
            worker = self._render_worker
            self._render_worker = None
        if worker is not None:
            worker.close()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class AerialClawBridgeServer:
    """Threaded loopback HTTP server around one AeroVerbs episode."""

    def __init__(
        self,
        episode: Any,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("the AerialClaw bridge must bind to loopback")
        self.episode = episode
        self._httpd = ThreadingHTTPServer((host, int(port)), self._handler_type())
        self._thread: Optional[threading.Thread] = None

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    def _handler_type(self):
        episode = self.episode

        class Handler(BaseHTTPRequestHandler):
            server_version = "AeroVerbsAerialClawBridge/1.0"

            def log_message(self, _format, *_args):
                return

            def _send(self, status: int, payload: Mapping[str, Any]) -> None:
                body = json.dumps(_json_safe(payload), allow_nan=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _payload(self) -> Dict[str, Any]:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length) if length else b"{}"
                    value = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError) as exc:
                    raise ValueError("request body must be JSON") from exc
                if not isinstance(value, dict):
                    raise ValueError("request body must be a JSON object")
                return value

            def _dispatch_get(self):
                parsed = urlparse(self.path)
                if parsed.path == "/health":
                    return {"ok": True}
                if parsed.path == "/state":
                    return episode.state()
                if parsed.path == "/termination":
                    return episode.termination_state()
                if parsed.path == "/lidar":
                    return episode.lidar_scan()
                if parsed.path == "/camera":
                    direction = parse_qs(parsed.query).get("direction", ["front"])[0]
                    if direction not in AeroVerbsEpisode._CAMERA_DIRECTIONS:
                        raise ValueError(f"unsupported camera direction: {direction}")
                    return episode.camera_base64(direction)
                if parsed.path == "/result":
                    return episode.result()
                raise KeyError(parsed.path)

            def _dispatch_post(self):
                parsed = urlparse(self.path)
                payload = self._payload()
                if parsed.path == "/action/fly_to_ned":
                    target = payload.get("target")
                    if not isinstance(target, list) or len(target) != 3:
                        raise ValueError("target must contain north, east, down")
                    return episode.fly_to_ned(target, speed=payload.get("speed"))
                if parsed.path == "/action/hover":
                    return episode.hover(float(payload.get("duration", 1.0)))
                if parsed.path == "/agent_done":
                    return episode.declare_agent_done()
                raise KeyError(parsed.path)

            def do_GET(self):
                self._handle(self._dispatch_get)

            def do_POST(self):
                self._handle(self._dispatch_post)

            def _handle(self, callback):
                try:
                    self._send(200, callback())
                except KeyError as exc:
                    self._send(404, {"error": f"unknown endpoint: {exc.args[0]}"})
                except ValueError as exc:
                    self._send(400, {"error": str(exc)})
                except Exception as exc:
                    self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        return Handler

    def start(self) -> "AerialClawBridgeServer":
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._httpd.serve_forever,
                name="aerialclaw-aeroverbs-bridge",
                daemon=True,
            )
            self._thread.start()
        return self

    def close(self) -> None:
        if self._thread is not None:
            self._httpd.shutdown()
            self._thread.join(timeout=5.0)
            self._thread = None
        self._httpd.server_close()
        close = getattr(self.episode, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> "AerialClawBridgeServer":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
