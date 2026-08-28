"""Habitat-Sim RGB rendering for AeroVerbs closed-loop evaluation.

The camera setup intentionally matches ``collect_single_trajectory_sample.py``
so a rollout pose can be rendered with the same coordinate conversion,
lighting, horizontal flip, and empty-background compositing used at collection
time. Habitat-Sim is imported lazily so the rest of AeroVerbs remains usable in
the lightweight radar-only environment.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R


PathLike = Union[str, Path]


@dataclass(frozen=True)
class SampleRGBFrame:
    """Scene and camera data needed to reproduce one stored sample frame."""

    scene_path: Path
    position_std: np.ndarray
    forward_std: np.ndarray
    reference_path: Optional[Path]


def std_to_habitat(point: Sequence[float]) -> np.ndarray:
    """Convert AeroVerbs' z-up ``(x, y, z)`` into Habitat ``(x, z, y)``."""

    x, y, z = np.asarray(point, dtype=np.float32)
    return np.array([x, z, y], dtype=np.float32)


def composite_empty_background(observation: np.ndarray) -> np.ndarray:
    """Replace Habitat's empty black pixels with the collection-time white."""

    rgba = np.asarray(observation, dtype=np.uint8)
    if rgba.ndim != 3 or rgba.shape[-1] < 3:
        raise ValueError(f"expected HxWx3/4 color observation, got {rgba.shape}")
    rgb = rgba[..., :3].copy()
    empty = np.all(rgb == 0, axis=-1)
    rgb[empty] = np.array([255, 255, 255], dtype=np.uint8)
    return rgb


def load_sample_frame(sample_dir: PathLike, frame_index: int = 0) -> SampleRGBFrame:
    """Load one stored camera pose and its corresponding render-scene path."""

    sample_path = Path(sample_dir)
    frames_path = sample_path / "frames.json"
    if not frames_path.is_file():
        raise FileNotFoundError(frames_path)
    frames = json.loads(frames_path.read_text(encoding="utf-8"))
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"frames.json contains no frames: {frames_path}")
    if frame_index < 0 or frame_index >= len(frames):
        raise IndexError(
            f"frame_index {frame_index} outside [0, {len(frames) - 1}] for {frames_path}"
        )

    frame = frames[frame_index]
    try:
        position = np.asarray(frame["camera_position_std"], dtype=np.float32)
        forward = np.asarray(frame["camera_forward_std"], dtype=np.float32)
    except KeyError as exc:
        raise ValueError(f"frame {frame_index} lacks stored camera pose: {frames_path}") from exc
    if position.shape != (3,) or forward.shape != (3,):
        raise ValueError(
            f"frame {frame_index} camera vectors must have shape (3,), got "
            f"position={position.shape}, forward={forward.shape}"
        )

    scene_path = (
        sample_path
        / "_render_scene"
        / f"{sample_path.name}_with_ground.glb"
    )
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)

    reference_path = sample_path / (
        "initial_rgb.png" if frame_index == 0 else f"frames/rgb/{frame_index:03d}.png"
    )
    if not reference_path.is_file():
        reference_path = None

    return SampleRGBFrame(
        scene_path=scene_path,
        position_std=position,
        forward_std=forward,
        reference_path=reference_path,
    )


class HabitatRGBRenderer:
    """Reusable front-camera RGB renderer for one AeroVerbs GLB scene."""

    def __init__(
        self,
        scene_path: PathLike,
        *,
        width: int = 640,
        height: int = 480,
        gpu_device_id: int = -1,
    ) -> None:
        self.scene_path = Path(scene_path)
        if not self.scene_path.is_file():
            raise FileNotFoundError(self.scene_path)
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        self.width = int(width)
        self.height = int(height)
        self.gpu_device_id = int(gpu_device_id)
        self._sim = None
        self._agent = None

    def start(self) -> "HabitatRGBRenderer":
        """Initialize Habitat-Sim and the collection-compatible color camera."""

        if self._sim is not None:
            return self
        try:
            import habitat_sim
            import habitat_sim.gfx as gfx
        except ImportError as exc:
            raise ImportError(
                "HabitatRGBRenderer requires habitat-sim; run it from the visfly "
                "environment used to collect AeroVerbs data."
            ) from exc

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = str(self.scene_path)
        sim_cfg.enable_physics = False
        sim_cfg.scene_light_setup = gfx.DEFAULT_LIGHTING_KEY
        sim_cfg.override_scene_light_defaults = True
        sim_cfg.gpu_device_id = self.gpu_device_id

        color_spec = habitat_sim.CameraSensorSpec()
        color_spec.uuid = "color"
        color_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_spec.resolution = [self.height, self.width]
        color_spec.position = [0.0, 0.0, 0.0]

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [color_spec]
        self._sim = habitat_sim.Simulator(
            habitat_sim.Configuration(sim_cfg, [agent_cfg])
        )
        self._sim.set_light_setup(
            [
                gfx.LightInfo(
                    vector=[-2.0, -2.0, -1.0, 0.0],
                    model=gfx.LightPositionModel.Camera,
                )
            ]
        )
        self._agent = self._sim.initialize_agent(0)
        return self

    def render(
        self,
        position_std: Sequence[float],
        forward_std: Sequence[float],
    ) -> np.ndarray:
        """Render an RGB uint8 frame at the given z-up world pose."""

        self.start()
        position = _vector3(position_std, "position_std")
        forward = _normalize(_vector3(forward_std, "forward_std"))
        eye_hab = std_to_habitat(position)
        target_hab = std_to_habitat(position + forward)

        import habitat_sim

        state = habitat_sim.AgentState()
        state.position = eye_hab
        state.rotation = _front_center_camera_rotation(eye_hab, target_hab)
        self._agent.set_state(state)
        observation = self._sim.get_sensor_observations()["color"]
        rgb = np.fliplr(composite_empty_background(observation))
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    def close(self) -> None:
        """Release the Habitat simulator and its graphics resources."""

        if self._sim is not None:
            self._sim.close()
        self._sim = None
        self._agent = None

    def __enter__(self) -> "HabitatRGBRenderer":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _vector3(value: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector with shape (3,), got {vector}")
    return vector


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return vector / norm


def _front_center_camera_rotation(
    eye_hab: np.ndarray,
    target_hab: np.ndarray,
) -> np.ndarray:
    forward = _normalize(target_hab - eye_hab)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = _normalize(right)
    up = _normalize(np.cross(-forward, right))
    camera_to_world = np.column_stack([right, up, -forward])
    return R.from_matrix(camera_to_world).as_quat()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one stored AeroVerbs sample frame with Habitat-Sim."
    )
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--gpu-device-id", type=int, default=-1)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    frame = load_sample_frame(args.sample_dir, args.frame_index)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with HabitatRGBRenderer(
        frame.scene_path,
        width=args.width,
        height=args.height,
        gpu_device_id=args.gpu_device_id,
    ) as renderer:
        image = renderer.render(frame.position_std, frame.forward_std)
    Image.fromarray(image).save(output_path)
    print(f"rendered_rgb={output_path.resolve()}")
    print(
        "reference_rgb="
        + (str(frame.reference_path.resolve()) if frame.reference_path else "unavailable")
    )


if __name__ == "__main__":
    main()
