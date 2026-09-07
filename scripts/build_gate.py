#!/usr/bin/env python3
"""Build a coarse 3D semantic gate from a four-point image region and LiDAR."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Gate:
    def __init__(
        self,
        *,
        points: np.ndarray,
        center_std: np.ndarray,
        normal_std: np.ndarray,
        left_axis_std: np.ndarray,
        up_axis_std: np.ndarray,
        width_m: float,
        height_m: float,
        distance_m: float,
        confidence: float,
        region_payload: dict,
        summary_payload: dict,
    ):
        self.points = points
        self.center_std = center_std
        self.normal_std = normal_std
        self.left_axis_std = left_axis_std
        self.up_axis_std = up_axis_std
        self.width_m = width_m
        self.height_m = height_m
        self.distance_m = distance_m
        self.confidence = confidence
        self.region_payload = region_payload
        self.summary_payload = summary_payload


@dataclass(frozen=True)
class GateBuildResult:
    sample_dir: str
    status: str
    message: str


def pixel_to_body_ray(u: float, v: float, *, width: int, height: int) -> np.ndarray:
    """Convert RGB pixel to body ray using the collection script's 90-degree convention.

    Body frame is x=forward, y=left, z=up. Image +u points to camera right, so it
    maps to negative body-y.
    """
    fx = height / 2.0
    fy = height / 2.0
    cx = width / 2.0
    cy = height / 2.0
    ray = np.array([1.0, -(float(u) - cx) / fx, (cy - float(v)) / fy], dtype=np.float32)
    return _unit(ray)


def build_gate_from_polygon_and_lidar(
    *,
    region_points: Sequence[Sequence[float]],
    range_image: np.ndarray,
    position_std: np.ndarray,
    forward_std: np.ndarray,
    image_size: tuple[int, int],
    lidar_pitch_min: float,
    lidar_pitch_max: float,
    lidar_max_depth: float,
    skill: str,
    object_id: str,
    preferred_distance: float = 3.0,
    range_margin: float = 0.5,
    min_distance: float = 0.8,
    grid_size: int = 9,
    max_region_samples: int = 2000,
    annotation_confidence: float = 1.0,
) -> Gate:
    width, height = int(image_size[0]), int(image_size[1])
    polygon = _validate_region_points(region_points, width=width, height=height)
    range_image = np.asarray(range_image, dtype=np.float32)
    position_std = np.asarray(position_std, dtype=np.float32)
    forward_std = np.asarray(forward_std, dtype=np.float32)

    u_center, v_center = _polygon_centroid(polygon)
    bbox_xyxy = [
        int(np.floor(polygon[:, 0].min())),
        int(np.floor(polygon[:, 1].min())),
        int(np.ceil(polygon[:, 0].max())),
        int(np.ceil(polygon[:, 1].max())),
    ]

    center_ray_body = pixel_to_body_ray(u_center, v_center, width=width, height=height)
    forward, left, up = _body_to_world_basis(forward_std)
    rot = np.column_stack([forward, left, up])
    normal_std = _unit(center_ray_body @ rot.T)

    ys, xs = _polygon_pixels(polygon, width=width, height=height)
    sampled_indices = _sample_region_pixels(len(xs), max_region_samples)
    ranges = []
    free_flags = []
    for index in sampled_indices:
        ray = pixel_to_body_ray(float(xs[index]), float(ys[index]), width=width, height=height)
        value = _range_for_body_ray(
            range_image,
            ray,
            pitch_min=lidar_pitch_min,
            pitch_max=lidar_pitch_max,
        )
        ranges.append(value)
        free_flags.append(value > min_distance + range_margin)
    ranges_arr = np.asarray(ranges, dtype=np.float32)
    free_ratio = float(np.mean(free_flags)) if free_flags else 0.0
    center_range = _range_for_body_ray(
        range_image,
        center_ray_body,
        pitch_min=lidar_pitch_min,
        pitch_max=lidar_pitch_max,
    )

    if ranges_arr.size:
        free_limit = float(np.percentile(ranges_arr, 20.0)) - float(range_margin)
    else:
        free_limit = float(center_range) - float(range_margin)
    distance_m = min(float(preferred_distance), max(0.0, free_limit))
    if distance_m < float(min_distance):
        distance_m = max(0.0, distance_m)
    confidence = float(np.clip(
        float(annotation_confidence)
        * free_ratio
        * (distance_m / max(float(preferred_distance), 1e-6)),
        0.0,
        1.0,
    ))

    center_std = position_std + normal_std * distance_m
    left_axis_std = left - float(np.dot(left, normal_std)) * normal_std
    if np.linalg.norm(left_axis_std) < 1e-6:
        left_axis_std = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float32), normal_std)
    left_axis_std = _unit(left_axis_std)
    up_axis_std = _unit(np.cross(normal_std, left_axis_std))
    if float(np.dot(up_axis_std, np.array([0.0, 0.0, 1.0], dtype=np.float32))) < 0.0:
        up_axis_std = -up_axis_std

    half_width, half_height = _gate_half_extents_from_bbox(
        bbox_xyxy,
        distance_m=distance_m,
        image_size=(width, height),
    )
    points = _sample_gate_points(center_std, left_axis_std, up_axis_std, half_width, half_height, grid_size)

    description = (
        "Four-point semantic gate sampled as a point region. "
        "Built from an initial-RGB polygon and the first-frame LiDAR free interval."
    )
    region_payload = {
        "skill": str(skill),
        "object_id": str(object_id),
        "clearance": 0.0,
        "description": description,
        "num_points": int(len(points)),
        "points": points.astype(float).tolist(),
    }
    summary_payload = {
        "skill": str(skill),
        "object_id": str(object_id),
        "center_std": center_std.astype(float).tolist(),
        "normal_std": normal_std.astype(float).tolist(),
        "left_axis_std": left_axis_std.astype(float).tolist(),
        "up_axis_std": up_axis_std.astype(float).tolist(),
        "width_m": float(half_width * 2.0),
        "height_m": float(half_height * 2.0),
        "distance_m": float(distance_m),
        "confidence": confidence,
        "region_points_uv": polygon.astype(float).tolist(),
        "region_centroid_uv": [u_center, v_center],
        "region_bbox_xyxy": bbox_xyxy,
        # Compatibility aliases for existing experiment readers.  New code
        # should use the region_* fields above.
        "heatmap_centroid_uv": [u_center, v_center],
        "heatmap_bbox_xyxy": bbox_xyxy,
        "lidar_center_range_m": float(center_range),
        "lidar_range_p20_m": float(np.percentile(ranges_arr, 20.0)) if ranges_arr.size else None,
        "lidar_free_ratio": free_ratio,
        "preferred_distance_m": float(preferred_distance),
        "range_margin_m": float(range_margin),
        "min_distance_m": float(min_distance),
        "image_size": [width, height],
        "lidar_shape": list(range_image.shape),
        "projection_note": "RGB ray uses collect_single_trajectory_sample.py's implicit fx=fy=height/2, cx=width/2, cy=height/2 convention.",
    }

    return Gate(
        points=points,
        center_std=center_std,
        normal_std=normal_std,
        left_axis_std=left_axis_std,
        up_axis_std=up_axis_std,
        width_m=half_width * 2.0,
        height_m=half_height * 2.0,
        distance_m=distance_m,
        confidence=confidence,
        region_payload=region_payload,
        summary_payload=summary_payload,
    )


def write_gate_region(gate: Gate, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gate.region_payload, indent=2) + "\n", encoding="utf-8")


def write_gate_summary(gate: Gate, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gate.summary_payload, indent=2) + "\n", encoding="utf-8")


def gate_feature_vector(gate: Gate) -> np.ndarray:
    """Pack gate geometry into a fixed-width vector for dataset loading.

    Layout:
    center[3], normal[3], left_axis[3], up_axis[3], width, height, distance, confidence.
    """
    return np.asarray(
        [
            *np.asarray(gate.center_std, dtype=np.float32).reshape(3).tolist(),
            *np.asarray(gate.normal_std, dtype=np.float32).reshape(3).tolist(),
            *np.asarray(gate.left_axis_std, dtype=np.float32).reshape(3).tolist(),
            *np.asarray(gate.up_axis_std, dtype=np.float32).reshape(3).tolist(),
            float(gate.width_m),
            float(gate.height_m),
            float(gate.distance_m),
            float(gate.confidence),
        ],
        dtype=np.float32,
    )


def write_gate_numpy(gate: Gate, feature_path: Path, points_path: Path) -> None:
    feature_path = Path(feature_path)
    points_path = Path(points_path)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    points_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feature_path, gate_feature_vector(gate))
    np.save(points_path, np.asarray(gate.points, dtype=np.float32))


def build_gate_for_sample_dir(
    sample_dir: Path,
    *,
    region_points: Optional[Sequence[Sequence[float]]] = None,
    annotation_name: str = "minimind_annotation.json",
    data_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
    output_region_name: str = "heatmap_gate_region.json",
    output_summary_name: str = "heatmap_gate_summary.json",
    output_npy_name: str = "heatmap_gate.npy",
    output_points_npy_name: str = "heatmap_gate_points.npy",
    preferred_distance: float = 3.0,
    range_margin: float = 0.5,
    min_distance: float = 0.8,
    grid_size: int = 9,
    max_region_samples: int = 2000,
) -> Optional[Gate]:
    sample_dir = Path(sample_dir)
    meta_path = sample_dir / "meta.json"
    frames_path = sample_dir / "frames.json"
    lidar_path = sample_dir / "frames" / "lidar_range_npy" / "000.npy"
    annotation_path = sample_dir / annotation_name
    if not (meta_path.exists() and frames_path.exists() and lidar_path.exists()):
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frames = json.loads(frames_path.read_text(encoding="utf-8"))
    if not frames:
        return None
    annotation_confidence = 1.0
    if region_points is None:
        if not annotation_path.exists():
            return None
        region_points, annotation_confidence = _load_region_annotation(
            annotation_path,
            skill=str(meta.get("skill", "around_left")),
        )
    frame0 = frames[0]
    image_size = tuple(meta.get("image_size", [640, 480]))
    pitch_min, pitch_max = meta.get("lidar_vertical_fov_deg", [-5.0, 35.0])
    gate = build_gate_from_polygon_and_lidar(
        region_points=region_points,
        range_image=np.load(lidar_path),
        position_std=np.asarray(frame0["camera_position_std"], dtype=np.float32),
        forward_std=np.asarray(frame0["camera_forward_std"], dtype=np.float32),
        image_size=(int(image_size[0]), int(image_size[1])),
        lidar_pitch_min=float(pitch_min),
        lidar_pitch_max=float(pitch_max),
        lidar_max_depth=float(meta.get("lidar_max_depth_m", 10.0)),
        skill=str(meta.get("skill", "around_left")),
        object_id=str(meta.get("object_id", "target")),
        preferred_distance=preferred_distance,
        range_margin=range_margin,
        min_distance=min_distance,
        grid_size=grid_size,
        max_region_samples=max_region_samples,
        annotation_confidence=annotation_confidence,
    )
    if output_root is None:
        region_path = sample_dir / output_region_name
        summary_path = sample_dir / output_summary_name
        npy_path = sample_dir / output_npy_name
        points_npy_path = sample_dir / output_points_npy_name
    else:
        region_path, summary_path = gate_output_paths(
            sample_dir,
            data_root=Path(data_root) if data_root is not None else sample_dir.parent.parent,
            output_root=Path(output_root),
            region_name=output_region_name,
            summary_name=output_summary_name,
        )
        npy_path = region_path.with_name(output_npy_name)
        points_npy_path = region_path.with_name(output_points_npy_name)
    write_gate_region(gate, region_path)
    write_gate_summary(gate, summary_path)
    write_gate_numpy(gate, npy_path, points_npy_path)
    return gate


def gate_output_paths(
    sample_dir: Path,
    *,
    data_root: Path,
    output_root: Path,
    region_name: str,
    summary_name: str,
) -> tuple[Path, Path]:
    sample_dir = Path(sample_dir)
    data_root = Path(data_root)
    output_root = Path(output_root)
    try:
        relative = sample_dir.relative_to(data_root)
    except ValueError:
        relative = Path(sample_dir.parent.name) / sample_dir.name
    output_dir = output_root / relative
    return output_dir / region_name, output_dir / summary_name


def find_sample_dirs(data_root: Path) -> Iterable[Path]:
    data_root = Path(data_root)
    for meta_path in sorted(data_root.glob("*/*/meta.json")):
        yield meta_path.parent


def progress_iterable(items: Sequence, *, enabled: bool = True, desc: str = "gate", total: int | None = None):
    if not enabled:
        return items
    try:
        from tqdm import tqdm
    except Exception:
        return items
    return tqdm(items, desc=desc, unit="sample", total=total)


def _build_gate_task(task: tuple) -> GateBuildResult:
    (
        sample_dir,
        data_root,
        output_root,
        output_region_name,
        output_summary_name,
        output_npy_name,
        output_points_npy_name,
        annotation_name,
        preferred_distance,
        range_margin,
        min_distance,
        grid_size,
        max_region_samples,
    ) = task
    sample_dir = Path(sample_dir)
    try:
        gate = build_gate_for_sample_dir(
            sample_dir,
            data_root=Path(data_root),
            output_root=Path(output_root) if output_root else None,
            output_region_name=output_region_name,
            output_summary_name=output_summary_name,
            output_npy_name=output_npy_name,
            output_points_npy_name=output_points_npy_name,
            annotation_name=annotation_name,
            preferred_distance=preferred_distance,
            range_margin=range_margin,
            min_distance=min_distance,
            grid_size=grid_size,
            max_region_samples=max_region_samples,
        )
        if gate is None:
            return GateBuildResult(str(sample_dir), "skipped", f"skip {sample_dir}")
        if output_root:
            region_path, _ = gate_output_paths(
                sample_dir,
                data_root=Path(data_root),
                output_root=Path(output_root),
                region_name=output_region_name,
                summary_name=output_summary_name,
            )
        else:
            region_path = sample_dir / output_region_name
        message = (
            f"built {region_path} "
            f"center={np.round(gate.center_std, 3).tolist()} "
            f"size=({gate.width_m:.2f},{gate.height_m:.2f}) "
            f"conf={gate.confidence:.2f}"
        )
        return GateBuildResult(str(sample_dir), "built", message)
    except Exception as exc:
        return GateBuildResult(str(sample_dir), "failed", f"failed {sample_dir}: {exc}")


def build_gate_tasks(args, data_root: Path, sample_dirs: Sequence[Path]) -> list[tuple]:
    return [
        (
            str(sample_dir),
            str(data_root),
            str(Path(args.output_root)) if args.output_root else None,
            args.output_region_name,
            args.output_summary_name,
            args.output_npy_name,
            args.output_points_npy_name,
            args.annotation_name,
            float(args.preferred_distance),
            float(args.range_margin),
            float(args.min_distance),
            int(args.grid_size),
            int(args.max_region_samples),
        )
        for sample_dir in sample_dirs
    ]


def inflight_limit(*, num_tasks: int, num_workers: int) -> int:
    return min(max(0, int(num_tasks)), max(1, int(num_workers)) * 2)


def run_gate_tasks(tasks: Sequence[tuple], *, num_workers: int, show_progress: bool) -> list[GateBuildResult]:
    num_workers = max(1, int(num_workers))
    if num_workers == 1:
        return [
            _build_gate_task(task)
            for task in progress_iterable(tasks, enabled=show_progress, desc="gates")
        ]

    results = []
    task_iter = iter(tasks)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        pending = set()
        for _ in range(inflight_limit(num_tasks=len(tasks), num_workers=num_workers)):
            try:
                pending.add(executor.submit(_build_gate_task, next(task_iter)))
            except StopIteration:
                break

        progress = progress_iterable(range(len(tasks)), enabled=show_progress, desc="gates")
        progress_iter = iter(progress)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                results.append(future.result())
                try:
                    next(progress_iter)
                except StopIteration:
                    pass
                try:
                    pending.add(executor.submit(_build_gate_task, next(task_iter)))
                except StopIteration:
                    pass
    return results


def main() -> int:
    args = _parse_args()
    data_root = Path(args.data_root)
    if args.sample_dir:
        sample_dirs = [Path(args.sample_dir)]
    else:
        sample_dirs = list(find_sample_dirs(data_root))
        if args.limit is not None:
            sample_dirs = sample_dirs[: args.limit]

    built = 0
    skipped = 0
    failed = 0
    results = run_gate_tasks(
        build_gate_tasks(args, data_root, sample_dirs),
        num_workers=args.num_workers,
        show_progress=not args.no_progress,
    )
    for result in results:
        if result.status == "built":
            built += 1
            print(result.message)
        elif result.status == "skipped":
            skipped += 1
            print(result.message)
        else:
            failed += 1
            print(result.message, file=sys.stderr)
    print(f"done: built={built} skipped={skipped} failed={failed}")
    return 1 if failed else 0


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build 3D semantic gates from four-point image regions and first-frame LiDAR.")
    parser.add_argument("--data-root", default="AeroVerbs/data/trajctorys")
    parser.add_argument("--sample-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-root", default=None, help="Optional mirrored output root. Default writes into each sample directory.")
    parser.add_argument("--output-region-name", default="heatmap_gate_region.json")
    parser.add_argument("--output-summary-name", default="heatmap_gate_summary.json")
    parser.add_argument("--output-npy-name", default="heatmap_gate.npy")
    parser.add_argument("--output-points-npy-name", default="heatmap_gate_points.npy")
    parser.add_argument("--annotation-name", default="minimind_annotation.json")
    parser.add_argument("--preferred-distance", type=float, default=3.0)
    parser.add_argument("--range-margin", type=float, default=0.5)
    parser.add_argument("--min-distance", type=float, default=0.8)
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--max-region-samples", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel worker processes.")
    parser.add_argument("--no-progress", action="store_true", help="Disable the tqdm progress bar.")
    return parser.parse_args(argv)


def _validate_region_points(
    points: Sequence[Sequence[float]], *, width: int, height: int
) -> np.ndarray:
    polygon = np.asarray(points, dtype=np.float32)
    if polygon.shape != (4, 2):
        raise ValueError(f"expected four [u, v] points, got shape {polygon.shape}")
    if not np.isfinite(polygon).all():
        raise ValueError("region points must be finite")
    if (
        np.any(polygon[:, 0] < 0)
        or np.any(polygon[:, 0] >= width)
        or np.any(polygon[:, 1] < 0)
        or np.any(polygon[:, 1] >= height)
    ):
        raise ValueError(f"region points fall outside image bounds {width}x{height}")
    x, y = polygon[:, 0], polygon[:, 1]
    twice_area = float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))
    if abs(twice_area) < 2.0:
        raise ValueError("four-point region is degenerate")
    return polygon


def _polygon_centroid(polygon: np.ndarray) -> tuple[float, float]:
    x, y = polygon[:, 0].astype(np.float64), polygon[:, 1].astype(np.float64)
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    denominator = 3.0 * float(cross.sum())
    if abs(denominator) < 1e-8:
        center = polygon.mean(axis=0)
        return float(center[0]), float(center[1])
    return (
        float(np.sum((x + np.roll(x, -1)) * cross) / denominator),
        float(np.sum((y + np.roll(y, -1)) * cross) / denominator),
    )


def _polygon_pixels(
    polygon: np.ndarray, *, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    x_min = max(0, int(np.floor(polygon[:, 0].min())))
    x_max = min(width - 1, int(np.ceil(polygon[:, 0].max())))
    y_min = max(0, int(np.floor(polygon[:, 1].min())))
    y_max = min(height - 1, int(np.ceil(polygon[:, 1].max())))
    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max + 1, dtype=np.float32),
        np.arange(y_min, y_max + 1, dtype=np.float32),
    )
    inside = np.zeros(grid_x.shape, dtype=bool)
    previous = polygon[-1]
    for current in polygon:
        x0, y0 = float(previous[0]), float(previous[1])
        x1, y1 = float(current[0]), float(current[1])
        crosses_row = (y0 > grid_y) != (y1 > grid_y)
        edge_x = (x1 - x0) * (grid_y - y0) / (y1 - y0 + 1e-12) + x0
        inside ^= crosses_row & (grid_x < edge_x)
        previous = current
    ys_local, xs_local = np.nonzero(inside)
    if len(xs_local) == 0:
        raise ValueError("four-point region contains no image pixels")
    return ys_local + y_min, xs_local + x_min


def _sample_region_pixels(count: int, max_samples: int) -> np.ndarray:
    if max_samples <= 0:
        raise ValueError("max_region_samples must be positive")
    if count <= max_samples:
        return np.arange(count)
    return np.linspace(0, count - 1, num=max_samples, dtype=np.int64)


def _load_region_annotation(annotation_path: Path, *, skill: str) -> tuple[np.ndarray, float]:
    payload = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    regions = payload.get("regions") or payload.get("parsed")
    if not isinstance(regions, dict) or len(regions) != 1:
        raise ValueError(f"expected exactly one region in {annotation_path}")
    points = next(iter(regions.values()))
    confidence = float(payload.get("confidence", 1.0))
    if payload.get("skill") not in (None, skill):
        raise ValueError(
            f"annotation skill {payload.get('skill')!r} does not match sample skill {skill!r}"
        )
    return np.asarray(points, dtype=np.float32), confidence


def _range_for_body_ray(range_image: np.ndarray, ray_body: np.ndarray, *, pitch_min: float, pitch_max: float) -> float:
    ray = _unit(ray_body)
    pitch = math.degrees(math.asin(float(np.clip(ray[2], -1.0, 1.0))))
    yaw = math.degrees(math.atan2(float(ray[1]), float(ray[0])))
    h, w = range_image.shape
    pitch_values = np.linspace(float(pitch_max), float(pitch_min), h, dtype=np.float32)
    yaw_values = np.linspace(180.0, -180.0, w, endpoint=False, dtype=np.float32)
    row = int(np.argmin(np.abs(pitch_values - pitch)))
    yaw_diff = np.abs(((yaw_values - yaw + 180.0) % 360.0) - 180.0)
    col = int(np.argmin(yaw_diff))
    return float(range_image[row, col])


def _body_to_world_basis(forward_std: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = _unit(np.asarray(forward_std, dtype=np.float32))
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    up = world_up - float(np.dot(world_up, forward)) * forward
    if np.linalg.norm(up) < 1e-6:
        up = world_up
    up = _unit(up)
    left = np.cross(up, forward)
    if np.linalg.norm(left) < 1e-6:
        left = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    left = _unit(left)
    return forward, left, up


def _gate_half_extents_from_bbox(
    bbox_xyxy: list[int],
    *,
    distance_m: float,
    image_size: tuple[int, int],
    min_width: float = 0.6,
    min_height: float = 0.4,
    max_width: float = 3.0,
    max_height: float = 2.0,
) -> tuple[float, float]:
    width, height = image_size
    x1, y1, x2, y2 = bbox_xyxy
    fx = height / 2.0
    fy = height / 2.0
    gate_width = max(float(min_width), float(distance_m) * max(1.0, x2 - x1 + 1) / fx)
    gate_height = max(float(min_height), float(distance_m) * max(1.0, y2 - y1 + 1) / fy)
    gate_width = min(float(max_width), gate_width)
    gate_height = min(float(max_height), gate_height)
    return gate_width / 2.0, gate_height / 2.0


def _sample_gate_points(
    center: np.ndarray,
    left_axis: np.ndarray,
    up_axis: np.ndarray,
    half_width: float,
    half_height: float,
    grid_size: int,
) -> np.ndarray:
    grid_size = max(2, int(grid_size))
    us = np.linspace(-float(half_width), float(half_width), grid_size, dtype=np.float32)
    vs = np.linspace(-float(half_height), float(half_height), grid_size, dtype=np.float32)
    points = [
        center + float(u) * left_axis + float(v) * up_axis
        for u in us
        for v in vs
    ]
    return np.asarray(points, dtype=np.float32)


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return vector / norm


if __name__ == "__main__":
    raise SystemExit(main())
