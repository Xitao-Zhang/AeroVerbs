from __future__ import annotations

import json
import math
import re
from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw


EXPECTED_REGION_KEY = {
    "around_left": "left",
    "around_right": "right",
    "through": "passage",
    "over": "over",
    "under": "under",
}


class RegionFormatError(ValueError):
    pass


def canonical_region_json(regions: Mapping[str, Any], skill: str, width: int, height: int) -> str:
    validated = validate_regions(regions, skill=skill, width=width, height=height)
    return json.dumps(validated, ensure_ascii=False, separators=(",", ":"))


def validate_regions(regions: Mapping[str, Any], *, skill: str, width: int, height: int) -> dict[str, list[list[int]]]:
    if skill not in EXPECTED_REGION_KEY:
        raise RegionFormatError(f"unsupported skill: {skill}")
    if not isinstance(regions, Mapping):
        raise RegionFormatError("region output must be a JSON object")
    key = EXPECTED_REGION_KEY[skill]
    if set(regions) != {key}:
        raise RegionFormatError(f"expected exactly the key {key!r}, got {sorted(regions)}")
    raw_points = regions[key]
    if not isinstance(raw_points, list) or len(raw_points) != 4:
        raise RegionFormatError("region must contain exactly four points")
    points: list[list[int]] = []
    for point in raw_points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise RegionFormatError(f"invalid point: {point!r}")
        if any(isinstance(v, bool) or not isinstance(v, int) for v in point):
            raise RegionFormatError(f"point coordinates must be integers: {point!r}")
        x, y = int(point[0]), int(point[1])
        if not (0 <= x < width and 0 <= y < height):
            raise RegionFormatError(f"point {(x, y)} is outside {width}x{height}")
        points.append([x, y])
    if abs(polygon_area(points)) < 1.0:
        raise RegionFormatError("region polygon is degenerate")
    return {key: points}


def parse_region_text(text: str, *, skill: str, width: int, height: int) -> dict[str, list[list[int]]]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise RegionFormatError(f"no JSON object found in output: {text!r}") from None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise RegionFormatError(f"invalid JSON output: {text!r}") from exc
    return validate_regions(payload, skill=skill, width=width, height=height)


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    arr = np.asarray(points, dtype=np.float64)
    x, y = arr[:, 0], arr[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def polygon_iou(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], width: int, height: int) -> float:
    ma = Image.new("1", (width, height), 0)
    mb = Image.new("1", (width, height), 0)
    ImageDraw.Draw(ma).polygon([tuple(p) for p in a], fill=1)
    ImageDraw.Draw(mb).polygon([tuple(p) for p in b], fill=1)
    aa, bb = np.asarray(ma, dtype=bool), np.asarray(mb, dtype=bool)
    union = np.logical_or(aa, bb).sum()
    return float(np.logical_and(aa, bb).sum() / union) if union else 0.0


def center_error(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float).mean(0) - np.asarray(b, dtype=float).mean(0)))


def vertex_error(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> float:
    """Minimum mean vertex distance over cyclic order and winding."""
    aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    candidates = []
    for reverse, shift in product((False, True), range(4)):
        candidate = bb[::-1] if reverse else bb
        candidate = np.roll(candidate, shift, axis=0)
        candidates.append(float(np.linalg.norm(aa - candidate, axis=1).mean()))
    return min(candidates)


def sample_metrics(predicted: Mapping[str, Any], target: Mapping[str, Any], *, skill: str, width: int, height: int) -> dict[str, float]:
    p = validate_regions(predicted, skill=skill, width=width, height=height)[EXPECTED_REGION_KEY[skill]]
    t = validate_regions(target, skill=skill, width=width, height=height)[EXPECTED_REGION_KEY[skill]]
    return {
        "iou": polygon_iou(p, t, width, height),
        "center_error_px": center_error(p, t),
        "vertex_error_px": vertex_error(p, t),
    }
