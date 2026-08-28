"""bbox -> one-shot maneuver heatmap conversion.

Implements the simple 2D rules from `UAV_maneuver_heatmap_context.md` §5.1:
shift a Gaussian centered on the target bbox according to the skill. This is a
reproducible reference generator; collected samples already ship a precomputed
``intention_heatmap.npy``.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

_ImageSize = Tuple[int, int]  # (W, H)


def bbox_to_heatmap(
    bbox: Sequence[float],
    image_size: _ImageSize = (640, 480),
    skill: str = "through",
    alpha: float = 0.8,
    beta: float = 0.8,
    sigma_scale: float = 1.0,
) -> np.ndarray:
    """Render a one-shot maneuver heatmap from a target bbox.

    Args:
        bbox: ``[x1, y1, x2, y2]`` in image pixels.
        image_size: ``(W, H)`` of the initial RGB frame.
        skill: one of ``around_left / around_right / through / over / under``.
        alpha, beta: offset scale relative to bbox width / height.
        sigma_scale: Gaussian width relative to bbox extent.

    Returns:
        float32 heatmap ``[H, W]`` in ``[0, 1]``.
    """
    W, H = int(image_size[0]), int(image_size[1])
    x1, y1, x2, y2 = (float(v) for v in bbox)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)

    cx_shift, cy_shift = cx, cy
    if skill == "around_left":
        cx_shift = x1 - alpha * bw
    elif skill == "around_right":
        cx_shift = x2 + alpha * bw
    elif skill == "over":
        cy_shift = y1 - beta * bh
    elif skill == "under":
        cy_shift = y2 + beta * bh
    # through: peak stays at bbox center

    sigma_x = max(bw * sigma_scale, 1.0)
    sigma_y = max(bh * sigma_scale, 1.0)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = np.exp(-((xx - cx_shift) ** 2 / (2 * sigma_x**2) + (yy - cy_shift) ** 2 / (2 * sigma_y**2)))
    return g.astype(np.float32)
