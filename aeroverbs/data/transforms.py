"""Input transforms: radar preprocessing and heatmap resizing.

Radar preprocessing follows `maneuver_policy_network_v1.md` §1.3. A raw range
image of shape ``[40, 500]`` (meters, ``r_max`` == no-hit) is expanded to three
channels:

- channel 0: inverse range  ``1 - r / r_max`` (0 where no hit)
- channel 1: valid mask      (1 == surface hit within range, 0 == no hit)
- channel 2: normalized elevation per row in [-1, 1]
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def preprocess_radar(
    raw_range: torch.Tensor,
    r_max: float = 10.0,
    fov_deg: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """Convert a single raw range image ``[H, W]`` to ``[3, H, W]``."""
    if raw_range.dim() == 2:
        r = raw_range
    elif raw_range.dim() == 3 and raw_range.shape[0] == 1:
        r = raw_range[0]
    else:
        raise ValueError(f"raw_range must be [H, W] or [1, H, W], got {raw_range.shape}")

    r = r.float()
    valid = (r < r_max).float()
    r_inv = torch.clamp(1.0 - r / r_max, 0.0, 1.0) * valid

    H, W = r.shape
    if fov_deg is not None and len(fov_deg) == 2:
        lo, hi = float(fov_deg[0]), float(fov_deg[1])
        if hi == lo:
            elev = torch.zeros(H)
        else:
            angles = torch.linspace(lo, hi, H)
            elev = (angles - lo) / (hi - lo) * 2.0 - 1.0
    else:
        elev = torch.linspace(-1.0, 1.0, H)
    elevation = elev.view(H, 1).expand(H, W)

    return torch.stack([r_inv, valid, elevation], dim=0)  # [3, H, W]


def resize_heatmap(
    heatmap: torch.Tensor, size: Tuple[int, int] = (120, 160)
) -> torch.Tensor:
    """Resize a heatmap ``[H, W]`` / ``[1, H, W]`` to ``[1, H', W']``."""
    if heatmap.dim() == 2:
        x = heatmap.unsqueeze(0).unsqueeze(0)
    elif heatmap.dim() == 3 and heatmap.shape[0] == 1:
        x = heatmap.unsqueeze(0)
    else:
        raise ValueError(f"heatmap must be [H, W] or [1, H, W], got {heatmap.shape}")
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    return x.squeeze(0)  # [1, H', W']


def load_heatmap_npy(path: str, size: Tuple[int, int] = (120, 160)) -> torch.Tensor:
    """Load a ``*.npy`` heatmap from disk and resize it."""
    arr = np.load(path).astype(np.float32)
    return resize_heatmap(torch.from_numpy(arr), size=size)
