"""Circular convolution for 360-degree radar range images.

A range image wraps around horizontally (azimuth 0 == 360), but not vertically
(pitch -5 deg is not adjacent to +35 deg). Convolutions therefore use circular
padding along the width axis only and ordinary zero padding along the height
axis.
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

_Pair = Union[int, Tuple[int, int]]


def _pair(v: _Pair) -> Tuple[int, int]:
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


class CircularConv2d(nn.Module):
    """Conv2d with circular padding on the horizontal (width) axis only.

    Args:
        in_channels, out_channels: channel counts.
        kernel_size: ``(kh, kw)``.
        stride: ``(sh, sw)``.
        padding: ``(ph, pw)``; ``ph`` is zero-padded vertically, ``pw`` is
            circular-padded horizontally.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _Pair,
        stride: _Pair = 1,
        padding: _Pair = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        kh, kw = _pair(kernel_size)
        self.ph, self.pw = _pair(padding)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kh, kw),
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Vertical zero-pad first (top/bottom), then horizontal circular pad.
        if self.ph > 0:
            x = F.pad(x, (0, 0, self.ph, self.ph), mode="constant", value=0.0)
        if self.pw > 0:
            x = F.pad(x, (self.pw, self.pw, 0, 0), mode="circular")
        return self.conv(x)


class ResidualCircularBlock(nn.Module):
    """Two stride-1 circular convs with GroupNorm + SiLU and a residual skip."""

    def __init__(self, channels: int, kernel_size: _Pair = 3, padding: _Pair = 1) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = CircularConv2d(channels, channels, kernel_size, stride=1, padding=padding)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = CircularConv2d(channels, channels, kernel_size, stride=1, padding=padding)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(identity + out)
