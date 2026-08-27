#!/usr/bin/env python
"""Rollout with a shifted intention heatmap.

Shifts are applied in the *native* heatmap pixel space (stored npy is
``[H, W] = [480, 640]``, matching meta ``image_size = [640, 480]``), before
the resize to the model input size ``[1, 120, 160]``.

- ``--shift-px``    : horizontal shift. Positive = content moves LEFT.
- ``--shift-up-px`` : vertical shift.   Positive = content moves UP.

Overflowing parts are cropped; the newly exposed part is zero-filled.
The scene, radar, collision geometry and semantic judge stay untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import rollout as R  # noqa: E402
from aeroverbs.data.transforms import resize_heatmap  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="Rollout with shifted heatmap (native pixel space).")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default="/root/autodl-tmp/data/trajctorys")
    ap.add_argument("--skill", required=True)
    ap.add_argument("--scene-id", required=True)
    ap.add_argument("--shift-px", type=int, default=0,
                    help="Horizontal shift in native pixels. Positive = move content left.")
    ap.add_argument("--shift-up-px", type=int, default=0,
                    help="Vertical shift in native pixels. Positive = move content up.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--step-size", type=float, default=0.1)
    ap.add_argument("--waypoint-index", type=int, default=2)
    ap.add_argument("--turn-rate", type=float, default=0.45)
    ap.add_argument("--drone-radius", type=float, default=0.35)
    ap.add_argument("--disable-collision-check", action="store_true")
    ap.add_argument("--goal-radius", type=float, default=1.0)
    ap.add_argument("--render-voxel-size", type=float, default=0.2)
    ap.add_argument("--render-voxel-stride", type=int, default=2)
    ap.add_argument("--render-region-stride", type=int, default=2)
    ap.add_argument("--no-render", action="store_true")
    return ap.parse_args()


def shift_native_heatmap(hm: torch.Tensor, shift_px: int, shift_up_px: int) -> torch.Tensor:
    """Shift native heatmap content horizontally and/or vertically.

    ``hm`` is ``[H, W]``. Positive ``shift_px`` moves content left; positive
    ``shift_up_px`` moves content up. Overflowing parts are cropped and the
    opposite side is zero-filled.
    """
    out = hm
    if shift_px > 0:
        tmp = torch.zeros_like(out)
        tmp[:, : -shift_px] = out[:, shift_px:]
        out = tmp
    elif shift_px < 0:
        tmp = torch.zeros_like(out)
        tmp[:, -shift_px:] = out[:, :shift_px]
        out = tmp

    if shift_up_px > 0:
        tmp = torch.zeros_like(out)
        tmp[: -shift_up_px, :] = out[shift_up_px:, :]
        out = tmp
    elif shift_up_px < 0:
        tmp = torch.zeros_like(out)
        tmp[-shift_up_px:, :] = out[:shift_up_px, :]
        out = tmp

    return out


def save_heatmap_png(arr: np.ndarray, path: str) -> str:
    """Save a native heatmap as a 640x480 grayscale PNG (no axes/title/colorbar)."""
    from PIL import Image

    a = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    img = Image.fromarray(a, mode="L")
    img.save(path)
    return path


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model, cfg = R._load_model(args.checkpoint, device)
    print(f"loaded checkpoint: {args.checkpoint}")

    shift_px = int(args.shift_px)
    shift_up_px = int(args.shift_up_px)
    print(f"shift: horizontal={shift_px:+d}px (left+), vertical={shift_up_px:+d}px (up+)")

    orig_load = R.load_heatmap_npy

    def shifted_load(path, size):
        arr = np.load(path).astype(np.float32)
        hm = torch.from_numpy(arr)  # [H, W] native
        hm = shift_native_heatmap(hm, shift_px, shift_up_px)
        return resize_heatmap(hm, size=size)  # [1, H', W']

    R.load_heatmap_npy = shifted_load

    sample_dirs = R._list_samples(args.data_root, args.skill, args.scene_id)

    # Save the shifted heatmap (native + model input + grayscale png).
    if sample_dirs:
        sample_dir = sample_dirs[0]
        hm_path = os.path.join(sample_dir, "intention_heatmap.npy")
        arr = np.load(hm_path).astype(np.float32)
        native = torch.from_numpy(arr)
        shifted_native = shift_native_heatmap(native, shift_px, shift_up_px).numpy()

        if shift_px == 0 and shift_up_px == 0:
            tag = "original"
        else:
            tag = "shift"
            if shift_px != 0:
                tag += f"_left{abs(shift_px)}" if shift_px > 0 else f"_right{abs(shift_px)}"
            if shift_up_px != 0:
                tag += f"_up{abs(shift_up_px)}" if shift_up_px > 0 else f"_down{abs(shift_up_px)}"

        inputs_dir = os.path.join(args.output_dir, "inputs")
        os.makedirs(inputs_dir, exist_ok=True)
        npy_path = os.path.join(inputs_dir, f"intention_heatmap_{tag}.npy")
        png_path = os.path.join(inputs_dir, f"intention_heatmap_{tag}.png")
        np.save(npy_path, shifted_native)
        save_heatmap_png(shifted_native, png_path)

        resized = resize_heatmap(torch.from_numpy(shifted_native),
                                 size=tuple(cfg.data.heatmap_size))
        np.save(os.path.join(inputs_dir, f"intention_heatmap_{tag}_model_input.npy"),
                resized.numpy())
        print(f"saved shifted heatmap: {npy_path} (native {shifted_native.shape})")
        print(f"saved shifted heatmap png: {png_path}")

    os.makedirs(os.path.join(args.output_dir, "trajectories"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "renders"), exist_ok=True)
    print(f"output dir: {args.output_dir}")

    print(f"rollout samples: {len(sample_dirs)}")
    results_path = os.path.join(args.output_dir, "results.jsonl")

    success = 0
    with open(results_path, "a", encoding="utf-8") as f:
        for i, sample_dir in enumerate(sample_dirs, 1):
            sample_name = os.path.basename(sample_dir)
            print(f"[{i}/{len(sample_dirs)}] {sample_name}")
            try:
                result = R._run_one_sample(sample_dir, model, cfg, device, args)
                result["heatmap_shift_px"] = shift_px
                result["heatmap_shift_up_px"] = shift_up_px
            except Exception as exc:
                result = {
                    "skill": args.skill,
                    "scene_id": sample_name,
                    "heatmap_shift_px": shift_px,
                    "heatmap_shift_up_px": shift_up_px,
                    "status": "error",
                    "reason": repr(exc),
                }
                print(f"  ERROR: {exc!r}")
            f.write(json.dumps(result) + "\n")
            f.flush()
            if result.get("status") == "success":
                success += 1
            print(
                f"  status={result.get('status')} steps={result.get('steps')} "
                f"goal_dist={result.get('goal_distance', float('nan')):.3f} "
                f"semantic_ok={result.get('semantic_ok')}"
            )

    print(f"success: {success}/{len(sample_dirs)} ({success / max(len(sample_dirs), 1):.1%})")
    print(f"results: {results_path}")


if __name__ == "__main__":
    main()
