#!/usr/bin/env python3
"""Probe SPF on rendered rollout poses without executing any action.

For every selected pose in ``frames.json`` this script renders the current
camera view, sends that rendered RGB image to SPF, and saves an overlay with
the returned screen point.  It is intended to isolate VLM point selection
from motion execution and coordinate conversion.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from aeroverbs.baselines.spf import SPFClient
from aeroverbs.env.rgb_renderer import HabitatRGBRenderer, load_sample_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", required=True, type=Path)
    parser.add_argument("--spf-url", default="http://127.0.0.1:5010")
    parser.add_argument("--prompt-mode", choices=("original", "relation"), default="relation")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None, help="Exclusive end frame; default: all frames")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--gpu-device-id", type=int, default=-1)
    args = parser.parse_args()

    sample_dir = args.sample_dir.resolve()
    frames_path = sample_dir / "frames.json"
    frames = json.loads(frames_path.read_text(encoding="utf-8"))
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"no frames in {frames_path}")
    if args.start < 0 or args.start >= len(frames):
        raise ValueError(f"--start must be in [0, {len(frames)-1}]")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")

    end = len(frames) if args.end is None else min(args.end, len(frames))
    if end <= args.start:
        raise ValueError("--end must be greater than --start")
    output_dir = (
        args.output_dir
        or (
            PROJECT_ROOT
            / "exp"
            / "spf_probe"
            / f"{sample_dir.parent.name}__{sample_dir.name}__{time.strftime('%Y%m%d-%H%M%S')}"
        )
    ).resolve()
    image_dir = output_dir / "frames"
    image_dir.mkdir(parents=True, exist_ok=True)

    # Use the scene and camera poses from frames.json; do not load initial_rgb.png.
    first = load_sample_frame(sample_dir, args.start)
    records = []
    with HabitatRGBRenderer(
        first.scene_path, width=640, height=480, gpu_device_id=args.gpu_device_id
    ) as renderer, SPFClient(args.spf_url, timeout_s=args.request_timeout_s) as client:
        for frame_index in range(args.start, end, args.stride):
            frame = load_sample_frame(sample_dir, frame_index)
            rgb = renderer.render(frame.position_std, frame.forward_std)
            prediction = client.predict(rgb, args.instruction, prompt_mode=args.prompt_mode)

            # Renderer returns RGB; OpenCV drawing/writing uses BGR.
            overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            x, y = prediction.screen_point
            cv2.drawMarker(overlay, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 28, 3)
            cv2.circle(overlay, (x, y), 10, (0, 255, 255), 2)
            cv2.putText(
                overlay,
                f"frame={frame_index} SPF=({x},{y}) dx={prediction.action_spf[0]:+.3f} dy={prediction.action_spf[1]:+.3f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
            )
            output_image = image_dir / f"{frame_index:04d}.png"
            cv2.imwrite(str(output_image), overlay)
            records.append({
                "frame_index": frame_index,
                "image": str(output_image.relative_to(output_dir)),
                "screen_point": [x, y],
                "action_spf": prediction.action_spf.tolist(),
                "action_aeroverbs": prediction.action_aeroverbs.tolist(),
                "raw_response": prediction.raw.get("raw_response", ""),
                "position_std": frame.position_std.tolist(),
                "forward_std": frame.forward_std.tolist(),
                "latency_s": prediction.latency_s,
            })
            print(
                f"frame {frame_index}: point=({x},{y}) "
                f"spf=({prediction.action_spf[0]:+.3f},{prediction.action_spf[1]:+.3f},{prediction.action_spf[2]:+.3f})",
                flush=True,
            )

    (output_dir / "probe_config.json").write_text(json.dumps({
        "sample_dir": str(sample_dir),
        "instruction": args.instruction,
        "prompt_mode": args.prompt_mode,
        "start": args.start, "end": end, "stride": args.stride,
        "executes_motion": False,
    }, indent=2), encoding="utf-8")
    (output_dir / "predictions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"saved {len(records)} overlays to {output_dir}")


if __name__ == "__main__":
    main()
