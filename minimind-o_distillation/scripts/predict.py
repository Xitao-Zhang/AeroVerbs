#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distill.coordinates import parse_region_text
from distill.model_loader import greedy_generate, inference_input, load_base_model, load_delta_checkpoint
from distill.paths import load_config
from distill.prompting import build_student_prompt, build_user_content


def predict_region(model, tokenizer, device, config, image_path: Path, skill: str, target_object: str) -> tuple[str, dict]:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        pixels = model.vision_processor(images=image, return_tensors="pt").pixel_values.to(device)
    prompt = build_user_content(build_student_prompt(
        width=width, height=height,
        skill=skill, target_object=target_object,
    ))
    ids, _ = inference_input(tokenizer, prompt, int(config["data"].get("image_token_len", 64)), device)
    text = greedy_generate(model, tokenizer, ids, pixels, max_new_tokens=int(config["training"].get("max_new_tokens", 96)))
    return text, parse_region_text(text, skill=skill, width=width, height=height)


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict one UAV passable region")
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--skill", required=True, choices=("around_left", "around_right", "through", "over", "under"))
    parser.add_argument("--target-object", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    model, tokenizer, device, _ = load_base_model(config)
    load_delta_checkpoint(model, args.checkpoint)
    model.eval()
    raw, parsed = predict_region(model, tokenizer, device, config, args.image, args.skill, args.target_object)
    print(json.dumps({"raw": raw, "regions": parsed}, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
