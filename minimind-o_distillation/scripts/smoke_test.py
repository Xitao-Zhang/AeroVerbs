#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distill.dataset import RegionCollator, RegionSFTDataset
from distill.model_loader import causal_text_loss, configure_trainable_parameters, load_base_model, thinker_forward
from distill.paths import load_config, resolve_project_path


def main() -> int:
    parser = argparse.ArgumentParser(description="One-batch forward smoke test; never updates weights")
    parser.add_argument("--config", default="configs/dev.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = resolve_project_path(config["paths"]["manifest"])
    if not manifest.is_file():
        raise FileNotFoundError(f"run scripts/prepare_dataset.py first: {manifest}")
    model, tokenizer, device, _ = load_base_model(config)
    counts = configure_trainable_parameters(model, config)
    dataset = RegionSFTDataset(
        manifest,
        tokenizer,
        model.vision_processor,
        max_seq_len=int(config["data"]["max_seq_len"]),
        image_token_len=int(config["data"].get("image_token_len", 64)),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=RegionCollator(tokenizer.pad_token_id), num_workers=0)
    batch = next(iter(loader))
    for key in ("input_ids", "labels", "attention_mask", "pixel_values"):
        batch[key] = batch[key].to(device)
    model.eval()
    with torch.inference_mode():
        output = thinker_forward(
            model,
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
        )
        loss = causal_text_loss(output.logits, batch["labels"])
    if not math.isfinite(float(loss.item())):
        raise RuntimeError(f"non-finite loss: {loss.item()}")
    report = {
        "sample_id": batch["sample_id"][0],
        "input_shape": list(batch["input_ids"].shape),
        "pixel_shape": list(batch["pixel_values"].shape),
        "logits_shape": list(output.logits.shape),
        "loss": float(loss.item()),
        "trainable_parameters": counts["trainable"],
        "total_parameters": counts["total"],
        "optimizer_steps": 0,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
