#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distill.dataset import read_jsonl
from distill.metrics import aggregate_metrics, evaluate_prediction
from distill.model_loader import greedy_generate, inference_input, load_base_model, load_delta_checkpoint
from distill.paths import ensure_local_output, load_config, resolve_project_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and score a distillation manifest")
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--name", default="dev")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = resolve_project_path(args.manifest or config["paths"].get("val_manifest") or config["paths"]["manifest"])
    rows = read_jsonl(manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    model, tokenizer, device, _ = load_base_model(config)
    load_delta_checkpoint(model, args.checkpoint)
    model.eval()
    records = []
    for index, row in enumerate(rows, 1):
        with Image.open(row["image_path"]) as image:
            pixels = model.vision_processor(images=image.convert("RGB"), return_tensors="pt").pixel_values.to(device)
        ids, _ = inference_input(tokenizer, row["prompt"], int(config["data"].get("image_token_len", 64)), device)
        prediction = greedy_generate(model, tokenizer, ids, pixels, max_new_tokens=int(config["training"].get("max_new_tokens", 96)))
        record = evaluate_prediction(row, prediction)
        records.append(record)
        if index % 20 == 0:
            print(f"evaluated {index}/{len(rows)}")
    pred_dir = ensure_local_output("outputs/predictions")
    metrics_dir = ensure_local_output("outputs/metrics")
    with (pred_dir / f"{args.name}.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = aggregate_metrics(records)
    (metrics_dir / f"{args.name}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
