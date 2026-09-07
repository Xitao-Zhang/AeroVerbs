#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distill.coordinates import validate_regions
from distill.dataset import read_jsonl, render_training_example
from distill.paths import ensure_local_output, load_config, resolve_project_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit prepared distillation manifests")
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--manifest", default="data/manifests/all.jsonl")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = resolve_project_path(args.manifest)
    rows, errors, token_lengths = read_jsonl(manifest), [], []
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["paths"]["tokenizer"])
    except Exception as exc:
        print(f"warning: tokenizer audit skipped: {exc}", file=sys.stderr)
    for row in rows:
        try:
            path = Path(row["image_path"])
            if path.name != "initial_rgb.png" or not path.is_file():
                raise ValueError(f"invalid image path: {path}")
            with Image.open(path) as image:
                if image.size != (int(row["width"]), int(row["height"])):
                    raise ValueError(f"image size mismatch: {image.size}")
            validate_regions(row["regions"], skill=row["skill"], width=int(row["width"]), height=int(row["height"]))
            if tokenizer is not None:
                ids, labels, _ = render_training_example(tokenizer, prompt=row["prompt"], answer=row["answer"], image_token_len=int(config["data"].get("image_token_len", 64)))
                token_lengths.append(len(ids))
                if len(ids) > int(config["data"]["max_seq_len"]):
                    raise ValueError(f"token length {len(ids)} exceeds max_seq_len")
                if not any(label != -100 for label in labels):
                    raise ValueError("no supervised labels")
        except Exception as exc:
            errors.append({"sample_id": row.get("sample_id"), "error": str(exc)})
    report = {
        "manifest": str(manifest), "samples": len(rows), "errors": len(errors),
        "skill_counts": dict(Counter(row["skill"] for row in rows)),
        "token_length_min": min(token_lengths) if token_lengths else None,
        "token_length_max": max(token_lengths) if token_lengths else None,
        "token_length_mean": sum(token_lengths) / len(token_lengths) if token_lengths else None,
    }
    report_dir = ensure_local_output("data/reports")
    (report_dir / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    errors_path = report_dir / "audit_errors.jsonl"
    if errors:
        with errors_path.open("w", encoding="utf-8") as handle:
            for error in errors:
                handle.write(json.dumps(error, ensure_ascii=False) + "\n")
    elif errors_path.exists():
        errors_path.unlink()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
