#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distill.coordinates import EXPECTED_REGION_KEY, sample_metrics
from distill.dataset import read_jsonl
from distill.model_loader import load_base_model, load_delta_checkpoint
from distill.paths import ensure_local_output, load_config, resolve_project_path
from scripts.predict import predict_region


def draw_panel(image: Image.Image, points: list[list[int]], title: str, color: str) -> Image.Image:
    panel = image.copy().convert("RGB")
    draw = ImageDraw.Draw(panel)
    polygon = [tuple(point) for point in points]
    draw.line(polygon + [polygon[0]], fill=color, width=5, joint="curve")
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), title, font=font)
    padding = 7
    draw.rectangle(
        (0, 0, box[2] - box[0] + padding * 2, box[3] - box[1] + padding * 2),
        fill=(0, 0, 0),
    )
    draw.text((padding, padding), title, fill=color, font=font)
    return panel


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare MiniMind-O and GPT-5.5 annotations")
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--sample-id")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = resolve_project_path(args.manifest or config["paths"].get("val_manifest") or config["paths"]["manifest"])
    rows = read_jsonl(manifest)
    if args.sample_id:
        matches = [row for row in rows if row["sample_id"] == args.sample_id]
        if not matches:
            raise ValueError(f"sample not found in {manifest}: {args.sample_id}")
        row = matches[0]
    else:
        if not 0 <= args.index < len(rows):
            raise IndexError(f"index {args.index} outside manifest with {len(rows)} rows")
        row = rows[args.index]

    model, tokenizer, device, _ = load_base_model(config)
    checkpoint = load_delta_checkpoint(model, args.checkpoint)
    model.eval()
    raw, predicted = predict_region(
        model, tokenizer, device, config, Path(row["image_path"]), row["skill"], row["target_object"]
    )

    key = EXPECTED_REGION_KEY[row["skill"]]
    teacher_points = row["regions"][key]
    student_points = predicted[key]
    with Image.open(row["image_path"]) as source:
        source = source.convert("RGB")
        teacher = draw_panel(source, teacher_points, "GPT-5.5 teacher", "#00ff66")
        student = draw_panel(source, student_points, "MiniMind-O student", "#ff3b30")
    canvas = Image.new("RGB", (teacher.width + student.width, teacher.height), "white")
    canvas.paste(teacher, (0, 0))
    canvas.paste(student, (teacher.width, 0))

    output = Path(args.output) if args.output else ensure_local_output("outputs/visualizations") / f"{row['sample_id'].replace('/', '__')}.png"
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)

    result = {
        "sample_id": row["sample_id"],
        "split": str(manifest),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "device": str(device),
        "raw_prediction": raw,
        "teacher": row["regions"],
        "student": predicted,
        "metrics": sample_metrics(
            predicted, row["regions"], skill=row["skill"], width=row["width"], height=row["height"]
        ),
        "output": str(output),
    }
    metadata = output.with_suffix(".json")
    metadata.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
