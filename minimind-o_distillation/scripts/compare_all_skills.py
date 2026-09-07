#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distill.coordinates import EXPECTED_REGION_KEY, sample_metrics
from distill.dataset import read_jsonl
from distill.model_loader import load_base_model, load_delta_checkpoint
from distill.paths import ensure_local_output, load_config, resolve_project_path
from scripts.compare_sample import draw_panel
from scripts.predict import predict_region


SKILLS = ("around_left", "around_right", "through", "over", "under")


def select_rows(rows: list[dict], per_skill: int) -> dict[str, list[dict]]:
    selected: dict[str, list[dict]] = defaultdict(list)
    families: dict[str, set[str]] = defaultdict(set)
    # First pass favors distinct scene/target families.
    for row in rows:
        skill = row["skill"]
        if skill not in SKILLS or len(selected[skill]) >= per_skill:
            continue
        if row["family"] not in families[skill]:
            selected[skill].append(row)
            families[skill].add(row["family"])
    # Fall back to any remaining sample if a skill has too few families.
    selected_ids = {row["sample_id"] for items in selected.values() for row in items}
    for row in rows:
        skill = row["skill"]
        if skill in SKILLS and len(selected[skill]) < per_skill and row["sample_id"] not in selected_ids:
            selected[skill].append(row)
            selected_ids.add(row["sample_id"])
    missing = {skill: per_skill - len(selected[skill]) for skill in SKILLS if len(selected[skill]) < per_skill}
    if missing:
        raise ValueError(f"manifest does not contain enough samples: {missing}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare MiniMind-O and GPT-5.5 on every AeroVerbs skill")
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--per-skill", type=int, default=2)
    parser.add_argument("--name", default="all-skills")
    parser.add_argument(
        "--sample-ids-from",
        type=Path,
        help="Reuse the ordered sample IDs recorded by an earlier results.json",
    )
    args = parser.parse_args()
    if args.per_skill < 1:
        raise ValueError("--per-skill must be positive")

    config = load_config(args.config)
    manifest = resolve_project_path(args.manifest or config["paths"].get("val_manifest") or config["paths"]["manifest"])
    rows = read_jsonl(manifest)
    if args.sample_ids_from:
        previous = json.loads(args.sample_ids_from.read_text(encoding="utf-8"))
        ordered_ids = [record["sample_id"] for record in previous["records"]]
        by_id = {row["sample_id"]: row for row in rows}
        missing = [sample_id for sample_id in ordered_ids if sample_id not in by_id]
        if missing:
            raise ValueError(f"sample IDs missing from manifest: {missing}")
        chosen = defaultdict(list)
        for sample_id in ordered_ids:
            row = by_id[sample_id]
            chosen[row["skill"]].append(row)
        wrong_counts = {skill: len(chosen[skill]) for skill in SKILLS if len(chosen[skill]) != args.per_skill}
        if wrong_counts:
            raise ValueError(f"reused sample counts do not match --per-skill={args.per_skill}: {wrong_counts}")
    else:
        chosen = select_rows(rows, args.per_skill)
    model, tokenizer, device, _ = load_base_model(config)
    checkpoint = load_delta_checkpoint(model, args.checkpoint)
    model.eval()

    output_dir = ensure_local_output("outputs/visualizations") / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    row_montages: list[Image.Image] = []

    for skill in SKILLS:
        comparisons: list[Image.Image] = []
        for sample_number, row in enumerate(chosen[skill], 1):
            raw, predicted = predict_region(
                model, tokenizer, device, config, Path(row["image_path"]), skill, row["target_object"]
            )
            key = EXPECTED_REGION_KEY[skill]
            with Image.open(row["image_path"]) as source_image:
                source = source_image.convert("RGB")
            teacher = draw_panel(source, row["regions"][key], f"{skill} | GPT-5.5 teacher", "#00ff66")
            student = draw_panel(source, predicted[key], f"{skill} | MiniMind-O student", "#ff3b30")
            comparison = Image.new("RGB", (teacher.width + student.width, teacher.height), "white")
            comparison.paste(teacher, (0, 0))
            comparison.paste(student, (teacher.width, 0))
            image_path = output_dir / f"{skill}_{sample_number}.png"
            comparison.save(image_path)
            comparisons.append(comparison.resize((640, 240), Image.Resampling.LANCZOS))

            metrics = sample_metrics(
                predicted, row["regions"], skill=skill, width=row["width"], height=row["height"]
            )
            records.append({
                "skill": skill,
                "sample_number": sample_number,
                "sample_id": row["sample_id"],
                "family": row["family"],
                "target_object": row["target_object"],
                "raw_prediction": raw,
                "teacher": row["regions"],
                "student": predicted,
                "metrics": metrics,
                "image": str(image_path),
            })
            print(f"{skill} {sample_number}/{args.per_skill}: IoU={metrics['iou']:.4f}")

        montage_row = Image.new("RGB", (640 * len(comparisons), 240), "white")
        for column, comparison in enumerate(comparisons):
            montage_row.paste(comparison, (column * 640, 0))
        row_montages.append(montage_row)

    summary = Image.new("RGB", (row_montages[0].width, 240 * len(row_montages)), "white")
    for row_number, montage_row in enumerate(row_montages):
        summary.paste(montage_row, (0, row_number * 240))
    summary_path = output_dir / "summary.png"
    summary.save(summary_path)

    result = {
        "manifest": str(manifest),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "device": str(device),
        "dtype": config["training"]["dtype"],
        "per_skill": args.per_skill,
        "records": records,
        "summary_image": str(summary_path),
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
