#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distill.coordinates import EXPECTED_REGION_KEY, parse_region_text, sample_metrics
from distill.model_loader import greedy_generate, inference_input, load_base_model, load_delta_checkpoint
from distill.paths import load_config, resolve_project_path
from distill.prompting import build_student_prompt, build_user_content


SKILLS = ("around_left", "around_right", "through", "over", "under")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def annotation_index(data_root: Path) -> dict[tuple[str, str], dict]:
    index = {}
    for skill in SKILLS:
        for row in read_jsonl(data_root / f"{skill}_annotations.jsonl"):
            key = (skill, row["trajectory"])
            if key in index:
                raise ValueError(f"duplicate source annotation: {key}")
            index[key] = row
    return index


def draw_prediction(image: Image.Image, points: list[list[int]], title: str) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    polygon = [tuple(point) for point in points]
    draw.line(polygon + [polygon[0]], fill="#ff3b30", width=5, joint="curve")
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), title, font=font)
    draw.rectangle((0, 0, box[2] + 14, box[3] + 14), fill="black")
    draw.text((7, 7), title, fill="#ff3b30", font=font)
    return output


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate AeroVerbs rollout batches with MiniMind-O")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    rollout_root = args.rollout_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_root = resolve_project_path(args.output)
    try:
        output_root.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"output must stay under {PROJECT_ROOT}: {output_root}") from exc
    # Refuse reuse so an earlier experiment can never be silently overwritten.
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=False)

    teacher_by_sample = annotation_index(data_root)
    model, tokenizer, device, _ = load_base_model(config)
    checkpoint = load_delta_checkpoint(model, args.checkpoint)
    model.eval()

    started = time.time()
    all_records = []
    batch_summaries = []
    seen: set[tuple[str, str]] = set()
    for batch_id in range(1, 6):
        source_batch = rollout_root / f"batch_{batch_id}"
        rollout_rows = read_jsonl(source_batch / "results.jsonl")
        batch_output = output_root / f"batch_{batch_id}"
        preview_dir = batch_output / "annotated"
        preview_dir.mkdir(parents=True)
        batch_records = []

        for rollout in rollout_rows:
            skill, scene_id = rollout["skill"], rollout["scene_id"]
            key = (skill, scene_id)
            if key in seen:
                raise ValueError(f"duplicate rollout sample across batches: {key}")
            seen.add(key)
            teacher = teacher_by_sample.get(key)
            if teacher is None:
                raise ValueError(f"missing target-object metadata for {key}")
            image_path = data_root / skill / scene_id / "initial_rgb.png"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)

            with Image.open(image_path) as source:
                image = source.convert("RGB")
                width, height = image.size
                pixels = model.vision_processor(images=image, return_tensors="pt").pixel_values.to(device)
            prompt = build_user_content(build_student_prompt(
                width=width,
                height=height,
                skill=skill,
                target_object=teacher["target_object"],
            ))
            ids, _ = inference_input(
                tokenizer, prompt, int(config["data"].get("image_token_len", 64)), device
            )
            raw = greedy_generate(
                model, tokenizer, ids, pixels,
                max_new_tokens=int(config["training"].get("max_new_tokens", 96)),
            )
            try:
                regions = parse_region_text(raw, skill=skill, width=width, height=height)
                metrics = sample_metrics(
                    regions, teacher["regions"], skill=skill, width=width, height=height
                )
                status, error = "ok", None
                preview_name = f"{skill}__{scene_id}.png"
                points = regions[EXPECTED_REGION_KEY[skill]]
                draw_prediction(image, points, f"MiniMind-O | {skill} | {teacher['target_object']}").save(
                    preview_dir / preview_name
                )
            except Exception as exc:
                regions, metrics, status, error, preview_name = None, None, "error", str(exc), None

            record = {
                "batch_id": batch_id,
                "skill": skill,
                "trajectory": scene_id,
                "scene_id": scene_id,
                "target_object": teacher["target_object"],
                "image": str(image_path),
                "model": "minimind-o",
                "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "status": status,
                "regions": regions,
                "raw_prediction": raw,
                "error": error,
                "annotated_preview": str(preview_dir / preview_name) if preview_name else None,
                "teacher_model": teacher.get("model"),
                "teacher_regions": teacher.get("regions"),
                "teacher_metrics": metrics,
                "original_rollout_status": rollout.get("status"),
                "original_rollout_result": str(source_batch / "results.jsonl"),
            }
            batch_records.append(record)
            all_records.append(record)
            if len(all_records) % 20 == 0:
                print(f"annotated {len(all_records)} samples")

        write_jsonl(batch_output / "minimind_annotations.jsonl", batch_records)
        summary = {
            "batch_id": batch_id,
            "count": len(batch_records),
            "ok": sum(row["status"] == "ok" for row in batch_records),
            "errors": sum(row["status"] != "ok" for row in batch_records),
            "skill_counts": dict(Counter(row["skill"] for row in batch_records)),
        }
        (batch_output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        batch_summaries.append(summary)

    write_jsonl(output_root / "minimind_annotations.jsonl", all_records)
    skill_ious: dict[str, list[float]] = defaultdict(list)
    for record in all_records:
        if record["teacher_metrics"]:
            skill_ious[record["skill"]].append(record["teacher_metrics"]["iou"])
    final_summary = {
        "source_rollout_root": str(rollout_root),
        "source_data_root": str(data_root),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "device": str(device),
        "dtype": config["training"]["dtype"],
        "total": len(all_records),
        "unique_samples": len(seen),
        "ok": sum(row["status"] == "ok" for row in all_records),
        "errors": sum(row["status"] != "ok" for row in all_records),
        "skill_counts": dict(Counter(row["skill"] for row in all_records)),
        "teacher_mean_iou_by_skill": {
            skill: sum(values) / len(values) for skill, values in skill_ious.items()
        },
        "batches": batch_summaries,
        "elapsed_sec": time.time() - started,
        "original_files_modified": False,
    }
    (output_root / "summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
