#!/usr/bin/env python3
"""Build a non-destructive AeroVerbs data overlay from MiniMind-O regions."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OVERRIDDEN = {
    "intention_heatmap.npy",
    "intention_heatmap.png",
    "initial_rgb_annotated.png",
    "heatmap_gate_region.json",
    "heatmap_gate_summary.json",
    "heatmap_gate.npy",
    "heatmap_gate_points.npy",
    "minimind_annotation.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def mirror_sample(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for item in source.iterdir():
        if item.name not in OVERRIDDEN:
            (destination / item.name).symlink_to(item, target_is_directory=item.is_dir())


def draw_annotation(image: Image.Image, points: list[list[int]]) -> Image.Image:
    output = image.convert("RGBA")
    overlay = Image.new("RGBA", output.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    polygon = [tuple(point) for point in points]
    draw.polygon(polygon, fill=(255, 59, 48, 70))
    draw.line(polygon + [polygon[0]], fill=(255, 59, 48, 255), width=4)
    return Image.alpha_composite(output, overlay).convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--source-data-root", type=Path, required=True)
    parser.add_argument("--aeroverbs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    annotations = args.annotations.expanduser().resolve()
    source_root = args.source_data_root.expanduser().resolve()
    aeroverbs_root = args.aeroverbs_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    try:
        output_root.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"output must stay under {PROJECT_ROOT}: {output_root}") from exc
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(exist_ok=False)

    sys.path.insert(0, str(aeroverbs_root))
    sys.path.insert(0, str(aeroverbs_root / "scripts"))
    from build_gate import build_gate_for_sample_dir

    scenes = source_root / "_scenes"
    if scenes.exists():
        (output_root / "_scenes").symlink_to(scenes, target_is_directory=True)

    records = read_jsonl(annotations)
    seen = set()
    for index, record in enumerate(records, 1):
        if record.get("status") != "ok" or not record.get("regions"):
            raise ValueError(f"annotation is not usable: {record.get('skill')}/{record.get('scene_id')}")
        skill = str(record["skill"])
        scene_id = str(record["scene_id"])
        identity = (skill, scene_id)
        if identity in seen:
            raise ValueError(f"duplicate annotation: {identity}")
        seen.add(identity)
        source = source_root / skill / scene_id
        destination = output_root / skill / scene_id
        if not source.is_dir():
            raise FileNotFoundError(source)
        mirror_sample(source, destination)

        region_key, points = next(iter(record["regions"].items()))
        with Image.open(source / "initial_rgb.png") as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            draw_annotation(rgb, points).save(destination / "initial_rgb_annotated.png")
        annotation_record = {
            "source": "minimind-o",
            "source_annotations": str(annotations),
            "batch_id": record.get("batch_id"),
            "skill": skill,
            "scene_id": scene_id,
            "target_object": record.get("target_object"),
            "region_key": region_key,
            "regions": record["regions"],
            "checkpoint": record.get("checkpoint"),
            "checkpoint_epoch": record.get("checkpoint_epoch"),
        }
        (destination / "minimind_annotation.json").write_text(
            json.dumps(annotation_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        gate = build_gate_for_sample_dir(destination, region_points=points)
        if gate is None:
            raise RuntimeError(f"failed to build gate for {skill}/{scene_id}")
        if index % 20 == 0:
            print(f"prepared {index}/{len(records)}", flush=True)

    provenance = {
        "schema_version": 1,
        "purpose": "AeroVerbs rollout overlay using MiniMind-O passable-region annotations",
        "annotations": str(annotations),
        "annotations_sha256": sha256(annotations),
        "source_data_root": str(source_root),
        "aeroverbs_root": str(aeroverbs_root),
        "sample_count": len(records),
        "original_files_modified": False,
    }
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
