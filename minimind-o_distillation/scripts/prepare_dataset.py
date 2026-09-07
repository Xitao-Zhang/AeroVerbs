#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distill.coordinates import canonical_region_json
from distill.paths import ensure_local_output, load_config
from distill.prompting import build_student_prompt, build_user_content

SKILLS = ("around_left", "around_right", "through", "over", "under")


def read_annotations(data_root: Path, skill: str) -> list[dict[str, Any]]:
    path = data_root / f"{skill}_annotations.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def family_key(trajectory: str) -> str:
    return re.sub(r"_\d+$", "", trajectory)


def prepare_rows(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths, data_cfg = config["paths"], config["data"]
    data_root = Path(paths["data_root"]).expanduser().resolve()
    annotation_script = Path(paths["annotation_script"]).expanduser().resolve()
    expected_size = (int(data_cfg["image_width"]), int(data_cfg["image_height"]))
    image_name = str(data_cfg.get("image_name", "initial_rgb.png"))
    prepared, invalid = [], []
    seen_ids: set[str] = set()
    for skill in SKILLS:
        for source in read_annotations(data_root, skill):
            sample_id = f"{skill}/{source.get('trajectory', '')}"
            try:
                if source.get("status") != "ok":
                    raise ValueError(f"annotation status is {source.get('status')!r}")
                if source.get("skill") != skill:
                    raise ValueError(f"skill mismatch: {source.get('skill')!r}")
                if sample_id in seen_ids:
                    raise ValueError("duplicate sample id")
                image_path = data_root / skill / source["trajectory"] / image_name
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                with Image.open(image_path) as image:
                    width, height = image.size
                if (width, height) != expected_size:
                    raise ValueError(f"unexpected image size {(width, height)}, expected {expected_size}")
                answer = canonical_region_json(source["regions"], skill, width, height)
                student_prompt = build_student_prompt(
                    width=width,
                    height=height,
                    skill=skill,
                    target_object=source["target_object"],
                )
                seen_ids.add(sample_id)
                prepared.append({
                    "sample_id": sample_id,
                    "skill": skill,
                    "trajectory": source["trajectory"],
                    "family": family_key(source["trajectory"]),
                    "target_object": source["target_object"],
                    "image_path": str(image_path),
                    "width": width,
                    "height": height,
                    "prompt": build_user_content(student_prompt),
                    "answer": answer,
                    "regions": json.loads(answer),
                    "teacher_model": source.get("model"),
                })
            except Exception as exc:
                invalid.append({"sample_id": sample_id, "error": str(exc), "source": source})
    return prepared, invalid


def group_split(rows: list[dict[str, Any]], fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("dev_val_fraction must be between 0 and 1")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["skill"], row["family"])].append(row)
    scored: dict[str, list[tuple[float, tuple[str, str]]]] = defaultdict(list)
    for key in groups:
        digest = hashlib.sha256(f"{seed}:{key[0]}:{key[1]}".encode()).digest()
        scored[key[0]].append((int.from_bytes(digest[:8], "big") / 2**64, key))
    val_keys: set[tuple[str, str]] = set()
    for candidates in scored.values():
        candidates.sort()
        target = max(1, round(sum(len(groups[key]) for _, key in candidates) * fraction))
        selected = 0
        for _, key in candidates:
            if selected >= target:
                break
            val_keys.add(key)
            selected += len(groups[key])
    train = [r for r in rows if (r["skill"], r["family"]) not in val_keys]
    val = [r for r in rows if (r["skill"], r["family"]) in val_keys]
    return train, val


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    payload = {"sample_id": [], "skill": [], "conversations": [], "image_bytes": []}
    for row in rows:
        payload["sample_id"].append(row["sample_id"])
        payload["skill"].append(row["skill"])
        payload["conversations"].append(json.dumps([
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["answer"]},
        ], ensure_ascii=False, separators=(",", ":")))
        payload["image_bytes"].append(Path(row["image_path"]).read_bytes())
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(payload), path, compression="zstd")


def counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare AeroVerbs region distillation data")
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--write-parquet", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    rows, invalid = prepare_rows(config)
    report_dir = ensure_local_output("data/reports")
    if invalid:
        write_jsonl(report_dir / "invalid_samples.jsonl", invalid)
    if len(rows) + len(invalid) != 2529:
        raise RuntimeError(
            f"expected 2529 source annotations, found {len(rows)} valid + "
            f"{len(invalid)} invalid"
        )
    train, val = group_split(rows, float(config["data"].get("dev_val_fraction", 0.1)), int(config["data"].get("split_seed", 42)))
    manifest_dir = ensure_local_output("data/manifests")
    write_jsonl(manifest_dir / "all.jsonl", rows)
    write_jsonl(manifest_dir / "dev_train.jsonl", train)
    write_jsonl(manifest_dir / "dev_val.jsonl", val)
    report = {
        "total": len(rows), "dev_train": len(train), "dev_val": len(val),
        "skill_counts": counts(rows, "skill"), "target_counts": counts(rows, "target_object"),
        "train_skill_counts": counts(train, "skill"), "val_skill_counts": counts(val, "skill"),
        "invalid": len(invalid),
        "invalid_report": str(report_dir / "invalid_samples.jsonl") if invalid else None,
    }
    (report_dir / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.write_parquet:
        parquet_dir = ensure_local_output("data/parquet")
        write_parquet(parquet_dir / "all.parquet", rows)
        write_parquet(parquet_dir / "dev_train.parquet", train)
        write_parquet(parquet_dir / "dev_val.parquet", val)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
