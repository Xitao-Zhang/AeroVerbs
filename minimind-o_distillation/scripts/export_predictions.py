#!/usr/bin/env python3
"""Split evaluate.py predictions into teacher-compatible per-skill JSONL files."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distill.paths import ensure_local_output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", default="outputs/predictions/exported")
    args = parser.parse_args()
    grouped = defaultdict(list)
    with args.predictions.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("valid"):
                skill, trajectory = row["sample_id"].split("/", 1)
                grouped[skill].append({
                    "skill": skill,
                    "trajectory": trajectory,
                    "model": "minimind-o-aeroverbs-distilled",
                    "status": "ok",
                    "regions": row["parsed"],
                })
    output = ensure_local_output(args.output_dir)
    for skill, rows in grouped.items():
        with (output / f"{skill}_predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({skill: len(rows) for skill, rows in grouped.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
