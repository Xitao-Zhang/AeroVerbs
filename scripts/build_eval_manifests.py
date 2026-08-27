#!/usr/bin/env python
"""Build deterministic, disjoint evaluation manifests from the fixed test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Mapping, Sequence


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from aeroverbs.data.dataset import build_sample_index, split_samples
from dagger_collect import _sample_category, _skill_seed


SKILLS = ("around_left", "around_right", "through", "over", "under")
DEFAULT_BATCH_COUNTS: Dict[str, List[int]] = {
    "around_left": [15, 15, 15, 15, 15],
    "around_right": [15, 15, 15, 15, 15],
    "through": [10, 10, 10, 10, 10],
    "over": [9, 9, 9, 9, 8],
    "under": [2, 2, 2, 2, 2],
}


def _balanced_category_batches(
    samples: Sequence[str],
    counts: Sequence[int],
    seed: int,
) -> List[List[str]]:
    """Spread each category across batches while respecting exact capacities."""
    groups: Dict[str, List[str]] = defaultdict(list)
    for sample in sorted(samples):
        groups[_sample_category(sample)].append(sample)

    rng = random.Random(int(seed))
    categories = sorted(groups)
    rng.shuffle(categories)
    for category in categories:
        rng.shuffle(groups[category])

    selected: List[List[str]] = [[] for _ in counts]
    category_counts = [Counter() for _ in counts]
    for category in categories:
        tie_order = list(range(len(counts)))
        rng.shuffle(tie_order)
        tie_rank = {batch_index: rank for rank, batch_index in enumerate(tie_order)}
        for sample in groups[category]:
            eligible = [
                batch_index
                for batch_index, capacity in enumerate(counts)
                if len(selected[batch_index]) < int(capacity)
            ]
            if not eligible:
                raise ValueError("batch capacities were exhausted before all samples")
            batch_index = min(
                eligible,
                key=lambda index: (
                    category_counts[index][category],
                    len(selected[index]) / int(counts[index]),
                    tie_rank[index],
                ),
            )
            selected[batch_index].append(sample)
            category_counts[batch_index][category] += 1
    return selected


def make_batches(
    test_samples: Sequence[str],
    batch_counts: Mapping[str, Sequence[int]],
    sample_seed: int,
) -> List[List[dict]]:
    """Partition test samples into deterministic category-stratified batches."""
    batch_lengths = {len(counts) for counts in batch_counts.values()}
    if len(batch_lengths) != 1:
        raise ValueError("every skill must define the same number of batches")
    num_batches = next(iter(batch_lengths), 0)
    if num_batches < 1:
        raise ValueError("at least one batch is required")

    samples_by_skill: Dict[str, List[str]] = {skill: [] for skill in batch_counts}
    for sample in test_samples:
        skill = os.path.basename(os.path.dirname(sample))
        if skill in samples_by_skill:
            samples_by_skill[skill].append(sample)

    batches: List[List[dict]] = [[] for _ in range(num_batches)]
    for skill, counts in batch_counts.items():
        samples = samples_by_skill[skill]
        expected = sum(int(count) for count in counts)
        if len(samples) != expected:
            raise ValueError(
                f"{skill}: expected {expected} test samples from batch counts, "
                f"found {len(samples)}"
            )
        selected_by_batch = _balanced_category_batches(
            samples,
            counts,
            seed=_skill_seed(sample_seed, skill),
        )
        for batch_index, selected in enumerate(selected_by_batch):
            for sample in selected:
                scene_id = os.path.basename(sample)
                batches[batch_index].append(
                    {
                        "skill": skill,
                        "scene_id": scene_id,
                        "relative_path": f"{skill}/{scene_id}",
                        "category": _sample_category(sample),
                    }
                )
    return batches


def _write_json(path: str, payload: dict) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_manifests(
    batches: Sequence[Sequence[dict]],
    output_dir: str,
    *,
    data_root: str,
    split_seed: int,
    sample_seed: int,
) -> str:
    """Write one readable JSON manifest per batch plus a checksummed index."""
    os.makedirs(output_dir, exist_ok=True)
    manifest_entries = []
    global_skill_counts: Counter = Counter()
    seen = set()

    for batch_index, batch in enumerate(batches, start=1):
        identifiers = [(item["skill"], item["scene_id"]) for item in batch]
        duplicates = seen.intersection(identifiers)
        if duplicates:
            raise ValueError(f"samples repeated across batches: {sorted(duplicates)}")
        seen.update(identifiers)
        skill_counts = Counter(item["skill"] for item in batch)
        global_skill_counts.update(skill_counts)
        filename = f"batch_{batch_index}.json"
        path = os.path.join(output_dir, filename)
        payload = {
            "schema_version": 1,
            "batch_id": batch_index,
            "split": "test",
            "data_root": os.path.abspath(data_root),
            "split_seed": int(split_seed),
            "sample_seed": int(sample_seed),
            "num_samples": len(batch),
            "skill_counts": dict(skill_counts),
            "samples": list(batch),
        }
        digest = _write_json(path, payload)
        manifest_entries.append(
            {
                "batch_id": batch_index,
                "file": filename,
                "num_samples": len(batch),
                "sha256": digest,
            }
        )

    index = {
        "schema_version": 1,
        "split": "test",
        "data_root": os.path.abspath(data_root),
        "split_seed": int(split_seed),
        "sample_seed": int(sample_seed),
        "total_samples": len(seen),
        "batch_sizes": [len(batch) for batch in batches],
        "skill_counts": dict(global_skill_counts),
        "manifests": manifest_entries,
    }
    index_path = os.path.join(output_dir, "manifest_index.json")
    _write_json(index_path, index)
    return index_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build five fixed, disjoint manifests covering the AeroVerbs test split."
    )
    parser.add_argument(
        "--data-root",
        default="/root/autodl-tmp/data/trajctorys",
    )
    parser.add_argument("--output-dir", default="eval_manifests")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = build_sample_index(args.data_root, list(SKILLS))
    _, _, test_samples = split_samples(samples, seed=args.split_seed)
    batches = make_batches(
        test_samples,
        DEFAULT_BATCH_COUNTS,
        sample_seed=args.sample_seed,
    )
    index_path = write_manifests(
        batches,
        args.output_dir,
        data_root=args.data_root,
        split_seed=args.split_seed,
        sample_seed=args.sample_seed,
    )
    print(f"wrote {len(batches)} batches and {sum(map(len, batches))} samples")
    print(f"index: {index_path}")


if __name__ == "__main__":
    main()
