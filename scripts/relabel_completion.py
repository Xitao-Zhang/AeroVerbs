#!/usr/bin/env python
"""Relabel BC samples with the unified completion judge.

For every sample under ``<data_root>/<skill>/<scene_id>/`` this script computes
the unified completion arrays and writes each array as a separate ``.npy`` file
inside that trajectory's own folder:

    completion.npy        float32 [T]  unified completion label (0/1)
    semantic_done.npy     float32 [T]  semantic relationship done (latched)
    far_side.npy          float32 [T]  current point past the object far side
    clearance.npy         float32 [T]  clearance to the nearest object surface
    safe_clearance.npy    float32 [T]  clearance >= d_exit
    ground_ok.npy         float32 [T]  ground clearance satisfied

``dataset.py`` only needs ``completion.npy``; the other files are diagnostics.

Samples are processed in parallel with ``ProcessPoolExecutor`` and progress is
printed as ``[done/total] percent | elapsed | eta | ok/skip/fail``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from aeroverbs.config import CompletionConfig, load_config
from aeroverbs.data.completion import CompletionJudge, make_clearance_provider
from aeroverbs.data.dataset import build_sample_index


def parse_args():
    ap = argparse.ArgumentParser(description="Relabel samples with the unified completion judge.")
    ap.add_argument("--data-root", default="/root/autodl-tmp/data/trajctorys")
    ap.add_argument("--skills", default="around_left,around_right,through,over,under")
    ap.add_argument("--config", default=None, help="Optional YAML config for completion thresholds.")
    ap.add_argument("--audit-out", default=None, help="Optional path for the audit JSONL.")
    ap.add_argument("--workers", type=int, default=0, help="Number of parallel workers (0 = auto).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip samples that already have completion.npy.")
    return ap.parse_args()


def _load_positions(sample_dir: str) -> np.ndarray:
    with open(os.path.join(sample_dir, "frames.json"), encoding="utf-8") as f:
        frames = json.load(f)
    return np.asarray([fr["camera_position_std"] for fr in frames], dtype=float)


def _relabel_one(
    sample_dir: str,
    data_root: str,
    completion_cfg: CompletionConfig,
    skip_existing: bool = False,
) -> dict:
    """Compute and save all completion arrays for one trajectory."""
    with open(os.path.join(sample_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    skill = meta.get("skill") or os.path.basename(os.path.dirname(sample_dir))
    scene_id = meta.get("scene_id") or os.path.basename(sample_dir)

    if skip_existing and os.path.exists(os.path.join(sample_dir, "completion.npy")):
        return {
            "skill": skill,
            "scene_id": scene_id,
            "status": "skipped",
            "reason": "exists",
        }

    positions = _load_positions(sample_dir)
    provider = make_clearance_provider(sample_dir, data_root, meta)
    judge = CompletionJudge(meta, clearance_provider=provider, cfg=completion_cfg)
    arrays = judge.label_sequence(positions)

    # Each array is saved separately into the trajectory's own folder.
    for key, arr in arrays.items():
        np.save(
            os.path.join(sample_dir, f"{key}.npy"),
            np.asarray(arr, dtype=np.float32),
        )

    completion = arrays["completion"]
    return {
        "skill": skill,
        "scene_id": scene_id,
        "frames": int(len(positions)),
        "valid_sides": bool(judge.geometry.valid_sides),
        "semantic_done_any": bool(arrays["semantic_done"].any()),
        "far_side_any": bool(arrays["far_side"].any()),
        "completion_any": bool(completion.any()),
        "completion_frames": int(completion.sum()),
        "completion_frac": round(float(completion.mean()), 4),
        "final_clearance": round(float(arrays["clearance"][-1]), 3),
        "status": "ok",
    }


def _format_progress(done: int, total: int, elapsed: float, statuses: dict, last: dict) -> str:
    pct = 100.0 * done / max(total, 1)
    eta = elapsed / max(done, 1) * (total - done)
    last_str = ""
    if last:
        last_str = f" | last={last['skill']}/{last['scene_id']}"
        if "completion_frac" in last:
            last_str += f" comp_frac={last['completion_frac']}"
    return (
        f"[{done}/{total}] {pct:5.1f}% | elapsed={elapsed:.1f}s | eta={eta:.1f}s | "
        f"ok={statuses.get('ok', 0)} skipped={statuses.get('skipped', 0)} "
        f"failed={statuses.get('failed', 0)}{last_str}"
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    completion_cfg = cfg.completion

    skills = [s.strip() for s in args.skills.split(",") if s.strip()]
    samples = build_sample_index(args.data_root, skills)
    if not samples:
        raise SystemExit(f"No samples found under {args.data_root} for skills {skills}")

    workers = int(args.workers) if args.workers > 0 else min(8, os.cpu_count() or 1)
    workers = max(1, workers)

    print(
        f"relabel_completion: data_root={args.data_root} skills={skills} "
        f"samples={len(samples)} workers={workers} completion_cfg={completion_cfg}"
    )

    statuses = {"ok": 0, "skipped": 0, "failed": 0}
    audit_rows = []
    t_start = time.time()
    done = 0

    def handle_result(res: dict) -> None:
        nonlocal done
        if res["status"] == "ok":
            statuses["ok"] += 1
            audit_rows.append(res)
        elif res["status"] == "skipped":
            statuses["skipped"] += 1
        else:
            statuses["failed"] += 1
            print(f"  FAILED {res.get('skill', '?')}/{res.get('scene_id', '?')}: {res.get('reason')}", flush=True)
        done += 1
        print(
            "\r" + _format_progress(done, len(samples), time.time() - t_start, statuses, res),
            end="",
            flush=True,
        )

    if workers <= 1:
        for sample_dir in samples:
            try:
                res = _relabel_one(sample_dir, args.data_root, completion_cfg,
                                   skip_existing=args.skip_existing)
            except Exception as exc:
                res = {
                    "skill": os.path.basename(os.path.dirname(sample_dir)),
                    "scene_id": os.path.basename(sample_dir),
                    "status": "failed",
                    "reason": repr(exc),
                }
            handle_result(res)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(_relabel_one, sample_dir, args.data_root, completion_cfg,
                          args.skip_existing)
                for sample_dir in samples
            ]
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                except Exception as exc:
                    res = {
                        "skill": "?",
                        "scene_id": "?",
                        "status": "failed",
                        "reason": repr(exc),
                    }
                handle_result(res)

    print()  # finish the progress line
    elapsed = time.time() - t_start
    print(
        f"done: ok={statuses['ok']} skipped={statuses['skipped']} "
        f"failed={statuses['failed']} total_elapsed={elapsed:.1f}s"
    )

    no_pos = [r for r in audit_rows if not r["completion_any"]]
    print(f"audit: {len(no_pos)}/{len(audit_rows)} samples have no completion positive frames")
    if no_pos:
        print("these samples need a longer exit segment or relaxed thresholds:")
        for r in no_pos[:20]:
            print(
                f"  - {r['skill']}/{r['scene_id']} (valid_sides={r['valid_sides']}, "
                f"semantic_done_any={r['semantic_done_any']}, far_side_any={r['far_side_any']})"
            )

    if args.audit_out:
        audit_rows.sort(key=lambda r: (r["skill"], r["scene_id"]))
        os.makedirs(os.path.dirname(os.path.abspath(args.audit_out)), exist_ok=True)
        with open(args.audit_out, "w", encoding="utf-8") as f:
            for row in audit_rows:
                f.write(json.dumps(row) + "\n")
        print(f"audit written to {args.audit_out}")


if __name__ == "__main__":
    main()
