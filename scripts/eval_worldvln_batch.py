#!/usr/bin/env python3
"""Resumable WorldVLN evaluation over AeroVerbs frozen manifests."""
from __future__ import annotations
import argparse, json, os, sys, time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from rollout_worldvln import run_one_sample

def load_manifest(path):
    d=json.loads(Path(path).read_text()); samples=d["samples"]
    if int(d.get("num_samples",len(samples))) != len(samples): raise ValueError("manifest count mismatch")
    if len({(s["skill"],s["scene_id"]) for s in samples}) != len(samples): raise ValueError("duplicate manifest sample")
    return d

def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest", required=True); p.add_argument("--data-root", required=True)
    p.add_argument("--server-url", default="http://127.0.0.1:8001"); p.add_argument("--output-dir", required=True)
    p.add_argument("--instruction", default=None)
    p.add_argument("--run-id", default=None); p.add_argument("--retry-infrastructure-errors", action="store_true")
    p.add_argument("--max-scenes", type=int, default=0); p.add_argument("--max-segments", type=int, default=12); p.add_argument("--step", type=int, default=16)
    p.add_argument("--no-collision-check", action="store_true")
    p.add_argument("--action-head-mode", default="tsformer_latent", choices=["tsformer_latent","actionhead_ref_vit"])
    p.add_argument("--request-timeout-s", type=float, default=600); p.add_argument("--drone-radius", type=float, default=.15)
    p.add_argument("--rgb-width", type=int, default=640); p.add_argument("--rgb-height", type=int, default=480); p.add_argument("--rgb-gpu-device-id", type=int, default=-1)
    args=p.parse_args(); manifest=load_manifest(args.manifest); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    result_path=out/"results.jsonl"; latest={}
    if result_path.exists():
        for line in result_path.read_text().splitlines():
            if line.strip():
                r=json.loads(line); latest[(r.get("skill"),r.get("scene_id"))]=r
    run_id=args.run_id or time.strftime("%Y%m%d-%H%M%S")
    pending=[]
    for s in manifest["samples"]:
        key=(s["skill"],s["scene_id"]); old=latest.get(key)
        if old is None or (args.retry_infrastructure_errors and old.get("status")=="infrastructure_error"): pending.append(s)
    if args.max_scenes>0: pending=pending[:args.max_scenes]
    for s in pending:
        sample=Path(args.data_root)/s["relative_path"]
        # ``args`` already contains the optional CLI run_id.  Pass the
        # resolved run id exactly once so Namespace construction is valid.
        ns=argparse.Namespace(**{**vars(args), "sample_dir": str(sample), "run_id": run_id})
        try: r=run_one_sample(sample, ns); r.update({"manifest":Path(args.manifest).name})
        except Exception as exc: r={"model":"worldvln","skill":s["skill"],"scene_id":s["scene_id"],"status":"infrastructure_error","reason":repr(exc)}
        # Keep the OpenVLA-compatible per-sample trajectory artifact while
        # retaining the raw WorldVLN result JSONL.
        traj_dir = out / "trajectories"; traj_dir.mkdir(parents=True, exist_ok=True)
        traj_name = f"{s['skill']}__{s['scene_id']}.json"
        traj_payload = {
            "points": r.get("trajectory", []),
            "semantic_label": s["skill"],
            "instruction": r.get("instruction"),
            "worldvln_steps": r.get("worldvln_steps", []),
        }
        (traj_dir / traj_name).write_text(json.dumps(traj_payload, ensure_ascii=False, indent=2) + "\n")
        r["trajectory"] = f"trajectories/{traj_name}"
        with result_path.open("a",encoding="utf-8") as f: f.write(json.dumps(r,ensure_ascii=False)+"\n"); f.flush(); os.fsync(f.fileno())
        latest[(r.get("skill"),r.get("scene_id"))]=r
        print(f"{r.get('skill')} {r.get('scene_id')} {r.get('status')}",flush=True)
    selected=[latest[(s["skill"],s["scene_id"])] for s in manifest["samples"] if (s["skill"],s["scene_id"]) in latest]
    counts=Counter(r.get("status","unknown") for r in selected); by_skill={}
    for skill in sorted({s["skill"] for s in manifest["samples"]}):
        rows=[r for r in selected if r.get("skill")==skill]; c=Counter(r.get("status","unknown") for r in rows); n=sum(1 for s in manifest["samples"] if s["skill"]==skill)
        by_skill[skill]={"expected":n,"processed":len(rows),"success":c.get("success",0),"success_rate":c.get("success",0)/n,"status_counts":dict(c)}
    summary={"model":"worldvln","manifest":str(Path(args.manifest).resolve()),"expected_samples":len(manifest["samples"]),"processed_samples":len(selected),"missing_samples":len(manifest["samples"])-len(selected),"complete":len(selected)==len(manifest["samples"]),"status_counts":dict(counts),"success_count":counts.get("success",0),"success_rate":counts.get("success",0)/len(manifest["samples"]),"per_skill":by_skill}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
