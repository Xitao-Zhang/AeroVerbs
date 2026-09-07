#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from distill.paths import load_config
from train import run_training


def main() -> int:
    parser = argparse.ArgumentParser(description="Final fit on all valid annotations")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--run-name", default="all-data")
    args = parser.parse_args()
    config = load_config(args.config)
    if not str(config["paths"]["manifest"]).endswith("all.jsonl"):
        raise ValueError("final training must use data/manifests/all.jsonl")
    checkpoint = run_training(config, run_name=args.run_name)
    print(f"final_checkpoint={checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
