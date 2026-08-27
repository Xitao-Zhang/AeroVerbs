#!/usr/bin/env python
"""Train the AeroVerbs policy.

Usage (from the AeroVerbs repo root)::

    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --experiment no_heatmap --name run1
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aeroverbs.config import load_config
from aeroverbs.training import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AeroVerbs policy")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment name (outputs/<experiment>/...)")
    parser.add_argument("--name", type=str, default=None, help="Run name; default is a timestamp")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {}
    if args.experiment is not None:
        overrides.setdefault("train", {})["experiment"] = args.experiment
    if args.name is not None:
        overrides.setdefault("train", {})["run_name"] = args.name
    cfg = load_config(args.config, overrides=overrides)
    train(cfg, resume_from=args.resume)


if __name__ == "__main__":
    main()
