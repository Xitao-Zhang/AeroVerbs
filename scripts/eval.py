#!/usr/bin/env python
"""Evaluate a trained AeroVerbs policy on the validation or test split.

Usage::

    python scripts/eval.py --checkpoint outputs/<exp>/<run>/final.pt --config configs/default.yaml --split test
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aeroverbs.config import load_config
from aeroverbs.training.trainer import _evaluate, make_dataloaders, make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AeroVerbs policy")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(cfg.resolve_device())

    state = torch.load(args.checkpoint, map_location=device)
    model = make_model(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    _, val_loader, test_loader = make_dataloaders(cfg)
    loader = val_loader if args.split == "val" else test_loader
    metrics = _evaluate(model, loader, cfg, device)
    print(f"evaluation metrics ({args.split}):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
