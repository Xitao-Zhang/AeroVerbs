#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distill.dataset import CoverageRepeatSampler, RegionCollator, RegionSFTDataset
from distill.metrics import aggregate_metrics, evaluate_prediction
from distill.model_loader import (
    causal_text_loss,
    checkpoint_payload,
    configure_trainable_parameters,
    greedy_generate,
    inference_input,
    load_base_model,
    thinker_forward,
    trainable_parameter_report,
)
from distill.paths import ensure_local_output, load_config, resolve_project_path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda" or dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def move_batch(batch, device: torch.device) -> dict:
    for key in ("input_ids", "labels", "attention_mask", "pixel_values"):
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


@torch.inference_mode()
def validation_loss(model, loader, device, dtype) -> float:
    model.eval()
    losses = []
    for batch in loader:
        batch = move_batch(batch, device)
        with autocast_context(device, dtype):
            output = thinker_forward(
                model,
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
            )
            loss = causal_text_loss(output.logits, batch["labels"])
        losses.append(float(loss.item()))
    return sum(losses) / len(losses) if losses else math.nan


def select_generation_rows(rows: list[dict], per_skill: int, seed: int) -> list[dict]:
    selected = []
    for skill in ("around_left", "around_right", "through", "over", "under"):
        candidates = [row for row in rows if row["skill"] == skill]
        random.Random(f"{seed}:{skill}").shuffle(candidates)
        selected.extend(candidates[:per_skill])
    return selected


@torch.inference_mode()
def generation_evaluation(model, tokenizer, rows, config, device) -> dict:
    model.eval()
    records = []
    for row in rows:
        with Image.open(row["image_path"]) as image:
            pixels = model.vision_processor(
                images=image.convert("RGB"), return_tensors="pt"
            ).pixel_values.to(device)
        ids, _ = inference_input(
            tokenizer, row["prompt"], int(config["data"].get("image_token_len", 64)), device
        )
        prediction = greedy_generate(
            model, tokenizer, ids, pixels,
            max_new_tokens=int(config["training"].get("max_new_tokens", 96)),
        )
        records.append(evaluate_prediction(row, prediction))
    summary = aggregate_metrics(records)
    skill_scores = {}
    for skill in ("around_left", "around_right", "through", "over", "under"):
        items = [record for record in records if record["skill"] == skill]
        skill_scores[skill] = {
            "count": len(items),
            "valid_rate": sum(bool(item.get("valid")) for item in items) / len(items) if items else 0.0,
            "effective_mean_iou": sum(float(item.get("iou", 0.0)) for item in items) / len(items) if items else 0.0,
        }
    active = [item["effective_mean_iou"] for item in skill_scores.values() if item["count"]]
    return {
        "count": len(records),
        "macro_iou": sum(active) / len(active) if active else 0.0,
        "overall": summary.get("overall", {}),
        "by_skill": skill_scores,
    }


def save_checkpoint(path: Path, model, config, epoch, optimizer, scheduler, metrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint_payload(
        model,
        config,
        epoch=epoch,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
    ), temp)
    temp.replace(path)


def run_training(config: dict, *, run_name: str | None = None) -> Path:
    train_cfg, data_cfg = config["training"], config["data"]
    seed_everything(int(train_cfg.get("seed", 42)))
    model, tokenizer, device, dtype = load_base_model(config)
    parameter_counts = configure_trainable_parameters(model, config)
    processor = model.vision_processor
    manifest = resolve_project_path(config["paths"]["manifest"])
    train_set = RegionSFTDataset(
        manifest,
        tokenizer,
        processor,
        max_seq_len=int(data_cfg["max_seq_len"]),
        image_token_len=int(data_cfg.get("image_token_len", 64)),
    )
    generation_per_skill = int(train_cfg.get("generation_eval_per_skill", 0))
    sampler = CoverageRepeatSampler(
        train_set.rows,
        data_cfg.get("oversample_weights", {}),
        seed=int(train_cfg.get("seed", 42)),
    )
    collator = RegionCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_set,
        batch_size=int(train_cfg["batch_size"]),
        sampler=sampler,
        collate_fn=collator,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )
    val_loader = None
    val_set = None
    val_value = config["paths"].get("val_manifest")
    if val_value:
        val_set = RegionSFTDataset(
            resolve_project_path(val_value), tokenizer, processor,
            max_seq_len=int(data_cfg["max_seq_len"]),
            image_token_len=int(data_cfg.get("image_token_len", 64)),
        )
        val_loader = DataLoader(
            val_set,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            collate_fn=collator,
            num_workers=int(train_cfg.get("num_workers", 4)),
            pin_memory=device.type == "cuda",
        )
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        parameters,
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    accumulation = int(train_cfg.get("accumulation_steps", 1))
    epochs = int(train_cfg["epochs"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_updates = max(1, epochs * updates_per_epoch)
    scheduler = LambdaLR(optimizer, lambda step: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step, total_updates) / total_updates)))
    run_name = run_name or time.strftime("%Y%m%d-%H%M%S")
    output_dir = ensure_local_output(train_cfg["output_dir"]) / run_name
    log_dir = ensure_local_output(train_cfg["log_dir"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (log_dir / "trainable_parameters.txt").write_text(trainable_parameter_report(model), encoding="utf-8")
    metrics_path = log_dir / "metrics.jsonl"
    print(
        f"device={device} dtype={dtype} train_unique={len(train_set)} "
        f"train_epoch_samples={len(sampler)} val={len(val_loader.dataset) if val_loader else 0}"
    )
    print(f"parameters trainable={parameter_counts['trainable']:,} total={parameter_counts['total']:,}")

    generation_source = val_set.rows if val_set is not None else train_set.rows
    generation_rows = select_generation_rows(
        generation_source, generation_per_skill, int(train_cfg.get("seed", 42))
    ) if generation_per_skill > 0 else []
    if generation_rows:
        source_name = "validation" if val_set is not None else "training-monitor"
        print(f"generation_eval={len(generation_rows)} source={source_name}")

    best_value = float("inf")
    best_generation_score = float("-inf")
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        model.train()
        running, steps, started = 0.0, 0, time.time()
        for step, batch in enumerate(train_loader, 1):
            batch = move_batch(batch, device)
            with autocast_context(device, dtype):
                output = thinker_forward(
                    model,
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                )
                raw_loss = causal_text_loss(output.logits, batch["labels"])
                loss = raw_loss / accumulation
            loss.backward()
            running += float(raw_loss.item())
            steps += 1
            if step % accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(parameters, float(train_cfg.get("grad_clip", 1.0)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % int(train_cfg.get("log_interval", 10)) == 0:
                print(f"epoch={epoch}/{epochs} step={step}/{len(train_loader)} loss={running/steps:.5f}")
        train_loss = running / max(steps, 1)
        val_loss = validation_loss(model, val_loader, device, dtype) if val_loader else None
        generation = generation_evaluation(
            model, tokenizer, generation_rows, config, device
        ) if generation_rows else None
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "elapsed_sec": time.time() - started,
            "generation": generation,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        save_checkpoint(output_dir / "last.pt", model, config, epoch, optimizer, scheduler, record)
        if generation is not None:
            selection = float(generation["macro_iou"])
            improved = selection > best_generation_score
            if improved:
                best_generation_score = selection
        else:
            selection = val_loss if val_loss is not None else train_loss
            improved = selection < best_value
            if improved:
                best_value = selection
        if improved:
            save_checkpoint(output_dir / "best.pt", model, config, epoch, optimizer, scheduler, record)
        print(json.dumps(record, ensure_ascii=False))
    return output_dir / "best.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MiniMind-O on AeroVerbs region annotations")
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--run-name")
    args = parser.parse_args()
    checkpoint = run_training(load_config(args.config), run_name=args.run_name)
    print(f"best_checkpoint={checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
