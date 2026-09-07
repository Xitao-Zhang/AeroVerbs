# AeroVerbs MiniMind-O Distillation

This standalone project distils the existing GPT-5.5 UAV passable-region
annotations into MiniMind-O without changing either upstream repository.

Inputs are read-only:

- `/data/UAV_Basic_Skills/trajctorys`
- `/home/zxt/UAV_Basic_Skills/scripts/annotate_table_passage.py`
- `/home/zxt/minimind-o`

All generated manifests, reports, checkpoints, predictions, logs, and temporary
files are written below this directory.

## Workflow

```bash
cd /home/zxt/UAV_Basic_Skills/AeroVerbs/minimind-o_distillation
conda activate minimind

# 1. Build and audit manifests. This does not call an API.
python3 scripts/prepare_dataset.py --config configs/dev.yaml --write-parquet
python3 scripts/audit_dataset.py --config configs/dev.yaml

# 2. Fast code/data checks. The model check loads one batch but does not train.
python3 -m unittest discover -s tests -v
python3 scripts/smoke_test.py --config configs/dev.yaml

# 3. Development training (run manually when ready).
python3 scripts/train.py --config configs/dev.yaml

# 4. Offline generation metrics.
python3 scripts/evaluate.py \
  --config configs/dev.yaml \
  --checkpoint outputs/checkpoints/dev/<run-id>/best.pt

# 5. Once hyperparameters are fixed, train from the original base weight on all
#    valid annotations. This command is intentionally never run automatically.
python3 scripts/train_final.py --config configs/final.yaml
```

The target is the teacher's original four-corner JSON, for example:

```json
{"left":[[148,214],[274,214],[274,268],[148,268]]}
```

The original JSONL `image` field is deliberately ignored because it contains
old paths. Images are always resolved as
`<data_root>/<skill>/<trajectory>/initial_rgb.png`.

## Training policy

SigLIP2, Talker, and all audio components are frozen. The default development
run trains `vision_proj`, the final two Thinker blocks, Thinker norm, and the
tied LM head/token embedding. A Thinker-only forward path avoids computing the
unused Talker branch; this changes training execution, not model architecture.

The development manifest has a deterministic group-aware validation split.
The final configuration uses `all.jsonl`, so every annotation participates in
the final fit. Generalisation of that final model must be assessed on newly
collected scenes rather than on its training images.

Dataset preparation rejects malformed or zero-area teacher regions and records
them in `data/reports/invalid_samples.jsonl`; unusable labels are never silently
fed to training. The current source contains 2529 rows: 2525 valid examples and
4 zero-area `over/refrigerator` labels that are excluded.
