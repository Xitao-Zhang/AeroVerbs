#!/bin/bash
cd /root/AeroVerbs
PY=/root/miniconda3/envs/aeroverbs/bin/python
CKPT=outputs/dagger_v2c/round3/train3/final.pt
DATA=/root/autodl-tmp/data/trajctorys
OUT=outputs/rollout/dagger_v2c_eval_10pct
mkdir -p "$OUT"

for sk in around_left around_right through over under; do
  n=$(ls "$DATA/$sk" 2>/dev/null | wc -l)
  echo "===== skill=$sk total=$n sample_10pct ====="
  for scene in $(ls "$DATA/$sk" 2>/dev/null | awk 'NR % 10 == 1'); do
    echo "--- $sk / $scene ---"
    $PY scripts/rollout.py \
      --checkpoint "$CKPT" \
      --data-root "$DATA" \
      --skill "$sk" --scene-id "$scene" \
      --output-dir "$OUT" \
      --no-render
  done
done
echo "ALL 10PCT ROLLOUTS DONE"
