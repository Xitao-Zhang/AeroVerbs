#!/bin/bash
cd /root/AeroVerbs
PY=/root/miniconda3/envs/aeroverbs/bin/python
CKPT=outputs/dagger_v2/round3/train3/final.pt
DATA=/root/autodl-tmp/data/trajctorys
SCENES="around_left:auto_around_left_asset_frl_apartment_bike_01_00000 around_right:auto_around_right_asset_frl_apartment_bike_01_00000 through:auto_door_opening_00000 over:auto_over_asset_frl_apartment_basket_00000 under:auto_under_table_frl_apartment_table_01_00000"

run_group () {
  local name=$1 wp=$2 tr=$3
  local out=outputs/rollout/dagger_v2_ablation/$name
  for s in $SCENES; do
    skill="${s%%:*}"; scene="${s##*:}";
    echo "===== group=$name skill=$skill scene=$scene waypoint_index=$wp turn_rate=$tr ====="
    $PY scripts/rollout.py \
      --checkpoint "$CKPT" \
      --data-root "$DATA" \
      --skill "$skill" --scene-id "$scene" \
      --waypoint-index "$wp" --turn-rate "$tr" \
      --output-dir "$out"
  done
}

run_group wp0_tr045 0 0.45
run_group wp4_tr045 4 0.45
run_group wp2_tr0   2 0
run_group wp2_tr02  2 0.2
echo "ALL ABLATIONS DONE"
