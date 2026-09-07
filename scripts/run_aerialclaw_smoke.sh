#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VISFLY_PYTHON="${VISFLY_PYTHON:-/home/zxt/miniconda3/envs/visfly/bin/python}"
AERIALCLAW_ROOT="${AERIALCLAW_ROOT:-/home/zxt/AerialClaw}"
AERIALCLAW_PYTHON="${AERIALCLAW_PYTHON:-/home/zxt/miniconda3/envs/aerialclaw-eval/bin/python}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/trajctorys}"
SKILL="${SKILL:-around_left}"
SCENE_ID="${SCENE_ID:-auto_around_left_asset_frl_apartment_lamp_02_00393}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/rollout/aerialclaw_smoke_prompt_v2}"

SAMPLE_DIR="$DATA_ROOT/$SKILL/$SCENE_ID"
TRAJECTORY_PATH="$OUTPUT_DIR/trajectories/${SKILL}__${SCENE_ID}.json"
RENDER_PATH="$OUTPUT_DIR/${SKILL}__${SCENE_ID}.png"
RESULT_PATH="$OUTPUT_DIR/result.json"

if [[ ! -x "$VISFLY_PYTHON" ]]; then
  echo "Missing visfly Python: $VISFLY_PYTHON" >&2
  exit 1
fi
if [[ ! -x "$AERIALCLAW_PYTHON" ]]; then
  echo "Missing AerialClaw Python: $AERIALCLAW_PYTHON" >&2
  exit 1
fi
if [[ ! -f "$SAMPLE_DIR/meta.json" ]]; then
  echo "Missing sample: $SAMPLE_DIR" >&2
  exit 1
fi
if [[ ! -f "$AERIALCLAW_ROOT/scripts/run_aeroverbs_episode.py" ]]; then
  echo "Missing AerialClaw runner under: $AERIALCLAW_ROOT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

echo "Running AerialClaw smoke episode: $SKILL/$SCENE_ID"
"$VISFLY_PYTHON" scripts/rollout_aerialclaw.py \
  --skill "$SKILL" \
  --scene-id "$SCENE_ID" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --aerialclaw-root "$AERIALCLAW_ROOT" \
  --aerialclaw-python "$AERIALCLAW_PYTHON" \
  --agent-max-iterations 40 \
  --max-steps 200 \
  --step-size 0.1 \
  --drone-radius 0.15 \
  --request-timeout-s 300 \
  --subprocess-timeout-s 1800 | tee "$RESULT_PATH"

echo "Rendering trajectory: $TRAJECTORY_PATH"
"$VISFLY_PYTHON" scripts/view_voxel_trajectory.py \
  --task "$SKILL" \
  --scene-id "$SCENE_ID" \
  --data-root "$DATA_ROOT" \
  --plan "$TRAJECTORY_PATH" \
  --voxel-size 0.2 \
  --voxel-stride 2 \
  --region-stride 2 \
  --output "$RENDER_PATH"

"$VISFLY_PYTHON" - "$RESULT_PATH" "$TRAJECTORY_PATH" <<'PY'
import json
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
trajectory = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
events = trajectory.get("agent_action_events", [])
skills = [str(event.get("skill", "")) for event in events]
print("\nSmoke summary")
print("  status:", result.get("status"))
print("  reason:", result.get("reason"))
print("  agent iterations:", result.get("agent_iterations"))
print("  physical steps:", result.get("steps"))
print("  action skills:", skills)
print("  visual observation used:", any(skill in {"observe", "perceive"} for skill in skills))
print("  trajectory:", sys.argv[2])
PY

echo "Rendered image: $RENDER_PATH"
