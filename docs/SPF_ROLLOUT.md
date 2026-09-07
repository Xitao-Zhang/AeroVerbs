# SPF rollout baseline

SPF is exposed as a small local HTTP service so the AeroVerbs rollout uses the
same RGB renderer and episode loop as the other baselines. AirSim is not used.

## Start the service

```bash
cd /home/zxt/see-point-fly
conda activate spf
python spf_server.py --host 127.0.0.1 --port 5010
```

The service reads `config_sim.yaml` and `.env`, and supports the existing
`qwen3-vl:8b-instruct` Ollama setup.

## Run one rollout

Use the same sample and instruction as another baseline. SPF's predicted
direction is normalized and executed with a fixed step size (default `0.1m`).
The raw distance predicted by SPF is still retained in the per-step logs.

```bash
cd /home/zxt/UAV_Basic_Skills/AeroVerbs
conda activate aeroverbs
python scripts/rollout_spf.py \
  --skill around_left \
  --scene-id auto_around_left_pillar_00141 \
  --spf-url http://127.0.0.1:5010 \
  --prompt-mode original \
  --max-steps 40 \
  --step-size 0.1 \
  --show-step-progress \
  --render-trajectory
```

`--prompt-mode relation` enables SPF's relation-aware prompt. Keep results
separate from `original`, since it is a prompt ablation rather than the
unmodified SPF baseline.

By default results are written under `exp/spf/batch_manual_<timestamp>/` with
the same layout as the OpenVLA experiments:

```text
run_config.json
trajectories/<skill>__<scene_id>.json
summary.json
results.jsonl
```
