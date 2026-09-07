# Rollout evaluation results

本目录按“原始评测、派生指标、抽样可视化”分层组织：

```text
exp/
├── rollouts/                         # 原始闭环评测结果
│   ├── bc/batch_{1..5}/
│   ├── dagger/batch_{1..5}/
│   ├── openvla/batch_{1..5}/
│   ├── spf/batch_{1..5}/
│   └── aerialclaw/batch_1/
├── metrics/                          # 从轨迹计算的派生指标
│   ├── trajectory_fidelity/
│   │   ├── max_x/                    # 当前主 nDTW/CLS
│   │   └── arc_length/               # 备用裁剪方案
│   ├── spl/                           # 所有模型的 SPL
│   ├── rear_region/                  # BC/DAgger/OpenVLA 后方区域指标
│   ├── spf/                          # SPF 专属指标与说明
│   └── aerialclaw/batch_1/           # AerialClaw Batch 1 指标
├── visualizations/sample20/          # 三模型同场景 20 条抽样图
│   ├── bc/
│   ├── dagger/
│   └── openvla/
├── EVALUATION_METRICS.md             # 当前指标总表
├── five_batch_summary.json           # BC/DAgger 五批机器可读汇总
└── five_batch_summary.md             # BC/DAgger 五批报告
```

每个 rollout batch 包含：

- `results.jsonl`：逐场景结果；重试时可能有重复场景，最新一条为准；
- `summary.json`：按每个场景最新结果生成的批次汇总；
- `run_config.json`：manifest、模型和 rollout 参数；
- `trajectories/`：逐场景三维轨迹；
- `renders/`：可选轨迹图片。

每个指标目录包含 `summary.json`、`per_scene.csv` 和 `per_scene.jsonl`。

重新计算主 nDTW/CLS：

```bash
python scripts/compute_trajectory_metrics.py \
  --exp-root exp \
  --expert-root /data/UAV_Basic_Skills/trajctorys \
  --output-dir exp/metrics/trajectory_fidelity/max_x \
  --models bc dagger openvla
```

重新计算主后方区域指标：

```bash
python scripts/compute_rear_region_metric.py \
  --exp-root exp \
  --data-root /data/UAV_Basic_Skills/trajctorys \
  --output-dir exp/metrics/rear_region \
  --models bc dagger openvla
```

重新计算 SPL（Success weighted by Path Length）：

```bash
python scripts/compute_spl_metrics.py \
  --exp-root exp \
  --reference-root eval_reference_paths \
  --output-dir exp/metrics/spl \
  --models bc dagger openvla spf aerialclaw
```

批量 rollout 的默认输出根目录现在是 `exp/rollouts/`。评测脚本兼容 `model/batch_N` 和 `batch_N/model` 两种内部布局，指标脚本也兼容旧的 `exp/<model>/` 布局。
