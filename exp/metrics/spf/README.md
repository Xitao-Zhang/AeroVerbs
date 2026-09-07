# SPF 五批闭环评测指标

评测范围：`batch_1` 至 `batch_5`，共 254 条结果记录。

## 总体结果

| 指标 | SPF |
|---|---:|
| Success Rate | 10/254 = 3.94% |
| Semantic Completion Rate | 61/254 = 24.02% |
| SPL | 3.17% |
| collision | 91 |
| timeout | 145 |
| infrastructure_error | 8 |
| 有效轨迹平均长度 | 6.803 m |
| 成功轨迹平均长度 | 5.240 m |

SPL 使用冻结的 Fast-Planner 参考路径，按
`mean[S_i × L_i / max(P_i, L_i)]` 计算。8 个没有轨迹的
`infrastructure_error` 样本对 SPL 的贡献为 0。此前此处的 3.94% 实际是成功率，现已更正为重新按轨迹长度计算得到的 3.17%。

## 各批结果

| 批次 | N | 成功率 | 语义完成率 | success | collision | timeout | infrastructure_error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Batch 1 | 51 | 1.96% | 21.57% | 1 | 23 | 26 | 1 |
| Batch 2 | 51 | 5.88% | 29.41% | 3 | 17 | 31 | 0 |
| Batch 3 | 51 | 5.88% | 31.37% | 3 | 17 | 29 | 2 |
| Batch 4 | 51 | 0.00% | 17.65% | 0 | 18 | 30 | 3 |
| Batch 5 | 50 | 6.00% | 20.00% | 3 | 16 | 29 | 2 |

## 各技能结果

| 技能 | N | 成功率 | 语义完成率 | success | collision | timeout | infrastructure_error |
|---|---:|---:|---:|---:|---:|---:|---:|
| around_left | 75 | 5.33% | 44.00% | 4 | 26 | 44 | 1 |
| around_right | 75 | 0.00% | 0.00% | 0 | 42 | 33 | 0 |
| through | 50 | 0.00% | 6.00% | 0 | 19 | 31 | 0 |
| over | 44 | 13.64% | 47.73% | 6 | 3 | 28 | 7 |
| under | 10 | 0.00% | 40.00% | 0 | 1 | 9 | 0 |

## 轨迹相似性

nDTW/CLS 使用项目既有 `compute_trajectory_metrics.py`、`max_x` 裁剪、0.1 m 重采样和 0.5 m 距离阈值计算。由于 8 条基础设施错误没有轨迹，此部分为 246 条有效轨迹。

| 指标 | 均值 | 中位数 |
|---|---:|---:|
| nDTW | 0.0612 | 0.0081 |
| CLS | 0.0798 | 0.0555 |
| Path Coverage | 0.3540 | 0.3503 |
| Length Score | 0.2001 | 0.1630 |

详细逐轨迹结果位于 `trajectory_metrics/per_scene.csv` 和 `trajectory_metrics/per_scene.jsonl`。

## 终点障碍物后方指标

该指标使用 `compute_rear_region_metric.py`，对 246 条有效轨迹计算；后方方向为 `+x`，体素尺寸 0.08 m，后方区域安全余量 0.15 m，横向余量 0.35 m。

| 指标 | 均值 | 中位数 |
|---|---:|---:|
| Reached rate | 20.73% (51/246) | — |
| Rear-region axial gap | 0.890 m | 0.295 m |
| Rear-region depth | 0.135 m | 0.000 m |
| Rear-region distance | 1.162 m | 0.486 m |
| Rear-region lateral distance | 0.523 m | 0.043 m |
| Rear-region score | 0.441 | 0.378 |

详细结果位于 `rear_region_metrics/per_scene.csv`、`rear_region_metrics/per_scene.jsonl` 和 `rear_region_metrics/summary.json`。
