# AeroVerbs 模型评估指标记录

记录日期：2026-09-05

本文档汇总 BC、DAgger、OpenVLA、SPF，以及使用 MiniMind-O 替代 GPT-5.5 生成可通行区域标注的 DAgger，在同一组 254 个 AeroVerbs 测试场景上的总体评估结果，并单独记录当前仅完成 Batch 1 的 AerialClaw。所有表格数值均由 `exp/` 下保存的 rollout 轨迹与汇总文件计算得到。

## 总体结果

| 模型 | 成功数 / 总数 | 成功率 ↑ | SPL ↑ | 平均 nDTW ↑ | 平均 CLS ↑ | 后方区域到达数 / 有效轨迹数 | RAR ↑ | RRD ↓ | RPS ↑ | RPD ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BC | 209 / 254 | 82.28% | 71.00% | 0.4587 | 0.3628 | 81 / 254 | 31.89% | 1.6353 m | 0.5207 | 0.1954 m |
| DAgger | 217 / 254 | 85.43% | 79.75% | 0.5700 | 0.4420 | 102 / 254 | 40.16% | 0.2871 m | 0.7099 | 0.2470 m |
| DAgger + MiniMind-O 标注 | 217 / 254 | 85.43% | 79.33% | 0.5702 | 0.4429 | 101 / 254 | 39.76% | 0.2736 m | 0.7138 | 0.2589 m |
| OpenVLA | 228 / 254 | 89.76% | 85.22% | 0.6099 | 0.4472 | 38 / 254 | 14.96% | 0.4662 m | 0.5031 | 0.0938 m |
| SPF | 10 / 254 | 3.94% | 3.17% | 0.0612† | 0.0798† | 51 / 246† | 20.73%† | 1.1622 m† | 0.4407† | 0.1346 m† |

其中，`↑` 表示数值越大越好，`↓` 表示数值越小越好。

† SPF 有 8 个 `infrastructure_error` 场景没有生成轨迹，因此 SPL 和成功率仍以完整的 254 个场景为分母，缺失场景对 SPL 的贡献为 0；nDTW、CLS 和后方区域指标只能在其余 246 条有效轨迹上计算。

## MiniMind-O 可通行区域标注替代实验

### 蒸馏训练配置与权重

本次用于 rollout 的 MiniMind-O 权重来自运行 `all-data-fp32-v2`。训练目标是根据无人机第一视角的 `initial_rgb.png` 和语义任务提示，直接生成对应可通行区域的四点像素坐标 JSON。

| 配置项 | 本次取值 |
| --- | --- |
| 数据清单 | `minimind-o_distillation/data/manifests/all.jsonl` |
| 原始数据根目录 | `/data/UAV_Basic_Skills/trajctorys` |
| 有效标注数 | 2525（另有 4 条零面积无效标注被排除） |
| 训练/验证划分 | 不划分；2525 条有效数据全部用于训练，`val_manifest=null` |
| 各任务原始数量 | around_left=750，around_right=750，over=425，through=500，under=100 |
| 重采样权重 | around_left=1，around_right=1，through=2，over=2，under=10 |
| 每 epoch 采样量 | 4350（保证每条原始数据至少访问一次，再按任务权重重复） |
| 图像 | `initial_rgb.png`，640 × 480 |
| 图像 token 数 | 64 |
| 最大序列长度 | 1024 |
| MiniMind-O 结构 | hidden size=768，8 个 Thinker block，非 MoE |
| 初始权重 | `/home/zxt/minimind-o/out/sft_omni_768.pth` |
| 视觉编码器 | `/home/zxt/minimind-o/model/siglip2-base-p32-256-ve` |
| 训练模块 | `vision_proj`、最后 2 个 Thinker block、Thinker norm、共享 LM head/token embedding |
| 可训练参数量 | 20,847,744 |
| 冻结模块 | SigLIP2、Talker 及音频相关模块 |
| 设备与精度 | CUDA，FP32 |
| Epoch | 8 |
| 单步 batch size | 4 |
| 梯度累积 | 8 |
| 有效 batch size | 32 |
| 优化器 | AdamW |
| 初始学习率 | 5e-5 |
| 学习率调度 | 余弦衰减至初始学习率的 10%（最终 5e-6） |
| Weight decay | 0.01 |
| 梯度裁剪 | 1.0 |
| 随机种子 | 42 |
| DataLoader workers | 4 |
| 生成长度上限 | 96 tokens |

训练 8 个 epoch 共耗时约 712.7 秒（11.9 分钟），train loss 从 0.6662 降至 0.2228。每个 epoch 使用每类 10 条、共 50 条训练样本进行生成监控；第 8 epoch 的 macro IoU 最高，为 0.8001，因此被保存为 `best.pt`。由于没有验证集，该指标是 training-monitor 指标，不应解释为独立验证集性能。

最终推荐使用的权重为：

- `minimind-o_distillation/outputs/checkpoints/final/all-data-fp32-v2/best.pt`
- 绝对路径：`/home/zxt/UAV_Basic_Skills/AeroVerbs/minimind-o_distillation/outputs/checkpoints/final/all-data-fp32-v2/best.pt`
- 文件大小：208,516,623 bytes；SHA-256：`8abe911145ee2486ccc51e1008949949cda113563b8160344c3f650e0f7f48a3`

训练结束时的最后一个权重另存为 `minimind-o_distillation/outputs/checkpoints/final/all-data-fp32-v2/last.pt`。本次 `best.pt` 和 `last.pt` 都对应第 8 epoch，但保存内容还包括优化器和调度器状态，因此两个文件的哈希并不相同。实际标注与 rollout 使用的是 `best.pt`。

完整配置和逐 epoch 日志分别保存在：

- `minimind-o_distillation/outputs/logs/final/all-data-fp32-v2/config.json`
- `minimind-o_distillation/outputs/logs/final/all-data-fp32-v2/metrics.jsonl`
- `minimind-o_distillation/outputs/logs/final/all-data-fp32-v2/trainable_parameters.txt`

### Rollout 替代实验设置

本实验保持 AeroVerbs DAgger 控制器权重、5 个测试 batch、场景清单和 rollout 参数不变，仅将输入中的 GPT-5.5 可通行区域标注替换为 MiniMind-O 生成的四点区域。两次实验使用的控制器检查点 SHA-256 均为 `cb224919ccbbd89d5a0bcfd24ba4e1acde71b53166fb38edb43052875e585e77`。

MiniMind-O 使用 `all-data-fp32-v2/best.pt`（第 8 epoch），共成功生成 254 / 254 条标注，没有解析失败。与 GPT-5.5 教师标注相比，区域 IoU 均值为 0.7975、中位数为 0.8245。

### 分 Batch 成功率

| Batch | GPT-5.5 标注 | MiniMind-O 标注 | 成功率变化 |
| --- | ---: | ---: | ---: |
| Batch 1 | 45 / 51（88.24%） | 44 / 51（86.27%） | -1.96 pp |
| Batch 2 | 45 / 51（88.24%） | 43 / 51（84.31%） | -3.92 pp |
| Batch 3 | 41 / 51（80.39%） | 42 / 51（82.35%） | +1.96 pp |
| Batch 4 | 45 / 51（88.24%） | 48 / 51（94.12%） | +5.88 pp |
| Batch 5 | 41 / 50（82.00%） | 40 / 50（80.00%） | -2.00 pp |
| **合计** | **217 / 254（85.43%）** | **217 / 254（85.43%）** | **0.00 pp** |

MiniMind-O 标注实验的最终状态为 `success=217`、`collision=36`、`timeout=1`；GPT-5.5 标注基线为 `success=217`、`collision=35`、`timeout=2`。

### 分任务结果

| 任务 | GPT-5.5 成功率 | MiniMind-O 成功率 | 成功数变化 | GPT-5.5 SPL | MiniMind-O SPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| around_left | 59 / 75（78.67%） | 61 / 75（81.33%） | +2 | 0.6891 | 0.6990 |
| around_right | 73 / 75（97.33%） | 73 / 75（97.33%） | 0 | 0.9044 | 0.9069 |
| over | 26 / 44（59.09%） | 25 / 44（56.82%） | -1 | 0.5880 | 0.5638 |
| through | 49 / 50（98.00%） | 48 / 50（96.00%） | -1 | 0.9476 | 0.9295 |
| under | 10 / 10（100.00%） | 10 / 10（100.00%） | 0 | 0.9785 | 0.9785 |

逐场景配对比较中，211 个场景在两次实验中均成功，31 个场景均碰撞；另有 6 个场景由失败变为成功、6 个场景由成功变为失败，因此成功数净变化为 0。

MiniMind-O 标注的平均 IoU 在成功场景中为 0.8197，在失败场景中为 0.6673。`over` 是目前最明显的薄弱项：其全部样本平均 IoU 为 0.6926，成功样本平均 IoU 为 0.7984，失败样本平均 IoU 为 0.5535。该相关性不能单独证明标注误差是失败的唯一原因，但说明后续优化应优先检查 `over` 的标注范围。

总体而言，MiniMind-O 标注将成功率保持在 85.43%，SPL 从 0.7975 小幅下降至 0.7933，nDTW 和 CLS 基本不变，说明它在当前场景集合上可以低成本替代 GPT-5.5 标注。需要注意：这 254 个场景包含在 MiniMind-O 的全量蒸馏训练数据中，因此本实验验证的是训练分布内替代能力，不能作为未见场景泛化能力的独立证据。

## AerialClaw：Batch 1 完整结果

首次运行中出现基础设施错误的 18 个场景已经重新运行。合并每个场景的最新结果后，`batch_1` 的 51 个场景均生成了有效轨迹，不再包含 `infrastructure_error`。

| 模型 | 成功数 / 总数 | 成功率 ↑ | SPL ↑ | 平均 nDTW ↑ | 平均 CLS ↑ | 后方区域到达数 / 有效轨迹数 | RAR ↑ | RRD ↓ | RPS ↑ | RPD ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AerialClaw（Batch 1） | 2 / 51 | 3.92% | 3.80% | 0.0371 | 0.0579 | 0 / 51 | 0.00% | 8.3374 m | 0.0439 | 0.0000 m |

51 条轨迹的最终状态为：`success=2`、`timeout=49`、`collision=0`。所有技能均有完整覆盖：`around_left=15`、`around_right=15`、`through=10`、`over=9`、`under=2`。两个成功样本均属于 `through`：

- `auto_door_opening_00111`：45 个物理步；
- `auto_door_opening_00132`：38 个物理步。

AerialClaw 的 Batch 1 已完整，但仍只覆盖 51 个场景；上方 BC、DAgger、OpenVLA 和 SPF 的总体结果覆盖 5 个 batch、共 254 个场景。因此，在完成 AerialClaw 的其余四个 batch 之前，不应把该行直接加入完整 254 场景模型排名。后方区域指标与 `CompletionJudge` 的 through 成功条件并不等价，所以两条 through 成功轨迹仍可能得到后方区域未到达；两类指标应分别报告。

## 指标说明

### 成功率

成功率由 rollout 结果中的 `status == "success"` 统计：

\[
\text{Success Rate}=\frac{N_{\text{status=success}}}{N_{\text{all}}}
\]

当前 rollout 的成功判定相对宽松，主要表示轨迹满足现有完成条件且未以碰撞或超时状态结束。该指标不能单独证明模型正确执行了“从左侧绕过”“从上方飞过”等语义，因此需要结合 nDTW、CLS 和后方区域到达程度一起分析。

### SPL

SPL（Success weighted by Path Length，成功加权路径长度）同时衡量任务成功和路径效率：

\[
\mathrm{SPL}=\frac{1}{N}\sum_{i=1}^{N} S_i\frac{L_i}{\max(P_i,L_i)}
\]

其中，只有 `status == "success"` 时 \(S_i=1\)，否则为 0；\(P_i\) 是 rollout 轨迹的三维弧长；\(L_i\) 是同一场景冻结的 Fast-Planner 参考路径长度。Fast-Planner 路径是统一的参考可行路径，不保证是数学意义上的全局最短路径。

SPL 范围为 0–1，越高越好。失败轨迹的 SPL 为 0；成功轨迹短于参考路径时该项最高为 1，不会因为路径短于参考值而超过 1。SPF 缺少轨迹的 8 个基础设施错误场景也按失败处理。AerialClaw 的 3.80% 只代表当前 Batch 1 的 51 个场景，不能直接与其余模型的完整 254 场景结果排名。

### nDTW

nDTW 衡量 rollout 轨迹与裁剪后的专家轨迹之间的时序空间相似度，范围为 0–1，越大表示轨迹形状及行进过程越接近专家轨迹。

本次记录采用以下配置：

- 专家轨迹裁剪规则：裁剪到 rollout 轨迹所达到的最远 `x` 位置；
- 空间重采样间隔：0.1 m；
- nDTW 距离阈值：0.5 m；
- 距离计算：三维欧氏距离；
- 不使用 Gate 数据。

### CLS

CLS（Coverage weighted by Length Score）同时衡量专家轨迹覆盖程度与 rollout 路径长度合理性：

\[
\text{CLS}=\text{Path Coverage}\times\text{Length Score}
\]

CLS 范围为 0–1，越大表示 rollout 不仅覆盖了更多专家路径，而且没有通过明显不合理的额外路程获得覆盖。

CLS 使用与 nDTW 相同的专家轨迹裁剪规则、0.1 m 空间重采样间隔和 0.5 m 距离阈值。

### 后方区域到达指标（Rear-Region Attainment Metrics, RRAM）

后方区域定义为：移除地面后的障碍物体素在 `yz` 平面上的投影，沿 `+x` 方向延伸形成的半无限三维区域。该指标只使用障碍物几何和 rollout 终点，不使用 Gate 或专家轨迹。

计算参数：

- 后方方向：`+x`；
- 障碍物体素大小：0.08 m；
- 横向投影膨胀：0.35 m，其中无人机半径为 0.15 m，额外安全余量为 0.20 m；
- 完全越过障碍物所需的后方净空：0.15 m；
- 分数距离尺度：0.5 m。

该指标组的英文名称为 **Rear-Region Attainment Metrics（RRAM）**。各子指标定义如下：

- **后方区域到达率（Rear-Region Arrival Rate, RAR）**：终点进入后方区域的轨迹比例；
- **后方区域平均距离（Rear-Region Distance, RRD）**：所有轨迹终点到后方区域的三维最短距离，越小越好；进入区域后距离为 0；表中报告其均值；
- **后方区域平均接近分数（Rear-Region Proximity Score, RPS）**：根据终点距离计算：

  \[
  \mathrm{RPS}=\exp\left(-\frac{D_{\text{rear}}}{0.5}\right)
  \]

  分数范围为 0–1，进入后方区域时为 1；
- **后方区域平均穿入深度（Rear-Region Penetration Depth, RPD）**：终点沿 `+x` 方向越过障碍物后表面及后方净空的距离。未到达后方区域的轨迹按 0 m 计入总体平均值。

后方区域到达程度只衡量终点是否到达障碍物后方，不能区分轨迹是从左侧、右侧、上方还是下方通过。因此，它应作为独立的任务完成进度指标，与任务方向语义指标及轨迹相似度指标联合使用。

## 结果解读注意事项

OpenVLA 的当前成功率最高，但后方区域到达率最低。这说明现有宽松成功判定与“真正到达障碍物后方”并不等价，也说明保留独立的后方区域指标是必要的。

BC 的后方区域平均距离明显较大，同时仍有 31.89% 的轨迹进入后方区域。这是因为部分 BC 轨迹能够到达后方，而另一些发散轨迹距离障碍物后方区域非常远；因此平均距离容易受极端失败轨迹影响。分析具体模型行为时，应同时查看到达率、平均距离和逐场景结果。

## 数据来源

- 成功状态、nDTW 和 CLS 汇总：`exp/metrics/trajectory_fidelity/max_x/summary.json`
- nDTW 和 CLS 逐场景结果：`exp/metrics/trajectory_fidelity/max_x/per_scene.csv`
- 后方区域指标汇总：`exp/metrics/rear_region/summary.json`
- 后方区域逐场景结果：`exp/metrics/rear_region/per_scene.csv`
- SPL 汇总：`exp/metrics/spl/summary.json`
- SPL 逐场景结果：`exp/metrics/spl/per_scene.csv`
- rollout 轨迹：`exp/rollouts/{bc,dagger,openvla,spf}/batch_*/trajectories/`
- AerialClaw Batch 1 原始结果：`exp/rollouts/aerialclaw/batch_1/{results.jsonl,summary.json,trajectories/}`
- AerialClaw Batch 1 的 nDTW/CLS：`exp/metrics/aerialclaw/batch_1/trajectory_metrics/`
- AerialClaw Batch 1 的后方区域指标：`exp/metrics/aerialclaw/batch_1/rear_region_metrics/`
- AerialClaw 两条成功轨迹图：`exp/rollouts/aerialclaw/batch_1/renders/`
- MiniMind-O 原始标注及预览：`exp/annotations/minimind_o_all_data_fp32_v2/`
- MiniMind-O 标注下的 DAgger rollout：`exp/rollouts/minimind_o_all_data_fp32_v2/`
- MiniMind-O 标注实验的 SPL：`exp/metrics/minimind_o_all_data_fp32_v2/spl/`
- MiniMind-O 标注实验的 nDTW/CLS：`exp/metrics/minimind_o_all_data_fp32_v2/trajectory_fidelity/`
- MiniMind-O 标注实验的后方区域指标：`exp/metrics/minimind_o_all_data_fp32_v2/rear_region/`

本文档采用 `max_x` 专家轨迹裁剪方案作为主要 nDTW/CLS 结果。按 rollout 三维弧长裁剪专家轨迹的备用结果保存在 `exp/metrics/trajectory_fidelity/arc_length/`，未纳入主表。
