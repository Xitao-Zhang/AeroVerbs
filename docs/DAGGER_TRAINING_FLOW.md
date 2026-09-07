# AeroVerbs DAgger 训练流程

> 更新：2026-08-26
> 当前实现：学生闭环执行 + Fast-Planner 在线局部重规划 Teacher + 累积数据再训练

## 1. 目标与基本原则

DAgger 用来修正 BC 模型在自身状态分布上的误差：rollout 时始终由 student 控制无人机，Teacher 只在 student 实际访问到的每个状态上生成监督标签。这样训练数据会覆盖“模型已经偏离专家轨迹后应该如何恢复”，而不只是专家轨迹上的理想状态。

当前一轮完整流程为：

1. 从当前 checkpoint 启动 student 闭环 rollout。
2. Fast-Planner Teacher 在 student 当前状态在线局部重规划并生成 waypoint/completion 标签。
3. 将本轮样本累积到所有历史 DAgger 数据中。
4. 使用 expert train split 与累积 DAgger 样本共同微调 student。
5. 在 expert val/test split 上检查监督指标，并另做固定场景闭环 rollout 评估。
6. 下一轮使用上轮新 checkpoint 重复以上过程。

## 2. 输入与固定约束

- 起始模型：当前 BC 或上一轮 DAgger 的 `final.pt`。
- 专家数据：默认 `/root/autodl-tmp/data/trajctorys`。
- 技能：`around_left,around_right,through,over,under`。
- SDF：直接读取数据集中预计算的 `_sdf/<skill>/<scene_id>.npz`，采集时不重新 voxelize、不重新计算 EDT/SDF。
- 硬碰撞半径：所有任务统一为 `0.15m`，同时用于 collector 碰撞检查和 Fast-Planner 硬占用膨胀。
- Teacher preferred clearance（相对原始障碍物表面的软目标）：

| 任务 | preferred clearance | optimizer `dist0 = preferred - 0.15` |
|---|---:|---:|
| `around_left` | `0.50m` | `0.35m` |
| `around_right` | `0.50m` | `0.35m` |
| `over` | `0.20m` | `0.05m` |
| `through` | `0.15m` | `0.00m` |
| `under` | `0.15m` | `0.00m` |

`preferred_clearance` 是 B-spline 优化阶段的软代价目标，不是新的硬碰撞半径。through/under 设为 `0.15m` 表示不额外要求软安全余量，硬碰撞限制仍然生效。若命令显式传入 `--teacher-preferred-clearance`，它会作为所有任务的全局覆盖值；正常训练应省略该参数以使用上表的分任务设置。

## 3. 固定种子的跨物体类别分层抽样

未显式传入 `--scene-ids` 时，collector 不再简单取数据索引的前 N 条，而是对每个 skill 独立执行确定性的类别分层排序：

1. 优先用 `asset_path` 的文件名作为物体类别。
2. 没有 asset 时使用有意义的 opening/object type，例如 `door`、`window`、`ring`。
3. 若 type 只是通用的 `obstacle`，则从程序化 scene id 恢复 `column`、`box` 等类别。
4. 每个类别内部和类别顺序都用固定 seed 洗牌。
5. 按 round-robin 交错类别，在重复某一类别前尽量先覆盖其他类别。

默认 `--sample-seed 42`。每个 skill 会从该 seed 生成稳定但不同的子 seed，因此同一命令可复现，同一任务之间又不会使用完全相同的排列。`--num-per-skill` 统计成功保存的轨迹数；某条 rollout 失败时 collector 会继续取后续候选补足，而不是让失败占掉名额。显式 `--scene-ids` 时尊重用户指定顺序，不做分层重排。

## 4. 在线采集与 Teacher 标注

### 4.1 Student 闭环执行

- 模型输入与当前 BC 保持一致，包括 radar、gate features、downward range、skill id 和 action history。
- student 预测 body-frame waypoints，collector 默认追踪 waypoint index 2，并用运动学步进更新位置与航向。
- 执行权属于 student；Teacher waypoint 不直接控制 rollout。
- 触发硬碰撞、超过最大步数或 Teacher completion 后终止当前轨迹。

### 4.2 Fast-Planner 在线局部重规划

- 每次重规划使用 student 当前 position 和由最近执行位移估计的 velocity。
- 默认每 5 个采集帧调用一次完整 Fast-Planner；中间帧复用缓存轨迹并重新投影生成 5 个 body-frame waypoint 标签。
- 完整流程包含 kinodynamic A*、Non-Uniform B-spline 优化和时间重分配。
- 默认单次预算 `2.0s`、kinodynamic A* `max_iter=2000`。
- 越过语义 guide plane 前规划 `current -> guide -> expert final goal`；越过后规划 `current -> expert final goal`。
- 重规划失败时优先验证并继续使用上一条安全 cached path；缓存也无效时，该帧 waypoint 权重设为0，不再使用 nearest-expert waypoint。旧 Teacher 仅保留基于语义的 completion 判断。
- 对 around，在 student 真正跨过 guide plane 前，监督用 waypoint index 2 最多指向 guide；落入 guide 后方回正段的更远 waypoint 会被截断或标为无效，避免提前切弯朝向最终目标。

### 4.3 可通行区域的使用方式

Teacher 会读取 `heatmap_gate_region.json`，但不要求“规划穿越点必须位于可通行区域内”。当前 guide 由专家轨迹 crossing anchor 与 region hint 软融合：约 `75%` 专家 crossing anchor + `25%` region hint；region 点的选择同时考虑到 anchor 的距离和 preferred-clearance deficit。这样可利用可通行区域，又不会因 gate/region 标注不精确而把几何可行路径判为失败。

### 4.4 每条采集样本的主要输出

- `frames.json`
- `frames/lidar_range_npy/*.npy`
- `downward_range.npy`
- `teacher_waypoints.npy [T,5,3]`
- `teacher_waypoint_valid.npy [T,5]`
- `teacher_completion.npy [T,1]`
- `teacher_label_source.npy [T]`
- `teacher_frame_weight.npy [T]`
- `teacher_planner_stats.json`
- intention heatmap、gate summary/region 及场景 meta

`teacher_planner_stats.json` 用于监控 replan 成功率、fallback fraction、mean/p95/max planning time、SDF 来源、preferred clearance 和 optimizer target。

## 5. 历史数据累积

每轮新样本写到：

```text
<output-root>/roundN_samples/<skill>/<sample>/
```

迭代器随后在以下目录建立指向各轮样本的 symlink：

```text
<output-root>/all_dagger/<skill>/<sample>/
```

若样本名重复，会自动加入 round 名和数字后缀。第 N 轮训练读取 `all_dagger`，因此会使用第 1 到 N 轮的全部历史 DAgger 数据，而不是只训练最新一轮。

## 6. Expert + DAgger 混合训练

### 6.1 数据划分

- expert 数据按 checkpoint 配置中的固定 seed 分为 train/val/test。
- 只有 expert train split 与 DAgger 样本混入训练集。
- expert val/test 保持纯专家数据，避免 DAgger 样本泄漏进监督验证指标。
- DAgger 标签文件存在时，Dataset 优先读取 Teacher waypoint/completion 标签。

### 6.2 混合比例的实际含义

`--mix-ratio 0.25` 是希望达到的 DAgger 目标占比，但 `--max-dagger-repeats 4` 会限制小数据集的重复次数。因此当 DAgger 样本很少时，实际占比会低于 25%。例如约 2021 条 expert train、25 条 DAgger、最多重复 4 次时，训练集为 `2021 + 100`，DAgger 实际占比约 `4.7%`。随着多轮数据累积，实际占比会逐渐接近目标。

### 6.3 Loss 与 BC 对齐

默认不在 DAgger 命令中传 lambda 覆盖项，而是继承起始 checkpoint 的 BC 配置。当前起点的有效权重为：

| Loss | 权重 |
|---|---:|
| completion | `0.05` |
| smooth (`lambda_smooth`) | `0.05` |
| temporal direction | `0.10` |
| action consistency | `0.00`（DAgger 中强制关闭） |
| gate cross | `0.50` |
| gate direction | `0.05` |
| progress | `0.05` |
| clearance | `1.00` |

clearance loss 保持开启，并直接使用数据集已有的 SDF/occupancy/free-region 数据。训练侧的软安全距离仍为 `clearance_drone_radius=0.15m + clearance_margin=0.20m = 0.35m`；它与 Teacher 的分任务 preferred clearance 是两个不同层面的配置。

当前 `danger_weight` 和 `lambda_temporal_smooth` 虽仍保留 CLI/配置兼容字段，但没有进入 `aeroverbs/training/losses.py` 的最终 loss 计算，因此目前调它们不会改变有效梯度。实际平滑项由 `lambda_smooth` 控制。`lambda_action_consistency` 在 DAgger 入口中强制设为0，避免把 student rollout 自己的动作重新当成专家动作学习。

### 6.4 旧 DAgger 状态的离线 relabel

旧采集轨迹中的 student 状态、雷达、downward range、heatmap 和预计算 SDF 可以继续利用，但没有逐帧来源记录的旧 Teacher waypoint 不应直接用于后续训练。使用：

```bash
/root/miniconda3/envs/aeroverbs/bin/python scripts/dagger_relabel.py \
  --input-root outputs/dagger_next/round1_samples \
  --output-root outputs/dagger_next/round1_samples_relabel_v2 \
  --data-root /root/autodl-tmp/data/trajctorys \
  --skills around_left,around_right,through,over,under \
  --drone-radius 0.15
```

relabel 时使用原专家样本构造 Teacher，再逐帧输入已保存的 student position/forward。新目录通过符号链接复用大文件，只写新的标签、来源、权重和 planner stats，不覆盖原采集目录。

### 6.5 训练与产物

- 沿用当前 trajectory GRU 训练方式，按 `chunk_len` 做截断反向传播。
- 每个 epoch 保存 `checkpoints/epoch_NNN.pt`。
- 训练完成保存 `final.pt`、`summary.json`、`metrics.jsonl` 和 TensorBoard logs。
- 每轮训练后的 `final.pt` 自动成为下一轮 collector 与训练的起始 checkpoint。

## 7. 推荐启动命令

先执行一轮、每类 20 条作为扩大采集前的稳定性检查：

```bash
cd /root/AeroVerbs
/root/miniconda3/envs/aeroverbs/bin/python scripts/dagger_iterate.py \
  --start-checkpoint outputs/bc_baseline/20260826-074509/final.pt \
  --data-root /root/autodl-tmp/data/trajctorys \
  --skills around_left,around_right,through,over,under \
  --num-rounds 1 \
  --num-per-skill 20 \
  --sample-seed 42 \
  --drone-radius 0.15 \
  --teacher-replan-interval 5 \
  --teacher-timeout 2.0 \
  --teacher-max-iter 2000 \
  --train-epochs 5 \
  --lr 0.0001 \
  --mix-ratio 0.25 \
  --max-dagger-repeats 4 \
  --output-root outputs/dagger_next
```

注意：

- 不传 `--teacher-preferred-clearance`，才能启用分任务默认值。
- 不传 `--lambda-clearance`、`--lambda-completion`、`--lambda-temporal-smooth`，以继承 BC checkpoint 配置。
- `--num-per-skill` 是每类任务成功保存的目标数量，不应解释为模型成功率。
- 首轮确认采集耗时和 fallback 后，再决定是否增加每类数量或轮数。

## 8. 每轮验收与监控

采集阶段至少检查：

- 每个 skill 的成功保存数量和类别覆盖；through 必须覆盖 door/window/ring。
- `teacher_planner_stats.json` 中的 replan success、fallback fraction、p95 planning time。
- waypoint valid 比例、completion 是否能及时终止轨迹。
- 碰撞、超时、规划失败是否集中在特定 task/object category。

训练阶段至少检查：

- 训练/验证 loss 是否有限，无 NaN/Inf。
- clearance、gate、progress 等各 loss 分量是否正常记录。
- 实际 expert/DAgger 样本数和实际 DAgger 占比。
- checkpoint 能否重新加载，参数是否均为有限值。

闭环评估应使用固定 seed、固定场景 split 和足够大的样本量，按 skill 与 object/opening category 分开报告。小规模 smoke 只用于验证流程可运行，不能作为稳定成功率结论。

## 9. 当前边界

- 尚未实现绕障后重连专家轨迹 suffix；当前重新连接的是专家最终目标。
- around 末端 y 回正没有专门 loss/约束，可能随 DAgger 数据改善，但不是本阶段硬保证。
- 可通行区域是软引导，不是硬 membership 判定。
- through/under 的 preferred clearance 等于硬碰撞半径，不提供额外软余量。
- 当前没有 beta 混合控制或 Teacher intervention；student 始终执行，Teacher 只标注。
- `danger_weight`、`lambda_temporal_smooth` 当前为兼容字段，不影响有效 loss；DAgger 的 action consistency 已完全关闭。
