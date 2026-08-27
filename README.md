# AeroVerbs

> 让长程无人机智能体不再用坐标说话，而是用可执行的运动动词说话。
> Long-horizon UAV agents should not speak in coordinates; they should speak in executable motion verbs.

## 项目定位

**AeroVerbs** 是一套面向无人机的可执行运动动词（maneuver verbs）系统，用于补齐长程 UAV-VLN / MLLM 导航系统中缺失的一层：

> 语言可寻址的 reactive 关系机动层。

上层大模型 / Agent 在规划长程任务时，不应再细粒度地输出轨迹点，而只需发出粗粒度的动作指令，例如：

```text
through(door)          —— 穿过门
around_left(pillar)    —— 从左侧绕过柱子
around_right(obstacle) —— 从右侧绕过障碍物
over(barrier)          —— 飞越障碍
under(table)           —— 从桌子下方通过
```

下层运动控制机构具备基础理解与执行能力，收到这些动作动词后即可自主完成局部机动，不再依赖上层提供运动坐标。目标是提高整个系统的效率与准确率。

## 核心思想

```text
Long-horizon task
      │
      ▼
High-level MLLM / Agent  (coarse maneuver verbs, no dense trajectory)
      │
      ▼
AeroVerbs  ──  maneuver heatmap / intent  ──  local policy
      │
      ▼
Reactive local execution  (radar + state + intent)
```

- 上层输出：粗粒度机动动词（skill / object）
- 下层执行：一次性机动意图 + 实时雷达观测驱动的局部策略
- 关键判断：heatmap 是子技能开始时的一次性机动意图先验，而非高频预测

## 模型（`aeroverbs/model/`）

按 `maneuver_policy_network_v1.md` 实现，模块如下：

- `IntentEncoder`：one-shot maneuver heatmap `[B,1,120,160]` → `z_intent [B,64]`
- `SkillEncoder`：`skill_id` → `z_skill [B,32]`
- `CircularRadarEncoder`：3 通道 360° 雷达 `[B,3,40,500]` → `z_radar [B,256]`（水平 circular padding）
- `FusionMLP` + `GRU` + `WaypointHead`(K=5) + `CompletionHead`

顶层接口：

```python
z_intent, z_skill, hidden = model.initialize(maneuver_heatmap, skill_id)
waypoints, completion, hidden = model.step(radar, z_intent, z_skill, hidden)
```

## 数据（`aeroverbs/data/`）

- 数据根目录：`data/trajctorys/<skill>/<scene_id>/`
- 每个样本包含 `meta.json`、`frames.json`、`trajectory.json`、`intention_heatmap.npy`
- 雷达预处理：inverse range + valid mask + normalized elevation（3 通道）
- 标签：机体系未来路径点 `[t+3, t+6, t+10, t+15, t+20]` + 机动完成标签
- 序列化训练：按 `seq_len` 切窗，GRU 学习机动进度

## 训练（`aeroverbs/training/`、`scripts/`）

```bash
# 激活环境（含 CUDA torch）
conda activate aeroverbs

# 冒烟测试（验证模型形状与数据加载）
python scripts/smoke_test.py

# 训练（默认实验）
python scripts/train.py --config configs/default.yaml

# 训练（指定实验名，用于消融）
python scripts/train.py --config configs/default.yaml --experiment no_heatmap --name run1

# 评估（在 held-out 测试集上）
python scripts/eval.py --checkpoint outputs/<experiment>/<run>/final.pt --config configs/default.yaml --split test
```

损失函数：

```text
L = L_waypoint + 0.2 * L_completion + 0.05 * L_smooth
```

### 数据划分

按**对象族**（`scene_id` 去掉末尾 `_NNNNN` 得到的前缀，如 `auto_around_left_pillar`、`auto_under_table_frl_apartment_table_01`）分层划分 **train / val / test**（默认 80% / 10% / 10%）。

每个对象族内部再按比例抽 train / val / test，保证**每个对象都有数据进入验证集和测试集**，而不是随机全局切分（那样可能让某些对象整族落到 train、测试集无法覆盖该对象的泛化）。样本不足 3 条的对象族整族放入 train。

配置项：

```yaml
data:
  train_frac: 0.8
  val_frac: 0.1     # test = 1 - train_frac - val_frac
```

### 输出组织（按实验隔离，便于消融对比）

```text
outputs/<experiment>/<run_name>/
├── config.yaml          # 本次运行的完整配置快照
├── metrics.jsonl        # 每步 train / val / test 指标（可直接做消融表格）
├── summary.json         # 最终测试指标
├── final.pt             # 最终模型
├── checkpoints/         # 每 epoch 权重
└── logs/                # tensorboard 日志
```

其中 `experiment` 用于区分消融类型（如 `bc_baseline`、`no_heatmap`、`object_embedding`），`run_name` 默认自动用时间戳，也可用 `--name` 指定。多次运行同一实验会落在同一 `outputs/<experiment>/` 下按 run 分隔。

## 目录结构

```text
AeroVerbs/
├── aeroverbs/
│   ├── config.py            # 类型化配置 + YAML 加载
│   ├── constants.py         # 技能词汇表 / skill_id 映射
│   ├── model/               # 网络结构
│   ├── data/                # 数据加载、雷达预处理、标签计算
│   └── training/            # 损失函数、训练器
├── scripts/
│   ├── train.py
│   ├── eval.py
│   └── smoke_test.py
├── configs/default.yaml
├── data/                    # 采集数据（trajctorys 为软链）
└── requirements.txt
```

## 备注

- 完成标签当前使用弧长进度启发式（轨迹末段），后续应替换为 privileged `TrajectorySemanticJudge` 的语义完成判定。
- 训练目标为机体系相对路径点；路径点跟踪由后处理 / 控制器完成，不属于本网络的监督目标。
