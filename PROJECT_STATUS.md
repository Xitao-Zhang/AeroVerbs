# AeroVerbs 项目状态记录

> 最后更新：2026-08-27（确定正式闭环对比评测方案）

---

## 1. 项目思路

### 1.1 目标
AeroVerbs = 语言可寻址的关系机动层。让无人机根据 `skill`（around_left / around_right / through / over / under）和一次性意图 heatmap，自主完成绕柱、穿门、越过、下穿等机动。

### 1.2 总体方法：Teacher-Student
分两个阶段：

1. **第一阶段：开环 BC（已完成）**
   - Teacher = 专家轨迹 + 规划器。
   - Student 只在专家轨迹数据上做监督学习，输入雷达 + heatmap + skill，输出机体系未来 waypoint + completion。

2. **第二阶段：DAgger（进行中）**
   - Student 闭环 rollout，Teacher 在 student 实际到达的状态上重新标注，加入训练集继续训练，迭代多轮。
   - 当前已完成 3 轮全技能 DAgger 训练，语义正确但仍有贴边碰撞。

### 1.3 模型输入输出
**一次性输入（每条轨迹只编码一次）**
- `intention_heatmap [1,120,160]` → IntentEncoder → `z_intent [64]`
- `skill_id` → SkillEncoder → `z_skill [32]`

**每步输入**
- `radar [3,40,500]`（inverse range / valid / elevation）→ CircularRadarEncoder → `z_radar [256]`
- `action_history [10,4]`（最近 10 步 `[dx_forward, dy_left, dz_up, delta_yaw]`）+ mask → ActionEncoder → `z_action [64]`

**融合与输出**
- `[z_radar, z_intent, z_skill, z_action]`（416 维）→ FusionMLP → GRU(256)
- WaypointHead → `waypoints [5,3]`：未来 t+3/6/10/15/20 的**机体系位移增量**（x=forward, y=left, z=up）
- CompletionHead → `completion [1]`

### 1.4 运动学控制器
保持简单：
```
body waypoint -> 用当前 forward 转世界 -> 选 t+10 -> 归一化方向 -> 1m/s 前进 0.1m -> 更新 forward -> 记录实际动作
```
不引入复杂控制器，不通过控制器掩盖训练分布问题。

### 1.5 雷达
- trimesh 射线求交，不依赖 habitat-sim。
- 必须使用 `<sample_dir>/_render_scene/<scene_id>_with_ground.glb`。
- 参数：40×500，pitch[-5,35]，rmax 10.0。
- 预处理 3 通道：inverse range + valid mask + elevation。
- embreex 提供 pyembree 兼容加速，单帧约 19.8ms。

---

## 2. 已完成内容

### 2.1 数据管线
- `aeroverbs/data/dataset.py`
  - `build_sample_index` / `split_samples`
  - `ManeuverDataset`（窗口式，兼容保留）
  - `TrajectoryDataset`（完整轨迹，轨迹级训练用）
  - `trajectory_collate_fn`（变长轨迹 padding + `time_mask`）
  - 支持 DAgger 样本目录：优先加载 `teacher_waypoints.npy` / `teacher_waypoint_valid.npy` / `teacher_completion.npy`
- `aeroverbs/data/labels.py`
  - waypoint 标签：世界未来点投影到当前 body 系
  - completion 标签：弧长最后 20%（progress completion）
  - `compute_action_labels`：相邻帧动作 `[dx_forward, dy_left, dz_up, delta_yaw]`
  - `build_action_history`：右对齐滑动窗口 + mask
- `aeroverbs/data/transforms.py`：雷达预处理 + heatmap resize
- `aeroverbs/data/heatmap.py`：heatmap 生成（样本已存好，直接读取）

### 2.2 模型
- `aeroverbs/model/encoders.py`
  - IntentEncoder / SkillEncoder / CircularRadarEncoder / ActionEncoder
- `aeroverbs/model/policy.py`
  - `AeroVerbsPolicy`：`initialize()` + `step()` + `forward()`
  - FusionMLP 输入 416（256+64+32+64）
- `aeroverbs/model/circular_conv.py`

### 2.3 训练
- `aeroverbs/training/trainer.py`
  - `make_model()`
  - 轨迹级 TBPTT 训练：`initialize` 一次，按 `chunk_len=20` 分块，hidden 跨块连续、块间 detach
  - 长度分桶 `_BucketBatchSampler` + `trajectory_batch_size=8`
  - `_evaluate_trajectory`：完整轨迹连续评估
- `aeroverbs/config.py`：新增 `action_history_len/action_dim/chunk_len/trajectory_batch_size`
- `scripts/train.py`：训练入口
- **开环 BC 训练完成**：
  - checkpoint: `outputs/bc_baseline/20260821-155734/final.pt`
  - test waypoint loss: `0.00114`

### 2.4 Teacher-Student / DAgger
- `aeroverbs/data/semantic_judge.py`
  - `trajectory_signature`：`(winding_class, vertical_class)`，轻量版 uniform visibility deformation 类
  - `SemanticCompletionJudge`：student 前缀 signature == teacher signature 即完成
- `aeroverbs/data/teacher.py`
  - `TeacherLabeler`：找最近专家帧 → 未来 waypoint 标签；completion 由 semantic judge 生成
- `scripts/dagger_collect.py`：Student 闭环 rollout + Teacher 标注 + 保存 DAgger 样本
- `scripts/train_dagger.py`：专家数据 + DAgger 数据混合训练，支持 bucket batch
- `scripts/dagger_iterate.py`：自动迭代“采集 → 软链合并 → 训练 → 更新 checkpoint”

### 2.5 Rollout 与可视化
- `scripts/rollout.py`
  - 闭环 rollout，结果写 `results.jsonl`
  - 动作历史维护（student 实际执行动作）
  - 碰撞检测：object-only mesh + trimesh proximity（半径 0.35m，地面高度 0.2m）
  - around_left/right 语义判定：`side_coord` 正/负
  - 渲染轨迹图：调用 `view_voxel_trajectory.make_figure`
- `scripts/view_voxel_trajectory.py`
  - 自包含（不再依赖 uav_maneuver_planning）
  - `--task + --scene-id` 自动补全 glb/plan/region/output
  - 不渲染地面（`_drop_ground_voxels`）
- `scripts/verify_radar.py`：雷达 bit 级一致验证，**PASS**

### 2.6 服务器环境
- 服务器：`ssh -p 23679 root@connect.cqa1.seetacloud.com`
- conda env: `aeroverbs`（python3.10）
- torch 2.2.1+cu121, numpy 1.26.4, trimesh 3.23.5 + embreex
- 已补装：`matplotlib`, `scipy`
- 数据：`/root/autodl-tmp/data/trajctorys`
- 代码：`/root/AeroVerbs`

### 2.7 训练历史
- BC 开环：`outputs/bc_baseline/20260821-155734/final.pt`
- DAgger round1（around only）：`outputs/dagger/round1/train1/final.pt`
- DAgger round1~3（全 5 技能）：`outputs/dagger_all/round{1,2,3}/train{1,2,3}/final.pt`
  - round1 test waypoint: `0.00646`
  - round2 test waypoint: `0.00477`
  - round3 test waypoint: `0.00544`

---

## 3. 当前存在的问题

### 3.1 闭环成功率仍为 0
最终 DAgger 模型（`outputs/dagger_all/round3/train3/final.pt`）评测：
- around_left 3 条：全部 `collision`，但 `semantic_ok=True`
- around_right 3 条：全部 `collision`，但 `semantic_ok=True`

**失败原因**：模型已经绕对边，但绕行时**贴对象太近**，离对象表面约 0.18~0.25m，小于碰撞半径 0.35m，触发 collision。

### 3.2 through/over/under 语义 judge 不精确
- `under` 轨迹的 vertical signature 被归为 `through`，因为当前 `_vertical_class` 对 under 的判定条件太苛刻。
- through/over/under 的 rollout 语义判定也还没实现（rollout.py 只支持 around_left/right）。

### 3.3 DAgger 样本多样性不足
- 每轮每 skill 只采 5 条，底层 DAgger 状态少。
- Teacher 标签只是“回专家轨迹”，没有显式避碰修正。
- 碰撞前危险帧没有被重点利用。

---

## 4. 后续解决思路

### 4.1 提高 DAgger 样本量
- 每轮每 skill 采集 20~30 条，而不是 5 条。
- 优先采集失败样本（collision/timeout）。

### 4.2 危险帧标记与加权
- 采集时计算每帧到对象表面距离。
- 距离 < 0.45m 的帧标记 `danger=1`，存 `danger_mask.npy`。
- 训练时对 danger 帧的 waypoint loss 加权（如 ×3）。

### 4.3 Teacher 避碰修正
- 当 student 离对象太近时，teacher 前几个 waypoint 不再指向专家轨迹，而是先指向“远离对象”的安全点。
- 具体：`push_dir = normalize(student_xy - object_center_xy)`，安全目标点 = student + push_dir × 补偿距离。
- 让模型学会在危险状态第一反应是“先远离”。

### 4.4 完善 through/over/under 语义判定
- 修正 `semantic_judge.py` 中 `_vertical_class`：
  - under：轨迹在对象 bbox 内部且 z 明显低于对象顶部；
  - over：z 高于对象顶部 + margin；
  - through：穿过 opening 平面。
- 在 `rollout.py` 中增加 through/over/under 的成功判定，复用 `trajectory_signature` 或手写判定。

### 4.5 迭代训练
- 用上述改进后，继续 DAgger 1~3 轮。
- 每轮后评测 around_left/right 成功率，并观察 through/over/under 的 rollout 状态。
- 目标：around_left/right 成功率明显提升，且不碰撞。

---

## 5. 关键路径与文件

### 代码
```text
AeroVerbs/
├── aeroverbs/
│   ├── config.py
│   ├── constants.py
│   ├── data/
│   │   ├── dataset.py
│   │   ├── labels.py
│   │   ├── teacher.py
│   │   ├── semantic_judge.py
│   │   └── transforms.py
│   ├── env/radar.py
│   ├── model/
│   │   ├── encoders.py
│   │   └── policy.py
│   └── training/
│       ├── trainer.py
│       └── losses.py
└── scripts/
    ├── train.py
    ├── rollout.py
    ├── dagger_collect.py
    ├── train_dagger.py
    ├── dagger_iterate.py
    ├── verify_radar.py
    └── view_voxel_trajectory.py
```

### 服务器关键产物
```text
/root/AeroVerbs/outputs/
├── bc_baseline/20260821-155734/final.pt        # BC 开环模型
├── dagger/round1/train1/final.pt               # around only DAgger round1
└── dagger_all/
    ├── round1/train1/final.pt
    ├── round2/train2/final.pt
    └── round3/train3/final.pt                  # 当前最终模型
```

### 关键命令
```bash
# 训练 BC
python scripts/train.py --config configs/default.yaml --experiment bc_baseline --name run1

# 单轮 DAgger
python scripts/dagger_iterate.py \
    --start-checkpoint outputs/bc_baseline/20260821-155734/final.pt \
    --data-root /root/autodl-tmp/data/trajctorys \
    --skills around_left,around_right \
    --num-rounds 1 --num-per-skill 5 --train-epochs 5 \
    --output-root outputs/dagger

# 3 轮全技能 DAgger
python scripts/dagger_iterate.py \
    --start-checkpoint outputs/dagger/round1/train1/final.pt \
    --data-root /root/autodl-tmp/data/trajctorys \
    --skills around_left,around_right,through,over,under \
    --num-rounds 3 --num-per-skill 5 --train-epochs 5 \
    --output-root outputs/dagger_all

# 评测
python scripts/rollout.py \
    --checkpoint outputs/dagger_all/round3/train3/final.pt \
    --data-root /root/autodl-tmp/data/trajctorys \
    --skill around_left \
    --scene-id auto_around_left_asset_frl_apartment_bike_01_00000 \
    --output-dir outputs/rollout/eval
```

---

## 6. 下一步行动清单

- [ ] 实现 DAgger 样本量增加 + 失败样本优先
- [ ] 实现 danger_mask 标记与 waypoint loss 加权
- [ ] 实现 Teacher 避碰修正
- [ ] 修正 under/over/through 的 vertical signature
- [ ] rollout.py 增加 through/over/under 成功判定
- [ ] 重新 DAgger 训练并评测 around 成功率

---

# 修复记录（2026-08-23）

> 记录时间：2026-08-23 13:18
> 记录范围：从本日第一轮修改开始，到 Teacher 避碰 v3（tube + 恢复轨迹 + 趋势外推）为止。

---

## 0. 起始状态

- BC 开环模型：`outputs/bc_baseline/20260821-155734/final.pt`
- 旧 3 轮全技能 DAgger：`outputs/dagger_all/round3/train3/final.pt`，test waypoint loss 0.00544
- 闭环成功率 0：around_left/right 语义正确但贴边碰撞（0.18~0.25m < 0.35m）；through/over/under 语义判定未实现
- DAgger 样本量：每轮每技能 5 条

---

## 1. 第一轮修改：DAgger 提量 + danger_mask + Teacher 避碰 v1

**修改内容**
- DAgger 采集量 `--num-per-skill` 5 → 20；新增 `--priority-results` 失败样本优先采集
- 新增 `danger_mask.npy`：采集时逐帧计算到物体表面距离，< 0.45m 标记 danger=1；训练时 danger 帧 waypoint loss ×3（`danger_weight=3.0`）
- Teacher 避碰 v1：危险帧前 3 个 waypoint 指向"水平远离物体中心"的安全点（所有技能统一）

**涉及文件**
`scripts/dagger_collect.py`、`scripts/dagger_iterate.py`、`scripts/rollout.py`（CollisionChecker.surface_distance）、`aeroverbs/data/dataset.py`、`aeroverbs/training/losses.py`、`aeroverbs/training/trainer.py`、`aeroverbs/config.py`、`scripts/train_dagger.py`

**训练**
- `trajectory_batch_size` 8 → 24（`configs/default.yaml` + start checkpoint 内部配置）
- 3 轮 DAgger 训练输出：`outputs/dagger_v2/`，总耗时约 70 分钟
- test waypoint loss：round1 0.00547 / round3 0.00546（与旧模型几乎持平）

**修改后问题**
- 5 task rollout 仍然全部 collision，成功率 0
- over 的螺旋改善（164 步 → 49 步），但出现水平方向无意义运动
- through/under 与旧模型基本持平，均未穿过 bbox 即碰撞
- around_left 当时 semantic_ok=False（事后确认是判定误判，见第 4 轮）
- 发现 Teacher 避碰 v1 对所有技能统一"水平远离物体中心"过于粗糙

---

## 2. 第二轮修改：rollout 四视角渲染

**修改内容**
- `scripts/view_voxel_trajectory.py`：`make_figure` 支持多视角 2×2 融合图
- 四视角：默认视角 + 正视 `front (x+ → x-)` + 侧视 `side (y- → y+)` + 俯视 `top (z+ → z-)`
- 合图分辨率 1200×900（dpi 100）；rollout 渲染不再绘制 semantic free region

**修改后问题**
- 视角 (elev, azim) 顺序写反导致侧视/俯视不对，修正为标准三视图角度
- 用户肉眼看图确认：around_left 语义实际成功但贴边碰撞；through/under 语义基本正确；over 纵向正确但水平乱动

---

## 3. 第三轮修改：through/under/over 语义判定

**修改内容**
- through/under 共用判定：轨迹穿过障碍物 bbox（起终点在 bbox 两侧），且 bbox 内轨迹点不落在体素占用内
- over 判定：轨迹存在点 `z > bbox_max.z + 0.2`，且越顶阶段水平偏离物体中心 ≤ 2×物体宽度
- rollout 统一在每步更新语义判定，结果写入 `semantic_ok` / `semantic_detail`

**修改后问题**
- around_left/right 仍用旧 side min/max 判定，终点稍过头就会误判（around_left 被判 False）
- through/under 因碰撞截断未穿过 bbox，semantic_ok=False（正确）
- over semantic_ok=True（与肉眼一致）

---

## 4. 第四轮修改：around 改用 UVD 判定

**修改内容**
- around_left/right 语义判定改为 UVD（uniform visibility deformation）类判断：
  - 学生轨迹与教师轨迹按弧长等分重采样 64 点
  - 按顺序对应点连线，用 object-only mesh 做线段-网格相交检测
  - 任一连线穿过障碍物 → 不属于同一 UVD 类 → 语义失败
  - 全部连线无碰撞 → 语义成功
- mesh 缺失时退化为 bbox slab 相交

**修改后问题**
- 判定结果与肉眼一致：around_left/right semantic_ok=True；through/under 未穿过 bbox 为 False；over True
- 但成功率仍为 0：语义正确 ≠ 不碰撞

---

## 5. 第五轮修改：rollout 控制器消融

**修改内容**
- 4 组参数 × 5 task：
  - `wp0_tr045`（追 t+3）
  - `wp4_tr045`（追 t+20）
  - `wp2_tr0`（无转向平滑）
  - `wp2_tr02`（弱平滑 0.2）
- 与默认 `wp2_tr045` 对比

**结论与问题**
- 控制器不是主要问题
- through/under 对所有参数均 collision，不受控制器影响
- around 对参数不敏感，默认参数最稳
- over 对参数敏感：追 t+20 或无平滑会 timeout + 水平漂移；默认参数是唯一语义成功的组合
- 问题仍在模型侧：around 贴边碰撞、through/under 未穿过 bbox、over 水平噪声

---

## 6. 第六轮修改：Teacher 避碰 v2（按 skill + 使用 bbox/opening）

**修改内容**
- 新增 `PushConfig`（skill + bbox + center + opening_center）
- around：目标点 = 物体表面外 0.5m（径向），使用 bbox 水平半径
- through：目标点 = opening 中心（door 无 opening_center 时用 bbox 中心）
- over：保留专家水平方向，z 强制 `bbox_max.z + 0.5`
- under：保留专家水平方向，z 限制 `≤ bbox_max.z - 0.4`

**修改后问题**
- 用户指出：危险帧才修复可能太晚，应判断未来趋势是否危险，提前修复
- 确认碰撞半径：`PlannerParams.drone_radius=0.15m`、`clearance_margin=0.2m`、`planning_margin=0.35m`；rollout `--drone-radius 0.35` 与专家一致
- 参考 SINGER 论文，teacher 应给"恢复轨迹 + tube"而非"最近专家点 + 单一逃逸方向"

---

## 7. 第七轮修改：Teacher 避碰 v3（tube + 恢复轨迹 + 趋势外推）

**修改内容**
- `PushConfig` 增加 `tube_radius`，按 skill 设置：around 0.6 / through 0.3 / over 0.5 / under 0.4
- 触发条件：学生在专家 tube 外（`d_expert > tube_radius`）且当前/趋势危险才修复；tube 内不修复
- 恢复轨迹式修复：`安全子目标 g_safe → 0.5*g_safe+0.5*p_tube → p_tube`（p_tube 为专家点 clamp 到 2×tube_radius）
- 趋势外推：用最近有效动作历史的平均每步位移外推未来 3/6/10 步位置，计算未来表面距离
- `danger_dist = min(当前表面距离, 未来表面距离)` 传给 teacher；`danger_mask` 同步用 `danger_dist` 标记
- 动作历史不足 2 步时退化为沿当前 heading 外推

**修改后问题**
- 尚未开始重训验证
- 预期待验证：提前修复是否降低 around 贴边碰撞；恢复轨迹是否改善 over 水平噪声；through/under 是否能更早纠正到开口/桌下

---

## 下一步（待办）

- [ ] 使用新 Teacher（v3）重训 DAgger（建议输出 `outputs/dagger_v2c`）
- [ ] 重训后 5 task rollout 评估：semantic_ok + collision + goal_distance
- [ ] 视结果决定是否调整 tube_radius / 趋势外推步数 / danger_weight

---

## 6. 2026-08-24 DAgger 稳定性与训练加速修复

### 6.1 问题复盘
- 现象：BC 模型在关闭碰撞时轨迹更平滑、任务完成更好；DAgger 后模型轨迹明显曲折，且不考虑碰撞也更容易 timeout。
- 直接原因：DAgger teacher 标签本身比专家标签抖动更大，尤其是 rollout 实际追踪的 `waypoint_index=2`。已观察到 DAgger `wp2` 帧间跳变显著高于专家轨迹。
- 训练放大因素：`danger_weight=3.0` 放大危险帧中的 push-away 不连续标签；DAgger completion 正样本极少，动态 pos_weight 容易产生不稳定梯度；`lambda_clearance=0.5` 的 swept-path SDF loss 训练开销很大。
- 数据累积问题：`outputs/*/all_dagger` 旧逻辑会覆盖同名 symlink，导致多轮 DAgger 实际只保留最新一轮样本，少量 DAgger 样本被高重复率训练。

### 6.2 已修改代码
- `scripts/dagger_iterate.py`
  - 修复 `_link_into_all_dagger`：同名 DAgger 样本不再删除覆盖，而是加 `roundN_samples__` 前缀生成唯一 symlink，使 `all_dagger` 真正累积历史样本。
  - DAgger 默认参数改为更保守：`mix_ratio=0.25`、`max_dagger_repeats=4`、`danger_weight=1.5`。
  - 新增并向训练脚本传递：`--lambda-completion`、`--lambda-clearance`、`--lambda-temporal-smooth`、`--completion-pos-weight`。
  - 新增并向采集脚本传递：`--push-blend-width`。
- `scripts/dagger_collect.py`
  - 新增 `--push-blend-width`，传入 `PushConfig`。
  - push-away 语义从“替换 waypoint”更新为“连续 blend 到 recovery waypoint”。
- `aeroverbs/data/teacher.py`
  - `nearest_expert_index` 改为单调前向匹配，只在上次匹配点之后的短窗口搜索，减少最近专家帧前后跳变。
  - push-away 改为按危险程度和偏离专家管道程度连续插值，避免阈值硬切换造成标签折线。
- `aeroverbs/training/losses.py`
  - 新增 `temporal_waypoint_smoothness_loss`：把 rollout 追踪的 waypoint 转成世界系方向，约束相邻时间步的二阶差分，用于抑制闭环蛇形。
  - `lambda_completion=0` 时跳过 completion BCE 对总 loss 的贡献。
- `aeroverbs/training/trainer.py`
  - 接入 temporal smoothness context，使轨迹级 TBPTT 训练可使用时间平滑监督。
- `aeroverbs/config.py` / `configs/default.yaml`
  - 新增 `lambda_temporal_smooth`、`temporal_smooth_waypoint_index`。
  - 新增 `data.use_sdf`，当 DAgger `lambda_clearance<=0` 时可跳过 SDF 加载。
- `aeroverbs/data/dataset.py`
  - `use_sdf=false` 时直接不加载 `_sdf.npz`，避免关闭 clearance loss 后仍产生 SDF I/O。

### 6.3 新的推荐 DAgger 训练策略
建议先从 BC checkpoint 重新跑一个稳定版 DAgger，而不是继续在已经退化的 DAgger checkpoint 上 fine-tune：

```bash
python scripts/dagger_iterate.py \
    --start-checkpoint outputs/bc_baseline/finetune_v2/final.pt \
    --data-root /root/autodl-tmp/data/trajctorys \
    --skills around_left,around_right,through,over,under \
    --num-rounds 3 \
    --num-per-skill 20 \
    --train-epochs 5 \
    --lr 1e-4 \
    --mix-ratio 0.25 \
    --max-dagger-repeats 4 \
    --danger-threshold 0.45 \
    --push-distance 0.5 \
    --push-waypoint-count 3 \
    --push-blend-width 0.15 \
    --danger-weight 1.5 \
    --lambda-completion 0.0 \
    --lambda-clearance 0.0 \
    --lambda-temporal-smooth 0.2 \
    --output-root outputs/dagger_stable_v1
```

### 6.4 当前 DAgger 训练耗时原因
- `lambda_clearance>0` 时，`swept_clearance_loss` 会把每帧预测的 `position -> wp0 -> ... -> wp4` 路径按 0.1m 采样，并对每个采样点做 SDF `grid_sample`；实现中还包含 batch/time Python 循环，是最重的训练项之一。
- 即使不需要 clearance，旧数据集也会在 `TrajectoryDataset.__getitem__` 中尝试加载 SDF，造成额外磁盘 I/O 与 CPU 开销。
- 模型训练按时间步逐帧调用 `model.step()`，每个 step 都跑 radar encoder + action encoder + GRU；长轨迹、`trajectory_batch_size=24`、`chunk_len=20` 时，一个 epoch 会有大量小 kernel 和 Python 循环开销。
- `mix_ratio=0.5` 且实际 DAgger 样本只有 100 条时，训练集被重复扩到约 4000 条轨迹，其中 100 条 DAgger 被重复约 20 次，既慢又容易过拟合。
- `num_workers=8` 会并行读大量 `lidar_range_npy`，如果数据在网络盘或系统盘 I/O 压力大，会出现 DataLoader 等待。

### 6.5 加速建议
- DAgger 稳定性验证阶段默认使用 `--lambda-clearance 0.0`，并由脚本自动设置 `data.use_sdf=false`，先省掉最重的 swept SDF loss 与 SDF 读取。
- DAgger 默认 `--lambda-completion 0.0`，先只优化 waypoint + temporal smoothness，减少稀疏 completion 标签造成的不稳定训练。
- 默认 `mix_ratio=0.25` 且 `max_dagger_repeats=4`，减少少量 DAgger 样本的重复训练；修复 symlink 累积后，后续轮次 DAgger 样本自然增加，不需要把 100 条样本重复 20 次。
- 若仍慢，可临时降低 `trajectory_batch_size` 到 12 或调低 `num_workers` 到 4，观察 GPU 利用率与 DataLoader wait；如果 GPU 利用率低，多半是雷达 npy/SDF I/O 和 Python 时间循环限制。
- 中长期优化方向：把 `AeroVerbsPolicy.forward()` 的逐帧 radar encoder 调用改成先展平 `[B*T,3,H,W]` 一次性编码，再 reshape 回 `[B,T,D]` 喂 GRU，可减少大量 Python 循环和小 kernel 调度开销。



---

## 7. Gate / Downward Range 网络修改状态

> 更新：2026-08-25 15:54:12（heatmap-derived gate + downward range + progress/loss 改造）

### 7.1 已实现

- **确认 DAgger teacher 尚未替换为 Fast-Planner**：当前 `aeroverbs/data/teacher.py` 仍是 nearest-expert-frame labeler；`scripts/dagger_collect.py` 仍调用 `TeacherLabeler(sample_dir)`，没有调用 Fast-Planner local replanner。
- **预计算输入已接入训练数据**：`TrajectoryDataset` / `ManeuverDataset` 现在读取每条轨迹的 `downward_range.npy [T,2]` 和 `heatmap_gate_summary.json`。
- **新增 gate 几何特征**：`aeroverbs/data/gate_features.py` 根据 gate center / normal / left / up / width / height / confidence 和当前 pose 生成每帧 `gate_features [12]`。
- **模型输入已修改**：GRU recurrent fusion 默认使用 `radar + gate_features + downward_range + skill_id + action_history`；`use_heatmap_latent=false`，heatmap 不再默认作为 pooled latent 进入 GRU。
- **新增模型模块**：`GateEncoder`、`DownwardRangeEncoder`、`ProgressHead`。
- **模型输出已修改**：保留 `waypoints [K,3]`、`completion_logits [1]`，新增 `progress_logits [1]`。
- **loss 已修改**：在 waypoint / completion / smoothness 之外，新增 `temporal_direction`、`action_consistency`、`gate_cross`、`gate_direction`、`progress`。`lambda_completion` 默认降到 `0.05`。
- **gate loss 启用条件改为坐标判断**：使用当前 UAV 相对 gate 平面的 signed distance，默认启用区间为 `[-4.0m, +0.5m]`，不依赖模型预测 progress。
- **rollout 已接入 gate 强制约束**：推理时在线计算 downward range 和 gate feature；若预测 waypoints 不穿过 gate 且当前位置仍处于 gate active 区间，则临时追踪 gate center，并记录 `gate_fallback_steps`。
- **DAgger 采集脚本已适配新输入格式**：新采集样本会保存 `downward_range.npy`，并复制 `heatmap_gate_summary.json` / `heatmap_gate_region.json`。
- **服务器验证已通过**：`scripts/smoke_test.py`、真实 over batch 的 model/loss forward、最小 rollout smoke 均已通过。

### 7.2 本轮训练配置

- 配置文件：`configs/gate_downward_server.yaml`
- 数据根目录：`/root/autodl-tmp/data/trajctorys`
- 实验名：`gate_downward_bc`
- 关键输入：`downward_range.npy`、`heatmap_gate_summary.json`
- 关键 loss 权重：
  - `lambda_completion=0.05`
  - `lambda_temporal_direction=0.1`
  - `lambda_action_consistency=0.1`
  - `lambda_gate_cross=0.1`
  - `lambda_gate_direction=0.05`
  - `lambda_progress=0.05`

### 7.3 后续仍需修改 / 验证

- **DAgger teacher 替换 Fast-Planner local replanner 尚未实现**：后续需要把当前 nearest-expert teacher 改为 student 当前状态 + skill + heatmap/gate/semantic constraints + SDF/ESDF 的局部重规划 teacher，并保留 fallback。
- **旧 checkpoint 迁移尚未实现**：新模型结构和旧 BC/DAgger checkpoint 不完全兼容；当前建议先从头训练 gate/downward BC，之后再单独实现 backbone 部分加载。
- **gate fallback 策略需要 rollout ablation**：需要比较无 fallback / gate center fallback / nearest-inside-gate fallback。
- **through / over / under 的 rollout success judge 仍需补全**：当前 rollout 语义成功判断主要支持 around_left/right。
- **训练后需要做 heatmap counterfactual ablation**：correct / zero / shuffled / opposite heatmap，验证 gate 和 heatmap 的因果作用。
- **可选后续**：轻量 clearance loss、Fast-Planner DAgger、多候选 trajectory head。


---

## 8. Clearance Loss 与 Through Occupancy 更新

> 更新：2026-08-26（clearance loss 对齐规划安全距离 + through 程序化体素占用 + 训练日志时间戳）

### 8.1 已实现

- **新增 clearance loss**：训练 loss 中新增 `clearance` 分量，并记录到 `metrics.jsonl` / TensorBoard 标量中。
- **around / over clearance**：将目标物体按球体近似，基于预测 waypoint 的 world-frame 坐标计算到 object sphere 的安全距离惩罚。
- **through / under clearance**：使用体素占用点 `occupied_points` 计算障碍物距离，并结合 `free_region_points` 约束 gate 附近 waypoint 靠近可通行区域。
- **through 程序化场景体素已补齐**：新增 `scripts/build_through_occupancy.py`，对 `_scenes/through/<scene_id>/<scene_id>.glb` 做 surface voxelization，保存为 under 同格式的 `occupancy/<object_id>_voxels.npz`。
- **through 全量体素生成完成**：`/root/autodl-tmp/data/trajctorys/_scenes/through` 下 500 个 scene 已生成 500 个 occupancy npz；Dataset 抽样验证可正常读取。
- **clearance 参数对齐采集/规划**：当前配置为 `clearance_drone_radius=0.15`、`clearance_margin=0.20`，总安全距离为 `0.35m`，与 `PlannerParams.planning_margin` 一致。
- **训练日志输出加入时间戳**：`aeroverbs/training/trainer.py` 中关键 print 输出和 `metrics.jsonl` 记录均加入 `timestamp`；tqdm epoch 描述也包含启动时间。
- **resume 兼容 model-only final.pt**：`train()` 现在支持从只包含 `model/config` 的 `final.pt` 加载权重，使用 fresh optimizer 开始新 run。

### 8.2 当前关键配置

- 配置文件：`configs/gate_downward_server.yaml`
- 关键 clearance 配置：
  - `lambda_clearance=1.0`
  - `clearance_drone_radius=0.15`
  - `clearance_margin=0.20`
  - `clearance_region_radius=0.35`
  - `clearance_occupied_points=1024`
  - `clearance_region_points=512`
- 继续训练起点：`outputs/bc_baseline/20260825-202126/final.pt`（model-only checkpoint，fresh optimizer）

### 8.3 后续建议

- 训练完成后，使用同一组 10 条轨迹做 rollout，对比 collision-on 与 collision-off 的成功率，并重点检查 around 水平偏移、through/under gate 内 clearance。
- 如果轨迹变得过于贴近障碍物，可增大 `clearance_margin`；如果轨迹过于保守，可保持当前与规划一致的 0.35m，并优先调低 `lambda_clearance` 或 `clearance_region_radius`。


---

## 9. 2026-08-26 Fast-Planner 在线 DAgger Teacher 与 0.15m 碰撞阈值

> 更新：2026-08-26（完整 Python Fast-Planner 在线局部重规划 + 预计算 SDF 直读 + DAgger 碰撞阈值统一）

### 9.1 已实现

- **DAgger Teacher 已替换为真实 Fast-Planner 在线局部重规划**：新增 `aeroverbs/data/fast_planner_teacher.py`，`scripts/dagger_collect.py` 不再直接使用 nearest-expert-frame Teacher 生成正常 waypoint 标签。
- **完整规划流程已保留**：Teacher 默认执行 kinodynamic A*、Non-Uniform B-spline 优化和时间重分配；`optimizer_enabled=true`，未使用仅 A* 的简化版本。
- **复用本地 Python 移植代码**：已将 `uav_maneuver_planning/fast_planner` 及其依赖集成到 `/root/AeroVerbs/uav_maneuver_planning`。
- **直接读取数据集预计算 SDF**：加载 `<data_root>/_sdf/<skill>/<scene_id>.npz` 中的 `grid [D=z,H=y,W=x]`、`origin`、`voxel_size`；DAgger 采集过程中不再 voxelize mesh，也不重新运行 EDT / distance transform。
- **SDF 范围扩展但不重算场**：原始 SDF 仅覆盖障碍物附近约 1m padding；加载时将原场嵌入包含专家起点、终点和边界余量的 free-space map，原 SDF 数值保持不变。
- **在线起始状态**：每次重规划使用学生当前 position，并由最近一次实际执行位移估计 start velocity（限幅到规划器最大速度）；第一帧使用零速度。
- **语义 guide**：使用 `heatmap_gate_region.json` 点集中心作为语义 guide；越过 guide plane 前规划 `current -> guide -> 最终目标`，越过后规划 `current -> 最终目标`，用于保持 around-left/right、through、over、under 的任务方向。
- **规划缓存**：默认每 5 个采集帧重规划一次，中间帧投影到缓存轨迹并按 `[3,6,10,15,20]` 时间偏移生成 body-frame waypoint 标签，避免逐帧运行规划器。
- **可靠 fallback**：规划失败、无效或超过时间预算时，当前帧回退到旧 `TeacherLabeler`；规划失败不会终止整条 DAgger 采集。
- **新增运行统计**：每条新 DAgger 样本保存 `teacher_planner_stats.json`，包含 replan attempts/success/failure/timeout、fallback frame/fraction、mean/p95/max planning time、SDF 来源和 optimizer 状态。

### 9.2 0.15m 碰撞阈值

- `scripts/rollout.py` 的 `--drone-radius` 默认值已由 `0.35m` 改为 `0.15m`。
- `scripts/dagger_collect.py` 的 `--drone-radius` 默认值已由 `0.35m` 改为 `0.15m`。
- Fast-Planner 从预计算 signed SDF 生成硬占用时使用 `signed_distance <= 0.15m`；collector 的碰撞检查和 Teacher 的硬规划半径保持一致。
- `0.15m` 是无人机硬碰撞半径；训练中的 `clearance_margin=0.20m` / 总软安全距离 `0.35m` 仍可作为 loss 的保守约束，两者含义不再混用。

### 9.3 Rollout 小样本结果与限制

- checkpoint：`outputs/bc_baseline/20260826-074509/final.pt`。
- 10 条/任务、共 50 条、碰撞半径 0.15m 的渲染测试输出：`outputs/rollout/bc_baseline_20260826-074509_eval_10each_r015_rendered/`。
- 该批次结果为 `43/50`：around_left `10/10`、around_right `6/10`、through `10/10`（door/ring/window 均覆盖）、over `7/10`、under `10/10`。
- **上述 43/50 只是在 50 条轨迹上的小样本观测，不能视为稳定成功率或后续 DAgger 的高成功率目标**；扩大测试集后成功率可能下降，后续必须做更大规模、固定 split 的评估。

### 9.4 规划速度优化与验证

- 初次真实完整规划约 `12.35s`。profile 确认瓶颈不是 kinodynamic A*，而是移植版 B-spline optimizer 未向 SciPy L-BFGS-B 提供 ESDF 解析梯度，导致对全部控制点做数值差分（单次约 8600 次目标函数计算）。
- `uav_maneuver_planning/fast_planner/bspline_optimizer.py` 已接入 smoothness、ESDF distance、velocity/acceleration feasibility 和 guide cost 的解析梯度；完整 B-spline 流程保持启用。
- 同一真实 around_left 场景规划耗时由约 `12.35s` 降至约 `0.80s`。
- 五类各取一个真实场景做首帧完整规划均成功、均生成 5 个有效 waypoint；实测 wall time：around_left `0.531s`、around_right `0.735s`、through `0.974s`、over `0.647s`、under `0.598s`。
- 闭环采集 smoke 输出：`outputs/dagger_smoke/fast_planner_teacher_20260826_r015_velocity/`；around_left 共 7 帧，完成 `2/2` 次重规划，`fallback=0`，平均规划时间约 `0.747s`。
- smoke 标签形状已验证：`teacher_waypoints [7,5,3]`、`teacher_waypoint_valid [7,5]`、`teacher_completion [7,1]`、`downward_range [7,2]`，并成功保存 planner stats。

### 9.5 测试与当前边界

- 服务器 `aeroverbs` 环境通过 10 个聚焦测试：预计算 SDF 轴序/膨胀/范围扩展、manager SDF 注入、真实 start velocity 转发、Teacher 5 帧缓存、guide crossing、fallback、解析梯度、collector 默认值和 stats 保存。
- `python -m compileall` 已通过：`aeroverbs/data/fast_planner_teacher.py`、`uav_maneuver_planning`、`scripts/dagger_collect.py`。
- **本阶段未实现**：绕障后重新接入专家轨迹 suffix、around 终点 y 回正的专门逻辑或 loss；当前只以语义 guide 和原专家最终位置作为在线规划约束。
- **下一步建议**：先用当前 Teacher 做小规模 DAgger round，统计真实采集中的 p95 planning time、fallback fraction 和各 skill 标签有效率；确认速度与稳定性后再扩大采集量。around 末端 y 偏移保持为非核心观察项，不应阻塞第一轮 DAgger。

### 9.6 DAgger 迭代编排兼容修复

- `scripts/dagger_iterate.py` 已适配新版 Fast-Planner collector，自动转发 `--drone-radius`、`--teacher-replan-interval`、`--teacher-timeout`、`--teacher-max-iter`。
- 新默认值与 collector 对齐：硬碰撞半径 `0.15m`、每 `5` 帧重规划、规划预算 `2.0s`、kinodynamic A* `max_iter=2000`。
- 旧 `danger-threshold` / `push-distance` / `push-waypoint-count` / `push-blend-width` / `priority-results` 参数仍可被迭代脚本解析以兼容旧启动命令，但不再传给 Fast-Planner collector，也不会再触发 `unrecognized arguments`。
- 新增 `tests/test_dagger_iterate_fast_planner.py`，验证新参数默认值、命令转发和旧参数过滤；服务器全量现有测试结果为 `13/13 passed`，迭代器生成的真实 collect command 已由 `dagger_collect.parse_args()` 成功解析。

### 9.7 Fast-Planner DAgger 首轮小规模闭环验证

> 更新：2026-08-26（25 条 Teacher 采集 + 3 epoch DAgger 训练 + 5 类闭环 smoke）

- **首次 pilot 暴露并修复了两个根因问题**：
  - collector 在 Teacher completion 已达成后仍继续采集到 `max_steps`，导致完成绕障后飞离目标区并大量 fallback；现已改为 Teacher completion 立即终止该轨迹。
  - Fast-Planner 扩展地图的 `bounds_margin=0.5m` 不足以覆盖 student 的横向恢复运动，局部状态会短暂超出 SDF map；默认边界余量已扩大为 `1.5m`。
- **修复后的原失败轨迹重放**：在 completion 帧停止，共 57 帧；Fast-Planner `12/12` 次重规划成功，`fallback=0`，平均单次规划约 `0.359s`。
- **第二次 pilot 采集完成**：输出位于 `outputs/dagger_fast_planner_pilot_v2/`，5 类各 5 条，共 25 条、1147 帧标签；采集耗时约 `130.5s`。
- **Teacher 采集统计**：
  - around_left：重规划 `58/60`，fallback `10/295 = 3.4%`，样本平均规划时间均值约 `0.389s`。
  - around_right：重规划 `30/39`，fallback `38/184 = 20.7%`，样本平均规划时间均值约 `0.453s`。
  - through：重规划 `40/46`，fallback `30/218 = 13.8%`，样本平均规划时间均值约 `0.390s`。
  - over：重规划 `40/43`，fallback `15/200 = 7.5%`，样本平均规划时间均值约 `0.334s`。
  - under：重规划 `45/52`，fallback `35/250 = 14.0%`，样本平均规划时间均值约 `0.326s`。
  - 总计重规划 `213/240 = 88.8%`，fallback `128/1147 = 11.2%`。around_right 的 fallback 率最高，扩大 DAgger 规模前需继续监控。
- **训练入口 API 问题已修复**：采集完成后首次启动训练时，`scripts/train_dagger.py` 向当前 `TrajectoryDataset` 传入了已不支持的 `completion_cfg`，引发 `TypeError`。现已改为当前两参数 API，并新增回归测试。已完成的 25 条采集数据被直接复用，未重复采集。
- **第 1 轮 DAgger 训练完成**：从 `outputs/bc_baseline/20260826-074509/final.pt` 启动，expert train `2021`、DAgger `25`、mixed train `2121`，训练 3 epoch，`mix_ratio=0.25`、`max_dagger_repeats=4`、`lr=1e-4`。
- **训练结果**：epoch train total loss 为 `0.0234 -> 0.0194 -> 0.0189`，val total loss 为 `0.0249 -> 0.0240 -> 0.0245`，最终 test total loss 为 `0.0236`；无 NaN/Inf 或运行时异常。最终 checkpoint：`outputs/dagger_fast_planner_pilot_v2/round1/train1/final.pt`。
- **最终验证**：checkpoint 大小约 16 MB，107 个参数张量均为有限值；服务器 `unittest` 全量结果为 `16/16 passed`。
- **闭环 rollout smoke**：从上一轮已成功样本中每类选 1 条，5 类均成功且 `semantic_ok=true`，渲染输出位于 `outputs/rollout/dagger_fast_planner_pilot_v2_round1_smoke/`。该 5/5 仅是可运行性 smoke，样本经过筛选，不能作为新模型的成功率结论。

### 9.8 DAgger Loss 对齐与 Region-Aware Preferred Clearance

> 更新：2026-08-26（DAgger 默认继承 BC loss + Teacher 硬/软安全距离解耦 + 可通行区域软引导）

- **DAgger loss 默认与起始 checkpoint 对齐**：`train_dagger.py` 不再默认强制 `lambda_completion=0`、`lambda_clearance=0`和 `lambda_temporal_smooth=0.2`；参数未显式传入时保留 checkpoint 配置，显式 CLI 值仍可覆盖。
- **当前 BC 起点的实际 loss 权重会被完整继承**：`lambda_completion=0.05`、`lambda_smooth=0.05`、`lambda_temporal_direction=0.1`、`lambda_action_consistency=0.1`、`lambda_gate_cross=0.5`、`lambda_gate_direction=0.05`、`lambda_progress=0.05`、`lambda_clearance=1.0`。
- **迭代器避免隐式覆盖**：`dagger_iterate.py` 只在用户显式指定时向 `train_dagger.py` 转发 lambda 参数；默认生成的 train command 不包含这些参数。
- **硬碰撞与 Teacher 软间距已解耦**：默认 `collision_radius=0.15m`保持不变；新增 `preferred_clearance=0.50m`，按原始物体表面距离定义；B-spline optimizer 使用 `optimization_dist0=max(0.50-0.15,0)=0.35m`，不再把 `0.15m` 当作行为层期望间距。
- **可通行区域改为软引导**：Teacher 不再直接使用 region 质心。先由专家轨迹与 gate 参考平面求 anchor，再按“距 anchor 距离 + preferred-clearance deficit”对 region 点加权，最终 guide 为 `75% expert crossing anchor + 25% region hint`。
- **region 不是硬约束**：没有新增“规划轨迹穿越点必须位于 region 内”的判定；region/gate 不精确时不会因此拒绝一条几何上可行的 Fast-Planner 路径。
- **采集参数和统计已打通**：`dagger_collect.py` 新增 `--teacher-preferred-clearance`，`dagger_iterate.py` 自动转发；`teacher_planner_stats.json` 新增 preferred clearance、optimizer target、guide source 和 guide 位置。
- **真实数据验证**：around_left 与 through 真实场景首次完整规划均生成 `5/5` 有效 waypoint、`fallback=0`，耗时分别约 `0.883s` / `0.715s`；Teacher waypoint 最小原始 SDF 距离分别约 `0.388m` / `0.618m`。
- **验证状态**：新增 loss 继承、preferred clearance、CLI 转发和 region 软引导回归测试；服务器全量 `unittest` 为 `21/21 passed`，修改模块 `compileall` 通过。
- **当前边界**：`preferred_clearance=0.50m` 目前为全任务软目标，窄 through/under 通道允许软代价违反但不允许进入 `0.15m` 硬碰撞区；后续可基于开口尺寸做 skill/scene-adaptive preferred clearance。专家 suffix 重连仍未实现。

### 9.9 分任务 Preferred Clearance、分层采样与流程文档

> 更新：2026-08-26（降低窄通道软间距 + 固定种子跨类别采样 + DAgger 流程固化）

- **分任务 Teacher preferred clearance 已启用**：`around_left=0.50m`、`around_right=0.50m`、`over=0.20m`、`through=0.15m`、`under=0.15m`；本节配置取代 9.8 中“全任务 0.50m”的阶段性设置。
- **硬碰撞阈值不变**：所有任务仍统一使用 `collision_radius/drone_radius=0.15m`。对应 B-spline optimizer 的软 `dist0` 分别为 around `0.35m`、over `0.05m`、through/under `0.00m`。
- **保留全局显式覆盖能力**：collector/iterator 的 `--teacher-preferred-clearance` 默认改为 `None`，省略参数时按 skill 选择上述值；仅在显式传参时覆盖全部任务。
- **collector 改为固定种子类别分层抽样**：新增 `--sample-seed`（默认 42），每个 skill 独立进行稳定洗牌和 round-robin 类别交错，不再简单取数据索引前 N 条；失败轨迹不会占用成功采集名额，会继续后续候选补足。
- **类别识别覆盖真实数据形态**：优先使用 asset 文件名，其次使用 door/window/ring 等有效 type；程序生成样本若只有通用 `type=obstacle`，会从 scene id 恢复 column/box 等类别。显式 `--scene-ids` 时不改变用户指定顺序。
- **真实数据抽样检查**：每类前 10 条均为确定性结果；around/over 能跨多个物体类别，through 覆盖 door/window/ring，under 覆盖现有 table 类别。
- **Loss 行为确认**：DAgger 默认继承 BC 的全部有效 loss，clearance 保持 `lambda_clearance=1.0`。`danger_weight` 与 `lambda_temporal_smooth` 当前仅为兼容字段，没有进入最终 loss；实际平滑项由 `lambda_smooth=0.05` 控制。
- **完整流程文档**：新增 `docs/DAGGER_TRAINING_FLOW.md`，记录抽样、在线 Teacher、预计算 SDF、历史数据累积、混合训练、loss、推荐命令、验收指标与当前边界。
- **验证状态**：服务器 `unittest discover` 全量 `26/26 passed`，修改模块 `compileall` 通过；真实数据前 10 条候选检查确认 through 覆盖 door/window/ring，通用 obstacle 能正确恢复为 column/box 类别。

### 9.10 Teacher Fallback 屏蔽、Around Guide 截断与旧数据 Relabel

> 更新：2026-08-26（移除 nearest-expert waypoint fallback + 安全缓存复用 + 100条旧状态离线重标）

- **around 退化诊断**：同场景对比确认首轮 DAgger 后 around_left 更晚开始侧移；距障碍物约 2m 时，两条样本的侧向偏移分别由 BC `0.674/0.652m` 降到 DAgger `0.480/0.405m`。Teacher 数据中，student 距障碍物 1.5–2m 时 waypoint index 2 已跨到 guide plane 后方的比例约为 around_left `51%`、around_right `59%`。
- **旧 fallback 污染确认**：首轮100条采集的加权 fallback 比例为 around_left `21.0%`、around_right `34.9%`；高 fallback 样本会明显把 waypoint 拉回障碍物方向。
- **nearest-expert waypoint 已屏蔽**：Fast-Planner 和安全缓存均不可用时，不再调用旧 Teacher 生成 waypoint；该帧返回零 waypoint validity，旧 Teacher 只继续提供 completion 判断。
- **安全缓存保留**：重规划失败不再立即清空上一条路径；仅当当前位置离缓存超过 `0.75m`、缓存超出 SDF map 或监督 horizon 内进入 `0.15m` 硬占用时才废弃缓存。
- **around stage-aware 标签**：student 未真正跨过 guide plane 时，waypoint index 2 不允许提前进入 guide 后方回正段；更远的 post-guide waypoint 标为无效。接近 guide 小于 `0.2m` 不再导致 around 提前丢弃 guide。
- **逐帧来源和权重**：新 collector 保存 `teacher_label_source.npy` 和 `teacher_frame_weight.npy`。来源包括 `fast_planner_replan`、`fast_planner_cache`、`invalid`；invalid 帧的 `teacher_waypoint_valid` 全部为0。
- **Action consistency 完全关闭**：`train_dagger.py` 在 DAgger 训练中强制设置 `lambda_action_consistency=0.0`，避免把 student rollout 的实际动作重新当成专家动作学习；clearance 等几何 loss 保持开启。
- **旧100条轨迹已无损复用**：新增 `scripts/dagger_relabel.py`，从原专家样本构造新 Teacher，对保存的 student 状态离线重标。输出位于 `outputs/dagger_next/round1_samples_relabel_v2/`；雷达等大文件使用符号链接，原 `round1_samples/` 标签哈希未变化。
- **relabel 完整性**：5类各20条、共100条和5310帧。来源统计为新规划833帧、安全缓存3940帧、invalid 537帧（10.1%）。按任务 invalid 比例：around_left `9.1%`、around_right `24.2%`、through `0.6%`、over `8.3%`、under `1.0%`。
- **around 截断验证**：40条 relabelled around 轨迹中，guide 前有效 index2 共1258帧，提前跨 guide plane 的帧数为0。

---

## 10. 正式闭环对比评测方案

> 制定日期：2026-08-27
> 目标：在控制单批运行成本、便于故障恢复的同时，覆盖完整固定 test split，并公平比较 AeroVerbs 与其他范式的方法。

### 10.1 评测对象

- AeroVerbs BC。
- AeroVerbs DAgger。
- 纯 Agent 方法，例如 AeroClaw。
- VLA 方法，例如 OpenVLA。
- 世界模型 / VLN 方法，例如 WorldVLN。
- 目标驱动方法；SINGER 若无法复现则选择可获得的替代方法。
- 传统路径规划方法，例如 EGO。

不同方法必须使用相同的场景、初始状态、任务指令、最大步数、碰撞半径和成功判定。若某个方法使用完整地图、额外传感器或其他 privileged information，需在结果中单独注明，避免将系统级比较描述成完全同观测条件的策略比较。

### 10.2 固定 test split

当前固定 test split（`split_samples(..., seed=0)`）共 254 条：

| Skill | Test 数量 |
|---|---:|
| around_left | 75 |
| around_right | 75 |
| through | 50 |
| over | 44 |
| under | 10 |
| **合计** | **254** |

正式评测不从全数据重新抽样，只使用该固定 test split；训练集、验证集和 DAgger 采集数据不得进入正式评测。

### 10.3 五个互斥批次

将 254 条 test 样本按 skill 分层、固定随机种子打乱，再切成 5 个互不重叠的批次：

| Skill | Batch 1 | Batch 2 | Batch 3 | Batch 4 | Batch 5 | 合计 |
|---|---:|---:|---:|---:|---:|---:|
| around_left | 15 | 15 | 15 | 15 | 15 | 75 |
| around_right | 15 | 15 | 15 | 15 | 15 | 75 |
| through | 10 | 10 | 10 | 10 | 10 | 50 |
| over | 9 | 9 | 9 | 9 | 8 | 44 |
| under | 2 | 2 | 2 | 2 | 2 | 10 |
| **每批合计** | **51** | **51** | **51** | **51** | **50** | **254** |

- skill 内部尽量让 asset/object family 均匀分布到五批，例如 through 的 door/window/ring。
- 生成后将场景列表固化为 `eval_manifests/batch_{1..5}.json`，保存随机种子和 test split 配置；后续不得按模型表现修改清单。
- 所有对比方法必须运行完全相同的五份 manifest，不能为每种方法单独抽样。
- 五批依次运行以隔离故障；五批合并后等价于对完整 254 条 test split 做一次评测。

**生成状态（2026-08-27）**

- 已生成目录：`/root/AeroVerbs/eval_manifests/`。
- 固定参数：`split_seed=0`、`sample_seed=42`。
- 文件：`batch_1.json` 至 `batch_5.json`，以及带每批 SHA-256 的 `manifest_index.json`。
- 每条记录包含 `skill`、`scene_id`、`relative_path` 和 `category`，后续 runner 可直接读取，不再重新抽样。
- 生成器：`scripts/build_eval_manifests.py`；回归测试：`tests/test_build_eval_manifests.py`。
- 已验证批次大小为 `[51,51,51,51,50]`，254 个 ID 全部唯一，五批并集严格等于固定 test split，类别跨批均衡且相同 seeds 可确定性复现。

### 10.4 固定运行条件

在正式测试前于 validation split 上确定全部超参数。正式 test 开始后不得根据结果调整：

- `drone_radius=0.15m`（硬碰撞半径）。
- `max_steps=200`，控制频率 10 Hz。
- `step_size=0.1m`。
- AeroVerbs 默认 `waypoint_index=2`、`turn_rate=0.45`。
- gate correction、CompletionJudge、碰撞 mesh、雷达预处理和成功阈值。
- 各随机方法的解码参数和随机种子策略。
- 各方法允许使用的传感器、地图、历史状态及外部工具权限。

### 10.5 成功定义与记录字段

AeroVerbs 的正式成功判定只看 rollout 输出：`status == "success"` 即记为成功，否则记为失败。`goal` 的位置本身并非绝对准确，因此 `goal_distance` 不参与成功判定，也不设置 goal-distance 阈值；成功案例无需额外满足接近最终 goal 的条件。正式结果至少记录：

- `status`：success / collision / timeout / infrastructure_error。
- `semantic_ok`、`far_side`、`safe_clearance`、`ground_ok`、`sustained`。
- 最小 clearance、完成步数、轨迹长度。
- gate correction/fallback 次数。
- 单步延迟、整条轨迹 wall time、硬件和显存占用（可获得时）。

所有方法都应通过统一评测器输出相同含义的 `status`，主成功率直接统计 `status == "success"`。`goal_distance` 如因调试需要而保留，只能作为可选诊断字段，不用于成功/失败判断、方法排名或主结论。人工复核全部失败案例，并随机抽查一部分自动判定成功案例。

### 10.6 结果汇总与统计

主结果基于五批合并后的全部 254 条，而不是五个批次百分比的简单平均：

```text
overall_micro_SR = status == "success" 的案例数 / 254
per_skill_SR     = 该 skill 中 status == "success" 的案例数 / 该 skill 的 test 总数
macro_SR         = 五个 per_skill_SR 的算术平均
```

同时报告 collision rate、timeout rate、平均完成步数、最小安全间距和运行耗时。五批的成功率 `mean ± SD` 可作为批次难度/运行稳定性的辅助指标，但不能替代合并后的 `success/254`。

方法间比较使用逐场景配对结果：

- 报告两种方法各自独有的成功/失败数量。
- 使用 McNemar exact test 检验配对成功率差异。
- 使用 scene-level paired bootstrap 计算成功率差值的 95% CI；同一次 bootstrap 对所有方法使用相同重采样索引。
- 多方法两两检验时使用 Holm 等多重比较校正。

### 10.7 故障恢复与重试规则

- 每个场景结束后立即追加写入 `results.jsonl`，并保留独立场景日志；不等待整批结束再统一保存。
- 重启批次时根据 `scene_id` 跳过已有完整结果，只运行未完成案例。
- collision、算法自身 timeout 和无法完成任务均视为算法失败，不得重跑至成功。
- 进程崩溃、OOM、API/网络断线、文件读取损坏等记为 `infrastructure_error`，按预先固定的次数重试（建议最多 2 次）。
- 超过重试次数仍失败的案例保留错误记录，不得静默删除或更换场景。
- 每批完成后先检查案例数、唯一 scene 数、状态字段和失败日志，再开始下一批。

### 10.8 测试集使用说明

此前 BC/DAgger 的 10 条/skill、共 50 条 rollout 已用于模型诊断并影响后续修改，因此应标记为 diagnostic evaluation，而不是严格的一次性 blind test。正式报告应说明部分 test 场景曾用于诊断；若后续能生成新的独立场景，应另建从未用于训练、DAgger 采集或调参的最终 blind test，尤其需要扩充当前仅有 10 条的 under 测试集。

### 10.9 Semantic SPL（基于 Fast-Planner 参考路径）

除成功率外，正式评测增加 Semantic SPL，用于衡量成功轨迹的路径效率。由于数据中的原始 `goal` 并非绝对准确，SPL 不使用原始 goal，也不使用各模型自己产生的终点作为参考终点；每个案例预先计算一条对所有方法固定、共享的语义参考路径。

#### 10.9.1 固定语义参考终点

第一版采用专家轨迹首次满足当前 CompletionJudge 的位置作为该案例的 canonical semantic endpoint：

1. 从该案例的固定起点开始，按顺序将完整专家轨迹输入与正式评测相同的 CompletionJudge。
2. 找到专家轨迹第一次满足 completion（即若作为 rollout 则会得到 `status == "success"`）的时刻。
3. 将该时刻的位置记为固定语义参考终点 `q_ref`；不使用数据中的原始 goal。
4. 若专家轨迹无法通过当前 CompletionJudge，则标记 `reference_invalid`，先修复案例或判定器，不能为不同方法临时更换终点。

`q_ref` 对同一个案例的所有方法保持完全相同。后续可升级为在合法语义完成区域内生成多个候选终点、分别规划并选择最短有效路径，但升级后必须一次性重新冻结所有案例的参考数据，不能针对单个方法调整。

#### 10.9.2 Fast-Planner 参考路径长度

从案例固定起点到 `q_ref` 运行 Fast-Planner，使用与 rollout 一致的场景、SDF、skill/semantic guide、动力学约束和 `drone_radius=0.15m`。规划完成后：

1. 用同一个 CompletionJudge 逐步验证规划轨迹。
2. 将规划轨迹截断到第一次满足 completion 的位置。
3. 用三维弧长计算参考路径长度：

```text
L_ref = sum_t ||q_t - q_(t-1)||_2
```

Fast-Planner 不能证明得到全局最短解，因此 `L_ref` 在文档和论文中称为 **Fast-Planner semantic reference path length**，不宣称为真实全局最短路径。若规划失败、碰撞或未通过 CompletionJudge，则记为 `planner_reference_invalid`，不得用模型路径或专家路径静默替代。

#### 10.9.3 模型实际路径长度

对任一待评测方法，实际路径长度 `P` 从固定起点累计到 rollout 首次得到 `status == "success"` 的位置：

```text
P = sum_t ||p_t - p_(t-1)||_2
```

对于 collision、timeout 或其他非 success 结果，仍可保存实际路径长度用于诊断，但该案例的 SPL 贡献固定为 0。

#### 10.9.4 单案例与总体 SPL

令 `S_i = 1` 当且仅当案例最终记录为 `status == "success"`，否则 `S_i = 0`。单案例 Semantic SPL 为：

```text
SPL_i = S_i * L_ref_i / max(P_i, L_ref_i)
```

总体及分技能结果为：

```text
overall_semantic_SPL = sum_i SPL_i / N
per_skill_SPL        = 该 skill 的 sum_i SPL_i / N_skill
macro_semantic_SPL   = 五个 per_skill_SPL 的算术平均
```

SPL 范围为 `[0,1]`，越高越好。失败案例贡献 0；成功但绕路越长，得分越低。五批分别记录 SPL，最终主结果仍由五批全部案例合并计算，不能简单平均五个批次的百分比。

#### 10.9.5 参考数据与审计字段

每个案例预计算并冻结以下字段：

```json
{
  "scene_id": "...",
  "skill": "around_left",
  "reference_endpoint_source": "expert_first_completion",
  "reference_endpoint": [0.0, 0.0, 0.0],
  "reference_path_length_m": 0.0,
  "planner_success": true,
  "completion_validated": true,
  "planner_config_hash": "...",
  "completion_config_hash": "..."
}
```

每次模型评测结果增加 `actual_path_length_m`、`reference_path_length_m` 和 `semantic_spl`。若出现成功模型路径 `P < L_ref`，公式中的 `max` 会将该案例得分限制为 1；同时单独统计 `P < L_ref` 的案例比例，因为这表明 Fast-Planner 参考路径可能不是足够短的近似解。

#### 10.9.6 参考路径生成状态（2026-08-27）

- 生成脚本：`scripts/build_spl_reference_paths.py`。
- 固定输入：`eval_manifests/manifest_index.json`；输出目录：`eval_reference_paths/`。
- 每个案例保存 `reference_path.npz`、`trajectory.json` 和 `metadata.json`；全局索引为 `eval_reference_paths/reference_index.json`，支持已有有效案例跳过、无效案例重试和逐案例原子落盘。
- Fast-Planner 对 horizon 返回的部分路径执行连续重规划，并按固定语义 route 逐段真正到达 guide 与 canonical endpoint，避免把未到达 waypoint 的部分路径错误拼接为完整路径。
- 默认使用 `drone_radius=0.15m`；若默认路径未通过语义或真实净空验证，依次尝试专家首次 semantic-done 硬引导点，以及仅用于规划搜索的 `0.18m` 保守膨胀。无论采用哪种搜索回退，最终均使用 CompletionJudge 的 mesh-aware clearance 强制验证整条路径真实最小净空不小于 `0.15m`，并再次验证 completion 后截断。
- 修复了 kinodynamic A* 在“起点已落入 near-end 粗网格范围、direct shot 失败”时直接返回 `NO_PATH` 的问题；现在会继续搜索短绕行。回归测试为 `tests/test_kinodynamic_near_end.py`。
- 脚本回归测试：`tests/test_build_spl_reference_paths.py`；独立完整性审计脚本：`audit_spl_references.py`。
- 全量生成结果：254/254 条 `valid`；技能分布为 around_left 75、around_right 75、through 50、over 44、under 10。
- 规划尝试分布：默认路径 246 条、专家硬语义引导 6 条、`0.18m` 保守规划 2 条；全部最终真实净空均不小于 `0.15m`，最小值为 `0.1520019m`。
- 参考路径三维长度范围为 `3.4757m` 至 `5.5568m`，平均 `4.4957m`。
- 审计通过：254 个 `(skill, scene_id)` 全部唯一；NPZ/JSON 点列一致；逐轨迹重算长度与 metadata 一致；配置 hash 一致；项目完整测试 `48/48` 通过。
- 冻结索引 SHA-256：`718262363d6083e8154ed2bc58f844c1a05598fae85ceda2275b7c1e7e47b2b0`。

### 10.10 待实现清单

- [x] 生成并冻结五个分层、互斥的 manifest，校验并集为全部 254 条且交集为空。
- [ ] 实现统一批次 runner、逐场景落盘和断点续跑。
- [ ] 实现统一结果汇总、Wilson/paired bootstrap CI 和 McNemar 检验。
- [x] 预计算并冻结每个 test 案例的专家首次 completion 终点和 Fast-Planner 语义参考路径长度。
- [ ] 在 rollout 结果中记录实际三维路径长度和 Semantic SPL，并验证失败案例 SPL 恒为 0。
- [ ] 统一不同方法的输入权限、超时、控制接口和成功判定。
- [ ] 完成所有方法的五批评测并生成最终对比表。

### 10.11 单批 BC/DAgger 闭环评测实现与 Batch 1 结果（2026-08-27）

- 新建实验根目录：`/root/AeroVerbs/exp/`。
- 新增单批评测脚本：`scripts/eval_rollout_batch.py`。脚本读取一份冻结 manifest，按清单顺序分别运行 BC、DAgger 或两者；复用 `scripts/rollout.py` 的模型加载、雷达闭环、碰撞检测和 CompletionJudge。
- 图片渲染被强制关闭；每个案例仍保存完整轨迹 JSON。每个模型独立保存 `run_config.json`、逐案例 `results.jsonl`、实时更新的 `summary.json` 和 `trajectories/`。
- 支持断点续跑：已有结果默认跳过；`results.jsonl` 每条写入后执行 flush/fsync；若最后一行因中断而截断，恢复前只删除该无效尾部；可用 `--retry-infrastructure-errors` 单独重试基础设施错误。
- 默认 checkpoint：BC 为 `/root/autodl-tmp/outputs/bc_baseline/20260826-074509/final.pt`；DAgger 为 `outputs/dagger_next_v2/round2/20260826-205424/final.pt`。
- 脚本测试：`tests/test_eval_rollout_batch.py`；Batch 1 审计脚本：`scripts/audit_eval_batch.py`。项目完整测试为 `54/54` 通过。
- 正式运行命令：

```bash
cd /root/AeroVerbs
/root/miniconda3/envs/aeroverbs/bin/python -u scripts/eval_rollout_batch.py \
  --manifest eval_manifests/batch_1.json \
  --output-root exp \
  --model both
```

- Batch 1 固定包含 51 条：around_left 15、around_right 15、through 10、over 9、under 2。正式输出位于 `exp/batch_1/{bc,dagger}/`。
- BC：`39/51 = 76.47%`；状态为 success 39、collision 6、timeout 6。
- DAgger：`45/51 = 88.24%`；状态为 success 45、collision 6、timeout 0。
- 分技能成功率：

| 模型 | around_left | around_right | through | over | under |
|---|---:|---:|---:|---:|---:|
| BC | 11/15 (73.33%) | 10/15 (66.67%) | 10/10 (100%) | 6/9 (66.67%) | 2/2 (100%) |
| DAgger | 13/15 (86.67%) | 14/15 (93.33%) | 10/10 (100%) | 6/9 (66.67%) | 2/2 (100%) |

- 逐场景配对：两者都成功 37 条、仅 BC 成功 2 条、仅 DAgger 成功 8 条、两者都失败 4 条。该结果仅代表 Batch 1，不能替代五批合并后的 254 条正式成功率。
- 独立审计通过：BC 和 DAgger 各 51 条结果均严格匹配 manifest 顺序且无重复；共 102 份轨迹 JSON；PNG 数量为 0；checkpoint hash、summary 状态计数和轨迹点数全部一致；无 `infrastructure_error`。
