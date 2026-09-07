# AeroVerbs 仓库整理：旧代码识别

更新时间：2026-09-07

## 判断基准

当前公开方法链路按以下范围判断：

1. MiniMind-O 从首帧 RGB 预测四边形像素坐标；
2. `scripts/build_gate.py` 使用四边形坐标与首帧 LiDAR 构造 3D gate；
3. AeroVerbs 使用 radar、3D gate、downward range、skill 与动作历史进行 rollout；
4. 保留 AeroVerbs、OpenVLA、SPF、AerialClaw 的 rollout 与统一评测；
5. 保留 AeroVerbs 训练、DAgger/Fast-Planner teacher、MiniMind-O 蒸馏及正式实验复现代码。

“旧代码”分为可以直接移除、需要联动清理、超出当前公开范围三类。第一批可直接移除项已于 2026-09-07 删除。

## 已删除：原“可以直接移除”项

### 已失效的 heatmap 实验入口

- `scripts/rollout_heatmap_shift.py`
- `scripts/rollout_ablation.py`

二者会访问或修改 `scripts/rollout.py` 中原有的 `load_heatmap_npy`，但当前 rollout 已不再定义该符号。因此 heatmap shift/flip 路径不仅不属于当前方法，而且已经不可运行。

- `scripts/run_ablation.sh`
- `scripts/run_rollout_10pct.sh`

这两个 shell 脚本固定引用 `/root/AeroVerbs`、旧的 `outputs/dagger_v2*` 权重和旧服务器数据路径，是历史实验命令，不是当前可复现入口。

### 被新版替代的评测入口

- `scripts/eval_rollout_batch_legacy.py`

其功能已被 `scripts/eval_rollout_batch.py` 覆盖。新版增加 OpenVLA、AerialClaw、断点恢复校验和更完整的 run config，旧副本没有独立保留价值。

### 明确排除的 sample-20/Qwen 流程

- `scripts/qwen_annotate_sample20.py`
- `scripts/visualize_qwen_sample20.py`

这两项属于 20 样本抽取、旧 Qwen bbox-to-heatmap 与可视化流程；对应结果也已由 `.gitignore` 排除。

### 生成缓存和编辑器残留

- 所有 `__pycache__/` 与 `*.pyc`
- `aeroverbs/data/.ipynb_checkpoints/`

这些不是源代码，已从工作区清理，并由 `.gitignore` 持续忽略。

## 已完成：Heatmap 联动清理

### Heatmap 数据与训练兼容层

以下旧 heatmap 输入链路已完成联动清理：

- `aeroverbs/data/heatmap.py`
- `aeroverbs/data/transforms.py` 中的 `resize_heatmap`、`load_heatmap_npy`
- `aeroverbs/data/dataset.py` 中的 `_load_heatmap`、`heatmap_size`、batch 的 `heatmap` 字段
- `aeroverbs/training/trainer.py` 中读取并传递 `batch["heatmap"]` 的逻辑
- `scripts/train_dagger.py` 中传递 heatmap 的逻辑
- `scripts/dagger_collect.py` 中加载、复制 `intention_heatmap.npy` 的逻辑
- `aeroverbs/data/__init__.py` 中 heatmap API 的导出
- `configs/default.yaml` 与 `aeroverbs/config.py` 中的 `heatmap_size`

训练、评测与 DAgger 现统一调用 `model.initialize(None, skill_id)`；数据集和 collate 已不再产生 heatmap 字段，DAgger 也不再读取或复制 `intention_heatmap.npy`。`aeroverbs/data/heatmap.py` 及 heatmap transform/API 已删除。3D gate 文件目前仍使用 `heatmap_gate_*` 历史文件名；这是 schema 命名兼容，不代表仍使用 heatmap，可在后续单独迁移。

### 窗口式旧训练/评测路径

正式训练已经使用完整轨迹：`TrajectoryDataset`、`make_trajectory_dataloaders`、`_evaluate_trajectory`。以下只剩兼容或测试用途：

- `aeroverbs/data/dataset.py` 中的 `ManeuverDataset` 与 `collate_fn`
- `aeroverbs/training/trainer.py` 中的 `make_dataloaders`、窗口 batch 训练/评测函数
- `scripts/eval.py`：当前仍调用旧的窗口式 `make_dataloaders + _evaluate`
- `scripts/smoke_test.py`

如果不保留窗口训练兼容性，可以删除上述实现；若还需要公开离线评测入口，应先将 `scripts/eval.py` 改为轨迹级评测，然后删除旧实现。

### 模型中的 IntentEncoder

- `aeroverbs/model/encoders.py` 中的 `IntentEncoder`
- `aeroverbs/model/policy.py` 中的 `use_heatmap_latent` 分支与 `intent_encoder`

当前正式配置 `use_heatmap_latent: false`，推理不会计算 heatmap latent。但 `final.pt` 的 state dict 仍包含这部分参数；直接删除模块会导致严格加载权重失败。它属于“结构遗留”，应暂时保留，或者先生成一个移除相应 key 的新版 checkpoint，并验证输出完全一致后再删。

## 超出当前公开方法范围，建议移除

### WorldVLN

- `aeroverbs/baselines/worldvln.py`
- `scripts/rollout_worldvln.py`
- `scripts/eval_worldvln_batch.py`

用户确认的对比方法是 OpenVLA、SPF、AerialClaw；统一批评测也把 WorldVLN 标记为 `UNIMPLEMENTED_MODELS`。除非后续决定正式发布 WorldVLN 实验，否则这组代码应移除，同时从 `scripts/eval_rollout_batch.py` 的 model choices 中删除占位项。

### 一次性测试、诊断与程序化 demo

以下不是当前正式训练或评测链路依赖，可按“tests 不公开”的原则移除：

- `scripts/create_procedural_test_scene.py`
- `scripts/plot_combined_procedural_rollouts.py`
- `scripts/render_rollout_radar_frames.py`
- `scripts/probe_spf_rendered_frames.py`
- `scripts/run_aerialclaw_smoke.sh`

其中 `probe_spf_rendered_frames.py` 和 `run_aerialclaw_smoke.sh` 对调试服务仍有帮助，但不是 rollout 的必要实现；若希望仓库保留最小诊断工具，可以保留后二者。

## 名称旧，但目前必须保留

以下代码容易被误判为旧代码，当前仍有真实依赖：

- `aeroverbs/data/teacher.py`：Fast-Planner teacher 规划失败时的 fallback；不能直接删除。
- `aeroverbs/data/semantic_judge.py`：`completion.py` 使用 `_winding_class`，旧 teacher 也使用 `SemanticCompletionJudge`。
- `aeroverbs/data/completion.py`：训练标签、rollout 完成判定和评测共用。
- `scripts/build_sdf.py`、`scripts/build_through_occupancy.py`：Fast-Planner/场景预处理依赖。
- `scripts/compute_downward_range.py`：当前 rollout 与 DAgger 都直接导入。
- `scripts/view_voxel_trajectory.py`：当前 rollout 渲染与 rear-region 指标直接导入。
- `uav_maneuver_planning/fast_planner/`：当前 DAgger teacher 与 SPL reference path 生成依赖。
- `aeroverbs/baselines/aerialclaw_bridge.py` 与 `scripts/aerialclaw_sim_server.py`：本仓库实现的 AerialClaw 推理桥接服务，应保留。

## 建议清理顺序

1. 删除缓存、sample-20、失效 heatmap rollout、legacy batch eval 和旧 shell 命令。
2. 移除 WorldVLN 及其统一评测占位项。
3. ~~把 AeroVerbs 训练、评测、DAgger 全部改为无 heatmap 数据字段，再删除 heatmap 模块。~~ 已完成。
4. 将 `scripts/eval.py` 切到轨迹级 API，删除窗口式 dataset/trainer 兼容层与 smoke tests。
5. 最后决定是否迁移 `final.pt`；完成 checkpoint 迁移后才删除 `IntentEncoder`。
6. 更新 `README.md` 和 `PROJECT_STATUS.md` 中仍将 heatmap 描述为主输入的历史说明。
