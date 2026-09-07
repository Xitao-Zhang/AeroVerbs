# 模型微调配置记录

本文档记录 AeroVerbs 数据集上的 OpenVLA 与 WorldVLN 微调配置。记录日期为 2026-09-01。

> 参数来源分为训练日志、启动脚本、模型配置和输出目录命名。未被训练产物完整记录的内容会单独注明，避免将推断值误认为完整配置快照。

## OpenVLA

### 训练任务

| 项目 | 配置 |
| --- | --- |
| 基础模型 | `/home/abc/UAV-Flow/OpenVLA-UAV/model` |
| 训练代码 | `/home/abc/UAV-Flow/OpenVLA-UAV/vla-scripts/finetune_uav.py` |
| 数据集实现 | `SimpleVLADataset` |
| 训练样本数 | 165,374 |
| Batch size | 16 |
| 梯度累积步数 | 1 |
| 有效 Batch size | 16 |
| 学习率 | `5e-4` |
| 最大训练步数 | 10,000 |
| 图像增强 | 关闭 |
| 量化训练 | 关闭 |
| 训练精度 | BF16 |
| 优化方式 | LoRA |
| LoRA rank | 32 |
| LoRA alpha | 16 |
| LoRA dropout | 0.0 |
| LoRA bias | `none` |
| DoRA | 关闭 |
| RS-LoRA | 关闭 |
| 可训练参数 | 110,828,288 |
| 模型总参数 | 7,652,065,472 |
| 可训练参数比例 | 约 1.4483% |

LoRA 的目标模块为：

```text
gate_proj, o_proj, proj, v_proj, fc2, fc1, qkv, q_proj,
kv, down_proj, q, fc3, up_proj, lm_head, k_proj
```

### 训练终止条件说明

本次输出目录名称中包含 `aeroverbs-b16-1epoch`，但训练日志明确显示训练在达到 `max_steps=10000` 时终止。由于 165,374 个样本在 batch size 16 下约为 10,336 个 iteration，因此本次训练接近一个 epoch，但不是严格完整的一个 epoch。

后续报告中建议表述为：

> OpenVLA 使用 batch size 16 进行 LoRA 微调，共训练 10,000 steps，约 0.97 epoch。

### 训练产物与参数来源

| 内容 | 服务器路径 |
| --- | --- |
| 训练日志 | `/home/abc/openvla_aeroverbs_train_b16/logs/train.log` |
| LoRA adapter | `/home/abc/openvla_aeroverbs_train_b16/adapters/model+b16+lr-0.0005+lora-r32+dropout-0.0--aeroverbs-b16-1epoch/` |
| LoRA 配置 | `/home/abc/openvla_aeroverbs_train_b16/adapters/model+b16+lr-0.0005+lora-r32+dropout-0.0--aeroverbs-b16-1epoch/adapter_config.json` |
| 合并后权重 | `/home/abc/openvla_aeroverbs_train_b16/weights/model+b16+lr-0.0005+lora-r32+dropout-0.0--aeroverbs-b16-1epoch/` |
| 合并模型配置 | `/home/abc/openvla_aeroverbs_train_b16/weights/model+b16+lr-0.0005+lora-r32+dropout-0.0--aeroverbs-b16-1epoch/config.json` |
| 冒烟测试目录 | `/home/abc/openvla_aeroverbs_smoke/` |

`adapter_config.json` 保存了 LoRA 结构和基础模型路径；合并模型的 `config.json` 保存了模型结构、BF16 类型和 `sim` 数据集的动作归一化统计。

### OpenVLA 配置完整性

OpenVLA 没有单独保存一份包含完整命令行参数的配置快照。目前可以从训练日志、输出目录名称、`adapter_config.json`、合并模型的 `config.json` 以及 `finetune_uav.py` 默认值恢复上述核心参数。

因此：

- 模型权重和核心训练参数已经保存；
- 完整的原始启动命令未在训练目录中单独落盘；
- 未显式出现在日志或模型配置中的参数，需要结合当时版本的 `finetune_uav.py` 默认值恢复。

## WorldVLN

### 当前 LoRA 微调任务

| 项目 | 配置 |
| --- | --- |
| 正式实验名称 | `aeroverbs_worldvln_lora_r32_e1` |
| 基础权重 | 原始 WorldVLN 权重，不加载旧部分全参 checkpoint |
| 训练范围 | 36层注意力和MLP的LoRA，以及两个输出头 |
| 动作解码器 | 不微调 |
| Epoch 数 | 1 |
| 每个 epoch 的训练 iterations | 11,072（训练日志实际值） |
| 单卡视频 Batch size | 1 |
| 梯度累积步数 `ac` | 1 |
| 学习率 `tlr` | `5e-5` |
| LoRA rank / alpha / dropout | 32 / 32 / 0.05 |
| LoRA目标模块 | `q/k/v/o_proj`、`gate/up/down_proj` |
| 完整训练模块 | `head`、`semantic_head2` |
| 可训练参数 | 102,105,248 |
| 可训练参数比例 | 约1.15%（注入后总参数8,870,487,360） |
| 视频采样频率 | 16 FPS |
| 每个样本视频帧数 | 49 |
| DataLoader workers | 16 |
| 最大 token 长度 | 20,480 |
| Flex Attention | 开启 |
| Activation checkpointing | `full-block` |
| 并行策略 | 单卡普通DDP，`zero=0` |
| Checkpoint 保存间隔 | 500 iterations |

此前解冻最后两个chunks的部分全参方案有2.9200B个可训练参数，占主干33.3%。该训练在1368/11072时因SSH断开收到`SIGHUP`，现已废弃；其step 500和step 1000 checkpoint目录已删除，训练日志保留。

本次单 epoch 训练的进度格式为 `[当前 iteration/11072]`，因此完整训练量是 11,072 iterations，而不是取整后的 11,000 steps。

### 训练产物与参数来源

| 内容 | 服务器路径 |
| --- | --- |
| LoRA单卡启动脚本 | `/home/abc/worldvln_aeroverbs_runtime/scripts/train_backbone_lora_single_gpu.sh` |
| LoRA训练入口 | `/home/abc/worldvln_aeroverbs_runtime/scripts/train_backbone_lora_entry.py` |
| LoRA运行时 | `/home/abc/worldvln_aeroverbs_runtime/scripts/worldvln_lora_runtime.py` |
| Adapter checkpoint | `/home/abc/worldvln_aeroverbs_runtime/scripts/worldvln_lora_checkpoint.py` |
| 权重合并工具 | `/home/abc/worldvln_aeroverbs_runtime/scripts/merge_worldvln_lora.py` |
| 启动与监控命令 | `/home/abc/worldvln_aeroverbs_runtime/outputs/FORMAL_LORA_TRAINING_COMMAND.md` |
| Python 兼容层 | `/home/abc/worldvln_aeroverbs_runtime/compat/sitecustomize.py` |
| LoRA环境检查 | `/home/abc/worldvln_aeroverbs_runtime/scripts/check_worldvln_lora.py` |
| 冒烟日志 | `/home/abc/worldvln_aeroverbs_runtime/outputs/aeroverbs_worldvln_lora_smoke.log` |
| 冒烟最终adapter | `/home/abc/worldvln_aeroverbs_runtime/outputs/adapters/aeroverbs_worldvln_lora_smoke/final/` |
| 冒烟合并权重 | `/home/abc/worldvln_aeroverbs_runtime/outputs/merged/aeroverbs_worldvln_lora_smoke/` |
| 主干预训练权重 | `/home/abc/WorldVLN.code/mode_weights` |

### 冒烟验证结果

- 实际8.77B模型成功加载原始分片后注入LoRA；
- 匹配36个Transformer blocks和252个目标Linear模块；
- 16个短iteration完整运行，loss从0.415下降至0.242；
- 8项轻量单元测试通过；
- 最终adapter约390MB；
- 合并权重生成8个safetensors分片，共约33GB；
- 未微调的`text_proj.weight`与原始权重逐元素完全相同；
- LoRA目标`q_proj.weight`和输出`head.weight`与原始权重存在预期差异；
- 正式训练尚未启动，必须由用户手动执行后台启动命令。

### WorldVLN 配置完整性

WorldVLN LoRA的配置保存相对完整：

- 启动脚本保存了显式设置的训练参数；
- 日志开头保存了解析后的完整 `Args` 和实际命令行；
- 日志可用于确认运行时最终采用的参数，而不只依赖脚本当前内容；
- checkpoint 生成后，应同时保留 checkpoint、对应日志和启动脚本，避免脚本日后修改造成配置歧义。

## 复现实验时的建议

1. 以训练日志中的最终解析参数为准，启动脚本作为复现入口。
2. 每次正式实验使用唯一实验名称，不覆盖已有日志、adapter 或 checkpoint。
3. 同时归档启动脚本、训练日志、环境依赖版本和模型权重路径。
4. OpenVLA 后续训练应额外保存完整启动命令或独立的 `training_config.json`，以免只能依赖源码默认值恢复参数。
5. 报告训练时长时同时记录 steps 和 epoch，避免目录名称中的 `1epoch` 与实际 `10000 steps` 混淆。
