# WorldVLN LoRA 微调设计

日期：2026-09-01

## 目标

在 AeroVerbs 视频数据上对原始 WorldVLN 视频世界模型进行 LoRA 微调，同时满足以下约束：

- 从原始 WorldVLN 权重开始，不加载先前未完成的部分全参数微调 checkpoint；
- 不微调动作解码器；
- 不修改 `/home/abc/WorldVLN.code/` 中的原始文件；
- 保存独立 LoRA adapter，并在训练完成后生成合并权重；
- 正式训练由用户手动启动，实施过程不得自动启动正式训练；
- 后台训练必须脱离 SSH controlling terminal，避免再次因 `SIGHUP` 终止。

## 当前状态与动机

WorldVLN 视频世界模型主干共有约 8.7684B 参数。先前方案冻结前四个 chunks，完整训练最后两个 chunks 和输出层，可训练参数约 2.9200B，占 33.3%。AeroVerbs 当前一个 epoch 包含约 11,073 个样本和 11,072 个 iterations。相对数据规模而言，2.92B 个可训练参数成本较高，并增加过拟合和预训练能力遗忘的风险。

先前单 epoch 训练在 1368/11072 时因 SSH 会话断开收到 `SIGHUP`，并生成了 step 500 和 step 1000 的部分全参 checkpoint。该训练不再使用。

## 选定方案

采用全 36 层 PEFT LoRA，并完整训练两个输出层。

### 基础配置

| 参数 | 配置 |
| --- | --- |
| 基础权重 | `/home/abc/WorldVLN.code/mode_weights/worldvln_runtime/gpt` |
| LoRA rank | 32 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| LoRA 覆盖范围 | 全部 36 个 Transformer blocks |
| 学习率 | `5e-5` |
| Batch size | 1 |
| Epoch | 1 |
| 每个 epoch iterations | 11,072 |
| Activation checkpointing | `full-block` |
| 视频采样率 | 16 FPS |
| 视频帧数 | 49 |

### LoRA 目标模块

注意力层：

```text
q_proj, k_proj, v_proj, o_proj
```

MLP 层：

```text
gate_proj, up_proj, down_proj
```

以下输出层使用完整参数训练，并随 adapter 一起保存：

```text
head, semantic_head2
```

其余原始 WorldVLN 参数全部冻结，包括归一化层、文本投影层和视频 token 输入投影层。

按当前模型维度估算，rank 32 的全 36 层 LoRA 参数约为 101,449,728；输出层约为 655,520 个参数，最终可训练参数约占 8.7684B 主干的 1.17%。启动日志必须打印并核验实际值。

## 架构

### 外部 LoRA 兼容层

所有新增文件位于：

```text
/home/abc/worldvln_aeroverbs_runtime/
```

外部训练入口在原始 `build_vae_gpt` 完成模型构建和预训练权重加载之后、DDP 包装和 optimizer 创建之前注入 LoRA。这样可以复用原始数据管线、损失函数和训练循环，同时不修改 WorldVLN 仓库。

LoRA 注入后执行以下检查：

1. 目标模块匹配数量符合 36 层结构；
2. 所有基础权重均被冻结；
3. 只有 LoRA 参数、`head` 和 `semantic_head2` 可训练；
4. 输出实际总参数、可训练参数和比例；
5. 若目标模块数量或参数状态不符合预期，立即终止，不进入训练。

### 单卡并行策略

LoRA 正式训练使用单卡普通 DDP，不使用 ZeRO-3。完整主干只需要承载冻结后的 BF16 权重，optimizer 只为约 102M 个可训练参数分配状态。该方案可以简化 adapter 保存、恢复和权重合并，避免 FSDP 对 adapter state dict 的额外处理。

训练仍通过单进程 `torchrun` 启动，以兼容 WorldVLN 现有分布式初始化逻辑。

## Checkpoint 与恢复

### 中间 checkpoint

每 500 iterations 保存一次，内容包括：

- `adapter_model.safetensors`；
- `adapter_config.json`；
- `head` 和 `semantic_head2` 参数；
- optimizer state；
- 当前 epoch、iteration 和 global step；
- 训练参数快照；
- 随机数状态。

保存过程先写入临时目录，完成后再原子重命名，避免中断时留下被误认为有效的 checkpoint。

不在每个中间 checkpoint 中复制或保存完整 8.7684B 基础模型。

### 断点恢复

恢复顺序固定为：

1. 从原始 WorldVLN 权重构建基础模型；
2. 使用保存的配置注入同结构 LoRA；
3. 加载 adapter 和输出层；
4. 恢复 optimizer、step 和随机数状态；
5. 从下一 iteration 继续训练。

adapter 配置不匹配时禁止宽松加载。

### 最终产物

训练结束后保存：

1. 最终独立 LoRA adapter；
2. 完整训练参数清单；
3. 合并 LoRA 后的 WorldVLN 主干权重；
4. 合并报告，记录缺失键、意外键、权重 dtype 和 shard 索引。

合并过程从原始权重重新加载基础模型，在 CPU 或受控显存环境中加载 adapter 并合并，输出到新目录。不得覆盖原始权重。

## 输出目录

实验使用独立名称，例如：

```text
aeroverbs_worldvln_lora_r32_e1
```

建议目录布局：

```text
/home/abc/worldvln_aeroverbs_runtime/outputs/
├── adapters/aeroverbs_worldvln_lora_r32_e1/
│   ├── checkpoint-000500/
│   ├── checkpoint-001000/
│   └── final/
├── merged/aeroverbs_worldvln_lora_r32_e1/
├── run_logs/aeroverbs_worldvln_lora_r32_e1/
├── aeroverbs_worldvln_lora_r32_e1.log
└── aeroverbs_worldvln_lora_r32_e1.pid
```

## 冒烟验证

正式训练前必须完成以下验证：

1. 在 `worldvln-blackwell` 环境中安装并导入兼容版本的 PEFT；
2. 检查 LoRA 目标模块数量和可训练参数量；
3. 使用受限数据运行 2 至 3 个训练 iterations；
4. 确认 loss 有限且反向传播成功；
5. 保存冒烟 adapter；
6. 从原始权重重新构建模型并加载 adapter；
7. 对同一固定输入比较保存前后输出，确认加载一致；
8. 生成一次合并权重并执行模型加载或最小前向验证。

任何一步失败都不得给出正式训练启动指令。

## 后台启动与进度查看

正式训练由用户手动启动。实施过程只提供命令，不执行该命令。

后台命令采用：

```bash
setsid nohup <训练启动脚本> > <日志文件> 2>&1 < /dev/null &
```

不能只使用普通 `nohup ... &`，因为 `torch.distributed.elastic` 会重新注册 `SIGHUP` 处理器。`setsid` 用于创建新会话并解除 SSH controlling terminal。

启动后必须提供：

- 查看 PID 和进程状态的命令；
- `tail -f` 查看训练日志的命令；
- 统计最新 iteration 和 checkpoint 的命令；
- 安全停止训练的命令。

## 旧实验清理

删除以下已明确废弃的部分全参 checkpoint 目录：

```text
/home/abc/worldvln_aeroverbs_runtime/outputs/checkpoints/aeroverbs_worldvln_backbone_partial_c4_e1/
```

删除前必须验证解析后的绝对路径与上述路径完全一致。保留以下内容：

- 部分全参训练日志；
- PID 文件和参数记录；
- 原始 WorldVLN 权重；
- WorldVLN 原始代码。

该目录包含 step 500 和 step 1000 checkpoint，删除后不可从这些 checkpoint 恢复先前训练。

## 非目标

- 不微调动作解码器；
- 不自动启动正式训练；
- 不修改 WorldVLN 原始仓库；
- 不加载或迁移旧部分全参 checkpoint；
- 不覆盖原始预训练权重；
- 不将 gate 或 rollout 成功判定逻辑纳入本次改动。
