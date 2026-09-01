# WorldVLN LoRA Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a non-invasive PEFT LoRA fine-tuning path for the original WorldVLN video-world-model weights, verify it with an actual-model smoke run, and provide—but do not execute—the detached formal-training command.

**Architecture:** External runtime files under `/home/abc/worldvln_aeroverbs_runtime` monkeypatch WorldVLN after pretrained-weight loading and before DDP/optimizer construction. A custom adapter-only saver replaces the original full-model checkpoint path; a separate merge utility rebuilds the original model, loads the adapter, merges it, and writes sharded weights without touching `/home/abc/WorldVLN.code`.

**Tech Stack:** Python 3.10, PyTorch 2.7.1+cu128, Transformers 4.57.6, PEFT 0.18.0, Accelerate 1.14.0, safetensors 0.8.0, single-process torchrun/DDP, Bash, pytest.

**Spec:** `docs/superpowers/specs/2026-09-01-worldvln-lora-finetuning-design.md`

## Global Constraints

- Start from `/home/abc/WorldVLN.code/mode_weights/worldvln_runtime/gpt`; never load the abandoned partial-full-finetuning checkpoints.
- Do not modify any file under `/home/abc/WorldVLN.code/`.
- Do not fine-tune the action decoder.
- Inject rank-32 LoRA into all 36 blocks for `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.
- Use `lora_alpha=32`, `lora_dropout=0.05`, learning rate `5e-5`, batch size 1, and one epoch of 11,072 iterations.
- Train `head` and `semantic_head2` fully; freeze every other base parameter.
- Save adapters every 500 iterations and save merged weights only after successful completion.
- Preserve original weights and previous logs.
- Delete only `/home/abc/worldvln_aeroverbs_runtime/outputs/checkpoints/aeroverbs_worldvln_backbone_partial_c4_e1/` after exact-path validation.
- Never launch the formal one-epoch training run during implementation. Only print and document its command.
- Do not create subagents; execute this plan inline.

---

## File Map

Files created on the server:

- `/home/abc/worldvln_aeroverbs_runtime/scripts/worldvln_lora_runtime.py`: LoRA configuration, injection, validation, adapter state extraction, atomic adapter save, and adapter load.
- `/home/abc/worldvln_aeroverbs_runtime/scripts/worldvln_lora_checkpoint.py`: replacement `CKPTSaver.sav` and `auto_resume` functions that serialize only trainable model state plus optimizer/training metadata.
- `/home/abc/worldvln_aeroverbs_runtime/scripts/train_backbone_lora_entry.py`: external monkeypatch entrypoint; injects LoRA and replaces checkpoint behavior before running original `train.py`.
- `/home/abc/worldvln_aeroverbs_runtime/scripts/train_backbone_lora_single_gpu.sh`: parameterized single-GPU torchrun launcher with dry-run and smoke controls.
- `/home/abc/worldvln_aeroverbs_runtime/scripts/merge_worldvln_lora.py`: loads original weights, loads adapter, merges LoRA, and writes a new sharded model directory.
- `/home/abc/worldvln_aeroverbs_runtime/scripts/check_worldvln_lora.py`: dependency, trainable-parameter, adapter-layout, and output-layout checks.
- `/home/abc/worldvln_aeroverbs_runtime/tests/test_worldvln_lora_runtime.py`: toy WorldVLN-shaped model tests for injection and trainable-state validation.
- `/home/abc/worldvln_aeroverbs_runtime/tests/test_worldvln_lora_checkpoint.py`: adapter checkpoint round-trip and optimizer-resume tests.
- `/home/abc/worldvln_aeroverbs_runtime/tests/test_worldvln_lora_merge.py`: adapter merge equivalence test using a toy model.
- `/home/abc/worldvln_aeroverbs_runtime/requirements-lora.txt`: exact LoRA-specific dependency versions.
- `/home/abc/worldvln_aeroverbs_runtime/outputs/FORMAL_LORA_TRAINING_COMMAND.md`: final user-facing launch, progress, checkpoint, and stop commands.

Existing external file modified on the server:

- `/home/abc/worldvln_aeroverbs_runtime/scripts/check_runtime.sh`: add PEFT/Accelerate checks without changing WorldVLN source.

No local source file outside the design and plan documents is modified.

---

### Task 1: Remove the abandoned partial-full-finetuning checkpoints

**Files:**
- Delete: `/home/abc/worldvln_aeroverbs_runtime/outputs/checkpoints/aeroverbs_worldvln_backbone_partial_c4_e1/`
- Preserve: `/home/abc/worldvln_aeroverbs_runtime/outputs/aeroverbs_worldvln_backbone_partial_c4_e1.log`
- Preserve: `/home/abc/worldvln_aeroverbs_runtime/outputs/aeroverbs_worldvln_backbone_partial_c4_e1.pid`

**Interfaces:**
- Consumes: explicit user approval to delete exactly one checkpoint directory.
- Produces: reclaimed disk space while retaining the failed-run audit log.

- [ ] **Step 1: Resolve and inspect the exact deletion target**

Run:

```bash
realpath /home/abc/worldvln_aeroverbs_runtime/outputs/checkpoints/aeroverbs_worldvln_backbone_partial_c4_e1
du -sh /home/abc/worldvln_aeroverbs_runtime/outputs/checkpoints/aeroverbs_worldvln_backbone_partial_c4_e1
find /home/abc/worldvln_aeroverbs_runtime/outputs/checkpoints/aeroverbs_worldvln_backbone_partial_c4_e1 -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
```

Expected: `realpath` equals the approved absolute path exactly; files include `global_step_500.pth` and `global_step_1000.pth`.

- [ ] **Step 2: Delete only the approved directory**

Run with the literal absolute path, without a glob or environment variable:

```bash
rm -r -- /home/abc/worldvln_aeroverbs_runtime/outputs/checkpoints/aeroverbs_worldvln_backbone_partial_c4_e1
```

- [ ] **Step 3: Verify deletion and preservation**

Run:

```bash
test ! -e /home/abc/worldvln_aeroverbs_runtime/outputs/checkpoints/aeroverbs_worldvln_backbone_partial_c4_e1
test -s /home/abc/worldvln_aeroverbs_runtime/outputs/aeroverbs_worldvln_backbone_partial_c4_e1.log
test -s /home/abc/worldvln_aeroverbs_runtime/outputs/aeroverbs_worldvln_backbone_partial_c4_e1.pid
```

Expected: exit code 0 for all three checks.

---

### Task 2: Install and lock PEFT dependencies

**Files:**
- Create: `/home/abc/worldvln_aeroverbs_runtime/requirements-lora.txt`
- Modify: `/home/abc/worldvln_aeroverbs_runtime/scripts/check_runtime.sh`

**Interfaces:**
- Consumes: `/home/abc/miniconda3/envs/worldvln-blackwell/bin/python`, Torch 2.7.1, Transformers 4.57.6.
- Produces: importable `peft==0.18.0` and `accelerate==1.14.0` without replacing Torch, Transformers, or safetensors.

- [ ] **Step 1: Add a failing runtime dependency check**

Append a check that runs:

```bash
"${PYTHON_BIN}" - <<'PY'
import accelerate
import peft
assert peft.__version__ == "0.18.0", peft.__version__
assert accelerate.__version__ == "1.14.0", accelerate.__version__
print(f"peft={peft.__version__} accelerate={accelerate.__version__}")
PY
```

- [ ] **Step 2: Run the check and verify it fails before installation**

Run:

```bash
bash /home/abc/worldvln_aeroverbs_runtime/scripts/check_runtime.sh
```

Expected: FAIL with `ModuleNotFoundError: No module named 'peft'` or `accelerate`.

- [ ] **Step 3: Create the dependency lock file**

Content:

```text
peft==0.18.0
accelerate==1.14.0
```

- [ ] **Step 4: Install only the missing LoRA packages**

Run:

```bash
/home/abc/miniconda3/envs/worldvln-blackwell/bin/pip install --no-deps -r /home/abc/worldvln_aeroverbs_runtime/requirements-lora.txt
```

- [ ] **Step 5: Verify core versions did not change**

Run:

```bash
/home/abc/miniconda3/envs/worldvln-blackwell/bin/python - <<'PY'
import accelerate, peft, safetensors, torch, transformers
print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('peft', peft.__version__)
print('accelerate', accelerate.__version__)
print('safetensors', safetensors.__version__)
assert torch.__version__ == '2.7.1+cu128'
assert transformers.__version__ == '4.57.6'
assert peft.__version__ == '0.18.0'
assert accelerate.__version__ == '1.14.0'
assert safetensors.__version__ == '0.8.0'
PY
```

Expected: all assertions pass.

---

### Task 3: Implement LoRA injection and trainable-parameter validation

**Files:**
- Create: `/home/abc/worldvln_aeroverbs_runtime/scripts/worldvln_lora_runtime.py`
- Create: `/home/abc/worldvln_aeroverbs_runtime/tests/test_worldvln_lora_runtime.py`

**Interfaces:**
- Consumes: `torch.nn.Module` with WorldVLN-style `block_chunks.N.module.M.attn.*_proj` and `.mlp.*_proj` names.
- Produces: `WorldVLNLoRAConfig`, `inject_worldvln_lora(model, config)`, `summarize_lora_trainables(model, config)`, `assert_lora_layout(model, config)`, `trainable_state_dict(model)`, `atomic_save_adapter(model, output_dir)`.

- [ ] **Step 1: Write failing toy-model injection tests**

The tests must define a two-block toy hierarchy with these exact leaf names:

```python
class ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 4, bias=False)
        self.v_proj = nn.Linear(8, 4, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)

class ToyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(8, 16, bias=False)
        self.up_proj = nn.Linear(8, 16, bias=False)
        self.down_proj = nn.Linear(16, 8, bias=False)
```

Required assertions:

```python
assert summary.matched_linear_modules == 14
assert summary.trainable_base_names == {"head", "semantic_head2"}
assert summary.unexpected_trainable_names == []
assert all("lora_" in name or "modules_to_save" in name
           for name, p in model.named_parameters() if p.requires_grad)
```

- [ ] **Step 2: Run tests and verify missing implementation failure**

Run:

```bash
cd /home/abc/worldvln_aeroverbs_runtime
/home/abc/miniconda3/envs/worldvln-blackwell/bin/python -m pytest tests/test_worldvln_lora_runtime.py -q
```

Expected: FAIL because `worldvln_lora_runtime` does not exist.

- [ ] **Step 3: Implement configuration and injection**

Use this public configuration shape:

```python
@dataclass(frozen=True)
class WorldVLNLoRAConfig:
    rank: int = 32
    alpha: int = 32
    dropout: float = 0.05
    expected_blocks: int = 36
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    modules_to_save: tuple[str, ...] = ("head", "semantic_head2")
```

Injection must call PEFT with:

```python
lora_config = LoraConfig(
    r=config.rank,
    lora_alpha=config.alpha,
    lora_dropout=config.dropout,
    bias="none",
    target_modules=list(config.target_modules),
    modules_to_save=list(config.modules_to_save),
)
model = get_peft_model(model, lora_config)
```

Before injection, derive block indices from full module names matching:

```text
block_chunks.<chunk>.module.<block>.<attn|mlp>.<target>
```

Map `(chunk, block)` to a global block index and require exactly 36 distinct blocks on the actual model. After injection, reject any trainable parameter that contains neither `lora_` nor `modules_to_save`.

- [ ] **Step 4: Implement atomic adapter saving**

Write to `<output>.tmp-<pid>`, call `model.save_pretrained(..., safe_serialization=True)`, verify `adapter_config.json` and `adapter_model.safetensors`, then rename the temporary directory to the final path. Refuse to replace an existing final directory.

- [ ] **Step 5: Run runtime tests**

Run:

```bash
cd /home/abc/worldvln_aeroverbs_runtime
/home/abc/miniconda3/envs/worldvln-blackwell/bin/python -m pytest tests/test_worldvln_lora_runtime.py -q
```

Expected: PASS.

---

### Task 4: Implement adapter-only checkpoint and auto-resume

**Files:**
- Create: `/home/abc/worldvln_aeroverbs_runtime/scripts/worldvln_lora_checkpoint.py`
- Create: `/home/abc/worldvln_aeroverbs_runtime/tests/test_worldvln_lora_checkpoint.py`

**Interfaces:**
- Consumes: `trainable_state_dict(model)`, `atomic_save_adapter(model, path)`, WorldVLN `trainer.gpt_wo_ddp`, `trainer.gpt_opt`, and original `CKPTSaver.sav` call arguments.
- Produces: `LoraCheckpointSaver`, `lora_auto_resume(args, pattern)`, and checkpoint directories named `checkpoint-000500`.

- [ ] **Step 1: Write failing checkpoint round-trip tests**

The test must:

1. inject LoRA into a toy model;
2. run one AdamW update;
3. call `LoraCheckpointSaver.sav(..., g_it=1, next_ep=0, next_it=1, trainer=toy_trainer)`;
4. assert adapter files and `training_state.pt` exist;
5. rebuild from the same seeded base model;
6. restore model and optimizer state;
7. assert every trainable tensor and optimizer tensor matches exactly.

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
cd /home/abc/worldvln_aeroverbs_runtime
/home/abc/miniconda3/envs/worldvln-blackwell/bin/python -m pytest tests/test_worldvln_lora_checkpoint.py -q
```

Expected: FAIL because `worldvln_lora_checkpoint` does not exist.

- [ ] **Step 3: Implement the saver**

`training_state.pt` must contain only:

```python
{
    "args": args.state_dict(),
    "epoch": next_ep,
    "iter": next_it,
    "g_it": g_it,
    "trainable_model": trainable_state_dict(raw_model),
    "optimizer": trainer.gpt_opt.state_dict(),
    "trainer_config": trainer.get_config(),
    "torch_rng_state": torch.get_rng_state(),
    "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    "numpy_rng_state": numpy.random.get_state(),
    "python_rng_state": random.getstate(),
}
```

The saver must not call `trainer.state_dict()` because that includes the full GPT and VAE.

- [ ] **Step 4: Implement auto-resume compatibility**

`lora_auto_resume(args, pattern)` must select the numerically greatest `checkpoint-NNNNNN`, load `training_state.pt`, and return the tuple expected by original `train.py`:

```python
(info, epoch, g_it, acc_str, milestones, trainer_state, args_state)
```

`trainer_state` must contain `gpt_wo_ddp`, `gpt_opt`, and `config` keys compatible with `InfinityTrainer.load_state_dict(..., strict=False, skip_vae=True)`. The trainable model state must retain full PEFT parameter names so `load_state_dict(strict=False)` restores LoRA and `modules_to_save` tensors.

- [ ] **Step 5: Run checkpoint tests**

Run:

```bash
cd /home/abc/worldvln_aeroverbs_runtime
/home/abc/miniconda3/envs/worldvln-blackwell/bin/python -m pytest tests/test_worldvln_lora_checkpoint.py -q
```

Expected: PASS with no full-model tensor in `training_state.pt`.

---

### Task 5: Add the external training entrypoint and single-GPU launcher

**Files:**
- Create: `/home/abc/worldvln_aeroverbs_runtime/scripts/train_backbone_lora_entry.py`
- Create: `/home/abc/worldvln_aeroverbs_runtime/scripts/train_backbone_lora_single_gpu.sh`
- Create: `/home/abc/worldvln_aeroverbs_runtime/scripts/check_worldvln_lora.py`

**Interfaces:**
- Consumes: Tasks 2–4 helpers and original `/home/abc/WorldVLN.code/train/train.py`.
- Produces: a dry-runnable launcher and an external entrypoint that installs all monkeypatches before executing original training code.

- [ ] **Step 1: Add a failing dry-run shell test**

Run before creating the launcher:

```bash
DRY_RUN=1 bash /home/abc/worldvln_aeroverbs_runtime/scripts/train_backbone_lora_single_gpu.sh
```

Expected: FAIL because the launcher does not exist.

- [ ] **Step 2: Implement external entrypoint monkeypatch order**

The entrypoint must perform this exact order:

```python
original_build_vae_gpt = model_load.build_vae_gpt

def build_vae_gpt_with_lora(*args, **kwargs):
    vae, gpt = original_build_vae_gpt(*args, **kwargs)
    gpt = inject_worldvln_lora(gpt, config_from_environment())
    assert_lora_layout(gpt, config_from_environment())
    register_active_lora_model(gpt)
    return vae, gpt

model_load.build_vae_gpt = build_vae_gpt_with_lora
save_and_load.CKPTSaver = LoraCheckpointSaver
save_and_load.auto_resume = lora_auto_resume
runpy.run_path("/home/abc/WorldVLN.code/train/train.py", run_name="__main__")
```

Because original `train.py` imports `CKPTSaver` and `auto_resume` at module import time, patch `infinity.utils.save_and_load` before `runpy.run_path`.

- [ ] **Step 3: Implement launcher defaults**

The shell launcher must default to:

```bash
EXP_NAME=aeroverbs_worldvln_lora_r32_e1
EPOCHS=1
LR=5e-5
LORA_RANK=32
LORA_ALPHA=32
LORA_DROPOUT=0.05
SAVE_FREQ_ITERS=500
RESTRICT_DATA_SIZE=-1
ZERO=0
```

It must retain the established data, T5, VAE, GPT, FPS, frame-count, cache, Flex Attention, and dynamic-scale arguments from `train_backbone_single_gpu.sh`, while changing the entrypoint, learning rate, output layout, `--zero=0`, and `--restrict_data_size` value.

- [ ] **Step 4: Add smoke and dry-run controls**

Support:

```bash
DRY_RUN=1
RESTRICT_DATA_SIZE=3
SAVE_FREQ_ITERS=1
EXP_NAME=aeroverbs_worldvln_lora_smoke
```

The launcher must print the complete shell-escaped torchrun command when `DRY_RUN=1` and exit without importing the model.

- [ ] **Step 5: Implement static configuration checker**

`check_worldvln_lora.py` must verify dependency versions, all required paths, original-weight shard index, output directories, and that no formal-training process with the same experiment name is already running. It must not load the 8.77B model.

- [ ] **Step 6: Run shell syntax and dry-run checks**

Run:

```bash
bash -n /home/abc/worldvln_aeroverbs_runtime/scripts/train_backbone_lora_single_gpu.sh
DRY_RUN=1 bash /home/abc/worldvln_aeroverbs_runtime/scripts/train_backbone_lora_single_gpu.sh
```

Expected: syntax check passes; dry run contains `--zero=0`, `--tlr 5e-5`, `--epoch 1`, original GPT path, and the LoRA entrypoint.

---

### Task 6: Implement adapter merge and sharded-weight output

**Files:**
- Create: `/home/abc/worldvln_aeroverbs_runtime/scripts/merge_worldvln_lora.py`
- Create: `/home/abc/worldvln_aeroverbs_runtime/tests/test_worldvln_lora_merge.py`

**Interfaces:**
- Consumes: original GPT shard directory, PEFT adapter directory, WorldVLN model builder, and a new output directory.
- Produces: merged safetensors shards, `model.safetensors.index.json`, copied non-weight metadata, and `merge_report.json`.

- [ ] **Step 1: Write failing toy merge equivalence test**

For a fixed input `x`, compare:

```python
adapter_output = peft_model(x)
merged_model = merge_adapter_into_base(base_factory, adapter_path)
merged_output = merged_model(x)
torch.testing.assert_close(adapter_output, merged_output, rtol=1e-5, atol=1e-6)
```

Also assert the base model directory is unchanged by hashing every original file before and after merge.

- [ ] **Step 2: Run merge test and verify it fails**

Run:

```bash
cd /home/abc/worldvln_aeroverbs_runtime
/home/abc/miniconda3/envs/worldvln-blackwell/bin/python -m pytest tests/test_worldvln_lora_merge.py -q
```

Expected: FAIL because the merge utility does not exist.

- [ ] **Step 3: Implement merge flow**

The command-line interface must be:

```text
merge_worldvln_lora.py --base-dir BASE --adapter-dir ADAPTER --output-dir OUTPUT --max-shard-size 5GB
```

Required behavior:

1. refuse if `OUTPUT` exists or resolves inside `BASE`;
2. build WorldVLN on CPU and load `BASE` with `strict=False` only for known non-weight metadata differences;
3. inject the adapter configuration from `adapter_config.json`;
4. load adapter tensors strictly;
5. call `merge_and_unload()`;
6. write safetensors shards using `huggingface_hub.split_torch_state_dict_into_shards` with 5GB maximum shards;
7. write `model.safetensors.index.json` and `merge_report.json` to a temporary output directory;
8. atomically rename the temporary directory to `OUTPUT`.

- [ ] **Step 4: Run merge tests**

Run:

```bash
cd /home/abc/worldvln_aeroverbs_runtime
/home/abc/miniconda3/envs/worldvln-blackwell/bin/python -m pytest tests/test_worldvln_lora_merge.py -q
```

Expected: PASS and base hashes unchanged.

---

### Task 7: Run the actual WorldVLN LoRA smoke test

**Files:**
- Create: `/home/abc/worldvln_aeroverbs_runtime/outputs/aeroverbs_worldvln_lora_smoke.log`
- Create: `/home/abc/worldvln_aeroverbs_runtime/outputs/adapters/aeroverbs_worldvln_lora_smoke/`
- Create: `/home/abc/worldvln_aeroverbs_runtime/outputs/merged/aeroverbs_worldvln_lora_smoke/`

**Interfaces:**
- Consumes: Tasks 2–6 and three restricted training samples.
- Produces: evidence that actual 8.77B model injection, backward, save, reload, and merge work.

- [ ] **Step 1: Run all lightweight tests first**

Run:

```bash
cd /home/abc/worldvln_aeroverbs_runtime
/home/abc/miniconda3/envs/worldvln-blackwell/bin/python -m pytest tests/test_worldvln_lora_runtime.py tests/test_worldvln_lora_checkpoint.py tests/test_worldvln_lora_merge.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run static checks and verify no formal process exists**

Run:

```bash
/home/abc/miniconda3/envs/worldvln-blackwell/bin/python /home/abc/worldvln_aeroverbs_runtime/scripts/check_worldvln_lora.py --exp-name aeroverbs_worldvln_lora_r32_e1
```

Expected: dependencies and paths pass; formal process count is zero.

- [ ] **Step 3: Run a foreground three-iteration smoke training**

Run:

```bash
EXP_NAME=aeroverbs_worldvln_lora_smoke RESTRICT_DATA_SIZE=3 SAVE_FREQ_ITERS=1 EPOCHS=1 bash /home/abc/worldvln_aeroverbs_runtime/scripts/train_backbone_lora_single_gpu.sh 2>&1 | tee /home/abc/worldvln_aeroverbs_runtime/outputs/aeroverbs_worldvln_lora_smoke.log
```

Expected:

- model reports exactly 36 matched blocks and 252 target linear modules;
- actual trainable parameters are near 102.1M and no unrelated base parameter is trainable;
- three iterations complete with finite loss;
- adapter checkpoints contain no full base-model weights;
- peak GPU memory is recorded.

- [ ] **Step 4: Verify adapter reload**

Run the checker against the final smoke adapter:

```bash
/home/abc/miniconda3/envs/worldvln-blackwell/bin/python /home/abc/worldvln_aeroverbs_runtime/scripts/check_worldvln_lora.py --adapter-dir /home/abc/worldvln_aeroverbs_runtime/outputs/adapters/aeroverbs_worldvln_lora_smoke/final --load-model
```

Expected: strict adapter load succeeds and trainable tensors match the saved adapter.

- [ ] **Step 5: Merge and validate smoke weights**

Run:

```bash
/home/abc/miniconda3/envs/worldvln-blackwell/bin/python /home/abc/worldvln_aeroverbs_runtime/scripts/merge_worldvln_lora.py --base-dir /home/abc/WorldVLN.code/mode_weights/worldvln_runtime/gpt --adapter-dir /home/abc/worldvln_aeroverbs_runtime/outputs/adapters/aeroverbs_worldvln_lora_smoke/final --output-dir /home/abc/worldvln_aeroverbs_runtime/outputs/merged/aeroverbs_worldvln_lora_smoke --max-shard-size 5GB
```

Expected: merged shard index and report exist; reload has no unexpected weight keys; original weight hashes remain unchanged.

---

### Task 8: Produce—but do not execute—the formal training command

**Files:**
- Create: `/home/abc/worldvln_aeroverbs_runtime/outputs/FORMAL_LORA_TRAINING_COMMAND.md`
- Modify: `/home/zxt/UAV_Basic_Skills/AeroVerbs/docs/MODEL_FINETUNING_CONFIGS.md`

**Interfaces:**
- Consumes: successful Task 7 evidence and actual verified launcher paths.
- Produces: copy-paste launch, progress, checkpoint, merge, and stop commands for the user.

- [ ] **Step 1: Generate the formal dry-run command**

Run only the dry run:

```bash
EXP_NAME=aeroverbs_worldvln_lora_r32_e1 EPOCHS=1 LR=5e-5 LORA_RANK=32 LORA_ALPHA=32 LORA_DROPOUT=0.05 SAVE_FREQ_ITERS=500 RESTRICT_DATA_SIZE=-1 DRY_RUN=1 bash /home/abc/worldvln_aeroverbs_runtime/scripts/train_backbone_lora_single_gpu.sh
```

Expected: prints the complete torchrun command and exits without creating a training process.

- [ ] **Step 2: Write the detached launch command without running it**

Document exactly:

```bash
setsid nohup env EXP_NAME=aeroverbs_worldvln_lora_r32_e1 EPOCHS=1 LR=5e-5 LORA_RANK=32 LORA_ALPHA=32 LORA_DROPOUT=0.05 SAVE_FREQ_ITERS=500 RESTRICT_DATA_SIZE=-1 bash /home/abc/worldvln_aeroverbs_runtime/scripts/train_backbone_lora_single_gpu.sh > /home/abc/worldvln_aeroverbs_runtime/outputs/aeroverbs_worldvln_lora_r32_e1.log 2>&1 < /dev/null & echo $! > /home/abc/worldvln_aeroverbs_runtime/outputs/aeroverbs_worldvln_lora_r32_e1.pid
```

This command must appear only in documentation/final response; do not execute it.

- [ ] **Step 3: Document progress and safety commands**

Include:

```bash
tail -f /home/abc/worldvln_aeroverbs_runtime/outputs/aeroverbs_worldvln_lora_r32_e1.log
```

```bash
grep -oE '\[[[:space:]]*[0-9]+/11072\]' /home/abc/worldvln_aeroverbs_runtime/outputs/aeroverbs_worldvln_lora_r32_e1.log | tail -1
```

```bash
find /home/abc/worldvln_aeroverbs_runtime/outputs/adapters/aeroverbs_worldvln_lora_r32_e1 -maxdepth 1 -type d -name 'checkpoint-*' | sort | tail
```

```bash
kill -TERM "$(cat /home/abc/worldvln_aeroverbs_runtime/outputs/aeroverbs_worldvln_lora_r32_e1.pid)"
```

- [ ] **Step 4: Update the local training configuration record**

Record the LoRA parameters, trainable-parameter count, smoke evidence paths, adapter layout, merge layout, and the explicit fact that formal training was not started by the implementation session.

- [ ] **Step 5: Final verification**

Run:

```bash
pgrep -af 'aeroverbs_worldvln_lora_r32_e1' || true
```

Expected: no formal training process.

Run all unit tests again and inspect the smoke log for `Traceback`, `CUDA out of memory`, `nan`, and `inf`. Completion requires all tests passing, successful adapter reload and merge, and zero formal processes.
