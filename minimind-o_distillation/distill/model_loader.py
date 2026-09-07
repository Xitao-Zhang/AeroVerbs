from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from .dataset import expand_image_tokens


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if value not in mapping:
        raise ValueError(f"unsupported dtype: {value}")
    return mapping[value]


def _import_minimind(root: str | Path):
    root = str(Path(root).expanduser().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from model.model_minimind import precompute_freqs_cis
    from model.model_omni import MiniMindOmni, OmniConfig
    return MiniMindOmni, OmniConfig, precompute_freqs_cis


def load_base_model(config: dict[str, Any], *, device: torch.device | None = None):
    from transformers import AutoTokenizer

    paths, model_cfg, training_cfg = config["paths"], config["model"], config["training"]
    device = device or resolve_device(str(training_cfg.get("device", "auto")))
    dtype = resolve_dtype(str(training_cfg.get("dtype", "bfloat16")), device)
    MiniMindOmni, OmniConfig, _ = _import_minimind(paths["minimind_root"])
    tokenizer = AutoTokenizer.from_pretrained(paths["tokenizer"])
    omni_config = OmniConfig(
        hidden_size=int(model_cfg.get("hidden_size", 768)),
        num_hidden_layers=int(model_cfg.get("num_hidden_layers", 8)),
        use_moe=bool(model_cfg.get("use_moe", False)),
    )
    model = MiniMindOmni(
        omni_config,
        audio_encoder_path="",
        vision_model_path=paths["vision_encoder"],
    )
    raw = torch.load(paths["initial_weight"], map_location="cpu")
    state = raw.get("model", raw) if isinstance(raw, dict) else raw
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = [k for k in incompatible.unexpected_keys if not k.startswith("audio_encoder.")]
    if unexpected:
        raise RuntimeError(f"unexpected base checkpoint keys: {unexpected[:20]}")
    model.to(device=device, dtype=dtype)
    if model.vision_encoder is None:
        raise RuntimeError("SigLIP2 vision encoder failed to load")
    model.vision_encoder.to(device=device, dtype=dtype).eval()
    for parameter in model.vision_encoder.parameters():
        parameter.requires_grad = False
    return model, tokenizer, device, dtype


def configure_trainable_parameters(model, config: dict[str, Any]) -> dict[str, int]:
    model_cfg = config["model"]
    for parameter in model.parameters():
        parameter.requires_grad = False
    if bool(model_cfg.get("train_vision_projector", True)):
        for parameter in model.vision_proj.parameters():
            parameter.requires_grad = True
    count = int(model_cfg.get("thinker_trainable_layers", 2))
    if count < 0 or count > len(model.thinker.layers):
        raise ValueError(f"invalid thinker_trainable_layers={count}")
    if count:
        for block in model.thinker.layers[-count:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
    if bool(model_cfg.get("train_thinker_norm", True)):
        for parameter in model.thinker.norm.parameters():
            parameter.requires_grad = True
    if bool(model_cfg.get("train_lm_head", True)):
        for parameter in model.lm_head.parameters():
            parameter.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError("no trainable parameters selected")
    return {"total": total, "trainable": trainable}


def trainable_parameter_report(model) -> str:
    lines = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            lines.append(f"{name}\t{parameter.numel()}\t{tuple(parameter.shape)}")
    return "\n".join(lines) + "\n"


def thinker_forward(
    model,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    pixel_values: torch.Tensor | None = None,
    past_key_values=None,
    use_cache: bool = False,
):
    """Run only MiniMind-O's visual Thinker branch."""
    # Import through the model module so this also works when the upstream path
    # is configured outside the default checkout.
    from model.model_minimind import precompute_freqs_cis

    past_key_values = past_key_values or [None] * len(model.thinker.layers)
    start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
    if model.thinker.freqs_cos[0, 0] == 0:
        cos, sin = precompute_freqs_cis(
            dim=model.config.head_dim,
            end=model.config.max_position_embeddings,
            rope_base=model.config.rope_theta,
            rope_scaling=model.config.rope_scaling,
        )
        model.thinker.freqs_cos = cos.to(input_ids.device)
        model.thinker.freqs_sin = sin.to(input_ids.device)
    hidden = model.thinker.dropout(model.thinker.embed_tokens(input_ids))
    if pixel_values is not None and start_pos == 0:
        vision = model.encode_image_inputs(pixel_values)
        hidden = model.count_vision_proj(input_ids, hidden, vision, input_ids.shape[1])
    positions = (
        model.thinker.freqs_cos[start_pos : start_pos + input_ids.shape[1]],
        model.thinker.freqs_sin[start_pos : start_pos + input_ids.shape[1]],
    )
    presents = []
    for layer, past in zip(model.thinker.layers, past_key_values):
        hidden, present = layer(
            hidden,
            positions,
            past_key_value=past,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        presents.append(present)
    hidden = model.thinker.norm(hidden)
    return SimpleNamespace(logits=model.lm_head(hidden), past_key_values=presents)


def causal_text_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(
        logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )


def inference_input(tokenizer, prompt: str, image_token_len: int, device: torch.device):
    content = expand_image_tokens(prompt, "<|image_pad|>", image_token_len)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=False,
    )
    ids = tokenizer(rendered, add_special_tokens=False).input_ids
    return torch.tensor([ids], dtype=torch.long, device=device), rendered


@torch.inference_mode()
def greedy_generate(model, tokenizer, input_ids: torch.Tensor, pixel_values: torch.Tensor, *, max_new_tokens: int = 96) -> str:
    prompt_len = input_ids.shape[1]
    attention_mask = torch.ones_like(input_ids)
    past = None
    for _ in range(max_new_tokens):
        current = input_ids if past is None else input_ids[:, -1:]
        result = thinker_forward(
            model,
            current,
            attention_mask=attention_mask,
            pixel_values=pixel_values if past is None else None,
            past_key_values=past,
            use_cache=True,
        )
        past = result.past_key_values
        next_token = result.logits[:, -1].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat((input_ids, next_token), dim=1)
        attention_mask = torch.cat((attention_mask, torch.ones_like(next_token)), dim=1)
        if int(next_token.item()) == tokenizer.eos_token_id:
            break
    return tokenizer.decode(input_ids[0, prompt_len:].tolist(), skip_special_tokens=True).strip()


def checkpoint_payload(model, config: dict[str, Any], *, epoch: int, optimizer=None, scheduler=None, metrics=None) -> dict[str, Any]:
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    delta = {name: value.detach().half().cpu() for name, value in model.state_dict().items() if name in trainable_names}
    payload: dict[str, Any] = {
        "format": "aeroverbs-minimind-region-delta-v1",
        "epoch": int(epoch),
        "model_delta": delta,
        "trainable_names": sorted(trainable_names),
        "config": config,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    return payload


def load_delta_checkpoint(model, path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != "aeroverbs-minimind-region-delta-v1":
        raise ValueError(f"unsupported checkpoint format: {payload.get('format')}")
    incompatible = model.load_state_dict(payload["model_delta"], strict=False)
    unexpected = incompatible.unexpected_keys
    if unexpected:
        raise RuntimeError(f"unexpected delta keys: {unexpected}")
    return payload
