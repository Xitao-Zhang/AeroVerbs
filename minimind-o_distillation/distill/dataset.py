from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            rows.append(row)
    return rows


def expand_image_tokens(text: str, image_token: str, image_token_len: int) -> str:
    if text.count("<image>") != 1:
        raise ValueError("each user prompt must contain exactly one <image> marker")
    return text.replace("<image>", image_token * image_token_len)


def render_training_example(
    tokenizer,
    *,
    prompt: str,
    answer: str,
    image_token_len: int,
) -> tuple[list[int], list[int], str]:
    user_content = expand_image_tokens(prompt, "<|image_pad|>", image_token_len)
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=False,
    )
    full = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ],
        tokenize=False,
        add_generation_prompt=False,
        open_thinking=False,
    )
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    input_ids = tokenizer(full, add_special_tokens=False).input_ids
    if input_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("chat template assistant prefix is not a prefix of the full sample")
    labels = [-100] * len(prefix_ids) + input_ids[len(prefix_ids) :]
    if not any(label != -100 for label in labels):
        raise ValueError("sample has no supervised assistant tokens")
    return input_ids, labels, full


class RegionSFTDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        tokenizer,
        image_processor,
        *,
        max_seq_len: int = 1024,
        image_token_len: int = 64,
    ) -> None:
        self.manifest = Path(manifest)
        self.rows = read_jsonl(self.manifest)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_seq_len = int(max_seq_len)
        self.image_token_len = int(image_token_len)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        input_ids, labels, _ = render_training_example(
            self.tokenizer,
            prompt=row["prompt"],
            answer=row["answer"],
            image_token_len=self.image_token_len,
        )
        if len(input_ids) > self.max_seq_len:
            raise ValueError(
                f"sample {row['sample_id']} has {len(input_ids)} tokens, "
                f"exceeding max_seq_len={self.max_seq_len}; refusing to truncate"
            )
        with Image.open(row["image_path"]) as image:
            pixel_values = self.image_processor(
                images=image.convert("RGB"), return_tensors="pt"
            ).pixel_values[0]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": pixel_values,
            "sample_id": row["sample_id"],
            "skill": row["skill"],
            "target": row["regions"],
            "width": int(row["width"]),
            "height": int(row["height"]),
            "prompt": row["prompt"],
        }


class RegionCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(sample["input_ids"].numel() for sample in samples)
        batch_size = len(samples)
        input_ids = torch.full(
            (batch_size, max_len), self.pad_token_id, dtype=torch.long
        )
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        for index, sample in enumerate(samples):
            length = sample["input_ids"].numel()
            input_ids[index, :length] = sample["input_ids"]
            labels[index, :length] = sample["labels"]
            attention_mask[index, :length] = 1
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": torch.stack([s["pixel_values"] for s in samples]),
            "sample_id": [s["sample_id"] for s in samples],
            "skill": [s["skill"] for s in samples],
            "target": [s["target"] for s in samples],
            "width": [s["width"] for s in samples],
            "height": [s["height"] for s in samples],
            "prompt": [s["prompt"] for s in samples],
        }


def skill_sample_weights(rows: Iterable[dict[str, Any]], weights: dict[str, float]) -> torch.Tensor:
    return torch.tensor(
        [float(weights.get(row["skill"], 1.0)) for row in rows], dtype=torch.double
    )


class CoverageRepeatSampler(Sampler[int]):
    """Visit every row once, then append configured integer repeats.

    Unlike WeightedRandomSampler, this never drops a source annotation from an
    epoch. A skill weight of 5 means one mandatory visit plus four repeats.
    """

    def __init__(self, rows: Sequence[dict[str, Any]], weights: dict[str, float], seed: int = 42) -> None:
        self.indices: list[int] = []
        self.seed = int(seed)
        self.epoch = 0
        for index, row in enumerate(rows):
            repeats = max(1, int(round(float(weights.get(row["skill"], 1.0)))))
            self.indices.extend([index] * repeats)

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.indices), generator=generator).tolist()
        self.epoch += 1
        return iter(self.indices[index] for index in order)
