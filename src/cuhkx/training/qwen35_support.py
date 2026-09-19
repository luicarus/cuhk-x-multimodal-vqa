"""Qwen3.5-specific training primitives.

The original QLoRA trainer is intentionally Qwen2.5-VL-specific.  This module
keeps the Qwen3.5 processor/model differences isolated so a future trainer can
use the same data contract without changing the established 7B lane.
"""
from __future__ import annotations

from pathlib import Path
import re

from PIL import Image

from cuhkx.config import require
from cuhkx.inference.prompt import build_mcq_prompt, detect_prompt_leakage
from cuhkx.training.collator import answer_labels


_TEXT_LAYER = re.compile(r"(?:model\.)?language_model\.layers\.\d+\.")
_TARGET_SUFFIXES = (
    ".linear_attn.in_proj_qkv",
    ".linear_attn.in_proj_z",
    ".self_attn.q_proj",
    ".self_attn.v_proj",
)


def qwen35_text_targets(names: list[str] | tuple[str, ...]) -> list[str]:
    """Select text-decoder LoRA targets from a Qwen3.5 model.

    Qwen3.5 has a hybrid decoder: most blocks use gated linear attention and
    every fourth block uses full attention.  The linear-attention QKV and gate
    projections are named differently from the full-attention q/v projections;
    selecting only ``self_attn.q_proj``/``v_proj`` would leave 75% of the text
    blocks untouched.  Vision modules and the MLP remain frozen.
    """
    selected = sorted(
        name for name in names
        if _TEXT_LAYER.search(name)
        and any(name.endswith(suffix) for suffix in _TARGET_SUFFIXES)
        and not any(part in name.split(".") for part in ("visual", "vision_model", "vision_tower", "merger"))
    )
    require(selected, "Qwen3.5 text decoder LoRA targets not found")
    require(any(name.endswith(".linear_attn.in_proj_qkv") for name in selected),
            "Qwen3.5 linear-attention targets not found")
    require(any(name.endswith(".self_attn.q_proj") for name in selected),
            "Qwen3.5 full-attention q_proj targets not found")
    return selected


def _image_messages(example: dict, data_root: Path) -> tuple[list[dict], list[Image.Image]]:
    """Build one Qwen3.5 user message and keep image ownership local."""
    qa = example["qa"]
    prompt = build_mcq_prompt(qa)
    require(not detect_prompt_leakage(prompt, qa), "training prompt leakage")
    require(len(example["frame_paths"]) == 4, "Qwen3.5 IR4 requires four images")
    owned: list[Image.Image] = []
    content = []
    for relative in example["frame_paths"]:
        path = Path(data_root) / relative
        with Image.open(path) as image:
            image.load()
            copied = image.copy()
        owned.append(copied)
        content.append({"type": "image", "image": copied,
                        "resized_height": 280, "resized_width": 280})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}], owned


class Qwen35CompletionCollator:
    """Answer-only multimodal collator using the Transformers 5 processor.

    Unlike the legacy Qwen2.5 path this does not call ``qwen_vl_utils``.  The
    Qwen3.5 processor consumes PIL images embedded in the chat messages and
    returns image tensors, grid metadata, and token tensors together.
    """

    def __init__(self, processor, data_root: Path, max_length: int = 4096):
        self.processor = processor
        self.data_root = Path(data_root)
        self.max_length = max_length

    def __call__(self, examples):
        import torch

        prefixes, fulls, owned = [], [], []
        try:
            for example in examples:
                user, images = _image_messages(example, self.data_root)
                owned.extend(images)
                prefixes.append(user)
                fulls.append(user + [{"role": "assistant", "content": example["answer"]}])
            full = self.processor.apply_chat_template(
                fulls, tokenize=True, add_generation_prompt=False, return_dict=True,
                return_tensors="pt", enable_thinking=False,
                processor_kwargs={"padding": True, "truncation": False},
            )
            prefix = self.processor.apply_chat_template(
                prefixes, tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", enable_thinking=False,
                processor_kwargs={"padding": True, "truncation": False},
            )
            tokenizer = self.processor.tokenizer
            eos = tokenizer.eos_token_id
            require(isinstance(eos, int), "Qwen3.5 training tokenizer must expose one EOS token")
            labels = []
            for index, example in enumerate(examples):
                target, trailing = answer_labels(
                    full["input_ids"][index].tolist(), full["attention_mask"][index].tolist(),
                    prefix["input_ids"][index].tolist(), prefix["attention_mask"][index].tolist(),
                    tokenizer.encode(example["answer"], add_special_tokens=False), eos, self.max_length,
                )
                require(not tokenizer.decode(trailing, skip_special_tokens=False).strip(),
                        "unexpected text after Qwen3.5 assistant EOS")
                labels.append(target)
            full["labels"] = torch.tensor(labels, dtype=torch.long)
            return full
        finally:
            for image in owned:
                image.close()
