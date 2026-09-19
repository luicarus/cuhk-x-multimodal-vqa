"""Cloud-only Qwen3.5-4B multimodal backend for the isolated comparison lane."""
from __future__ import annotations

import importlib.metadata
import time
from pathlib import Path

from cuhkx.config import require


def _token_ids(value):
    if value is None:
        return ()
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(item) for item in value)


def token_constraint(tokenizer, outputs, prompt_length, max_new_tokens, *, eos_token_id=None):
    encoded = [tuple(tokenizer.encode(value, add_special_tokens=False)) for value in outputs]
    require(bool(encoded) and all(encoded), "Qwen3.5 answer candidates must tokenize")
    require(len(set(encoded)) == len(encoded), "Qwen3.5 answer candidates collide after tokenization")
    eos_ids = _token_ids(tokenizer.eos_token_id if eos_token_id is None else eos_token_id)
    require(eos_ids and all(not set(eos_ids).intersection(values) for values in encoded),
            "Qwen3.5 EOS token is invalid")
    require(max(len(values) for values in encoded) + 1 <= max_new_tokens,
            "Qwen3.5 answer space exceeds max_new_tokens")

    def allowed(_batch, input_ids):
        generated = tuple(int(token) for token in input_ids[prompt_length:].tolist())
        choices = set()
        for values in encoded:
            if values[:len(generated)] != generated:
                continue
            choices.update((values[len(generated)],) if len(generated) < len(values) else eos_ids)
        require(bool(choices), "Qwen3.5 decoding reached an illegal answer prefix")
        return sorted(choices)

    return allowed


class Qwen35Backend:
    def __init__(self, baseline: dict, weights: Path, adapter: Path | None = None):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        require(torch.cuda.is_available(), "Qwen3.5 test requires a cloud CUDA GPU")
        self.torch = torch
        self.image_size = baseline["frames"]["input_image_size"]
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(
            str(weights), local_files_only=True, trust_remote_code=False, use_fast=False,
        )
        self.model = AutoModelForMultimodalLM.from_pretrained(
            str(weights), local_files_only=True, trust_remote_code=False,
            dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True,
        )
        if adapter is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(adapter), is_trainable=False,
                                                   local_files_only=True)
        self.model.eval()
        self.device = next((parameter.device for parameter in self.model.parameters()
                            if parameter.device.type == "cuda"), None)
        require(self.device is not None, "Qwen3.5 model was not placed on CUDA")
        self.load_seconds = time.perf_counter() - started

    def generate(self, images, prompt, *, allowed_outputs, max_new_tokens):
        require(len(images) == 4, "Qwen3.5 IR4 requires exactly four images")
        content = [{"type": "image", "image": image,
                    "resized_height": self.image_size, "resized_width": self.image_size}
                   for image in images]
        messages = [{"role": "user", "content": content + [{"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", enable_thinking=False,
        )
        inputs = inputs.to(self.device)
        generation_eos = self.model.generation_config.eos_token_id
        constraint = token_constraint(
            self.processor.tokenizer,
            allowed_outputs,
            inputs["input_ids"].shape[-1],
            max_new_tokens,
            eos_token_id=generation_eos,
        )
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                do_sample=False, num_beams=1, use_cache=True, eos_token_id=generation_eos,
                prefix_allowed_tokens_fn=constraint)
        trimmed = [output[len(original):] for original, output in
                    zip(inputs["input_ids"], generated, strict=True)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0].strip()

    def metadata(self):
        return {"backend": "qwen35_4b_transformers", "model_class": type(self.model).__name__,
                "image_size": self.image_size, "device": str(self.device),
                "load_seconds": self.load_seconds,
                "versions": {name: importlib.metadata.version(name)
                    for name in ("torch", "transformers", "accelerate")},
                "device_map": {str(key): str(value)
                    for key, value in getattr(self.model, "hf_device_map", {}).items()}}
