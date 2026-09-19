"""Cloud-only Qwen2.5-VL-7B NF4 backend; imports GPU packages only on construction."""
from __future__ import annotations

import importlib.metadata
import time
from pathlib import Path

from cuhkx.config import require


def token_constraint(tokenizer, outputs, prompt_length, max_new_tokens):
    candidates = [tuple(tokenizer.encode(value, add_special_tokens=False)) for value in outputs]
    require(bool(candidates) and all(candidates), "allowed answers must tokenize to nonempty sequences")
    require(len(set(candidates)) == len(candidates), "different answers share the same tokens")
    eos = tokenizer.eos_token_id
    require(eos is not None and all(eos not in values for values in candidates), "invalid EOS token")
    require(max(len(v) for v in candidates) + 1 <= max_new_tokens, "answer space exceeds frozen token budget")

    def allowed(_batch, ids):
        generated = tuple(int(token) for token in ids[prompt_length:].tolist())
        choices = set()
        for values in candidates:
            if values[:len(generated)] == generated:
                choices.add(values[len(generated)] if len(generated) < len(values) else eos)
        require(bool(choices), "decoding reached an illegal answer prefix")
        return sorted(choices)
    return allowed


class QwenBackend:
    def __init__(self, baseline: dict, weights: Path, offload: Path, adapter: Path | None = None):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        require(torch.cuda.is_available(), "7B NF4 requires a cloud CUDA GPU; do not run predict on this CPU host")
        self.torch = torch
        self.baseline = baseline
        self.image_size = baseline["frames"]["input_image_size"]
        budget = baseline["runtime"]
        memory = {}
        for index in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(index)
            available = min(budget["gpu_memory_mib"], int(free / 1024**2) - 512)
            require(available >= 2048, f"insufficient free memory on GPU {index}")
            memory[index] = f"{available}MiB"
        memory["cpu"] = f"{budget['cpu_memory_gib']}GiB"
        self.memory = memory
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True)
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(str(weights), local_files_only=True,
            trust_remote_code=False, min_pixels=self.image_size**2, max_pixels=self.image_size**2)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(weights), local_files_only=True,
            trust_remote_code=False, quantization_config=quantization, dtype=torch.float16,
            device_map="auto", max_memory=memory, offload_folder=str(offload), offload_state_dict=True,
            low_cpu_mem_usage=True, attn_implementation="sdpa")
        if adapter is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(adapter), is_trainable=False, local_files_only=True)
        self.model.eval()
        self.device = next((p.device for p in self.model.parameters() if p.device.type == "cuda"), None)
        require(self.device is not None, "model was not placed on CUDA")
        self.load_seconds = time.perf_counter() - started
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)

    def generate(self, images, prompt, *, allowed_outputs, max_new_tokens):
        require(len(images) == 4, "IR4 requires exactly four input images")
        from qwen_vl_utils import process_vision_info

        content = [{"type": "image", "image": image, "resized_height": self.image_size,
                    "resized_width": self.image_size} for image in images]
        messages = [{"role": "user", "content": content + [{"type": "text", "text": prompt}]}]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[rendered], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to(self.device)
        constraint = token_constraint(self.processor.tokenizer, allowed_outputs,
                                      inputs["input_ids"].shape[-1], max_new_tokens)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                do_sample=False, num_beams=1, use_cache=True, prefix_allowed_tokens_fn=constraint)
        trimmed = [output[len(original):] for original, output in zip(inputs.input_ids, generated, strict=True)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0].strip()

    def metadata(self):
        return {"backend": "qwen25_vl_7b_nf4", "load_seconds": self.load_seconds,
                "device_map": {str(k): str(v) for k, v in self.model.hf_device_map.items()},
                "memory_budget": self.memory, "versions": {name: importlib.metadata.version(name)
                    for name in ("torch", "transformers", "bitsandbytes", "qwen-vl-utils")},
                "peak_cuda_allocated_mib": {str(i): self.torch.cuda.max_memory_allocated(i) / 1024**2
                    for i in range(self.torch.cuda.device_count())}}
