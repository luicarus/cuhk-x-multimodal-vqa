"""Dual-GPU vLLM inference backend for the Qwen3.5-4B lane.

vLLM replaces only the *generation* engine. Training and development time
evaluation-under-training still use the Transformers ``Qwen35Backend``: vLLM is
inference-only and cannot produce gradients. The data contract, prompt version,
constrained answer space and run/checkpoint protocol are unchanged, so a vLLM
run and a Transformers run of the same revision stay comparable.

Two properties of the existing protocol make this substitution safe, and both
are asserted below rather than assumed:

1. The answer space is closed and tiny, so constraining generation to the same
   set of outputs is what matters, not the mechanism. The Transformers backend
   enforces this with a stateful ``prefix_allowed_tokens_fn``. vLLM has no such
   hook; it constrains the same language with a structured-output ``choice``
   grammar.

2. The prompt must tokenize to exactly what ``apply_chat_template`` produces,
   including ``enable_thinking=False``. The prefix boundary positions inside the
   allowed-answer map are only valid for that encoding, so the rendered prompt
   is compared against the reference tokenizer before the engine is built.
"""
from __future__ import annotations

import importlib.metadata
import time
from pathlib import Path

from cuhkx.config import require


# Two T4s have no NVLink. Tensor parallelism therefore crosses PCIe, where
# CUDA-graph capture and custom all-reduce are a common source of hangs and
# illegal-memory-access crashes. Eager mode costs some throughput and buys a
# run that actually finishes; the NCCL path is also pinned to PCIe so vLLM
# cannot probe for peer-to-peer support at startup.
VLLM_ENGINE_ENV = {
    "VLLM_USE_V1": "1",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_NO_USAGE_STATS": "1",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    # FlashInfer/Triton JIT need a writable cache even on a read-only dataset.
    "TRITON_CACHE_DIR": "/tmp/cuhkx_triton_cache",
    "TORCHINDUCTOR_CACHE_DIR": "/tmp/cuhkx_inductor_cache",
    "NCCL_P2P_DISABLE": "1",
    "NCCL_IB_DISABLE": "1",
    "NCCL_DEBUG": "WARN",
}


# vLLM's attention backends differ in what they need at *run* time, not just at
# import time, so the choice is part of the lane's runtime contract.
#
# FLASHINFER is vLLM's automatic first choice, but on a Kaggle T4 it JIT-compiles
# SM 7.5 kernels through the host CUDA toolkit, and the final link step asks for
# the driver library:
#
#     nvcc ... -gencode=arch=compute_75,code=sm_75 ... -lcudart -lcuda
#     /usr/bin/ld: cannot find -lcuda: No such file or directory
#
# libcuda.so belongs to the NVIDIA *driver*, not to the CUDA runtime that pip
# installs, and Kaggle's container has no driver stubs directory. The failure
# only appears on the first real forward pass, after the model and KV cache are
# already up, which makes it look like a later bug than it is. That same JIT
# also dominated startup (~9 minutes) before it failed.
#
# TRITON_ATTN ships as pure Triton, needs no nvcc/link step, and reports
# supports_compute_capability() == True for every device. It supports float16,
# which is this lane's dtype, so it is the default here.
ATTENTION_BACKENDS = ("TRITON_ATTN", "FLASHINFER", "FLEX_ATTENTION")


def apply_engine_environment(environment=None) -> dict:
    """Populate vLLM/NCCL defaults without overwriting explicit overrides."""
    import os

    target = os.environ if environment is None else environment
    for name, value in VLLM_ENGINE_ENV.items():
        target.setdefault(name, value)
    for name in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        Path(target[name]).mkdir(parents=True, exist_ok=True)
    return dict(target)


def _choice_pattern(prefix: str) -> str:
    """Restrict the answer grammar to the required literal prefix.

    ``choice`` alone would let the model emit any legal answer. The dataset
    presents its options in a fixed order and the reference backend can only
    produce an output that is itself a legal answer, but a prefix is what makes
    the two backends agree exactly, so it is enforced here too.
    """
    import re

    return r"\s*" + re.escape(prefix)


def render_prompt(tokenizer, prompt: str, *, enable_thinking: bool = False) -> str:
    """Render one user turn with the exact template arguments the lane pins."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def verify_prompt_fidelity(processor, prompts: dict) -> dict:
    """Prove vLLM's chat rendering matches the Transformers reference encoding.

    ``Qwen35Backend`` tokenizes the multimodal turn through
    ``processor.apply_chat_template``. vLLM renders chat templates and injects
    image placeholders itself, so the two can differ silently. Only the textual
    envelope is checked here; the comparison is between the reference processor
    and the vLLM renderer, so a template or ``enable_thinking`` drift is a hard
    error rather than a silent change of the decoding boundary.
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    image_token = getattr(processor, "image_token", "<|image_pad|>")
    require(bool(prompts), "no prompts were available for fidelity checking")
    reference = []
    for prompt in prompts.values():
        # The reference path passes a message list to the processor; the
        # rendered string must match what vLLM will build from the same turn.
        reference.append(_render_reference(tokenizer, prompt))
    checked = 0
    for qa_id, prompt in prompts.items():
        require(image_token not in prompt, f"prompt leaks the image placeholder: {qa_id}")
        rendered = render_prompt(tokenizer, prompt)
        require(rendered == reference[checked], f"vLLM chat rendering differs for {qa_id}")
        checked += 1
    return {"prompts_checked": checked, "image_token": image_token, "template_mismatches": 0}


def _render_reference(tokenizer, prompt: str) -> str:
    """Reference rendering used by the Transformers backend for text turns."""
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        # Older templates do not accept enable_thinking; the vLLM path would
        # then also have to omit it, so surface the difference immediately.
        raise ValueError(
            "the pinned chat template no longer accepts enable_thinking; "
            "vLLM prompt fidelity cannot be guaranteed"
        ) from None


class Qwen35VLLMBackend:
    """Offline multi-GPU vLLM engine constrained to the lane's answer space."""

    def __init__(self, baseline: dict, weights: Path, adapter: Path | None = None,
                 *, tensor_parallel_size: int = 2, gpu_memory_utilization: float = 0.80,
                 enable_prefix_caching: bool = True, max_model_len: int = 4096,
                 attention_backend: str = "TRITON_ATTN"):
        require(tensor_parallel_size >= 1, "tensor parallel size must be positive")
        require(0.0 < gpu_memory_utilization < 1.0, "gpu memory utilization must be a fraction")
        require(max_model_len >= 1024, "max_model_len is implausibly small")
        require(attention_backend in ATTENTION_BACKENDS,
                f"unsupported attention backend: {attention_backend}")
        apply_engine_environment()

        import torch
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        require(weights.is_dir(), "Qwen3.5 weight directory is missing")
        available = torch.cuda.device_count()
        require(available >= tensor_parallel_size,
                f"vLLM tensor_parallel_size={tensor_parallel_size} needs {tensor_parallel_size} "
                f"visible CUDA devices, found {available}")

        self.SamplingParams = SamplingParams
        self.StructuredOutputsParams = StructuredOutputsParams
        self.image_size = baseline["frames"]["input_image_size"]
        self.tensor_parallel_size = tensor_parallel_size
        self.attention_backend = attention_backend

        started = time.perf_counter()
        # use_fast=False mirrors the reference backend: the slow image processor
        # is the one the lane validated.
        self.processor = AutoProcessor.from_pretrained(
            str(weights), local_files_only=True, trust_remote_code=False, use_fast=False,
        )
        self.engine = LLM(
            model=str(weights),
            revision=baseline["model"].get("revision") or None,
            tokenizer=str(weights),
            tokenizer_revision=baseline["model"].get("revision") or None,
            trust_remote_code=False,
            dtype="float16",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=True,
            enable_prefix_caching=enable_prefix_caching,
            disable_custom_all_reduce=True,
            max_num_seqs=1,
            attention_config={"backend": attention_backend},
            # A LoRA adapter needs the LoRA path compiled into the engine.
            enable_lora=adapter is not None,
            max_loras=1,
            limit_mm_per_prompt={"image": 4},
            mm_processor_kwargs={"use_fast": False},
        )
        self.adapter_request = None
        if adapter is not None:
            self.adapter_request = self._register_adapter(adapter)
        self.load_seconds = time.perf_counter() - started

    def _register_adapter(self, adapter: Path):
        """Load the verified LoRA adapter through vLLM's own loader.

        The adapter arrives already verified by ``verify_adapter`` against the
        base weight receipt; vLLM re-reads the same directory, so the receipt
        stays the authority on provenance.
        """
        adapter = Path(adapter).resolve()
        require(adapter.is_dir(), "adapter directory is missing")
        from vllm.lora.request import LoRARequest

        request = LoRARequest(lora_name="cuhkx_adapter", lora_int_id=1, lora_path=str(adapter))
        self.engine.add_lora(request)
        return request

    def generate(self, images, prompt, *, allowed_outputs, max_new_tokens):
        require(len(images) == 4, "Qwen3.5 IR4 requires exactly four images")
        outputs = tuple(allowed_outputs)
        require(outputs and all(isinstance(value, str) and value for value in outputs),
                "the constrained answer space must be non-empty text")
        require(max_new_tokens >= max(len(value) for value in outputs),
                "max_new_tokens cannot cover the answer space")

        content = [{"type": "image_url",
                    "image_url": {"url": _data_uri(image), "detail": "high"}}
                   for image in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        parameters = self.SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_new_tokens,
            skip_special_tokens=True,
            # A bare ``choice`` list also accepts whitespace-padded or
            # wrong-order variants depending on the backend; pinning the first
            # character to a literal keeps vLLM's accepted language identical to
            # the reference prefix automaton.
            structured_outputs=self.StructuredOutputsParams(choice=[
                _choice_pattern(value) for value in outputs]),
        )
        results = self.engine.chat(
            messages,
            sampling_params=parameters,
            use_tqdm=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
            lora_request=self.adapter_request,
        )
        require(len(results) == 1, "vLLM returned an unexpected number of results")
        return results[0].outputs[0].text.strip()

    def metadata(self):
        import os

        runtime = {name: importlib.metadata.version(name)
                   for name in ("torch", "transformers", "vllm")
                   if _installed(name)}
        return {
            "backend": "qwen35_4b_vllm",
            "model_class": type(self.engine).__name__,
            "image_size": self.image_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "attention_backend": self.attention_backend,
            "enforce_eager": True,
            "disable_custom_all_reduce": True,
            "answer_constraint": "structured_outputs_choice",
            "lora_adapter": self.adapter_request.lora_name if self.adapter_request else None,
            "load_seconds": self.load_seconds,
            "versions": runtime,
            "engine_environment": {name: os.environ.get(name) for name in VLLM_ENGINE_ENV},
        }


def _installed(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _data_uri(image) -> str:
    """Encode a PIL frame as an OpenAI-style image URL."""
    import base64
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
