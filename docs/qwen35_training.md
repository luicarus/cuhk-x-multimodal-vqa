# Qwen3.5-4B QLoRA release — vLLM dual-GPU

This lane fine-tunes `Qwen/Qwen3.5-4B` with the existing IR4 data contract and complete five-fold IR8 cache. It has its own Transformers 5.17.0 / vLLM 0.21.0 environment; the Qwen2.5-VL-7B v3 lane is unchanged.

- Notebook: `notebooks/qwen35-4b-qlora-vllm.ipynb`
- Package: `artifacts/cloud_training/qwen35_4b_qlora.zip`
- Config: `configs/training_qwen35.yaml`
- Training profile: `qwen35_4b_qlora_v1`
- vLLM backend: `src/cuhkx/inference/qwen35_vllm.py`

This is the **post-training** lane. The inference-only lane produces `artifacts/cloud/qwen35_4b.zip` and does not carry the training stack; see `docs/qwen35_4b.md`.

The package does not contain model weights. The Notebook resolves and pins a Qwen3.5 commit, downloads or reuses a verified weight directory, runs CPU data checks, then performs a cloud CUDA smoke run before any full training. Formal test submission stays gated on dev and confirm improvements.

The local package command prints `manifest_sha256`. Copy that value through a trusted channel into the Notebook's `EXPECTED_MANIFEST_SHA256` before extraction. Qwen3.5 uses the native Transformers implementation with remote model code disabled.

## Engine split

vLLM is inference-only and cannot produce gradients, so it replaces the generation engine and nothing else:

| Stage | Engine |
|---|---|
| Training and evaluation-under-training | Transformers 5.17.0 |
| Adapter reload smoke, dev/confirm evaluation, test inference | vLLM 0.21.0, tensor parallel across both T4 GPUs |

Both engines share the same data contract, prompt version, image protocol and closed answer space, so their scores are directly comparable. The signed run contract records `engine` and `engine_options`; the same `run-id` cannot mix results from two engines.

The reference backend constrains decoding with a stateful `prefix_allowed_tokens_fn`. vLLM has no equivalent hook, so `Qwen35VLLMBackend` constrains the same language with `StructuredOutputsParams(choice=[...])`, pinning each legal answer to a literal prefix. Chat rendering is checked against the reference processor with `enable_thinking=False` before the engine is built, because the answer boundary depends on that exact encoding.

## Dual-T4 operation

Two T4s have no NVLink, so tensor parallelism crosses PCIe. The backend therefore runs with `enforce_eager=True`, `disable_custom_all_reduce=True`, and `NCCL_P2P_DISABLE=1`; CUDA-graph capture and peer-to-peer probing over PCIe are the usual cause of hangs on this hardware. `gpu_memory_utilization` defaults to `0.80` so the KV cache does not crowd out the rest of the session.

Training stays pinned to a single GPU (`TRAIN_GPU = 0`) to avoid competing with the vLLM tensor-parallel workers for memory.

## Dependencies

- `requirements/qwen35.lock.txt`: Transformers 5.17.0 + vLLM 0.21.0 + torch 2.11.0+cu126
- `requirements/train_qwen35.lock.txt`: adds PEFT 0.18.0 and bitsandbytes 0.49.2

vLLM 0.21.0 is the last release whose acceleration stack targets CUDA 12: it declares a bare `nvidia-cutlass-dsl==4.4.2`. From 0.22.1 onward vLLM declares `nvidia-cutlass-dsl[cu13]`, which pulls a CUDA 13 runtime next to torch's cu126 build and fails at import with `libcudart.so.13: cannot open shared object file`, because `torch==2.11.0+cu126` only ships `libcudart.so.12`. Kaggle's T4 is SM 7.5 on a CUDA 12 driver, so the CUDA 13 line is unusable there regardless of vLLM version.

vLLM 0.21.0 pins `torch==2.11.0`, which is why the lane moved off torch 2.7.1, and requires `transformers>=4.56`, satisfied by 5.17.0. The 7B lane keeps torch 2.7.1 + Transformers 4.57.6 and never receives vLLM.

Locks are checked for CUDA-major-version consistency: a lock that contains both `cu12` and `cu13` packages is rejected by `tests/test_qwen35_training.py`.

The Qwen3.5 trainer selects both hybrid text decoder families: `linear_attn.in_proj_qkv`/`in_proj_z` and full-attention `self_attn.q_proj`/`v_proj`. Vision parameters remain frozen.
