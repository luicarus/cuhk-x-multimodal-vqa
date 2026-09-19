# Qwen3.5-4B QLoRA release

This lane fine-tunes `Qwen/Qwen3.5-4B` with the existing IR4 data contract and complete five-fold IR8 cache. It has its own Transformers 5.17.0 and PEFT environment; the Qwen2.5-VL-7B v3 lane is unchanged.

- Notebook: `notebooks/qwen35-4b-qlora-full-v1.ipynb`
- Package: `artifacts/cloud_training/qwen35_4b_qlora_full_v1.zip`
- Config: `configs/training_qwen35.yaml`
- Training profile: `qwen35_4b_qlora_v1`

The package does not contain model weights. The Notebook resolves and pins a Qwen3.5 commit, downloads or reuses a verified weight directory, runs CPU data checks, then performs a cloud CUDA smoke run before any full training. Formal test submission stays gated on dev and confirm improvements.

The local package command prints `manifest_sha256`. Copy that value through a trusted channel into the Notebook's `EXPECTED_MANIFEST_SHA256` before extraction. Qwen3.5 uses the native Transformers implementation with remote model code disabled.

The Qwen3.5 trainer selects both hybrid text decoder families: `linear_attn.in_proj_qkv`/`in_proj_z` and full-attention `self_attn.q_proj`/`v_proj`. Vision parameters remain frozen.
