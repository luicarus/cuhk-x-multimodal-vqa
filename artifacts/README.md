# Cloud release artifacts

`artifacts/cloud/` contains cloud inference packages and baseline provenance:

- `ir4_7b_v1.zip`: Qwen2.5-VL-7B IR4 baseline.
- `qwen35_4b_test_v1.zip`: independent Qwen3.5-4B test comparison.
- `ir4_7b_v1/`: baseline weight receipt and cloud environment evidence.

`artifacts/cloud_training/` contains complete post-training packages:

- `cuhkx-ir4-qlora-full-v3.zip`: current Qwen2.5-VL-7B complete release.
- `qwen35_4b.zip`: Qwen3.5-4B complete post-training release (vLLM dual-GPU inference).

Generate packages with `scripts/package_cloud.py`, `scripts/package_qwen35.py`, `scripts/package_training.py`, and `scripts/package_qwen35_training.py`. Each command validates the archive and prints a `manifest_sha256`; copy that digest into the corresponding Notebook through a trusted channel. ZIPs and provenance files are ignored by Git and should be kept in private storage.
