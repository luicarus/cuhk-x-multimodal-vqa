"""Pinned Qwen3.5-4B weight download and content verification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cuhkx.config import inside, require
from cuhkx.data.validate import fingerprint
from cuhkx.inference.storage import write_json


MODEL_ID = "Qwen/Qwen3.5-4B"
RECEIPT = "cuhkx_qwen35_weights.json"
ALLOWED_SUFFIXES = {".json", ".safetensors", ".txt", ".model", ".tiktoken", ".jinja"}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file()
                  and ".cache" not in path.relative_to(root).parts and path.name != RECEIPT)


def validate_layout(root: Path, files: list[Path]) -> None:
    require(files and all(path.suffix.lower() in ALLOWED_SUFFIXES for path in files),
            "unexpected Qwen3.5 model files")
    config_path = root / "config.json"
    require(config_path.is_file(), "Qwen3.5 config.json is missing")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(config.get("model_type") == "qwen3_5", "model config is not Qwen3.5")
    require(config.get("architectures") == ["Qwen3_5ForConditionalGeneration"],
            "wrong Qwen3.5 architecture")
    for name in ("config.json", "tokenizer_config.json", "preprocessor_config.json"):
        path = root / name
        require(path.is_file(), f"Qwen3.5 {name} missing")
        value = json.loads(path.read_text(encoding="utf-8"))
        require(not value.get("auto_map"), "Qwen3.5 dynamic model code is unsupported")
    weights = {path.name for path in files if path.suffix.lower() == ".safetensors"}
    index = root / "model.safetensors.index.json"
    if index.exists():
        index_data = json.loads(index.read_text(encoding="utf-8"))
        shards = set(index_data.get("weight_map", {}).values())
        require(shards and shards == weights, "Qwen3.5 shard index differs from weights")
    else:
        require(weights == {"model.safetensors"}, "Qwen3.5 model weights missing")


def verify_qwen35_weights(root: Path, model: dict) -> dict:
    root = Path(root).resolve()
    require(model["id"] == MODEL_ID and model["revision"], "Qwen3.5 model ID/revision is not pinned")
    receipt_path = root / RECEIPT
    require(receipt_path.is_file(), "Qwen3.5 weight provenance receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(receipt.get("schema_version") == 1 and
            receipt.get("method") == "huggingface_pinned_force_download" and
            receipt.get("model_id") == MODEL_ID and receipt.get("revision") == model["revision"],
            "Qwen3.5 weight source differs from profile")
    files = model_files(root)
    validate_layout(root, files)
    expected = {}
    for entry in receipt.get("files", []):
        path = inside(root, entry["path"])
        require(path not in expected, "duplicate Qwen3.5 receipt entry")
        expected[path] = entry
    require(set(files) == set(expected), "Qwen3.5 weight file set differs from receipt")
    for path in files:
        entry = expected[path]
        require(path.stat().st_size == entry["bytes"] and file_hash(path) == entry["sha256"],
                f"Qwen3.5 weight content changed: {path.name}")
    return {"model_id": MODEL_ID, "revision": model["revision"], "method": receipt["method"],
            "receipt_sha256": fingerprint(receipt), "files": receipt["files"]}


def fetch_qwen35_weights(root: Path, model: dict) -> dict:
    require(model["id"] == MODEL_ID and model["revision"],
            "pin Qwen3.5 revision before downloading")
    root = Path(root).resolve()
    if (root / RECEIPT).exists():
        return verify_qwen35_weights(root, model)
    require(not root.exists() or not any(root.iterdir()),
            "Qwen3.5 weight directory must be empty when provenance is absent")
    from huggingface_hub import snapshot_download

    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=MODEL_ID, revision=model["revision"], local_dir=str(root),
                      force_download=True, allow_patterns=[f"*{suffix}" for suffix in sorted(ALLOWED_SUFFIXES)])
    files = model_files(root)
    validate_layout(root, files)
    receipt = {"schema_version": 1, "method": "huggingface_pinned_force_download",
               "model_id": MODEL_ID, "revision": model["revision"],
               "files": [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                          "sha256": file_hash(path)} for path in files]}
    write_json(root / RECEIPT, receipt)
    return verify_qwen35_weights(root, model)
