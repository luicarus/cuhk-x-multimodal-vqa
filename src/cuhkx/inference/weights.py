"""Pinned Hub downloads and offline content verification; no GPU imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cuhkx.config import inside, require
from cuhkx.data.validate import fingerprint
from cuhkx.inference.storage import write_json


RECEIPT = "cuhkx_weights.json"
ALLOWED_SUFFIXES = {".json", ".safetensors", ".txt", ".model", ".tiktoken", ".jinja"}


def file_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def model_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file()
                  and ".cache" not in path.relative_to(root).parts and path.name != RECEIPT)


def validate_layout(root: Path, files: list[Path]) -> None:
    require(files and all(path.suffix in ALLOWED_SUFFIXES for path in files), "unexpected model files (adapters and bin weights are unsupported)")
    require(all(not path.is_symlink() for path in files), "weight files must be materialized, not arbitrary snapshot symlinks")
    model_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    require(model_config.get("model_type") == "qwen2_5_vl", "model config is not Qwen2.5-VL")
    require(model_config.get("architectures") == ["Qwen2_5_VLForConditionalGeneration"], "wrong model architecture")
    require((root / "tokenizer_config.json").is_file(), "tokenizer config missing")
    weights = {p.name for p in files if p.suffix == ".safetensors"}
    index = root / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        require(bool(shards) and shards == weights, "missing or extra model weight shards")
    else:
        require(weights == {"model.safetensors"}, "model weights missing")


def verify_weights(root: Path, model: dict) -> dict:
    root = root.resolve()
    require(model["revision"] is not None, "model revision is not pinned")
    require((root / RECEIPT).is_file(), "weight provenance receipt missing; use fetch-weights for a pinned download")
    receipt = json.loads((root / RECEIPT).read_text(encoding="utf-8"))
    require(receipt.get("schema_version") == 1 and receipt.get("method") == "huggingface_pinned_force_download", "unsupported weight provenance")
    require(receipt.get("model_id") == model["id"] and receipt.get("revision") == model["revision"], "weight source differs from baseline")
    files = model_files(root)
    validate_layout(root, files)
    expected = {}
    for entry in receipt["files"]:
        path = inside(root, entry["path"])
        require(path not in expected, "duplicate weight receipt entry")
        expected[path] = entry
    require(set(files) == set(expected), "weight file set differs from receipt")
    for path in files:
        entry = expected[path]
        require(path.stat().st_size == entry["bytes"] and file_hash(path) == entry["sha256"], f"weight content changed: {path.name}")
    return {"model_id": model["id"], "revision": model["revision"], "method": receipt["method"],
            "receipt_sha256": fingerprint(receipt), "files": receipt["files"]}


def fetch_weights(root: Path, model: dict) -> dict:
    require(model["revision"] is not None, "pin the baseline model revision before downloading")
    root = root.resolve()
    if (root / RECEIPT).exists():
        return verify_weights(root, model)
    require(not root.exists() or not any(root.iterdir()), "use an empty weights directory; existing unverified files are not adopted")
    from huggingface_hub import snapshot_download

    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model["id"], revision=model["revision"], local_dir=str(root),
                      force_download=True, allow_patterns=[f"*{suffix}" for suffix in sorted(ALLOWED_SUFFIXES)])
    files = model_files(root)
    validate_layout(root, files)
    receipt = {"schema_version": 1, "method": "huggingface_pinned_force_download", "model_id": model["id"],
               "revision": model["revision"], "files": [{"path": path.relative_to(root).as_posix(),
               "bytes": path.stat().st_size, "sha256": file_hash(path)} for path in files]}
    write_json(root / RECEIPT, receipt)
    return verify_weights(root, model)
