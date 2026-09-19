"""Adapter provenance and text-decoder-only LoRA targets; CPU validation only."""
import json
import re
from pathlib import Path

from cuhkx.config import require
from cuhkx.data.validate import fingerprint
from cuhkx.inference.storage import write_json
from cuhkx.inference.weights import file_hash


RECEIPT = "cuhkx_adapter.json"
TARGET = re.compile(r"(?:model\.)?(?:language_model\.)?layers\.\d+\.self_attn\.(?:q_proj|v_proj)$")


def text_targets(names):
    selected = sorted(name for name in names if TARGET.fullmatch(name) and
                      not any(part in name.split(".") for part in ("visual", "vision_model", "vision_tower", "merger")))
    require(selected and any(name.endswith(".q_proj") for name in selected) and
            any(name.endswith(".v_proj") for name in selected), "language decoder q_proj/v_proj modules not found")
    layers = {}
    for name in selected:
        layer, projection = name.rsplit(".", 1)
        layers.setdefault(layer, set()).add(projection)
    require(all(values == {"q_proj", "v_proj"} for values in layers.values()), "each selected decoder layer requires both q_proj and v_proj")
    return selected


def targets_for_model(names, model_id):
    """Select and validate LoRA targets for the model family being trained."""
    if model_id == "Qwen/Qwen3.5-4B":
        from cuhkx.training.qwen35_support import qwen35_text_targets
        return qwen35_text_targets(names)
    return text_targets(names)


def save_adapter_receipt(root, base_source, training_signature, data_signature, lora, targets, purpose):
    root = Path(root)
    config_path = root / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = base_source["model_id"]
    config["revision"] = base_source["revision"]
    write_json(config_path, config)
    value = {"schema_version": 1, "base_model_id": base_source["model_id"], "base_revision": base_source["revision"],
             "base_receipt_sha256": base_source["receipt_sha256"], "training_signature": training_signature,
             "data_signature": data_signature, "lora": lora, "target_modules": targets, "purpose": purpose,
             "files": [{"path": name, "bytes": (root/name).stat().st_size, "sha256": file_hash(root/name)}
                       for name in ("adapter_config.json", "adapter_model.safetensors")]}
    write_json(root / RECEIPT, value)
    return value


def verify_adapter(root, base_source):
    root = Path(root).resolve()
    value = json.loads((root / RECEIPT).read_text(encoding="utf-8"))
    require(value["schema_version"] == 1 and value["purpose"] in ("sft", "smoke"), "unsupported adapter receipt")
    require(value["base_model_id"] == base_source["model_id"] and value["base_revision"] == base_source["revision"] and
            value["base_receipt_sha256"] == base_source["receipt_sha256"], "adapter base model differs")
    entries = value["files"]
    require(len(entries) == 2 and {e["path"] for e in entries} == {"adapter_config.json", "adapter_model.safetensors"}, "adapter file set differs")
    for e in entries:
        path = root / e["path"]
        require(not path.is_symlink() and path.is_file() and path.stat().st_size == e["bytes"] and file_hash(path) == e["sha256"], "adapter file hash mismatch")
    cfg = json.loads((root / "adapter_config.json").read_text(encoding="utf-8"))
    require(cfg["peft_type"] == "LORA" and cfg["task_type"] == "CAUSAL_LM" and cfg.get("bias") == "none", "unsupported adapter type")
    require(not cfg.get("modules_to_save") and not cfg.get("use_dora"), "unexpected trainable adapter modules")
    targets = targets_for_model(value["target_modules"], base_source["model_id"])
    saved_targets = cfg["target_modules"]
    # PEFT 0.17.x may compact a long list of fully-qualified target paths to
    # the minimal suffixes ["q_proj", "v_proj"] when serializing the config.
    # The receipt retains the exact module paths selected from the live model;
    # accept that canonical PEFT representation while rejecting broader targets.
    if sorted(saved_targets) != targets:
        allowed_suffixes = {name.rsplit(".", 1)[-1] for name in targets}
        require(isinstance(saved_targets, list) and set(saved_targets) <= allowed_suffixes and
                all(any(name.endswith("." + suffix) for name in targets) for suffix in saved_targets),
                "adapter targets do not match the selected text decoder modules")
    require(cfg["r"] == value["lora"]["r"] and cfg["lora_alpha"] == value["lora"]["alpha"] and
            cfg["lora_dropout"] == value["lora"]["dropout"], "adapter LoRA config mismatch")
    require(cfg["base_model_name_or_path"] == base_source["model_id"] and cfg["revision"] == base_source["revision"], "adapter config base mismatch")
    return {"receipt_sha256": fingerprint(value), **value}
