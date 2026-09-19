"""Single baseline configuration, with no model or GPU imports."""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


class InputError(ValueError):
    """Configuration or cached input cannot safely identify the requested run."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InputError(message)


class UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader: UniqueLoader, node: yaml.MappingNode) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        require(isinstance(key, str) and key not in result, f"duplicate/invalid YAML key: {key!r}")
        result[key] = loader.construct_object(value_node)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueLoader)
    require(isinstance(value, dict), f"expected YAML mapping: {path}")
    return value


def keys(value: dict, expected: set[str], label: str) -> None:
    require(isinstance(value, dict) and set(value) == expected, f"unexpected or missing keys in {label}")


def same(value: Any, expected: Any, label: str) -> None:
    # JSON comparison distinguishes true from 1, unlike Python equality.
    require(json.dumps(value, sort_keys=True) == json.dumps(expected, sort_keys=True), f"unsupported {label}: {value!r}")


def inside(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and bool(relative) and "\\" not in relative, "expected relative POSIX path")
    parts = PurePosixPath(relative)
    require(not parts.is_absolute() and ".." not in parts.parts and ":" not in relative, f"unsafe path: {relative}")
    resolved = (root / relative).resolve()
    require(resolved.is_relative_to(root.resolve()), f"path escapes root: {relative}")
    return resolved


def load_config(project_root: Path | None = None, data_root: Path | None = None,
                *, require_revision: bool = False) -> dict:
    project = (project_root or Path(__file__).resolve().parents[2]).resolve()
    require((project / "configs/baseline.yaml").is_file(), "project configs missing; provide --project-root")
    data = (data_root or project / "data").resolve()
    baseline = read_yaml(project / "configs/baseline.yaml")
    keys(baseline, {"schema_version", "baseline_id", "model", "frames", "prompt_version", "generation", "runtime"}, "baseline")
    same(baseline["schema_version"], 1, "baseline schema")
    same(baseline["baseline_id"], "ir4_7b_v1", "baseline ID")
    keys(baseline["model"], {"id", "revision", "quantization", "double_quant", "compute_dtype", "attention"}, "model")
    revision = baseline["model"]["revision"]
    require(revision is None or (isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) is not None), "revision must be null or a 40-character lowercase commit SHA")
    require(not require_revision or revision is not None, "model revision is not pinned; verify actual weights before inference")
    same({k: v for k, v in baseline["model"].items() if k != "revision"}, {
        "id": "Qwen/Qwen2.5-VL-7B-Instruct", "quantization": "nf4", "double_quant": True,
        "compute_dtype": "float16", "attention": "sdpa"}, "model protocol")
    same(baseline["frames"], {"modality": "IR", "protocol_version": "uniform_time_v1",
        "config_hash": "520837f5b798f45a", "cache_num_frames": 8, "input_num_frames": 4,
        "selection": "temporal_bin_centers", "indices": [1, 3, 5, 7], "cache_image_size": 448,
        "input_image_size": 280, "cache_resize": "letterbox"}, "frame protocol")
    same(baseline["prompt_version"], "cuhkx_mcq_v2", "prompt")
    same(baseline["generation"], {"do_sample": False, "num_beams": 1, "max_new_tokens": 8,
                               "constrained_decoding": True}, "generation")
    runtime = baseline["runtime"]
    keys(runtime, {"batch_size", "gpu_memory_mib", "cpu_memory_gib"}, "runtime")
    same(runtime["batch_size"], 1, "batch size")
    for field, minimum in (("gpu_memory_mib", 1024), ("cpu_memory_gib", 4)):
        require(type(runtime[field]) is int and runtime[field] >= minimum, f"invalid {field}")
    datasets = read_yaml(project / "configs/datasets.yaml")
    keys(datasets, {"schema_version", "asset_manifest", "datasets"}, "datasets")
    same(datasets["schema_version"], 1, "dataset schema")
    keys(datasets["datasets"], {"test", "pilot"}, "dataset names")
    for name, binding in datasets["datasets"].items():
        keys(binding, {"qa", "expected_qa", "split", "frames_root", "frame_index"}, name)
        require(type(binding["expected_qa"]) is int and binding["expected_qa"] > 0, "invalid expected QA count")
        same(binding["split"], "test" if name == "test" else "train", "dataset split")
        for field in ("qa", "frames_root", "frame_index"):
            inside(data, binding[field])
    inside(data, datasets["asset_manifest"])
    submission = read_yaml(project / "configs/submission.yaml")
    keys(submission, {"schema_version", "columns", "require_template_id_order", "require_unique_ids",
                      "allow_extra_rows", "allow_extra_columns", "test_qa", "template", "references"}, "submission")
    for field, expected in {"schema_version": 1, "columns": ["qa_id", "prediction"],
                            "require_template_id_order": True, "require_unique_ids": True,
                            "allow_extra_rows": False, "allow_extra_columns": False}.items():
        same(submission[field], expected, field)
    keys(submission["references"], {"pilot"}, "references")
    for value in (submission["test_qa"], submission["template"], submission["references"]["pilot"]):
        inside(data, value)
    same(submission["test_qa"], datasets["datasets"]["test"]["qa"], "submission test binding")
    return {"project_root": str(project), "data_root": str(data), "baseline": baseline,
            "datasets": datasets, "submission": submission}


def load_qwen35_config(project_root: Path | None = None, data_root: Path | None = None,
                       *, require_revision: bool = False) -> dict:
    """Load Qwen3.5 while reusing the existing dataset and submission contracts."""
    common = load_config(project_root, data_root, require_revision=False)
    project = Path(common["project_root"])
    profile_path = project / "configs/qwen35_4b.yaml"
    require(profile_path.is_file(), "Qwen3.5 profile is missing")
    profile = read_yaml(profile_path)
    keys(profile, {"schema_version", "baseline_id", "model", "frames", "prompt_version", "generation", "runtime"}, "qwen35 profile")
    same(profile["schema_version"], 1, "Qwen3.5 schema")
    same(profile["baseline_id"], "qwen35_4b_v1", "Qwen3.5 baseline ID")
    keys(profile["model"], {"id", "revision", "architecture", "quantization", "compute_dtype", "trust_remote_code", "transformers"}, "Qwen3.5 model")
    same(profile["model"]["id"], "Qwen/Qwen3.5-4B", "Qwen3.5 model ID")
    same(profile["model"]["architecture"], "qwen3_5", "Qwen3.5 architecture")
    same(profile["model"]["quantization"], "none", "Qwen3.5 quantization")
    same(profile["model"]["compute_dtype"], "float16", "Qwen3.5 dtype")
    same(profile["model"]["trust_remote_code"], False, "Qwen3.5 remote-code policy")
    keys(profile["model"]["transformers"], {"version", "source"}, "Qwen3.5 Transformers")
    require(isinstance(profile["model"]["transformers"]["version"], str) and
            re.fullmatch(r"5\.\d+\.\d+", profile["model"]["transformers"]["version"]),
            "Qwen3.5 Transformers version must be a release pin")
    same(profile["model"]["transformers"]["source"], "pypi", "Qwen3.5 Transformers source")
    revision = profile["model"]["revision"]
    require(revision is None or (isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision)),
            "Qwen3.5 revision must be null or a 40-character lowercase commit SHA")
    require(not require_revision or revision is not None, "Qwen3.5 model revision is not pinned")
    common_baseline = common["baseline"]
    same(profile["frames"], common_baseline["frames"], "Qwen3.5 frame protocol")
    same(profile["prompt_version"], common_baseline["prompt_version"], "Qwen3.5 prompt")
    keys(profile["generation"], {"do_sample", "num_beams", "max_new_tokens", "constrained_decoding", "enable_thinking"}, "Qwen3.5 generation")
    same(profile["generation"], {"do_sample": False, "num_beams": 1, "max_new_tokens": 8,
                                  "constrained_decoding": True, "enable_thinking": False}, "Qwen3.5 generation")
    same(profile["runtime"], common_baseline["runtime"], "Qwen3.5 runtime")
    common["baseline"] = profile
    # The frames were generated under the IR4 data protocol before this
    # model-comparison profile existed. Keep that data identity separate from
    # the model identity used in run signatures.
    common["asset_baseline_id"] = "ir4_7b_v1"
    return common
