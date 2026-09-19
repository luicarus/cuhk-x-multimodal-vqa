from __future__ import annotations

import csv
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from cuhkx.cli import main
from cuhkx.config import InputError, load_config, load_qwen35_config
from cuhkx.inference.qwen35 import Qwen35Backend, token_constraint
from cuhkx.inference.qwen35_weights import (
    RECEIPT,
    file_hash,
    verify_qwen35_weights,
)


PROJECT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3.5-4B"
REVISION = "a" * 40


def test_qwen35_profile_is_separate_but_keeps_ir4_protocol():
    config = load_qwen35_config()
    assert config["baseline"]["baseline_id"] == "qwen35_4b_v1"
    assert config["baseline"]["model"]["id"] == MODEL
    assert config["baseline"]["model"]["revision"] is None
    assert config["baseline"]["frames"] == load_config()["baseline"]["frames"]
    assert config["baseline"]["generation"]["enable_thinking"] is False
    assert config["baseline"]["model"]["transformers"]["version"] == "5.17.0"


@pytest.mark.parametrize(
    "field,value",
    [("id", "Qwen/Qwen2.5-VL-7B-Instruct"), ("architecture", "qwen2_5"),
     ("quantization", "nf4"), ("transformers", {"version": "4.57.6", "source": "pypi"})],
)
def test_qwen35_profile_rejects_baseline_values(tmp_path, field, value):
    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    for name in ("baseline.yaml", "datasets.yaml", "submission.yaml", "qwen35_4b.yaml"):
        source = PROJECT / "configs" / name
        (root / "configs" / name).write_bytes(source.read_bytes())
    # Reuse the real data as an explicit root; only profile is mutated.
    path = root / "configs/qwen35_4b.yaml"
    profile = yaml.safe_load(path.read_text())
    profile["model"][field] = value
    path.write_text(yaml.safe_dump(profile))
    with pytest.raises(InputError):
        load_qwen35_config(root, PROJECT / "data")


class Tokens(list):
    def __getitem__(self, key):
        value = super().__getitem__(key)
        return Tokens(value) if isinstance(key, slice) else value

    def tolist(self):
        return list(self)


def test_qwen35_constraint_supports_multiple_eos_tokens():
    class Tokenizer:
        eos_token_id = [900, 901]

        def encode(self, value, **kwargs):
            return [ord(char) for char in value]

    allowed = token_constraint(Tokenizer(), ("A", "AB", "B"), 1, 8)
    assert allowed(0, Tokens([999])) == [65, 66]
    assert allowed(0, Tokens([999, 65])) == [66, 900, 901]
    assert allowed(0, Tokens([999, 65, 66])) == [900, 901]
    with pytest.raises(InputError, match="illegal"):
        allowed(0, Tokens([999, 67]))


def test_qwen35_constraint_can_use_model_eos_after_answer():
    class Tokenizer:
        # Qwen3.5's tokenizer EOS and model generation EOS are different IDs.
        eos_token_id = 248046

        def encode(self, value, **kwargs):
            return [ord(char) for char in value]

    allowed = token_constraint(Tokenizer(), ("A", "B"), 1, 8, eos_token_id=248044)
    assert allowed(0, Tokens([999, 65])) == [248044]


def test_qwen35_backend_uses_official_multimodal_processor_and_direct_answers(monkeypatch):
    seen = {}

    class InputIds(list):
        shape = (1, 2)

    class Batch(dict):
        def __init__(self):
            super().__init__(input_ids=InputIds([Tokens([111, 222])]))
            self.input_ids = self["input_ids"]

        def to(self, device):
            seen["device"] = device
            return self

    class Tokenizer:
        eos_token_id = [0, 1]

        def encode(self, value, **kwargs):
            return [65] if value == "A" else [66]

    class Processor:
        tokenizer = Tokenizer()

        def apply_chat_template(self, messages, **kwargs):
            seen["messages"] = messages
            seen["template_kwargs"] = kwargs
            return Batch()

        def batch_decode(self, rows, **kwargs):
            seen["decoded_rows"] = rows
            return ["A"]

    class Torch:
        inference_mode = staticmethod(nullcontext)

    def generate(**kwargs):
        seen["generation"] = kwargs
        assert kwargs["prefix_allowed_tokens_fn"](0, Tokens([111, 222])) == [65, 66]
        assert kwargs["prefix_allowed_tokens_fn"](0, Tokens([111, 222, 65])) == [900]
        return [Tokens([111, 222, 65, 900])]

    backend = Qwen35Backend.__new__(Qwen35Backend)
    backend.processor = Processor()
    backend.model = SimpleNamespace(
        generate=generate,
        generation_config=SimpleNamespace(eos_token_id=900),
    )
    backend.torch = Torch()
    backend.device = "cuda:0"
    backend.image_size = 280
    images = [Image.new("RGB", (448, 448)) for _ in range(4)]
    try:
        assert backend.generate(images, "prompt", allowed_outputs=("A", "B"), max_new_tokens=8) == "A"
        content = seen["messages"][0]["content"]
        assert len(content) == 5 and content[-1]["text"] == "prompt"
        assert all(item["resized_height"] == item["resized_width"] == 280 for item in content[:4])
        assert seen["template_kwargs"]["enable_thinking"] is False
        assert "chat_template_kwargs" not in seen["template_kwargs"]
        assert seen["generation"]["do_sample"] is False and seen["generation"]["num_beams"] == 1
        assert seen["generation"]["eos_token_id"] == 900
    finally:
        for image in images:
            image.close()


def test_qwen35_weight_receipt_requires_architecture_and_hashes(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
    }))
    (tmp_path / "tokenizer_config.json").write_text("{}")
    (tmp_path / "preprocessor_config.json").write_text("{}")
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"fixture")
    receipt = {
        "schema_version": 1, "method": "huggingface_pinned_force_download", "model_id": MODEL,
        "revision": REVISION, "files": [
            {"path": "config.json", "bytes": (tmp_path / "config.json").stat().st_size, "sha256": file_hash(tmp_path / "config.json")},
            {"path": "tokenizer_config.json", "bytes": (tmp_path / "tokenizer_config.json").stat().st_size, "sha256": file_hash(tmp_path / "tokenizer_config.json")},
            {"path": "preprocessor_config.json", "bytes": (tmp_path / "preprocessor_config.json").stat().st_size, "sha256": file_hash(tmp_path / "preprocessor_config.json")},
            {"path": "model.safetensors", "bytes": weight.stat().st_size, "sha256": file_hash(weight)},
        ],
    }
    (tmp_path / RECEIPT).write_text(json.dumps(receipt))
    result = verify_qwen35_weights(tmp_path, {"id": MODEL, "revision": REVISION})
    assert result["model_id"] == MODEL
    weight.write_bytes(b"tampered")
    with pytest.raises(InputError, match="changed"):
        verify_qwen35_weights(tmp_path, {"id": MODEL, "revision": REVISION})


def test_qwen35_weight_policy_rejects_python_and_remote_code():
    from cuhkx.inference.qwen35_weights import ALLOWED_SUFFIXES

    assert ".py" not in ALLOWED_SUFFIXES
    assert load_qwen35_config()["baseline"]["model"]["trust_remote_code"] is False
    assert "trust_remote_code=True" not in (PROJECT / "src/cuhkx/inference/qwen35.py").read_text(encoding="utf-8")
    trainer = (PROJECT / "src/cuhkx/training/trainer.py").read_text(encoding="utf-8")
    assert "trust_remote_code=qwen35" not in trainer


def test_qwen35_weights_reject_dynamic_model_code(tmp_path):
    config = {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
              "auto_map": {"AutoModelForMultimodalLM": "modeling_qwen35.CustomModel"}}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "modeling_qwen35.py").write_text("raise RuntimeError('must never import')", encoding="utf-8")
    files = [path for path in tmp_path.iterdir() if path.name != RECEIPT]
    receipt = {"schema_version": 1, "method": "huggingface_pinned_force_download",
               "model_id": MODEL, "revision": REVISION,
               "files": [{"path": path.name, "bytes": path.stat().st_size,
                           "sha256": file_hash(path)} for path in files]}
    (tmp_path / RECEIPT).write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(InputError, match="dynamic|unexpected"):
        verify_qwen35_weights(tmp_path, {"id": MODEL, "revision": REVISION})


def test_qwen35_weights_reject_auto_map_without_python_files(tmp_path):
    config = {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
              "auto_map": {"AutoModelForMultimodalLM": "unlisted_module.CustomModel"}}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"weights")
    files = [tmp_path / name for name in ("config.json", "tokenizer_config.json",
                                          "preprocessor_config.json", "model.safetensors")]
    receipt = {"schema_version": 1, "method": "huggingface_pinned_force_download",
               "model_id": MODEL, "revision": REVISION,
               "files": [{"path": path.name, "bytes": path.stat().st_size,
                           "sha256": file_hash(path)} for path in files]}
    (tmp_path / RECEIPT).write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(InputError, match="dynamic"):
        verify_qwen35_weights(tmp_path, {"id": MODEL, "revision": REVISION})


def test_qwen35_check_is_cpu_only_and_predict_requires_pin(capsys):
    assert main(["check", "--profile", "qwen35", "--dataset", "pilot"]) == 0
    output = capsys.readouterr().out
    assert '"gpu_model_loaded": false' in output
    assert main(["predict", "--profile", "qwen35", "--dataset", "test", "--run-id", "q",
                 "--weights-dir", str(PROJECT / "missing")]) == 2
    assert "revision" in capsys.readouterr().err


def test_qwen35_files_keep_baseline_model_separate():
    baseline = PROJECT / "notebooks/cuhk-x-base7b.ipynb"
    assert baseline.is_file()
    assert (PROJECT / "configs/baseline.yaml").read_text(encoding="utf-8").find("Qwen/Qwen2.5-VL-7B-Instruct") >= 0
