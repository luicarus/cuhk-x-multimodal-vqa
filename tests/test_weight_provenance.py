import json
import sys
from types import SimpleNamespace

import pytest

from cuhkx.config import InputError
from cuhkx.inference.weights import fetch_weights, verify_weights


MODEL = {"id": "Qwen/Qwen2.5-VL-7B-Instruct", "revision": "a" * 40}


def test_pinned_download_and_offline_verification(tmp_path, monkeypatch):
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2_5_vl", "architectures": ["Qwen2_5_VLForConditionalGeneration"]}))
        (tmp_path / "tokenizer_config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_bytes(b"fixture-only")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    source = fetch_weights(tmp_path, MODEL)
    assert calls[0]["revision"] == "a" * 40 and calls[0]["force_download"] is True
    assert verify_weights(tmp_path, MODEL) == source
    assert fetch_weights(tmp_path, MODEL) == source and len(calls) == 1
    with pytest.raises(InputError, match="differs"):
        verify_weights(tmp_path, {**MODEL, "revision": "b" * 40})
    (tmp_path / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(InputError, match="changed"):
        verify_weights(tmp_path, MODEL)


def test_arbitrary_sha_directory_is_not_provenance(tmp_path):
    directory = tmp_path / MODEL["revision"]
    directory.mkdir()
    (directory / "config.json").write_text("{}")
    with pytest.raises(InputError, match="receipt missing"):
        verify_weights(directory, MODEL)
    with pytest.raises(InputError, match="empty weights"):
        fetch_weights(directory, MODEL)
