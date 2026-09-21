from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import zipfile
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("package_qwen35", ROOT / "scripts/package_qwen35.py")
PACKAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE)


def test_qwen35_package_allowlist_and_extraction(tmp_path):
    archive = tmp_path / "qwen35.zip"
    result = PACKAGE.build(ROOT, archive, Path(sys.executable), run_checks=True)
    assert result["status"] == "PASS"
    assert result["package_id"] == PACKAGE.PACKAGE_ID
    with zipfile.ZipFile(archive) as zipped:
        manifest = json.loads(zipped.read(PACKAGE.MANIFEST))
        assert manifest["model_id"] == "Qwen/Qwen3.5-4B"
        assert manifest["model_revision"] is None
        assert manifest["training_included"] is False
        assert manifest["inference_engine"] == PACKAGE.INFERENCE_ENGINE
        assert manifest["tensor_parallel_size"] == 2
        names = set(zipped.namelist())
        assert "qwen35_repo/notebooks/qwen35-4b-vllm.ipynb" in names
        assert "qwen35_repo/notebooks/cuhk-x-base7b.ipynb" not in names
        assert not any("training/" in name or "artifacts/" in name for name in names)
        assert zipped.read("qwen35_repo/src/cuhkx/inference/qwen35.py") == (ROOT / "src/cuhkx/inference/qwen35.py").read_bytes()
        # The vLLM engine is the whole point of this lane, so it must ship.
        assert zipped.read("qwen35_repo/src/cuhkx/inference/qwen35_vllm.py") == (ROOT / "src/cuhkx/inference/qwen35_vllm.py").read_bytes()
        assert zipped.read("qwen35_repo/configs/qwen35_4b.yaml") == (ROOT / "configs/qwen35_4b.yaml").read_bytes()
        for entry in manifest["files"]:
            content = zipped.read(entry["path"])
            assert len(content) == entry["bytes"]
            assert hashlib.sha256(content).hexdigest() == entry["sha256"]
    unpacked = tmp_path / "unpacked"
    extracted = PACKAGE.extract_verified(archive, unpacked)
    assert extracted["package_id"] == PACKAGE.PACKAGE_ID
    assert (unpacked / "qwen35_repo/data/frames/test").is_dir()


def test_qwen35_package_rejects_existing_output_and_tampered_zip(tmp_path):
    archive = tmp_path / "qwen35.zip"
    PACKAGE.build(ROOT, archive, Path(sys.executable), run_checks=False)
    with pytest.raises(ValueError, match="already exists"):
        PACKAGE.build(ROOT, archive, Path("C:/Program Files/Python311/python.exe"), run_checks=False)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(bad, "w") as target:
        for name in source.namelist():
            target.writestr(name, b"changed" if name.endswith("qwen35.py") else source.read(name))
    with pytest.raises(ValueError, match="hash"):
        PACKAGE.extract_verified(bad, tmp_path / "bad_extract")
