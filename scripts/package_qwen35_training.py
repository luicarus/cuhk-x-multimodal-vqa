"""Build the independent full-data Qwen3.5-4B QLoRA release (vLLM dual-GPU)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from cuhkx.release_security import extract_verified_archive

PACKAGE_ID = "cuhkx-qwen35-4b-qlora-vllm-v1"
MANIFEST = "qwen35_training_bundle_manifest.json"
PREFIX = "qwen35_training_repo/"
NOTEBOOK = "notebooks/qwen35-4b-qlora-vllm.ipynb"
ARCHIVE_NAME = "qwen35_4b_qlora.zip"
INFERENCE_ENGINE = "vllm_0.24.0_tensor_parallel"
DATA_FILES = {
    "data/qa/test.csv", "data/qa/pilot.csv", "data/qa/sample_submission.csv",
    "data/references/pilot_answers.csv", "data/references/training_qa.csv",
    "data/references/folds/subject_grouped_v1/qa_folds.csv",
}
FRAME_PREFIXES = (
    "data/frames/test/", "data/frames/pilot/", "data/frames/fold_0/",
    "data/frames/fold_1/", "data/frames/fold_2/", "data/frames/fold_3/",
    "data/frames/fold_4/",
)
FILES = (
    "pyproject.toml", "src/cuhkx/__init__.py", "src/cuhkx/config.py", "src/cuhkx/cli.py",
    "src/cuhkx/release_security.py",
    "src/cuhkx/data/__init__.py", "src/cuhkx/data/inputs.py", "src/cuhkx/data/validate.py",
    "src/cuhkx/inference/__init__.py", "src/cuhkx/inference/prompt.py", "src/cuhkx/inference/qwen.py",
    "src/cuhkx/inference/qwen35.py", "src/cuhkx/inference/qwen35_weights.py",
    "src/cuhkx/inference/qwen35_vllm.py",
    "src/cuhkx/inference/runner.py", "src/cuhkx/inference/storage.py", "src/cuhkx/inference/weights.py",
    "src/cuhkx/evaluation/__init__.py", "src/cuhkx/evaluation/metric.py",
    "src/cuhkx/submission/__init__.py", "src/cuhkx/submission/validator.py", "src/cuhkx/submission/export.py",
    "src/cuhkx/training/__init__.py", "src/cuhkx/training/dataset.py", "src/cuhkx/training/collator.py",
    "src/cuhkx/training/qwen35_support.py", "src/cuhkx/training/trainer.py",
    "src/cuhkx/training/adapter.py", "src/cuhkx/training/evaluate.py",
    "configs/baseline.yaml", "configs/qwen35_4b.yaml", "configs/training_qwen35.yaml",
    "configs/datasets.yaml", "configs/submission.yaml",
    "requirements/bootstrap.lock.txt", "requirements/cpu.in", "requirements/cpu.lock.txt", "requirements/qwen35.in",
    "requirements/qwen35.lock.txt", "requirements/train_qwen35.in", "requirements/train_qwen35.lock.txt",
    "requirements/README.md", "docs/qwen35_4b.md", "docs/qwen35_training.md", NOTEBOOK,
)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or ":" in name or "\\" in name
            or path.as_posix() != name):
        raise ValueError(f"unsafe package path: {name}")


def collect(project: Path):
    project = project.resolve()
    payloads = {}
    for relative in FILES:
        safe_name(relative)
        path = (project / relative).resolve()
        if not path.is_relative_to(project) or path.is_symlink() or not path.is_file():
            raise ValueError(f"required source is missing or unsafe: {relative}")
        payloads[PREFIX + relative] = path.read_bytes()

    assets = json.loads((project / "data/asset_manifest.json").read_text(encoding="utf-8"))
    selected = {}
    for entry in assets["files"]:
        relative = entry["path"]
        if relative not in DATA_FILES and not relative.startswith(FRAME_PREFIXES):
            continue
        safe_name(relative)
        path = (project / relative).resolve()
        if not path.is_relative_to(project) or not path.is_file():
            raise ValueError(f"registered asset is missing: {relative}")
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if len(content) != entry["bytes"] or digest != entry["sha256"]:
            raise ValueError(f"asset differs from manifest: {relative}")
        selected[relative] = entry
        payloads[PREFIX + relative] = content
    if not DATA_FILES <= set(selected):
        raise ValueError("required QA/fold references are missing")
    assets = {"schema_version": assets["schema_version"], "baseline_id": "ir4_7b_v1",
              "files": [selected[name] for name in sorted(selected)]}
    payloads[PREFIX + "data/asset_manifest.json"] = encoded(assets)
    payloads[PREFIX + "README.md"] = (
        "# Independent Qwen3.5-4B QLoRA release (vLLM dual-GPU)\n\n"
        f"Use {NOTEBOOK}. The package embeds the complete five-fold IR8 cache and "
        "contains no model weights or raw videos. Adapter reload checks, dev/confirm "
        "evaluation and test inference run on vLLM 0.24.0 with tensor parallelism "
        "across both Kaggle T4 GPUs; training stays on Transformers.\n"
    ).encode("utf-8")
    manifest = {
        "schema_version": 1, "package_id": PACKAGE_ID,
        "model_id": "Qwen/Qwen3.5-4B", "model_revision": None,
        "training_cache_mode": "embedded_complete",
        # Declared so the notebook can refuse a package built for another engine.
        "inference_engine": INFERENCE_ENGINE,
        "tensor_parallel_size": 2,
        "files": [{"path": name, "bytes": len(content),
                   "sha256": hashlib.sha256(content).hexdigest()}
                  for name, content in sorted(payloads.items())],
    }
    payloads[MANIFEST] = encoded(manifest)
    return payloads


def extract_verified(archive: Path, destination: Path):
    manifest, _ = extract_verified_archive(
        archive, destination, MANIFEST,
        lambda value: value.get("schema_version") == 1 and value.get("package_id") == PACKAGE_ID,
        validate_name=safe_name,
    )
    return manifest


def build(project: Path, output: Path, python=sys.executable, run_checks=True):
    project, output = project.resolve(), output.resolve()
    if output.exists():
        raise ValueError("release already exists; choose a new output filename")
    payloads = collect(project)
    with tempfile.TemporaryDirectory(prefix="cuhkx_qwen35_training_release_") as temporary:
        stage = Path(temporary)
        archive = stage / (PACKAGE_ID + ".zip")
        with zipfile.ZipFile(archive, "w") as zipped:
            for name, content in sorted(payloads.items()):
                info = zipfile.ZipInfo(name, (2026, 9, 11, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                zipped.writestr(info, content)
        unpacked = stage / "unpacked"
        manifest = extract_verified(archive, unpacked)
        checks = {}
        if run_checks:
            repo = unpacked / "qwen35_training_repo"
            env = {**os.environ, "PYTHONPATH": str(repo / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
            commands = (
                ("test", ["check", "--profile", "qwen35", "--dataset", "test"]),
                ("pilot", ["check", "--profile", "qwen35", "--dataset", "pilot"]),
                ("training", ["training-check", "--profile", "qwen35",
                               "--training-config", "configs/training_qwen35.yaml"]),
            )
            for name, args in commands:
                process = subprocess.run(
                    [str(python), "-B", "-m", "cuhkx.cli", *args,
                     "--project-root", str(repo)], cwd=stage, env=env,
                    capture_output=True, text=True, encoding="utf-8",
                )
                if process.returncode != 0:
                    raise RuntimeError(f"{name} check failed: {process.stderr or process.stdout}")
                try:
                    checks[name] = json.loads(process.stdout)
                except ValueError as error:
                    raise RuntimeError(f"{name} check did not emit JSON: {process.stdout}") from error
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as handle:
            handle.write(archive.read_bytes())
    return {"status": "PASS", "archive": str(output), "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(payloads[MANIFEST]).hexdigest(),
            "files": len(manifest["files"]), "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=PROJECT / "artifacts/cloud_training" / ARCHIVE_NAME)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--no-checks", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(PROJECT, args.output, args.python, not args.no_checks),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
