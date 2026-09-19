"""Build an independent Qwen3.5-4B test package; the Qwen2.5-VL baseline is untouched."""
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

PACKAGE_ID = "cuhkx-qwen35-4b-test-v1"
MANIFEST = "qwen35_bundle_manifest.json"
PREFIX = "qwen35_repo/"
CODE_FILES = (
    "pyproject.toml", "src/cuhkx/__init__.py", "src/cuhkx/config.py", "src/cuhkx/cli.py",
    "src/cuhkx/release_security.py",
    "src/cuhkx/data/__init__.py", "src/cuhkx/data/inputs.py", "src/cuhkx/data/validate.py",
    "src/cuhkx/inference/__init__.py", "src/cuhkx/inference/prompt.py", "src/cuhkx/inference/qwen35.py",
    "src/cuhkx/inference/qwen35_weights.py", "src/cuhkx/inference/runner.py", "src/cuhkx/inference/storage.py",
    "src/cuhkx/evaluation/__init__.py", "src/cuhkx/evaluation/metric.py",
    "src/cuhkx/submission/__init__.py", "src/cuhkx/submission/validator.py", "src/cuhkx/submission/export.py",
    "configs/baseline.yaml", "configs/qwen35_4b.yaml", "configs/datasets.yaml", "configs/submission.yaml",
    "requirements/bootstrap.lock.txt", "requirements/cpu.in", "requirements/cpu.lock.txt",
    "requirements/qwen35.in", "requirements/qwen35.lock.txt",
    "requirements/README.md", "docs/qwen35_4b.md", "notebooks/qwen35-4b-test.ipynb",
)
DATA_FILES = {"data/qa/test.csv", "data/qa/pilot.csv", "data/qa/sample_submission.csv"}
DATA_PREFIXES = ("data/frames/test/", "data/frames/pilot/")


def encoded(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def safe_relative(name: str) -> None:
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name
            or path.as_posix() != name):
        raise ValueError(f"unsafe package path: {name}")


def collect(project: Path) -> dict[str, bytes]:
    project = project.resolve()
    payloads: dict[str, bytes] = {}
    for relative in CODE_FILES:
        path = (project / relative).resolve()
        if not path.is_file() or (project / relative).is_symlink() or not path.is_relative_to(project):
            raise ValueError(f"missing/unsafe package source: {relative}")
        payloads[PREFIX + relative] = path.read_bytes()
    payloads[PREFIX + "README.md"] = (
        "# Qwen3.5-4B IR4 test lane\n\n"
        "Independent model comparison. Use notebooks/qwen35-4b-test.ipynb.\n"
        "The Qwen2.5-VL-7B baseline package and results are not included or modified.\n"
    ).encode("utf-8")
    source_manifest = json.loads((project / "data/asset_manifest.json").read_text(encoding="utf-8"))
    selected: list[dict] = []
    for entry in source_manifest["files"]:
        relative = entry["path"]
        if relative not in DATA_FILES and not relative.startswith(DATA_PREFIXES):
            continue
        safe_relative(relative)
        path = (project / relative).resolve()
        if not path.is_file() or not path.is_relative_to(project):
            raise ValueError(f"missing/unsafe data asset: {relative}")
        content = path.read_bytes()
        if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"data asset changed: {relative}")
        key = PREFIX + relative
        if key in payloads:
            raise ValueError(f"duplicate package asset: {relative}")
        payloads[key] = content
        selected.append(entry)
    if not DATA_FILES <= {entry["path"] for entry in selected}:
        raise ValueError("test/pilot QA or template missing")
    filtered_manifest = {"schema_version": 1, "baseline_id": "ir4_7b_v1",
                         "scope": "Qwen3.5 test inputs only", "files": selected}
    payloads[PREFIX + "data/asset_manifest.json"] = encoded(filtered_manifest)
    manifest = {
        "schema_version": 1,
        "package_id": PACKAGE_ID,
        "model_id": "Qwen/Qwen3.5-4B",
        "model_revision": None,
        "transformers_version": "5.17.0",
        "training_included": False,
        "baseline_reference": "Qwen2.5-VL-7B IR4 baseline remains outside this package",
        "files": [{"path": name, "bytes": len(content),
                   "sha256": hashlib.sha256(content).hexdigest()}
                  for name, content in sorted(payloads.items())],
    }
    payloads[MANIFEST] = encoded(manifest)
    return payloads


def extract_verified(archive: Path, destination: Path) -> dict:
    manifest, _ = extract_verified_archive(
        archive, destination, MANIFEST,
        lambda value: value.get("schema_version") == 1 and value.get("package_id") == PACKAGE_ID,
        validate_name=safe_relative,
    )
    return manifest


def build(project: Path, output: Path, python: Path = Path(sys.executable), run_checks: bool = True) -> dict:
    project, output, python = project.resolve(), output.resolve(), python.resolve()
    if output.exists():
        raise ValueError("Qwen3.5 package already exists; choose a new output path")
    payloads = collect(project)
    with tempfile.TemporaryDirectory(prefix="cuhkx_qwen35_package_") as temporary:
        stage = Path(temporary)
        archive = stage / (PACKAGE_ID + ".zip")
        with zipfile.ZipFile(archive, "w") as zipped:
            for name, content in sorted(payloads.items()):
                info = zipfile.ZipInfo(name, date_time=(2026, 9, 11, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                zipped.writestr(info, content)
        unpacked = stage / "unpacked"
        manifest = extract_verified(archive, unpacked)
        checks = {}
        if run_checks:
            repo = unpacked / "qwen35_repo"
            env = {**os.environ, "PYTHONPATH": str(repo / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
            for dataset in ("test", "pilot"):
                process = subprocess.run(
                    [str(python), "-B", "-m", "cuhkx.cli", "check", "--profile", "qwen35",
                     "--dataset", dataset, "--project-root", str(repo)],
                    cwd=stage, env=env, capture_output=True, text=True, encoding="utf-8",
                )
                if process.returncode:
                    raise RuntimeError(f"unpacked {dataset} check failed: {process.stderr}")
                checks[dataset] = json.loads(process.stdout)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as handle:
            handle.write(archive.read_bytes())
    return {"status": "PASS", "package_id": manifest["package_id"], "archive": str(output),
            "bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(payloads[MANIFEST]).hexdigest(),
            "manifest_files": len(manifest["files"]), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/cloud/qwen35_4b_test_v1.zip")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    print(json.dumps(build(PROJECT, args.output, args.python), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
