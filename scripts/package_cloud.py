"""Build the IR4 cloud ZIP from an explicit allowlist; verify and check after extraction."""
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

CODE_FILES = (
    "pyproject.toml", "src/cuhkx/__init__.py", "src/cuhkx/config.py", "src/cuhkx/cli.py",
    "src/cuhkx/release_security.py",
    "src/cuhkx/data/__init__.py", "src/cuhkx/data/inputs.py", "src/cuhkx/data/validate.py",
    "src/cuhkx/inference/__init__.py", "src/cuhkx/inference/prompt.py", "src/cuhkx/inference/qwen.py",
    "src/cuhkx/inference/runner.py", "src/cuhkx/inference/storage.py", "src/cuhkx/inference/weights.py",
    "src/cuhkx/evaluation/__init__.py", "src/cuhkx/evaluation/metric.py",
    "src/cuhkx/submission/__init__.py", "src/cuhkx/submission/validator.py", "src/cuhkx/submission/export.py",
    "src/cuhkx/training/__init__.py", "src/cuhkx/training/dataset.py", "src/cuhkx/training/collator.py",
    "src/cuhkx/training/trainer.py", "src/cuhkx/training/adapter.py", "src/cuhkx/training/evaluate.py",
    "configs/baseline.yaml", "configs/datasets.yaml", "configs/submission.yaml",
    "configs/training.yaml", "requirements/train.in", "requirements/train.lock.txt", "docs/post_training.md",
    "requirements/bootstrap.lock.txt", "requirements/cpu.in", "requirements/cpu.lock.txt",
    "requirements/cloud.in", "requirements/cloud.lock.txt",
    "requirements/README.md", "docs/cloud.md", "notebooks/cuhk-x-base7b.ipynb",
)
DATA_FILES = {"data/qa/test.csv", "data/qa/pilot.csv", "data/qa/sample_submission.csv",
              "data/references/pilot_answers.csv"}
DATA_PREFIXES = ("data/frames/test/", "data/frames/pilot/")
README = """# IR4 + 云端 7B

唯一基线：IR 缓存 8 帧，取第 2/4/6/8 张，Qwen2.5-VL-7B-Instruct NF4。

运行说明：[docs/cloud.md](docs/cloud.md)。云端入口：[Notebook](notebooks/cuhk-x-base7b.ipynb)。
将 ZIP 作为 Kaggle Dataset 输入，或使用已解压且包含 repo/ 的内容。运行 Notebook 前，须把本地打包命令输出的 `manifest_sha256` 填入 `EXPECTED_MANIFEST_SHA256`。
CPU 校验：`python -m cuhkx.cli check --dataset test`。
本包不含原视频、模型权重或历史预测。pilot 参考答案仅用于评测。
"""


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def safe_relative(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name or path.as_posix() != name:
        raise ValueError(f"unsafe bundle path: {name}")
    return path


def collect(project, include_training=False):
    payloads = {}
    for relative in CODE_FILES:
        source = project / relative
        if source.is_symlink():
            raise ValueError(f"symlink source not allowed: {relative}")
        payloads["repo/" + relative] = source.read_bytes()
    payloads["repo/README.md"] = README.encode("utf-8")
    assets = json.loads((project / "data/asset_manifest.json").read_text(encoding="utf-8"))
    selected = []
    for entry in assets["files"]:
        relative = entry["path"]
        if relative in DATA_FILES or relative.startswith(DATA_PREFIXES):
            safe_relative(relative)
            path = (project / relative).resolve()
            if not path.is_relative_to(project.resolve()) or (project / relative).is_symlink():
                raise ValueError(f"asset escapes project: {relative}")
            content = path.read_bytes()
            if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError(f"asset changed since P1: {relative}")
            key = "repo/" + relative
            if key in payloads:
                raise ValueError(f"duplicate asset: {relative}")
            payloads[key] = content
            selected.append(entry)
    if not DATA_FILES <= {entry["path"] for entry in selected}:
        raise ValueError("required QA/reference files missing")
    if include_training:
        sys.path.insert(0, str(PROJECT / "src"))
        from cuhkx.config import load_config
        from cuhkx.training.dataset import load_training_config, prepare_data, require_ready
        config = load_config(project)
        prepared = prepare_data(config, load_training_config(config))
        require_ready(prepared, ("train", "dev", "confirm"))
        for relative, digest in prepared["identity"]["inputs"].items():
            name = "data/" + relative
            safe_relative(name)
            path = (project/name).resolve()
            if not path.is_relative_to(project.resolve()):
                raise ValueError("training asset escapes project")
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError("training asset changed during packaging")
            key = "repo/" + name
            if key in payloads:
                if payloads[key] != content:raise ValueError("conflicting training asset")
                continue
            payloads[key] = content
            selected.append({"path":name,"bytes":len(content),"sha256":digest,
                             "source":name,"source_sha256":digest,"operation":"validated existing training cache"})
        assets["training"] = {"data_signature":prepared["data_signature"],"coverage":prepared["coverage"]}
    assets["files"] = selected
    payloads["repo/data/asset_manifest.json"] = encoded(assets)
    manifest = {"schema_version": 1, "baseline_id": "ir4_7b_v1", "files": [
        {"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        for name, payload in sorted(payloads.items())]}
    payloads["bundle_manifest.json"] = encoded(manifest)
    return payloads


def verify_extract(archive, destination):
    manifest, _ = extract_verified_archive(
        archive, destination, "bundle_manifest.json",
        lambda value: value.get("schema_version") == 1 and value.get("baseline_id") == "ir4_7b_v1",
        validate_name=safe_relative,
    )
    return manifest


def build(project, output, python=sys.executable, run_checks=True, include_training=False):
    python = Path(python).resolve()
    if output.exists():
        raise ValueError("output archive already exists; choose a new path")
    payloads = collect(project, include_training=include_training)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ir4_bundle_") as temporary:
        stage = Path(temporary)
        archive = stage / "ir4_7b_v1.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
            for name, content in sorted(payloads.items()):
                info = zipfile.ZipInfo(name, date_time=(2026, 9, 8, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                zipped.writestr(info, content)
        unpacked = stage / "unpacked"
        manifest = verify_extract(archive, unpacked)
        checks = {}
        if run_checks:
            env = {**os.environ, "PYTHONPATH": str(unpacked / "repo/src"), "PYTHONDONTWRITEBYTECODE": "1"}
            for dataset in ("test", "pilot"):
                proc = subprocess.run([str(python), "-m", "cuhkx.cli", "check", "--dataset", dataset,
                                       "--project-root", str(unpacked / "repo")], cwd=stage,
                                      env=env, capture_output=True, text=True, encoding="utf-8")
                if proc.returncode:
                    raise RuntimeError(f"unpacked {dataset} check failed: {proc.stderr}")
                checks[dataset] = json.loads(proc.stdout)
            if include_training:
                proc = subprocess.run([str(python), "-m", "cuhkx.cli", "training-check", "--project-root", str(unpacked/"repo")],
                                      cwd=stage, env=env, capture_output=True, text=True, encoding="utf-8")
                if proc.returncode:
                    raise RuntimeError(f"unpacked training check failed: {proc.stderr}")
                checks["training"] = json.loads(proc.stdout)
        # Publication only after integrity and optional isolated CPU checks pass.
        with output.open("xb") as handle:
            handle.write(archive.read_bytes())
    return {"status": "PASS", "archive": str(output.resolve()), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(payloads["bundle_manifest.json"]).hexdigest(),
            "bytes": output.stat().st_size, "manifest_files": len(manifest["files"]), "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/cloud/ir4_7b_v1.zip")
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="CPU Python for extracted-package checks")
    parser.add_argument("--verification-report", type=Path, help="Optional report path; otherwise only print verification to stdout")
    parser.add_argument("--include-training", action="store_true", help="Include validated fold caches and training references; fails if incomplete")
    args = parser.parse_args()
    result = build(PROJECT, args.output.resolve(), args.python, include_training=args.include_training)
    if args.verification_report:
        args.verification_report.parent.mkdir(parents=True, exist_ok=True)
        with args.verification_report.open("xb") as handle:
            handle.write(encoded(result))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
