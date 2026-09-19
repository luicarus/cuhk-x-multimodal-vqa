"""Create the independent Qwen3.5-4B test notebook without touching the baseline notebook."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.notebook_security import secure_loader_source
except ModuleNotFoundError:  # direct `python scripts/...` execution
    from notebook_security import secure_loader_source


def cell(kind: str, source: str) -> dict:
    value = {"cell_type": kind, "metadata": {}, "source": source.strip() + "\n"}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


CELLS = [
    cell("markdown", """
# Qwen3.5-4B IR4 test （修复版）
此 Notebook 是独立模型对照，不读取或修改 Qwen2.5-VL-7B baseline 的 Notebook、运行目录或 ZIP。
固定相同的 IR4 输入：缓存 8 帧，使用第 2/4/6/8 张。只有模型、模型加载接口和依赖环境不同。
将 `qwen35_4b_test_v1.zip` 作为 Kaggle 私有输入，开启 GPU 和 Internet；先执行 smoke，再执行完整 test。
"""),
    cell("code", '''
from pathlib import Path, PurePosixPath
import hashlib, json, os, re, shutil, subprocess, sys, tempfile, zipfile

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
BUNDLE_INPUT = None  # ZIP，或含 qwen35_bundle_manifest.json 的已解压目录
WEIGHTS_INPUT = None  # 可选：含 cuhkx_qwen35_weights.json 的完整权重目录
PINNED_REVISION = ""  # 留空则由固定环境中的 HfApi 解析并写入运行副本
'''),
    cell("markdown", "## 1. 验证 Qwen3.5 包并准备独立工作目录"),
    cell("code", '''
PACKAGE_ID = "cuhkx-qwen35-4b-test-v1"
MARKER = "qwen35_bundle_manifest.json"
if BUNDLE_INPUT is None:
    candidates = [p.parent for p in INPUT.rglob(MARKER)]
    if not candidates:
        candidates = list(INPUT.rglob("qwen35_4b_test_v1.zip"))
    if len(candidates) != 1:
        raise RuntimeError(f"需要恰好一个 Qwen3.5 包，找到 {len(candidates)} 个；请设置 BUNDLE_INPUT")
    BUNDLE_INPUT = candidates[0]
BUNDLE_INPUT = Path(BUNDLE_INPUT)
archive = zipfile.ZipFile(BUNDLE_INPUT) if BUNDLE_INPUT.is_file() else None
try:
    def read_member(name):
        return archive.read(name) if archive else (BUNDLE_INPUT / name).read_bytes()
    manifest = json.loads(read_member(MARKER))
    if manifest.get("schema_version") != 1 or manifest.get("package_id") != PACKAGE_ID:
        raise RuntimeError("不是当前 Qwen3.5 test 包")
    entries = manifest["files"]
    names = [entry["path"] for entry in entries]
    if len(names) != len(set(names)):
        raise RuntimeError("包清单存在重复路径")
    for name in names:
        path = PurePosixPath(name)
        if not name.startswith("qwen35_repo/") or path.is_absolute() or ".." in path.parts or "\\\\" in name or ":" in name or path.as_posix() != name:
            raise RuntimeError("包内路径不安全")
    if archive and set(archive.namelist()) != set(names) | {MARKER}:
        raise RuntimeError("ZIP 与清单的文件集合不同")
    for entry in entries:
        content = read_member(entry["path"])
        if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise RuntimeError("包内文件哈希不匹配: " + entry["path"])
    package_sha = hashlib.sha256(read_member(MARKER)).hexdigest()
    RUNTIME = WORK / ("qwen35_runtime_" + package_sha[:12])
    for entry in entries:
        target = (RUNTIME / entry["path"]).resolve()
        if not target.is_relative_to(RUNTIME.resolve()):
            raise RuntimeError("工作目录路径越界")
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError("工作副本已修改，请换一个 Kaggle 工作目录")
    for entry in entries:
        target = RUNTIME / entry["path"]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(read_member(entry["path"]))
    REPO = RUNTIME / "qwen35_repo"
    (RUNTIME / MARKER).write_bytes(read_member(MARKER))
finally:
    if archive:
        archive.close()
print("Qwen3.5 project:", REPO)
print("Package SHA:", package_sha)
print("Training included:", manifest["training_included"])
'''),
    cell("markdown", "## 2. 独立 Python 3.11 环境（Qwen3.5 专用）"),
    cell("code", '''
VENV = RUNTIME / "venv"
PYTHON = VENV / "bin/python"
if not PYTHON.exists():
    with tempfile.TemporaryDirectory(prefix="cuhkx_uv_bootstrap_", dir=WORK) as bootstrap_dir:
        BOOT = Path(bootstrap_dir)
        subprocess.run([sys.executable, "-m", "pip", "install", "--target", str(BOOT), "--no-deps",
                        "--require-hashes", "--only-binary=:all:",
                        "-r", str(REPO / "requirements/bootstrap.lock.txt")], check=True)
        subprocess.run([sys.executable, "-m", "uv", "venv", "--python", "3.11", "--seed", str(VENV)],
                       env={**os.environ, "PYTHONPATH": str(BOOT)}, check=True)
subprocess.run([str(PYTHON), "-m", "pip", "install", "--require-hashes", "--only-binary=:all:",
                "-r", str(REPO / "requirements/qwen35.lock.txt")], check=True)
subprocess.run([str(PYTHON), "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", str(REPO)], check=True)
subprocess.run([str(PYTHON), "-m", "pip", "check"], check=True)
CLOUD_ENV = {**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1",
             "PYTHONUNBUFFERED": "1", "CUHKX_TRACEBACK": "1"}

def command(*args):
    return [str(PYTHON), "-m", "cuhkx.cli", *args, "--project-root", str(REPO)]

def cloud(*args):
    result = subprocess.run(command(*args), cwd=REPO, env=CLOUD_ENV, capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    if result.returncode:
        raise RuntimeError(f"{args[0]} exited with code {result.returncode}\\n{result.stdout}\\n{result.stderr}")
    return result
'''),
    cell("markdown", "## 3. 固定 Qwen3.5 revision、准备权重并检查缓存"),
    cell("code", '''
if not PINNED_REVISION:
    PINNED_REVISION = subprocess.check_output(
        [str(PYTHON), "-c", "from huggingface_hub import HfApi; print(HfApi().model_info('Qwen/Qwen3.5-4B').sha)"],
        env=CLOUD_ENV, text=True).strip()
if not re.fullmatch(r"[0-9a-f]{40}", PINNED_REVISION):
    raise RuntimeError("Qwen3.5 revision 不是 40 位小写 commit SHA")
import yaml
profile = REPO / "configs/qwen35_4b.yaml"
profile_value = yaml.safe_load(profile.read_text(encoding="utf-8"))
old_revision = profile_value["model"].get("revision")
if old_revision not in (None, PINNED_REVISION):
    raise RuntimeError("包内 Qwen3.5 revision 与本次指定版本不同")
profile_value["model"]["revision"] = PINNED_REVISION
profile.write_text(yaml.safe_dump(profile_value, sort_keys=False), encoding="utf-8")
if WEIGHTS_INPUT is None:
    candidates = []
    for receipt in INPUT.rglob("cuhkx_qwen35_weights.json"):
        try:
            value = json.loads(receipt.read_text(encoding="utf-8"))
            if value.get("model_id") == "Qwen/Qwen3.5-4B" and value.get("revision") == PINNED_REVISION:
                candidates.append(receipt.parent)
        except (OSError, ValueError):
            continue
    if len(candidates) > 1:
        raise RuntimeError("找到多份 Qwen3.5 权重，请手动设置 WEIGHTS_INPUT")
    WEIGHTS = candidates[0] if candidates else Path("/tmp") / ("qwen35_weights_" + PINNED_REVISION[:12])
else:
    WEIGHTS = Path(WEIGHTS_INPUT)
cloud("fetch-qwen35-weights", "--weights-dir", str(WEIGHTS), "--revision", PINNED_REVISION)
cloud("check", "--profile", "qwen35", "--dataset", "test")
cloud("check", "--profile", "qwen35", "--dataset", "pilot")
print("Qwen3.5 weights:", WEIGHTS)
print("Qwen3.5 revision:", PINNED_REVISION)
'''),
    cell("markdown", "## 4. Smoke：test 前 16 QA"),
    cell("code", '''
cloud("predict", "--profile", "qwen35", "--dataset", "test", "--limit", "16",
      "--run-id", "qwen35_4b_smoke", "--weights-dir", str(WEIGHTS), "--resume")
cloud("verify-run", "--profile", "qwen35", "--run-id", "qwen35_4b_smoke")
'''),
    cell("markdown", "## 5. 完整 test：682 QA 与提交文件"),
    cell("code", '''
cloud("predict", "--profile", "qwen35", "--dataset", "test",
      "--run-id", "qwen35_4b_test", "--weights-dir", str(WEIGHTS), "--resume")
cloud("verify-run", "--profile", "qwen35", "--run-id", "qwen35_4b_test")
cloud("submit", "--profile", "qwen35", "--run-id", "qwen35_4b_test")
print("Qwen3.5 submission:", REPO / "outputs/qwen35_4b_test/submission.csv")
print("Qwen3.5 run evidence:", REPO / "outputs/qwen35_4b_test")
'''),
]

CELLS[1]["source"] = CELLS[1]["source"].replace(
    "BUNDLE_INPUT = None",
    "BUNDLE_INPUT = None\nEXPECTED_MANIFEST_SHA256 = None  # paste manifest_sha256 from the trusted package command",
)
CELLS[3]["source"] = secure_loader_source(
    package_id="cuhkx-qwen35-4b-test-v1",
    identity_key="package_id",
    marker="qwen35_bundle_manifest.json",
    prefix="qwen35_repo/",
    zip_name="qwen35_4b_test_v1.zip",
    runtime_prefix="qwen35_runtime_",
    repository_name="qwen35_repo",
    extra_prints='print("Training included:", manifest["training_included"])',
)


CELLS[0]["source"] = """# Qwen3.5-4B IR4 test

This independent comparison reuses the existing IR8 cache and selects frames 2, 4, 6, and 8. Create the package locally, copy its printed `manifest_sha256` into `EXPECTED_MANIFEST_SHA256` in the first code cell, and attach that exact ZIP or extracted package as a private Kaggle input. The digest must come from a trusted local build.

Use a Kaggle CUDA GPU and run the smoke check before complete test inference.
"""


def build(*, force: bool = False) -> Path:
    target = Path(__file__).resolve().parents[1] / "notebooks/qwen35-4b-test.ipynb"
    if target.exists() and not force:
        raise FileExistsError("Qwen3.5 notebook already exists; use --force to regenerate it")
    for index, item in enumerate(CELLS):
        if item["cell_type"] == "code":
            compile(item["source"], f"qwen35_cell_{index}", "exec")
    notebook = {"nbformat": 4, "nbformat_minor": 4,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "cells": CELLS}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Regenerate the generated Qwen3.5 notebook")
    args = parser.parse_args()
    print(build(force=args.force))
