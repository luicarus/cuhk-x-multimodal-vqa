"""Create the independent full-data Qwen3.5-4B QLoRA notebook (vLLM dual-GPU)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.notebook_security import secure_loader_source
except ModuleNotFoundError:  # direct `python scripts/...` execution
    from notebook_security import secure_loader_source


def cell(kind: str, source: str):
    result = {"cell_type": kind, "metadata": {}, "source": source.strip() + "\n"}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


CELLS = [
    cell("markdown", """
# Qwen3.5-4B QLoRA：vLLM 双卡加速版

这个 Notebook 是独立的 Qwen3.5-4B 后训练入口。它复用已有 IR4 缓存（缓存 8 帧、实际输入 4 帧），只改变模型、训练环境和推理引擎；不会读取或覆盖 7B baseline、Qwen3.5 test 或旧训练运行目录。五折训练缓存已内置到 ZIP。请在 2×T4 的 Kaggle GPU session 中运行。

推理引擎分工：

| 阶段 | 引擎 | 原因 |
|---|---|---|
| 训练 / 训练中 dev 评估 | Transformers 5.17.0 | vLLM 只做推理，无法反传梯度 |
| adapter 重载、dev/confirm 评估、test 推理 | **vLLM 0.19.1，DP=2** | 每卡一份完整模型，卡间零通信 |

推理阶段与推理 lane 的 DP2-B32 对齐：每个 replica 调度上限 16，Runner 全局批 32。T4 之间没有 NVLink，张量并行每层都要跨 PCIe all-reduce，因此改用数据并行。

两边共用同一份数据契约、prompt、答案空间和运行校验，因此 vLLM 的分数与 Transformers 可直接比较。运行合同记录 `engine` 字段，同一 run-id 不会混用两种引擎的结果。
"""),
    cell("code", r'''
from pathlib import Path, PurePosixPath
from collections import deque
import hashlib, json, os, re, shutil, subprocess, sys, tempfile, zipfile

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
BUNDLE_INPUT = None  # ZIP，或含 qwen35_training_bundle_manifest.json 的目录
WEIGHTS_INPUT = None  # 可选：含 cuhkx_qwen35_weights.json 的完整 Qwen3.5 权重目录
EXPERIMENT = "qwen35_vllm_full_v1"
TRAIN_GPU = 0          # 训练固定单卡，避免与 vLLM 进程争抢显存
# 双卡拓扑：DP（每卡一份完整模型，卡间零通信）或 TP（一份模型切两卡，跨 PCIe
# all-reduce）。T4 无 NVLink，本实验改用 DP。
PARALLEL_MODE = "dp"
DATA_PARALLEL = 2 if PARALLEL_MODE == "dp" else 1
TENSOR_PARALLEL = 1 if PARALLEL_MODE == "dp" else 2
GPU_MEMORY_UTILIZATION = 0.80
# FlashInfer 会为 SM 7.5 现场 JIT 编译，最后一步需要链接 libcuda.so（属于 NVIDIA
# 驱动，Kaggle 容器没有 stubs），报 "cannot find -lcuda"。TRITON_ATTN 是纯 Triton
# 实现，不需要 nvcc/链接，因此作为默认值。
ATTENTION_BACKEND = "TRITON_ATTN"
# 后训练 lane 的推理阶段（adapter 重载 smoke、dev/confirm 评估、test 推理）与
# 推理 lane 的 DP2-B32 对齐：每个 replica 调度上限 16，Runner 全局批 32。
# 训练本身仍走 Transformers 单卡，只有这些推理步骤用 vLLM。
MAX_NUM_SEQS = 16
RUNNER_BATCH_SIZE = 32
RUN_CONFIRMATION = False
RUN_TEST = False
PACKAGE_ID = "cuhkx-qwen35-4b-qlora-vllm-v1"
MARKER = "qwen35_training_bundle_manifest.json"
'''),
    cell("markdown", "## 1. 验证独立训练包并建立隔离工作目录"),
    cell("code", r'''
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", EXPERIMENT):
    raise RuntimeError("invalid experiment name")
if BUNDLE_INPUT is None:
    candidates = []
    for marker in INPUT.rglob(MARKER):
        try:
            if json.loads(marker.read_text(encoding="utf-8")).get("package_id") == PACKAGE_ID:
                candidates.append(marker.parent)
        except (OSError, ValueError):
            pass
    if not candidates:
        candidates = list(INPUT.rglob("qwen35_4b_qlora.zip"))
    if len(candidates) != 1:
        raise RuntimeError(f"found {len(candidates)} matching training packages; set BUNDLE_INPUT")
    BUNDLE_INPUT = candidates[0]
BUNDLE_INPUT = Path(BUNDLE_INPUT)
archive = zipfile.ZipFile(BUNDLE_INPUT) if BUNDLE_INPUT.is_file() else None
try:
    def bundle_bytes(name):
        return archive.read(name) if archive else (BUNDLE_INPUT / name).read_bytes()
    raw_manifest = bundle_bytes(MARKER)
    manifest = json.loads(raw_manifest)
    if manifest.get("schema_version") != 1 or manifest.get("package_id") != PACKAGE_ID:
        raise RuntimeError("wrong Qwen3.5 vLLM training package")
    names = [entry["path"] for entry in manifest["files"]]
    if len(names) != len(set(names)):
        raise RuntimeError("manifest contains duplicate paths")
    for name in names:
        path = PurePosixPath(name)
        if (not name.startswith("qwen35_training_repo/") or path.is_absolute()
                or ".." in path.parts or "\\" in name or ":" in name
                or path.as_posix() != name):
            raise RuntimeError("unsafe package path")
    if archive and (len(archive.namelist()) != len(set(archive.namelist()))
                    or set(archive.namelist()) != set(names) | {MARKER}):
        raise RuntimeError("ZIP does not match its manifest")
    for entry in manifest["files"]:
        content = bundle_bytes(entry["path"])
        if (len(content) != entry["bytes"]
                or hashlib.sha256(content).hexdigest() != entry["sha256"]):
            raise RuntimeError("package hash mismatch: " + entry["path"])
    PACKAGE_SHA = hashlib.sha256(raw_manifest).hexdigest()
    RUNTIME = WORK / ("qwen35_qlora_" + PACKAGE_SHA[:12]) / EXPERIMENT
    for entry in manifest["files"]:
        target = (RUNTIME / entry["path"]).resolve()
        if not target.is_relative_to(RUNTIME.resolve()):
            raise RuntimeError("runtime path escapes package root")
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError("runtime copy was modified; use a new experiment name")
    for entry in manifest["files"]:
        target = RUNTIME / entry["path"]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle_bytes(entry["path"]))
    REPO = RUNTIME / "qwen35_training_repo"
    (RUNTIME / MARKER).write_bytes(raw_manifest)
finally:
    if archive:
        archive.close()
print("Training repository:", REPO)
print("Package:", PACKAGE_ID)
'''),
    cell("markdown", "## 2. 安装独立 Python 3.11 环境并检查完整五折数据"),
    cell("code", r'''
VENV = RUNTIME / "train_env"
PYTHON = VENV / "bin/python"
if not PYTHON.exists():
    with tempfile.TemporaryDirectory(prefix="cuhkx_uv_bootstrap_", dir=WORK) as bootstrap_dir:
        BOOT = Path(bootstrap_dir)
        subprocess.run([sys.executable, "-m", "pip", "install", "--target", str(BOOT),
                        "--no-deps", "--require-hashes", "--only-binary=:all:",
                        "-r", str(REPO / "requirements/bootstrap.lock.txt")], check=True)
        subprocess.run([sys.executable, "-m", "uv", "venv", "--python", "3.11",
                        "--seed", str(VENV)], env={**os.environ, "PYTHONPATH": str(BOOT)}, check=True)
subprocess.run([str(PYTHON), "-m", "pip", "install", "--require-hashes",
                "--only-binary=:all:", "-r", str(REPO / "requirements/cpu.lock.txt")], check=True)
subprocess.run([str(PYTHON), "-m", "pip", "install", "--no-deps",
                "--no-build-isolation", "-e", str(REPO)], check=True)

def command(*args):
    return [str(PYTHON), "-m", "cuhkx.cli", *args, "--project-root", str(REPO)]

CLOUD_ENV = {**os.environ, "PYTHONPATH": str(REPO / "src"),
             "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
             "CUHKX_TRACEBACK": "1"}

def cloud(*args):
    tail = deque(maxlen=120)
    with subprocess.Popen(command(*args), cwd=REPO, env=CLOUD_ENV,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            tail.append(line)
        code = process.wait()
    if code:
        raise RuntimeError(f"{args[0]} exited with code {code}; last output:\n" + "".join(tail))

TRAINING_CONFIG = REPO / "configs/training_qwen35.yaml"
data_state = json.loads(subprocess.check_output(command(
    "training-check", "--profile", "qwen35", "--training-config", str(TRAINING_CONFIG)),
    cwd=REPO, env=CLOUD_ENV, text=True))
if data_state["status"] != "PASS":
    raise RuntimeError("embedded five-fold caches are incomplete")
print(json.dumps(data_state["coverage"], indent=2))
'''),
    cell("markdown", "## 3. 安装 Qwen3.5 后训练依赖（含 vLLM 0.19.1）并固定双卡环境"),
    cell("code", r'''
subprocess.run([str(PYTHON), "-m", "pip", "install", "--require-hashes",
                "--only-binary=:all:", "--index-url", "https://pypi.org/simple",
                "--extra-index-url", "https://download.pytorch.org/whl/cu128",
                "-r", str(REPO / "requirements/train_qwen35.lock.txt")], check=True)
subprocess.run([str(PYTHON), "-m", "pip", "check"], check=True)
compatibility_probe = (
    "import json, peft, transformers, vllm, torch; "
    "from transformers import AutoModelForMultimodalLM, Trainer; "
    "print(json.dumps({'transformers': transformers.__version__, 'peft': peft.__version__, "
    "'vllm': vllm.__version__, 'torch': torch.__version__, "
    "'multimodal_class': AutoModelForMultimodalLM.__name__}))"
)
print(subprocess.check_output([str(PYTHON), "-c", compatibility_probe], text=True))
probe = ("import json,sys,torch; assert sys.version_info[:2]==(3,11); "
         "assert torch.cuda.is_available(), 'a cloud CUDA GPU is required'; "
         "count=torch.cuda.device_count(); "
         "assert count>=TENSOR_PARALLEL_COUNT, f'vLLM TP={TENSOR_PARALLEL_COUNT} needs that many GPUs'; "
         "print(json.dumps({'python':sys.version,'torch':torch.__version__,"
         "'cuda':torch.version.cuda,'device_count':count,"
         "'devices':[torch.cuda.get_device_name(i) for i in range(count)]}))")
environment = subprocess.check_output(
    [str(PYTHON), "-c", f"TENSOR_PARALLEL_COUNT={TENSOR_PARALLEL};{probe}"], text=True)
(RUNTIME / "environment.json").write_text(environment, encoding="utf-8")
(RUNTIME / "environment.freeze.txt").write_text(
    subprocess.check_output([str(PYTHON), "-m", "pip", "freeze", "--all"], text=True),
    encoding="utf-8")
print(environment)
'''),
    cell("markdown", "## 4. 解析并固定 Qwen3.5 revision，下载或复用已验证权重"),
    cell("code", r'''
profile_path = REPO / "configs/qwen35_4b.yaml"
profile = json.loads(json.dumps(__import__("yaml").safe_load(profile_path.read_text(encoding="utf-8"))))
if WEIGHTS_INPUT is None:
    candidates = []
    for receipt_path in INPUT.rglob("cuhkx_qwen35_weights.json"):
        try:
            value = json.loads(receipt_path.read_text(encoding="utf-8"))
            if value.get("model_id") == "Qwen/Qwen3.5-4B":
                candidates.append(receipt_path.parent)
        except (OSError, ValueError):
            pass
    if len(candidates) > 1:
        raise RuntimeError("found multiple Qwen3.5 weight directories; set WEIGHTS_INPUT")
    WEIGHTS = candidates[0] if candidates else Path("/tmp") / "qwen35_4b_weights"
else:
    WEIGHTS = Path(WEIGHTS_INPUT)
if not (WEIGHTS / "cuhkx_qwen35_weights.json").exists():
    from huggingface_hub import HfApi
    revision = HfApi().model_info("Qwen/Qwen3.5-4B").sha
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("Hugging Face did not return an immutable revision")
else:
    revision = json.loads((WEIGHTS / "cuhkx_qwen35_weights.json").read_text(encoding="utf-8"))["revision"]
profile["model"]["revision"] = revision
profile_path.write_text(__import__("yaml").safe_dump(profile, sort_keys=False), encoding="utf-8")
cloud("fetch-qwen35-weights", "--weights-dir", str(WEIGHTS), "--revision", revision)
cloud("check", "--profile", "qwen35", "--dataset", "test")
cloud("check", "--profile", "qwen35", "--dataset", "pilot")
'''),
    cell("markdown", "## 5. 训练与训练中评估走 Transformers（vLLM 不能训练）"),
    cell("code", r'''
cloud("train", "--profile", "qwen35", "--training-config", str(TRAINING_CONFIG),
      "--weights-dir", str(WEIGHTS), "--run-id", "qwen35_pt_smoke", "--smoke-steps", "4",
      "--gpu", str(TRAIN_GPU), "--resume")
SMOKE_ADAPTER = REPO / "artifacts/training/qwen35_pt_smoke/adapter"
cloud("train", "--profile", "qwen35", "--training-config", str(TRAINING_CONFIG),
      "--weights-dir", str(WEIGHTS), "--run-id", "qwen35_pt_sft", "--gpu", str(TRAIN_GPU), "--resume")
ADAPTER = REPO / "artifacts/training/qwen35_pt_sft/adapter"
'''),
    cell("markdown", "## 6. vLLM 双卡短跑：验证 DP=2 引擎、答案约束与 adapter 重载"),
    cell("code", r'''
VLLM = ["--backend", "vllm",
        "--tensor-parallel-size", str(TENSOR_PARALLEL),
        "--data-parallel-size", str(DATA_PARALLEL),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--attention-backend", ATTENTION_BACKEND,
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--runner-batch-size", str(RUNNER_BATCH_SIZE)]
cloud("predict", "--profile", "qwen35", *VLLM,
      "--dataset", "pilot", "--limit", "16",
      "--weights-dir", str(WEIGHTS), "--adapter-dir", str(SMOKE_ADAPTER),
      "--run-id", "qwen35_vllm_smoke_reload", "--resume")
smoke = json.loads((REPO / "outputs/qwen35_vllm_smoke_reload/run_summary.json").read_text())
backend_metadata = smoke["backend"]
if backend_metadata.get("backend") != "qwen35_4b_vllm":
    raise RuntimeError("smoke run did not use the vLLM engine")
if backend_metadata.get("tensor_parallel_size") != TENSOR_PARALLEL:
    raise RuntimeError(f"vLLM ran with TP={backend_metadata.get('tensor_parallel_size')}, expected {TENSOR_PARALLEL}")
if PARALLEL_MODE == "dp" and backend_metadata.get("data_parallel_size") != DATA_PARALLEL:
    raise RuntimeError(f"vLLM ran with DP={backend_metadata.get('data_parallel_size')}, expected {DATA_PARALLEL}")
if backend_metadata.get("attention_backend") != ATTENTION_BACKEND:
    raise RuntimeError(f"vLLM ran with attention backend {backend_metadata.get('attention_backend')}, expected {ATTENTION_BACKEND}")
# The two batch limits must be the ones requested, not whatever the engine
# defaulted to: they are recorded in the signed contract and a silent fallback
# would make this run incomparable with the aligned inference lane.
smoke_options = json.loads(
    (REPO / "outputs/qwen35_vllm_smoke_reload/resume_state.json").read_text()
)["contract"]["engine_options"]
if smoke_options.get("max_num_seqs") != MAX_NUM_SEQS:
    raise RuntimeError(f"engine max_num_seqs={smoke_options.get('max_num_seqs')}, expected {MAX_NUM_SEQS}")
if smoke_options.get("runner_batch_size") != RUNNER_BATCH_SIZE:
    raise RuntimeError(f"runner_batch_size={smoke_options.get('runner_batch_size')}, expected {RUNNER_BATCH_SIZE}")
workers = backend_metadata.get("workers") or {}
if not workers.get("workers"):
    raise RuntimeError(f"worker memory telemetry is empty: {workers.get('error')}")
prefix = backend_metadata.get("prefix_cache") or {}
if prefix.get("reported_requests"):
    print("prefix cache hit rate:", prefix)
else:
    print("WARNING: prefix cache counters were not reported by this engine build:",
          prefix)
    print("engine metrics field inventory:",
          json.dumps(backend_metadata.get("metrics_shape"), indent=2, default=str))
print(json.dumps({"backend": backend_metadata, "prefix_cache": prefix}, indent=2))
'''),
    cell("markdown", "## 7. vLLM dev 基座/候选对照与 confirm 门禁"),
    cell("code", r'''
VLLM = ["--backend", "vllm",
        "--tensor-parallel-size", str(TENSOR_PARALLEL),
        "--data-parallel-size", str(DATA_PARALLEL),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--attention-backend", ATTENTION_BACKEND,
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--runner-batch-size", str(RUNNER_BATCH_SIZE)]
cloud("evaluate-training", "--profile", "qwen35", "--training-config", str(TRAINING_CONFIG),
      *VLLM, "--split", "dev", "--weights-dir", str(WEIGHTS),
      "--run-id", "qwen35_pt_base_dev", "--resume")
cloud("evaluate-training", "--profile", "qwen35", "--training-config", str(TRAINING_CONFIG),
      *VLLM, "--split", "dev", "--weights-dir", str(WEIGHTS), "--adapter-dir", str(ADAPTER),
      "--run-id", "qwen35_pt_adapter_dev", "--resume")
def accuracy(run_id):
    return json.loads((REPO / "outputs" / run_id / "metrics.json").read_text())["metrics"]["overall_accuracy"]
print("vLLM dev baseline:", accuracy("qwen35_pt_base_dev"), "adapter:", accuracy("qwen35_pt_adapter_dev"))

CONFIRMED = False
if RUN_CONFIRMATION:
    cloud("verify-run", "--profile", "qwen35", "--training-config", str(TRAINING_CONFIG),
          "--run-id", "qwen35_pt_base_dev")
    cloud("verify-run", "--profile", "qwen35", "--training-config", str(TRAINING_CONFIG),
          "--run-id", "qwen35_pt_adapter_dev")
    if accuracy("qwen35_pt_adapter_dev") <= accuracy("qwen35_pt_base_dev"):
        raise RuntimeError("adapter has no dev improvement; formal test is gated")
    cloud("evaluate-training", "--profile", "qwen35", "--training-config", str(TRAINING_CONFIG),
          *VLLM, "--split", "confirm", "--weights-dir", str(WEIGHTS),
          "--run-id", "qwen35_pt_base_confirm", "--resume")
    cloud("evaluate-training", "--profile", "qwen35", "--training-config", str(TRAINING_CONFIG),
          *VLLM, "--split", "confirm", "--weights-dir", str(WEIGHTS), "--adapter-dir", str(ADAPTER),
          "--run-id", "qwen35_pt_adapter_confirm", "--resume")
    CONFIRMED = accuracy("qwen35_pt_adapter_confirm") > accuracy("qwen35_pt_base_confirm")
    print("confirm improved:", CONFIRMED)
'''),
    cell("markdown", "## 8. 只有 confirm 提升后才用 vLLM 导出 test submission"),
    cell("code", r'''
if RUN_TEST:
    if not RUN_CONFIRMATION or not CONFIRMED:
        raise RuntimeError("confirm gate is not satisfied")
    cloud("verify-run", "--profile", "qwen35", "--training-config", str(TRAINING_CONFIG),
          "--run-id", "qwen35_pt_adapter_confirm")
    cloud("predict", "--profile", "qwen35", *VLLM, "--dataset", "test", "--weights-dir", str(WEIGHTS),
          "--adapter-dir", str(ADAPTER), "--run-id", "qwen35_pt_test", "--resume")
    cloud("submit", "--profile", "qwen35", "--run-id", "qwen35_pt_test")
    print("submission:", REPO / "outputs/qwen35_pt_test/submission.csv")
print("adapter/checkpoints:", REPO / "artifacts/training")
print("run evidence:", REPO / "outputs")
print("runtime/environment backup:", RUNTIME)
'''),
]

CELLS[1]["source"] = CELLS[1]["source"].replace(
    "BUNDLE_INPUT = None",
    "BUNDLE_INPUT = None\nEXPECTED_MANIFEST_SHA256 = None  # paste manifest_sha256 from the trusted package command",
)
CELLS[3]["source"] = secure_loader_source(
    package_id="cuhkx-qwen35-4b-qlora-vllm-v1",
    identity_key="package_id",
    marker="qwen35_training_bundle_manifest.json",
    prefix="qwen35_training_repo/",
    zip_name="qwen35_4b_qlora.zip",
    runtime_prefix="qwen35_qlora_",
    repository_name="qwen35_training_repo",
    experiment=True,
    extra_validation='''
if manifest.get("training_cache_mode") != "embedded_complete":
    raise RuntimeError("training package does not contain the complete cache")
if manifest.get("inference_engine") != "vllm_0.19.1_dual_gpu":
    raise RuntimeError("training package was not built for the vLLM dual-GPU lane")
''',
    extra_prints='print("Package:", PACKAGE_ID)',
)


CELLS[0]["source"] = """# Qwen3.5-4B QLoRA: vLLM Dual-GPU

This independent training notebook reuses the existing IR4 input protocol and embeds the complete five-fold cache. Create the package locally, copy its printed `manifest_sha256` into `EXPECTED_MANIFEST_SHA256` in the first code cell, and attach that exact `qwen35_4b_qlora.zip` as a private Kaggle input. This authenticates the manifest before any project code is copied or installed.

Training and evaluation-under-training stay on Transformers, because vLLM is inference-only. Adapter reload checks, dev/confirm evaluation, and test inference run on **vLLM 0.19.1 with one full replica per T4 (data parallel, DP=2, 16 sequences per replica and a 32-request Runner batch)**, matching the inference lane's DP2-B32 configuration. Both engines share the same data contract and constrained answer space, and the run contract records which engine produced each result.

The package contains no model weights. Use a Kaggle 2x T4 session; the notebook verifies that both GPUs are visible before vLLM starts.
"""


def build(output: Path, *, force: bool = False):
    if output.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing notebook: {output}")
    for index, item in enumerate(CELLS):
        if item["cell_type"] == "code":
            compile(item["source"], f"qwen35_training_cell_{index}", "exec")
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": CELLS,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "notebooks/qwen35-4b-qlora-vllm.ipynb")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate the generated Qwen3.5 vLLM notebook")
    arguments = parser.parse_args()
    build(arguments.output, force=arguments.force)
