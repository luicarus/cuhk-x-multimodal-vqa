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
# Qwen3.5-4B IR4 test（vLLM 双卡版）
此 Notebook 是独立模型对照，不读取或修改 Qwen2.5-VL-7B baseline 的 Notebook、运行目录或 ZIP。
固定相同的 IR4 输入：缓存 8 帧，使用第 2/4/6/8 张。只有模型、模型加载接口和依赖环境不同。
将 `qwen35_4b.zip` 作为 Kaggle 私有输入，开启 2×T4 GPU 和 Internet；先执行 smoke，再执行完整 test。

推理引擎：**vLLM 双卡，数据并行（DP=2，每卡一份完整模型，卡间零通信）**，
调度并发 16。这个包只做推理，不含训练栈。
此前几轮用的是张量并行（TP=2，一份模型切两卡，每层跨 PCIe all-reduce）。
T4 之间没有 NVLink，TP 的通信开销在小 prefill + 极短 decode 的负载上无法被 batch 摊薄，
因此本轮改成 DP 对照，并把 batch 从 32 减半到 16。
vLLM 用 `StructuredOutputsParams(choice=[...])` 约束到与 Transformers 相同的答案空间，
引擎启动前会校验 chat 渲染，因此两条引擎的分数可直接比较。运行合同记录 `engine` 字段。
本 Notebook 还会断言 worker 显存遥测非空、以及前缀缓存命中率是实测值而非缺失。
"""),
    cell("code", '''
from pathlib import Path, PurePosixPath
import hashlib, json, os, re, shutil, subprocess, sys, tempfile, zipfile

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
BUNDLE_INPUT = None  # ZIP，或含 qwen35_bundle_manifest.json 的已解压目录
WEIGHTS_INPUT = None  # 可选：含 cuhkx_qwen35_weights.json 的完整权重目录
PINNED_REVISION = ""  # 留空则由固定环境中的 HfApi 解析并写入运行副本
# 双卡拓扑：两种方式，互斥。
#   DP（data parallel）：每张卡一份完整模型，各自独立服务请求，卡间零通信。
#   TP（tensor parallel）：一份模型切到两张卡，每次前向都要跨 PCIe all-reduce。
# T4 之间没有 NVLink，TP 的通信开销在小 prefill + 极短 decode 的负载上占比很高，
# 所以本实验改为 DP，并把 batch 减半对照。
PARALLEL_MODE = "dp"          # "dp" 或 "tp"
DATA_PARALLEL = 2 if PARALLEL_MODE == "dp" else 1
TENSOR_PARALLEL = 1 if PARALLEL_MODE == "dp" else 2
GPU_COUNT_REQUIRED = DATA_PARALLEL * TENSOR_PARALLEL
GPU_MEMORY_UTILIZATION = 0.80
# FlashInfer 会为 SM 7.5 现场 JIT 编译，最后一步需要链接 libcuda.so（属于 NVIDIA
# 驱动，Kaggle 容器没有 stubs），报 "cannot find -lcuda"。TRITON_ATTN 是纯 Triton
# 实现，不需要 nvcc/链接，因此作为默认值。
ATTENTION_BACKEND = "TRITON_ATTN"
# 上一轮 B 为 32；本轮减半到 16，配合 DP 观察 batch 维度的收益曲线。
MAX_NUM_SEQS = 16
'''),
    cell("markdown", "## 1. 验证 Qwen3.5 vLLM 包并准备独立工作目录"),
    cell("code", '''
PACKAGE_ID = "cuhkx-qwen35-4b-vllm-v1"
MARKER = "qwen35_bundle_manifest.json"
if BUNDLE_INPUT is None:
    candidates = [p.parent for p in INPUT.rglob(MARKER)]
    if not candidates:
        candidates = list(INPUT.rglob("qwen35_4b.zip"))
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
        raise RuntimeError("不是当前 Qwen3.5 vLLM test 包")
    if manifest.get("inference_engine") != "vllm_0.19.1_dual_gpu":
        raise RuntimeError("这个包不是为 vLLM 双卡 lane 构建的")
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
print("Inference engine:", manifest["inference_engine"])
'''),
    cell("markdown", "## 2. 独立 Python 3.11 环境（Qwen3.5 专用，含 vLLM 0.19.1）"),
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
compatibility_probe = (
    "import json, transformers, torch, vllm; "
    "print(json.dumps({'transformers': transformers.__version__, 'vllm': vllm.__version__, "
    "'torch': torch.__version__}))"
)
print(subprocess.check_output([str(PYTHON), "-c", compatibility_probe], text=True))
probe = ("import json,sys,torch; assert sys.version_info[:2]==(3,11); "
         "assert torch.cuda.is_available(), 'a cloud CUDA GPU is required'; "
         "count=torch.cuda.device_count(); "
         "assert count>=REQUIRED_GPUS, f'{PARALLEL_LABEL} needs {REQUIRED_GPUS} GPUs'; "
         "print(json.dumps({'python':sys.version,'torch':torch.__version__,"
         "'cuda':torch.version.cuda,'device_count':count,"
         "'devices':[torch.cuda.get_device_name(i) for i in range(count)]}))")
environment = subprocess.check_output(
    [str(PYTHON), "-c",
     f"REQUIRED_GPUS={GPU_COUNT_REQUIRED};PARALLEL_LABEL={PARALLEL_MODE!r};{probe}"], text=True)
(RUNTIME / "environment.json").write_text(environment, encoding="utf-8")
print(environment)
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
    cell("markdown", "## 4. vLLM 双卡 smoke：test 前 16 QA，验证拓扑、答案约束与前缀缓存命中率"),
    cell("code", '''
VLLM = ["--backend", "vllm",
        "--tensor-parallel-size", str(TENSOR_PARALLEL),
        "--data-parallel-size", str(DATA_PARALLEL),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--attention-backend", ATTENTION_BACKEND,
        "--max-num-seqs", str(MAX_NUM_SEQS)]
cloud("predict", "--profile", "qwen35", *VLLM, "--dataset", "test", "--limit", "16",
      "--run-id", "qwen35_4b_smoke", "--weights-dir", str(WEIGHTS), "--resume")
cloud("verify-run", "--profile", "qwen35", "--run-id", "qwen35_4b_smoke")
smoke = json.loads((REPO / "outputs/qwen35_4b_smoke/run_summary.json").read_text())
backend_metadata = smoke["backend"]
if backend_metadata.get("backend") != "qwen35_4b_vllm":
    raise RuntimeError("smoke run did not use the vLLM engine")
if backend_metadata.get("tensor_parallel_size") != TENSOR_PARALLEL:
    raise RuntimeError(f"vLLM ran with TP={backend_metadata.get('tensor_parallel_size')}, expected {TENSOR_PARALLEL}")
if PARALLEL_MODE == "dp" and backend_metadata.get("data_parallel_size") != DATA_PARALLEL:
    raise RuntimeError(f"vLLM ran with DP={backend_metadata.get('data_parallel_size')}, expected {DATA_PARALLEL}")
if backend_metadata.get("attention_backend") != ATTENTION_BACKEND:
    raise RuntimeError(f"vLLM ran with attention backend {backend_metadata.get('attention_backend')}, expected {ATTENTION_BACKEND}")
# Worker 显存遥测必须真的采到数据：序列化失败时 workers 为空、error 非空。
workers = backend_metadata.get("workers") or {}
if not workers.get("workers"):
    raise RuntimeError(f"worker memory telemetry is empty: {workers.get('error')}")
# 前缀缓存命中率：能测到就打印；这个 vLLM 构建不上报该字段时只告警，
# 并把 metrics_shape 打出来，让下一次运行自己说明字段名。
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
    cell("markdown", "## 5. 完整 test：682 QA 与提交文件（vLLM）"),
    cell("code", '''
cloud("predict", "--profile", "qwen35", *VLLM, "--dataset", "test",
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
    package_id="cuhkx-qwen35-4b-vllm-v1",
    identity_key="package_id",
    marker="qwen35_bundle_manifest.json",
    prefix="qwen35_repo/",
    zip_name="qwen35_4b.zip",
    runtime_prefix="qwen35_runtime_",
    repository_name="qwen35_repo",
    extra_validation='''
if manifest.get("inference_engine") != "vllm_0.19.1_dual_gpu":
    raise RuntimeError("package was not built for the vLLM dual-GPU lane")
''',
    extra_prints='print("Training included:", manifest["training_included"])',
)


CELLS[0]["source"] = """# Qwen3.5-4B IR4 test (vLLM dual-GPU)

This independent comparison reuses the existing IR8 cache and selects frames 2, 4, 6, and 8. Create the package locally, copy its printed `manifest_sha256` into `EXPECTED_MANIFEST_SHA256` in the first code cell, and attach that exact ZIP or extracted package as a private Kaggle input. The digest must come from a trusted local build.

Test inference runs on **vLLM 0.19.1 with one full replica per T4 (data parallel, DP=2, scheduler concurrency 16)**. Two T4s have no NVLink, so tensor parallelism would pay a PCIe all-reduce on every forward pass; independent replicas remove that cost entirely. Set `PARALLEL_MODE = "tp"` to reproduce the earlier tensor-parallel rounds. The engine constrains decoding to the same closed answer space as the Transformers backend and checks its chat rendering against the reference processor before starting, so its scores stay comparable with the other lanes. The signed run contract records the topology, so one run-id cannot mix results from two engines.

The smoke cell also asserts that worker GPU memory telemetry is non-empty and that the prefix-cache hit rate is measured rather than missing.

Use a Kaggle 2x T4 session and run the smoke check before complete test inference.
"""


def build(*, force: bool = False) -> Path:
    target = Path(__file__).resolve().parents[1] / "notebooks/qwen35-4b-vllm.ipynb"
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
