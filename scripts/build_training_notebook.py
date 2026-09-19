"""Create only the independent training notebook; refuse to overwrite existing notebooks."""
import json
import argparse
from copy import deepcopy
from pathlib import Path

try:
    from scripts.notebook_security import secure_loader_source
except ModuleNotFoundError:  # direct `python scripts/...` execution
    from notebook_security import secure_loader_source


def cell(kind, source):
    value={"cell_type":kind,"metadata":{},"source":source.strip()+"\n"}
    if kind=="code":value.update(execution_count=None,outputs=[])
    return value


CELLS=[
cell("markdown", """
# Independent IR4 + 7B QLoRA
独立后训练入口，不读取或修改原 baseline Notebook/运行目录。
模型 revision、训练配置、代码和环境锁来自本训练包；不会重新解析模型 main。
当前 embedded_complete 版本需另挂载五折 IR8 缓存。缺缓存时在安装 GPU 依赖和下载权重前停止。
准备：私有训练包、五折缓存、云端 CUDA GPU、网络或已验证权重。此 Notebook 尚未完成真实训练验收。
"""),
cell("code", '''
from pathlib import Path, PurePosixPath
import hashlib, json, os, re, shutil, subprocess, sys, tempfile, zipfile

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
BUNDLE_INPUT = None  # ZIP 文件，或含 training_bundle_manifest.json 的已解压目录
TRAIN_CACHE_INPUT = None  # 含 fold_0 ... fold_4 的已挂载 IR8 缓存目录
WEIGHTS_INPUT = None  # 可选：含原 cuhkx_weights.json 和全部权重文件的目录
EXPERIMENT = "sft_v1"  # 新配置使用新的实验名/训练包
GPU = 0
RUN_CONFIRMATION = False  # 开发集选定候选后开启
RUN_TEST = False          # 确认有收益后开启
'''),
cell("markdown", "## 1. 验证独立训练包并准备工作副本（只用标准库）"),
cell("code", '''
PACKAGE_ID = "cuhkx-ir4-qlora-full-v3"
MARKER = "training_bundle_manifest.json"
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", EXPERIMENT):
    raise RuntimeError("无效的实验名")
if BUNDLE_INPUT is None:
    candidates = [p.parent for p in INPUT.rglob(MARKER)]
    if not candidates:
        candidates = list(INPUT.rglob(PACKAGE_ID + ".zip"))
    if len(candidates) != 1:
        raise RuntimeError(f"找到 {len(candidates)} 个训练包，请手动设置 BUNDLE_INPUT")
    BUNDLE_INPUT = candidates[0]
BUNDLE_INPUT = Path(BUNDLE_INPUT)
archive = zipfile.ZipFile(BUNDLE_INPUT) if BUNDLE_INPUT.is_file() else None
try:
    def bundle_bytes(name):
        return archive.read(name) if archive else (BUNDLE_INPUT / name).read_bytes()
    raw_manifest = bundle_bytes(MARKER)
    manifest = json.loads(raw_manifest)
    if manifest.get("package_id") != PACKAGE_ID or manifest.get("schema_version") != 1:
        raise RuntimeError("不是本版本的独立训练包")
    names = [e["path"] for e in manifest["files"]]
    if len(names) != len(set(names)):
        raise RuntimeError("清单存在重复路径")
    for name in names:
        path = PurePosixPath(name)
        if not name.startswith("training_repo/") or path.is_absolute() or ".." in path.parts or "\\\\" in name or ":" in name or path.as_posix()!=name:
            raise RuntimeError("不安全的包路径")
        if not archive and not (BUNDLE_INPUT/name).resolve().is_relative_to(BUNDLE_INPUT.resolve()):
            raise RuntimeError("输入路径越界")
    if archive and (len(archive.namelist()) != len(set(archive.namelist())) or set(archive.namelist()) != set(names) | {MARKER}):
        raise RuntimeError("ZIP 与清单文件集合不同")
    for entry in manifest["files"]:
        content = bundle_bytes(entry["path"])
        if len(content)!=entry["bytes"] or hashlib.sha256(content).hexdigest()!=entry["sha256"]:
            raise RuntimeError("包内文件哈希不匹配: " + entry["path"])
    PACKAGE_SHA = hashlib.sha256(raw_manifest).hexdigest()
    RUNTIME = WORK / ("cuhkx_qlora_" + PACKAGE_SHA[:12]) / EXPERIMENT
    # Preflight all destinations before writing. Never overwrite a changed working copy.
    for entry in manifest["files"]:
        target = (RUNTIME / entry["path"]).resolve()
        if not target.is_relative_to(RUNTIME.resolve()):
            raise RuntimeError("工作路径越界")
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=entry["sha256"]:
            raise RuntimeError("工作副本已修改，请使用新实验名: " + entry["path"])
    for entry in manifest["files"]:
        target = RUNTIME / entry["path"]
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(bundle_bytes(entry["path"]))
    REPO = RUNTIME / "training_repo"
    (RUNTIME / MARKER).write_bytes(raw_manifest)
finally:
    if archive: archive.close()
print("Training project:", REPO)
print("Cache mode:", manifest["training_cache_mode"])
print("Pinned revision:", manifest["model_revision"])
'''),
cell("markdown", "## 2. 补充既有五折缓存；不读取原视频、不抽帧"),
cell("code", '''
if TRAIN_CACHE_INPUT is None and manifest["training_cache_mode"] == "embedded_complete":
    raise RuntimeError("本包尚不含五折训练缓存。先设置 TRAIN_CACHE_INPUT（目录下应有 fold_0 到 fold_4），再继续。")
if TRAIN_CACHE_INPUT is not None:
    source = Path(TRAIN_CACHE_INPUT).resolve()
    destination = (REPO / "data/frames/train").resolve()
    pending = []
    def file_digest(path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8*1024*1024), b""):
                digest.update(block)
        return digest.hexdigest()
    for fold in range(5):
        folder = source / f"fold_{fold}"
        if not folder.is_dir(): raise RuntimeError(f"缺少 {folder}")
        for path in folder.rglob("*"):
            if not path.is_file(): continue
            if not path.resolve().is_relative_to(source) or path.suffix.lower() not in (".jpg", ".json", ".csv"):
                raise RuntimeError("缓存中有不支持的文件或路径: " + str(path))
            target = (destination / path.relative_to(source)).resolve()
            if not target.is_relative_to(destination): raise RuntimeError("目标路径越界")
            if target.exists():
                if file_digest(target) != file_digest(path): raise RuntimeError("不覆盖不同缓存: " + str(target))
            else:
                pending.append((path,target))
    for source_file,target in pending:
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source_file,target)
print("缓存已就位；下一步执行逐文件、QA 和分折检查。")
'''),
cell("markdown", "## 3. 独立 Python 3.11 环境与 CPU 数据检查"),
cell("code", '''
VENV = RUNTIME / "train_env"
PYTHON = VENV / "bin/python"
if not PYTHON.exists():
    with tempfile.TemporaryDirectory(prefix="cuhkx_uv_bootstrap_",dir=WORK) as bootstrap_dir:
        BOOT = Path(bootstrap_dir)
        subprocess.run([sys.executable,"-m","pip","install","--target",str(BOOT),"--no-deps","--require-hashes","--only-binary=:all:","-r",str(REPO/"requirements/bootstrap.lock.txt")],check=True)
        subprocess.run([sys.executable,"-m","uv","venv","--python","3.11","--seed",str(VENV)],
                       env={**os.environ,"PYTHONPATH":str(BOOT)},check=True)
subprocess.run([str(PYTHON),"-m","pip","install","--require-hashes","--only-binary=:all:","-r",str(REPO/"requirements/cpu.lock.txt")],check=True)
subprocess.run([str(PYTHON),"-m","pip","install","--no-deps","--no-build-isolation","-e",str(REPO)],check=True)

def command(*args):
    return [str(PYTHON),"-m","cuhkx.cli",*args,"--project-root",str(REPO)]

CLOUD_ENV = {**os.environ,"PYTHONPATH":str(REPO/"src"),"PYTHONDONTWRITEBYTECODE":"1"}

def cloud(*args):
    from collections import deque
    tail = deque(maxlen=80)
    env = {**CLOUD_ENV,"PYTHONUNBUFFERED":"1","CUHKX_TRACEBACK":"1"}
    with subprocess.Popen(command(*args),cwd=REPO,env=env,stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",bufsize=1) as process:
        for line in process.stdout:
            print(line,end="",flush=True)
            tail.append(line)
        code = process.wait()
    if code:
        raise RuntimeError(f"{args[0]} exited with code {code}; last output:\\n" + "".join(tail))

data_state = json.loads(subprocess.check_output(command("training-check"),cwd=REPO,env=CLOUD_ENV,text=True))
if data_state["status"] != "PASS": raise RuntimeError("五折缓存尚未完整")
if manifest["training_data_signature"] is not None and data_state["data_signature"] != manifest["training_data_signature"]:
    raise RuntimeError("完整数据包的数据指纹发生变化")
print(json.dumps(data_state["coverage"],indent=2))
'''),
cell("markdown", "## 4. 安装固定训练依赖；只在云端 CUDA 环境执行"),
cell("code", '''
subprocess.run([str(PYTHON),"-m","pip","install","--require-hashes","--only-binary=:all:","-r",str(REPO/"requirements/train.lock.txt")],check=True)
subprocess.run([str(PYTHON),"-m","pip","check"],check=True)
probe = "import json,sys,torch; assert sys.version_info[:2]==(3,11); assert torch.cuda.is_available(), '需要云端 CUDA GPU'; print(json.dumps({'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))"
environment = subprocess.check_output([str(PYTHON),"-c",probe],text=True)
(RUNTIME/"environment.json").write_text(environment)
(RUNTIME/"environment.freeze.txt").write_text(subprocess.check_output([str(PYTHON),"-m","pip","freeze","--all"],text=True))
print(environment)
'''),
cell("markdown", "## 5. 准备与 baseline 同一 revision 的权重，不使用浮动 main"),
cell("code", '''
expected_receipt = json.loads((REPO/"provenance/base_weights.json").read_text())
revision = manifest["model_revision"]
if not re.fullmatch(r"[0-9a-f]{40}", revision): raise RuntimeError("包内 revision 未锁定")
if WEIGHTS_INPUT is None:
    candidates = []
    for receipt_path in INPUT.rglob("cuhkx_weights.json"):
        candidate = json.loads(receipt_path.read_text())
        if candidate.get("model_id") == manifest["model_id"] and candidate.get("revision") == revision:
            if candidate == expected_receipt and all((receipt_path.parent/e["path"]).is_file() for e in candidate["files"]):
                candidates.append(receipt_path.parent)
    if len(candidates)>1: raise RuntimeError("找到多份匹配权重，请设置 WEIGHTS_INPUT")
    WEIGHTS = candidates[0] if candidates else Path("/tmp")/("cuhkx_qlora_weights_"+revision[:12])
else:
    WEIGHTS = Path(WEIGHTS_INPUT)
if not (WEIGHTS/"cuhkx_weights.json").exists():
    needed = sum(e["bytes"] for e in expected_receipt["files"]) + 2*1024**3
    if shutil.disk_usage(WEIGHTS.parent if WEIGHTS.parent.exists() else Path("/tmp")).free < needed:
        raise RuntimeError("权重存储空间不足，请挂载完整的已验证权重目录")
cloud("fetch-weights","--weights-dir",str(WEIGHTS))
if json.loads((WEIGHTS/"cuhkx_weights.json").read_text()) != expected_receipt:
    raise RuntimeError("权重与原 baseline 的来源清单不同")
(RUNTIME/"session.json").write_text(json.dumps({"package_manifest_sha256":PACKAGE_SHA,"training_data_signature":data_state["data_signature"],"model_revision":revision,"experiment":EXPERIMENT},indent=2))
'''),
cell("markdown", "## 6. 短跑与 adapter 重载验证；这不是正式性能结果"),
cell("code", '''
cloud("train","--weights-dir",str(WEIGHTS),"--run-id","pt_smoke","--smoke-steps","4","--gpu",str(GPU),"--resume")
SMOKE_ADAPTER = REPO/"artifacts/training/pt_smoke/adapter"
cloud("predict","--dataset","pilot","--limit","16","--weights-dir",str(WEIGHTS),"--adapter-dir",str(SMOKE_ADAPTER),"--run-id","pt_smoke_reload","--resume")
'''),
cell("markdown", "## 7. 记录相同 dev 集合上的基座对照，然后正式训练"),
cell("code", '''
cloud("evaluate-training","--split","dev","--weights-dir",str(WEIGHTS),"--run-id","pt_base_dev","--resume")
cloud("train","--weights-dir",str(WEIGHTS),"--run-id","pt_sft","--gpu",str(GPU),"--resume")
ADAPTER = REPO/"artifacts/training/pt_sft/adapter"
cloud("evaluate-training","--split","dev","--weights-dir",str(WEIGHTS),"--adapter-dir",str(ADAPTER),"--run-id","pt_adapter_dev","--resume")
def accuracy(run_id):
    return json.loads((REPO/"outputs"/run_id/"metrics.json").read_text())["metrics"]["overall_accuracy"]
print("dev baseline:",accuracy("pt_base_dev"),"adapter:",accuracy("pt_adapter_dev"))
'''),
cell("markdown", "## 8. 候选选定后才开启确认；本轮不据确认结果反复调参"),
cell("code", '''
CONFIRMED = False
if RUN_CONFIRMATION:
    cloud("verify-run","--run-id","pt_base_dev")
    cloud("verify-run","--run-id","pt_adapter_dev")
    if accuracy("pt_adapter_dev") <= accuracy("pt_base_dev"):
        raise RuntimeError("开发集尚无提升，停止确认/测试生产")
    cloud("evaluate-training","--split","confirm","--weights-dir",str(WEIGHTS),"--run-id","pt_base_confirm","--resume")
    cloud("evaluate-training","--split","confirm","--weights-dir",str(WEIGHTS),"--adapter-dir",str(ADAPTER),"--run-id","pt_adapter_confirm","--resume")
    CONFIRMED = accuracy("pt_adapter_confirm") > accuracy("pt_base_confirm")
    print("confirm baseline:",accuracy("pt_base_confirm"),"adapter:",accuracy("pt_adapter_confirm"),"improved:",CONFIRMED)
else:
    print("确认阶段未开启。先固定配置与候选，再设置 RUN_CONFIRMATION。")
'''),
cell("markdown", "## 9. 有收益的候选单独导出测试提交；不自动上传"),
cell("code", '''
if RUN_TEST:
    if not RUN_CONFIRMATION or not CONFIRMED:
        raise RuntimeError("确认集尚未显示提升，不生成正式测试候选")
    cloud("verify-run","--run-id","pt_adapter_confirm")
    cloud("predict","--dataset","test","--weights-dir",str(WEIGHTS),"--adapter-dir",str(ADAPTER),"--run-id","pt_test","--resume")
    cloud("submit","--run-id","pt_test")
    print("新提交:",REPO/"outputs/pt_test/submission.csv")
print("保存最终 adapter 与所需 checkpoint:",REPO/"artifacts/training")
print("保存真实预测与评测:",REPO/"outputs")
print("保存环境与训练包身份:",RUNTIME)
'''),
]


def cells_for(complete=True):
    cells = deepcopy(CELLS)
    if not complete:
        raise ValueError("Incomplete training notebooks have been retired")
    cells[0]["source"] = """# IR4 + 7B QLoRA — 完整数据版 v3
独立训练包已内置五折 IR8 缓存、全部训练 QA、原始分折和 test/pilot 缓存。
不需要另挂载五折缓存；只需训练包、云端 CUDA GPU 和已验证权重或下载网络。
模型 revision、代码、训练配置和依赖锁均固定在包内，原 baseline Notebook 与结果不受影响。
数据/哈希/CPU 验证通过不代表真实 GPU 训练通过；先执行短跑及 adapter 重载验证。
"""
    cells[1]["source"] = "\n".join(line for line in cells[1]["source"].splitlines() if not line.startswith("TRAIN_CACHE_INPUT")) + "\n"
    cells[1]["source"] = cells[1]["source"].replace('EXPERIMENT = "sft_v1"', 'EXPERIMENT = "sft_full_v3"')
    cells[1]["source"] = cells[1]["source"].replace(
        "BUNDLE_INPUT = None",
        "BUNDLE_INPUT = None\nEXPECTED_MANIFEST_SHA256 = None  # paste manifest_sha256 from the trusted package command",
    )
    cells[3]["source"] = secure_loader_source(
        package_id="cuhkx-ir4-qlora-full-v3",
        identity_key="package_id",
        marker="training_bundle_manifest.json",
        prefix="training_repo/",
        zip_name="cuhkx-ir4-qlora-full-v3.zip",
        runtime_prefix="cuhkx_qlora_",
        repository_name="training_repo",
        experiment=True,
        extra_validation='''
if manifest.get("training_cache_mode") != "embedded_complete":
    raise RuntimeError("training package does not contain the complete cache")
''',
        extra_prints='''
print("Cache mode:", manifest["training_cache_mode"])
print("Pinned revision:", manifest["model_revision"])
''',
    )
    cells[4]["source"] = "## 2. 确认包内完整缓存，无需复制外挂数据\n"
    cells[5]["source"] = '''
if manifest.get("training_cache_mode") != "embedded_complete":
    raise RuntimeError("请选择完整数据版 ZIP")
coverage = manifest.get("training_coverage", {})
if set(coverage) != {"train", "dev", "confirm"} or any(c["missing_qa"] for c in coverage.values()):
    raise RuntimeError("训练包声明的数据不完整")
print("包内五折缓存:", manifest.get("training_cache_summary"))
print("分组:", coverage)
print("下一步独立核验全部图片、QA 关联与数据指纹。")
'''.strip()+"\n"
    cells[0]["source"] += "\nCreate the package locally and copy its printed `manifest_sha256` into `EXPECTED_MANIFEST_SHA256` before extraction. Attach only that exact private package to Kaggle.\n"
    return cells


def build(complete=True):
    if not complete:
        raise ValueError("Incomplete training notebooks have been retired")
    name = "cuhk-x-qlora-full-v3.ipynb"
    target=Path(__file__).resolve().parents[1]/"notebooks"/name
    if target.exists():raise FileExistsError("Training notebook already exists; not overwriting it")
    cells = cells_for(complete)
    for index,c in enumerate(cells):
        if c["cell_type"]=="code":compile(c["source"],f"training_cell_{index}","exec")
    notebook={"nbformat":4,"nbformat_minor":4,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"cells":cells}
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(target)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete", action="store_true", default=True, help="Create only the v3 full-data notebook (default)")
    build(parser.parse_args().complete)
