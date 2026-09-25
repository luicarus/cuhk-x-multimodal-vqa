"""Independent, pinned post-training release. Does not consume or overwrite baseline ZIPs."""
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

FULL_PACKAGE_ID = "cuhkx-ir4-qlora-full-v3"
FULL_NOTEBOOK = "notebooks/cuhk-x-qlora-full-v3.ipynb"
MANIFEST = "training_bundle_manifest.json"
PREFIX = "training_repo/"
FILES = (
    "pyproject.toml", "src/cuhkx/__init__.py", "src/cuhkx/config.py", "src/cuhkx/cli.py",
    "src/cuhkx/release_security.py",
    "src/cuhkx/data/__init__.py", "src/cuhkx/data/inputs.py", "src/cuhkx/data/validate.py",
    "src/cuhkx/inference/__init__.py", "src/cuhkx/inference/prompt.py", "src/cuhkx/inference/qwen.py",
    "src/cuhkx/inference/runner.py", "src/cuhkx/inference/storage.py", "src/cuhkx/inference/weights.py",
    # The training profiler reuses the NVML sampler that lives here.
    "src/cuhkx/inference/profiling.py",
    "src/cuhkx/evaluation/__init__.py", "src/cuhkx/evaluation/metric.py",
    "src/cuhkx/submission/__init__.py", "src/cuhkx/submission/validator.py", "src/cuhkx/submission/export.py",
    "src/cuhkx/training/__init__.py", "src/cuhkx/training/dataset.py", "src/cuhkx/training/collator.py",
    "src/cuhkx/training/trainer.py", "src/cuhkx/training/adapter.py", "src/cuhkx/training/evaluate.py",
    "src/cuhkx/training/profiling.py",
    "configs/baseline.yaml", "configs/datasets.yaml", "configs/submission.yaml", "configs/training.yaml",
    "requirements/bootstrap.lock.txt", "requirements/cpu.in", "requirements/cpu.lock.txt",
    "requirements/cloud.in", "requirements/cloud.lock.txt",
    "requirements/train.in", "requirements/train.lock.txt", "requirements/README.md",
    "docs/post_training.md", "docs/training_release.md", "notebooks/cuhk-x-qlora-full-v3.ipynb",
)
DATA = {"data/qa/test.csv", "data/qa/pilot.csv", "data/qa/sample_submission.csv",
        "data/references/pilot_answers.csv", "data/references/training_qa.csv",
        "data/references/folds/subject_grouped_v1/qa_folds.csv"}
FRAME_PREFIXES = ("data/frames/test/", "data/frames/pilot/")


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+"\n").encode("utf-8")


def safe_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or ":" in name or "\\" in name or path.as_posix()!=name:
        raise ValueError(f"unsafe package path: {name}")


def collect(project):
    project = project.resolve()
    sys.path.insert(0, str(PROJECT/"src"))
    from cuhkx.config import load_config
    from cuhkx.training.dataset import load_training_config, prepare_data, require_ready
    from cuhkx.data.validate import fingerprint
    config = load_config(project, require_revision=True)
    model = config["baseline"]["model"]
    payloads = {}
    for relative in FILES:
        path = (project/relative).resolve()
        if not path.is_relative_to(project) or (project/relative).is_symlink():
            raise ValueError(f"unsafe source: {relative}")
        payloads[PREFIX+relative] = path.read_bytes()
    payloads[PREFIX+"README.md"] = (
        "# Independent IR4 + 7B QLoRA release\n\n"
        "Use notebooks/cuhk-x-qlora-full-v3.ipynb and docs/training_release.md. "
        "No baseline notebook, previous results, model weights, or raw videos are included. "
        "The pinned model revision and all file hashes are in ../training_bundle_manifest.json.\n"
    ).encode()
    assets = json.loads((project/"data/asset_manifest.json").read_text(encoding="utf-8"))
    selected = {}
    for entry in assets["files"]:
        relative = entry["path"]
        if relative not in DATA and not relative.startswith(FRAME_PREFIXES):continue
        safe_name(relative)
        path = (project/relative).resolve()
        if not path.is_relative_to(project):raise ValueError("asset escapes project")
        content = path.read_bytes()
        if len(content)!=entry["bytes"] or hashlib.sha256(content).hexdigest()!=entry["sha256"]:
            raise ValueError(f"asset differs from original manifest: {relative}")
        if relative in selected:raise ValueError("duplicate asset entry")
        selected[relative] = entry
        payloads[PREFIX+relative] = content
    if not DATA <= set(selected):raise ValueError("required QA/fold references missing")
    readiness = prepare_data(config,load_training_config(config))
    require_ready(readiness,("train","dev","confirm"))
    for relative,digest in readiness["identity"]["inputs"].items():
        name = "data/"+relative
        safe_name(name)
        path = (project/name).resolve()
        if not path.is_relative_to(project):raise ValueError("training source escapes project")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest()!=digest:raise ValueError("training source changed")
        payloads[PREFIX+name] = content
        selected[name] = {"path":name,"bytes":len(content),"sha256":digest,
                          "source":name,"source_sha256":digest,"operation":"validated cache copy"}
    assets["files"] = [selected[name] for name in sorted(selected)]
    assets.pop("training",None)
    payloads[PREFIX+"data/asset_manifest.json"] = encoded(assets)
    receipt_path = project/"artifacts/cloud/ir4_7b_v1/cuhkx_weights.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["model_id"]!=model["id"] or receipt["revision"]!=model["revision"]:
        raise ValueError("baseline weight provenance differs from pinned config")
    payloads[PREFIX+"provenance/base_weights.json"] = encoded(receipt)
    manifest = {"schema_version":1,"package_id":FULL_PACKAGE_ID,
                "model_id":model["id"],"model_revision":model["revision"],
                "base_receipt_sha256":fingerprint(receipt),
                "training_cache_mode":"embedded_complete",
                "training_data_signature":readiness["data_signature"],
                "training_coverage":readiness["coverage"],
                "training_cache_summary":readiness["cache_summary"],
                "files":[{"path":name,"bytes":len(content),"sha256":hashlib.sha256(content).hexdigest()}
                         for name,content in sorted(payloads.items())]}
    payloads[MANIFEST] = encoded(manifest)
    return payloads


def extract_verified(archive, destination, expected_package_id=None):
    def identity(value):
        return (value.get("schema_version") == 1 and value.get("package_id") == FULL_PACKAGE_ID
                and (expected_package_id is None or value.get("package_id") == expected_package_id))
    manifest, _ = extract_verified_archive(
        archive, destination, MANIFEST, identity, validate_name=safe_name)
    return manifest


def build(project, output, python=sys.executable, run_checks=True):
    project, output, python = project.resolve(), output.resolve(), Path(python).resolve()
    if output.exists():raise ValueError("release already exists; choose a new output filename")
    payloads = collect(project)
    with tempfile.TemporaryDirectory(prefix="cuhkx_training_release_") as temporary:
        stage = Path(temporary)
        archive = stage/(FULL_PACKAGE_ID+".zip")
        with zipfile.ZipFile(archive,"w") as zipped:
            for name,content in sorted(payloads.items()):
                info=zipfile.ZipInfo(name,(2026,9,9,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
                zipped.writestr(info,content)
        unpacked=stage/"unpacked"
        manifest=extract_verified(archive,unpacked,FULL_PACKAGE_ID)
        checks={}
        if run_checks:
            repo=unpacked/"training_repo"
            env={**os.environ,"PYTHONPATH":str(repo/"src"),"PYTHONDONTWRITEBYTECODE":"1"}
            for name,args in (("test",["check","--dataset","test"]),("pilot",["check","--dataset","pilot"]),
                              ("training",["training-check"])):
                process=subprocess.run([str(python),"-B","-m","cuhkx.cli",*args,"--project-root",str(repo)],
                                       cwd=stage,env=env,capture_output=True,text=True,encoding="utf-8")
                try:result=json.loads(process.stdout)
                except ValueError:raise RuntimeError(process.stderr or process.stdout)
                if process.returncode!=0:raise RuntimeError(f"unexpected {name} result: {process.stderr} {result}")
                checks[name]=result
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open("xb") as handle:handle.write(archive.read_bytes())
    return {"status":"PASS","archive":str(output),"bytes":output.stat().st_size,
            "sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"files":len(manifest["files"]),
            "manifest_sha256":hashlib.sha256(payloads[MANIFEST]).hexdigest(),
            "training_cache_mode":manifest["training_cache_mode"],"checks":checks}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--python",type=Path,default=Path(sys.executable))
    args=parser.parse_args()
    output = args.output or PROJECT/"artifacts/cloud_training"/(FULL_PACKAGE_ID+".zip")
    print(json.dumps(build(PROJECT,output,args.python),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
