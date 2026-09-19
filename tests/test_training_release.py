import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import pytest
import yaml

from test_cached_inputs import PROJECT, project, seal, write_csv
from test_training import training
from cuhkx.config import load_config
from cuhkx.training.dataset import load_training_config, prepare_data


spec=importlib.util.spec_from_file_location("training_release",PROJECT/"scripts/package_training.py")
release=importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


@pytest.fixture
def release_project(training):
    config,_=training
    project=Path(config["project_root"])
    for relative in (*release.FILES,release.FULL_NOTEBOOK):
        target=project/relative
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(PROJECT/relative,target)
    path=project/"configs/baseline.yaml"
    value=yaml.safe_load(path.read_text())
    value["model"]["revision"]="a"*40
    path.write_text(yaml.safe_dump(value))
    reference=project/"data/references/pilot_answers.csv"
    write_csv(reference,["qa_id","answer"],[{"qa_id":"t5","answer":"A"}])
    receipt=project/"artifacts/cloud/ir4_7b_v1/cuhkx_weights.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"schema_version":1,"method":"huggingface_pinned_force_download",
        "model_id":value["model"]["id"],"revision":"a"*40,
        "files":[{"path":"model.safetensors","bytes":1,"sha256":"0"*64}]}))
    seal(project/"data")
    return project


def notebook_scope(bundle, work):
    notebook=json.loads((PROJECT/release.FULL_NOTEBOOK).read_text(encoding="utf-8"))
    scope={}
    exec(notebook["cells"][1]["source"],scope)
    if Path(bundle).is_file():
        with zipfile.ZipFile(bundle) as zipped:
            raw_manifest = zipped.read(release.MANIFEST)
    else:
        raw_manifest = (Path(bundle) / release.MANIFEST).read_bytes()
    scope.update(BUNDLE_INPUT=bundle, WORK=work,
                 EXPECTED_MANIFEST_SHA256=hashlib.sha256(raw_manifest).hexdigest())
    exec(notebook["cells"][3]["source"],scope)
    return notebook,scope


def test_default_release_is_complete_and_independent(release_project,tmp_path):
    archive=tmp_path/"training.zip"
    result=release.build(release_project,archive,run_checks=False)
    assert result["training_cache_mode"]=="embedded_complete"
    with zipfile.ZipFile(archive) as zipped:
        names=zipped.namelist()
        assert "training_bundle_manifest.json" in names
        assert all(not name.startswith("repo/") for name in names)
        assert not any("cuhk-x-base7b.ipynb" in name or "/outputs/" in name for name in names)
        assert "training_repo/data/references/training_qa.csv" in names
        assert any("data/frames/fold_" in name for name in names)
        assert "training_repo/notebooks/cuhk-x-qlora-full-v3.ipynb" in names
        assert zipped.read("training_repo/"+release.FULL_NOTEBOOK)== (PROJECT/release.FULL_NOTEBOOK).read_bytes()
    notebook,scope=notebook_scope(archive,tmp_path/"working")
    assert "cuhkx_qlora_" in str(scope["REPO"])
    exec(notebook["cells"][5]["source"],scope)
    config=load_config(scope["REPO"])
    assert prepare_data(config,load_training_config(config))["status"]=="PASS"


def test_complete_release_preserves_training_identity_and_directory_input(release_project,tmp_path):
    archive=tmp_path/"complete.zip"
    release.build(release_project,archive,run_checks=False)
    source=tmp_path/"input"
    manifest=release.extract_verified(archive,source)
    assert manifest["training_cache_mode"]=="embedded_complete"
    assert manifest["package_id"]==release.FULL_PACKAGE_ID
    assert manifest["training_cache_summary"]["images_decoded"]==48
    notebook,scope=notebook_scope(source,tmp_path/"working")
    exec(notebook["cells"][5]["source"],scope)
    config=load_config(scope["REPO"])
    assert prepare_data(config,load_training_config(config))["data_signature"]==manifest["training_data_signature"]
    exec(notebook["cells"][3]["source"],scope)
    (scope["REPO"]/"configs/training.yaml").write_text("tampered")
    with pytest.raises(RuntimeError):exec(notebook["cells"][3]["source"],scope)


@pytest.mark.parametrize('name',[release.FULL_NOTEBOOK])
def test_no_floating_revision_and_all_cells_compile(name):
    notebook=json.loads((PROJECT/name).read_text(encoding="utf-8"))
    for index,cell in enumerate(notebook["cells"]):
        if cell["cell_type"]=="code":
            compile(cell["source"],f"cell_{index}","exec")
            assert not cell["outputs"] and cell["execution_count"] is None
    source="\n".join(c["source"] for c in notebook["cells"] if c["cell_type"]=="code")
    assert "model_info(" not in source and 'WORK / "repo"' not in source


def test_complete_release_cannot_claim_missing_caches(release_project,tmp_path):
    (release_project/"data/frames/fold_0/uniform_time_v1/ir/520837f5b798f45a/frame_index.csv").unlink()
    with pytest.raises(ValueError,match="missing"):
        release.build(release_project,tmp_path/"invalid.zip",run_checks=False)
    assert not (tmp_path/"invalid.zip").exists()


def test_full_notebook_ignores_other_release_during_discovery(release_project,tmp_path):
    inputs=tmp_path/'inputs'
    new=tmp_path/'new.zip'
    (inputs/'old').mkdir(parents=True)
    (inputs/'old'/release.MANIFEST).write_text(json.dumps({'package_id':'other-release'}))
    release.build(release_project,new,run_checks=False)
    release.extract_verified(new,inputs/'new')
    notebook=json.loads((PROJECT/release.FULL_NOTEBOOK).read_text(encoding='utf-8'))
    scope={}
    exec(notebook['cells'][1]['source'],scope)
    raw_manifest=(inputs/'new'/release.MANIFEST).read_bytes()
    scope.update(INPUT=inputs,WORK=tmp_path/'work',
                 EXPECTED_MANIFEST_SHA256=hashlib.sha256(raw_manifest).hexdigest())
    exec(notebook['cells'][3]['source'],scope)
    assert scope['BUNDLE_INPUT']==inputs/'new'
    assert 'TRAIN_CACHE_INPUT' not in scope
