import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile

import pytest

from test_cached_inputs import project, seal
from test_training import training

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("package_cloud", ROOT / "scripts/package_cloud.py")
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


@pytest.fixture
def bundle(project, tmp_path):
    for relative in package.CODE_FILES:
        target = project / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
    (project / "data/qa/pilot.csv").write_bytes((project / "data/qa/test.csv").read_bytes())
    (project / "data/references").mkdir(exist_ok=True)
    (project / "data/references/pilot_answers.csv").write_text("qa_id,answer\nq0,A\n")
    seal(project / "data")
    (project / "secret.env").write_text("must-not-package")
    (project / "src/cuhkx/inference/pipeline.py").write_text("must-not-package")
    archive = tmp_path / "ir4_7b_v1.zip"
    result = package.build(project, archive, run_checks=False)
    return archive, result


def test_allowlist_and_verified_notebook_preserved(bundle, tmp_path):
    archive, result = bundle
    assert result["status"] == "PASS"
    with zipfile.ZipFile(archive) as zipped:
        assert not any("secret.env" in n or "pipeline.py" in n or "/outputs/" in n or "/.git/" in n for n in zipped.namelist())
        assert zipped.read("repo/notebooks/cuhk-x-base7b.ipynb") == (ROOT / "notebooks/cuhk-x-base7b.ipynb").read_bytes()
    package.verify_extract(archive, tmp_path / "dataset")
    assert (tmp_path / "dataset/repo/src/cuhkx/cli.py").exists()


def test_zip_tamper_and_traversal_rejected(bundle, tmp_path):
    archive, _ = bundle
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(bad, "w") as target:
        for name in source.namelist():
            target.writestr(name, b"tampered" if name == "repo/README.md" else source.read(name))
    with pytest.raises(ValueError, match="hash"):
        package.verify_extract(bad, tmp_path / "bad_extract")
    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, "w") as zipped:
        zipped.writestr("../escape", b"no")
    with pytest.raises(ValueError, match="unsafe"):
        package.verify_extract(traversal, tmp_path / "traversal")


def test_verified_notebook_compiles_without_executing_or_clearing_outputs():
    path = ROOT / "notebooks/cuhk-x-base7b.ipynb"
    before = path.read_bytes()
    notebook = json.loads(before)
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = cell["source"]
            compile("".join(source) if isinstance(source, list) else source, f"cell_{i}", "exec")
    assert path.read_bytes() == before


def test_notebook_cli_calls_remain_compatible(monkeypatch):
    import ast
    from cuhkx.cli import main
    from cuhkx import cli
    path = ROOT / "notebooks/cuhk-x-base7b.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    class Parsed(Exception):
        pass
    def no_execution(*args, **kwargs):
        raise Parsed()
    monkeypatch.setattr(cli, "load_config", no_execution)
    commands = set()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        for node in ast.walk(ast.parse("".join(source) if isinstance(source, list) else source)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "cloud":
                args = [arg.value if isinstance(arg, ast.Constant) else "weights" for arg in node.args]
                commands.add(args[0])
                with pytest.raises(Parsed):
                    main([*args, "--project-root", str(ROOT)])
    assert commands == {"fetch-weights", "check", "predict", "verify-run", "evaluate", "submit"}


def test_check_interpreter_resolved_before_switching_cwd(bundle, project, tmp_path, monkeypatch):
    from types import SimpleNamespace
    calls = []
    monkeypatch.chdir(project)
    def checked(command, **kwargs):
        assert Path(command[0]).is_absolute()
        assert Path(command[0]) == project / "env/python.exe"
        assert Path(kwargs["cwd"]) != project
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout='{"status":"PASS"}', stderr="")
    monkeypatch.setattr(package.subprocess, "run", checked)
    package.build(project, tmp_path / "relative_interpreter.zip", python="env/python.exe")
    assert len(calls) == 2


def test_training_pack_contains_unregistered_but_validated_caches(bundle, training, project, tmp_path):
    config,settings=training
    seal(project/'data')
    manifest_path=project/'data/asset_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest['files']=[e for e in manifest['files'] if not e['path'].startswith('data/frames/fold_')]
    manifest_path.write_text(json.dumps(manifest))
    archive=tmp_path/'training.zip'
    package.build(project,archive,run_checks=False,include_training=True)
    unpacked=tmp_path/'training_unpacked'
    package.verify_extract(archive,unpacked)
    from cuhkx.config import load_config
    from cuhkx.training.dataset import load_training_config, prepare_data
    loaded=load_config(unpacked/'repo')
    result=prepare_data(loaded,load_training_config(loaded))
    assert result['status']=='PASS' and result['coverage']['train']['available_qa']==3
