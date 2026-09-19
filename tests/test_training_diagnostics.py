import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cuhkx import cli


def test_debug_cli_preserves_actual_traceback(monkeypatch,capsys):
    def failed(*args,**kwargs):
        raise ValueError("specific training failure")
    monkeypatch.setattr(cli,"load_config",failed)
    monkeypatch.setenv("CUHKX_TRACEBACK","1")
    assert cli.main(["train","--weights-dir","missing","--run-id","test"])==2
    error=capsys.readouterr().err
    assert "Traceback" in error and "ValueError: specific training failure" in error
    assert "train failed: specific training failure" in error


def test_notebook_wrapper_shows_stderr_in_exception(tmp_path,capsys):
    notebook=Path(__file__).resolve().parents[1]/"notebooks/cuhk-x-qlora-full-v3.ipynb"
    cells=json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    source=next(c["source"] for c in cells if c["cell_type"]=="code" and "def cloud(" in c["source"])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="cloud")
    code=compile(ast.Module(body=[node],type_ignores=[]),"cloud_wrapper","exec")
    scope={"subprocess":subprocess,"REPO":tmp_path,"CLOUD_ENV":dict(os.environ),
           "command":lambda *args:[sys.executable,"-c","import sys; print('train failed: actual cause',file=sys.stderr);sys.exit(2)"]}
    exec(code,scope)
    with pytest.raises(RuntimeError,match="actual cause"):
        scope["cloud"]("train")
    assert "actual cause" in capsys.readouterr().out
