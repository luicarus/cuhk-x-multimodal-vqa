"""Run CPU tests with bytecode disabled and all disposable files outside the project."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    project = Path(__file__).resolve().parents[1]
    temporary_root = Path(tempfile.gettempdir()).resolve()
    with tempfile.TemporaryDirectory(prefix="cuhkx_tests_", dir=temporary_root) as temporary:
        scratch = Path(temporary).resolve()
        if scratch.parent != temporary_root:
            raise RuntimeError("test temporary directory escaped the intended root")
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TEMP": str(scratch),
               "TMP": str(scratch), "TMPDIR": str(scratch)}
        command = [sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider",
                   "--basetemp", str(scratch / "pytest"), *sys.argv[1:]]
        return subprocess.run(command, cwd=project, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
