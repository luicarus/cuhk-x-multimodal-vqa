import hashlib
import json
from pathlib import Path
import re
import zipfile

import pytest

from cuhkx.release_security import ArchiveLimits, extract_verified_archive
from scripts.notebook_security import NOTEBOOK_ARCHIVE_GUARD, secure_loader_source


PROJECT = Path(__file__).resolve().parents[1]


def write_archive(path, payloads, marker="manifest.json"):
    manifest = {
        "schema_version": 1,
        "package_id": "test-package",
        "files": [
            {"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in sorted(payloads.items())
        ],
    }
    raw = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for name, content in payloads.items():
            zipped.writestr(name, content)
        zipped.writestr(marker, raw)
    return raw


def test_bounded_archive_extracts_verified_payload(tmp_path):
    archive = tmp_path / "bundle.zip"
    raw = write_archive(archive, {"repo/file.txt": b"safe"})
    destination = tmp_path / "out"

    manifest, digest = extract_verified_archive(
        archive, destination, "manifest.json",
        lambda value: value["package_id"] == "test-package",
        limits=ArchiveLimits(max_archive_bytes=4096, max_entries=4, max_expanded_bytes=4096,
                             max_member_bytes=2048, max_compression_ratio=50,
                             free_space_reserve=0),
    )

    assert manifest["package_id"] == "test-package"
    assert digest == hashlib.sha256(raw).hexdigest()
    assert (destination / "repo/file.txt").read_bytes() == b"safe"


@pytest.mark.parametrize("limit_field,limit", [
    ("max_entries", 1),
    ("max_expanded_bytes", 4),
    ("max_member_bytes", 3),
    ("max_compression_ratio", 1),
])
def test_archive_limits_fail_before_extraction(tmp_path, limit_field, limit):
    archive = tmp_path / "bundle.zip"
    write_archive(archive, {"repo/file.txt": b"A" * 100})
    values = dict(max_archive_bytes=4096, max_entries=4, max_expanded_bytes=4096,
                  max_member_bytes=2048, max_compression_ratio=50, free_space_reserve=0)
    values[limit_field] = limit
    destination = tmp_path / "out"

    with pytest.raises(ValueError):
        extract_verified_archive(archive, destination, "manifest.json", lambda value: True,
                                 limits=ArchiveLimits(**values))

    assert not destination.exists() or not any(destination.rglob("*"))


def test_notebook_guard_requires_an_external_manifest_digest():
    assert "EXPECTED_MANIFEST_SHA256" in NOTEBOOK_ARCHIVE_GUARD
    assert "hmac.compare_digest" in NOTEBOOK_ARCHIVE_GUARD
    assert "infolist" in NOTEBOOK_ARCHIVE_GUARD
    assert "MAX_EXPANDED_BYTES" in NOTEBOOK_ARCHIVE_GUARD
    assert "source.is_symlink()" in NOTEBOOK_ARCHIVE_GUARD
    loader = secure_loader_source(package_id="id", identity_key="package_id", marker="m.json",
                                  prefix="repo/", zip_name="b.zip", runtime_prefix="runtime_",
                                  repository_name="repo")
    assert "if RUNTIME.is_symlink()" in loader
    assert "RUNTIME_ROOT.is_relative_to(WORK_ROOT)" in loader


def test_notebook_loader_rejects_self_consistent_forged_bundle(tmp_path):
    archive = tmp_path / "bundle.zip"
    raw = write_archive(archive, {"repo/payload.txt": b"attacker source"}, marker="manifest.json")
    work = tmp_path / "work"
    work.mkdir()
    scope = {
        "Path": Path, "PurePosixPath": __import__("pathlib").PurePosixPath,
        "hashlib": hashlib, "hmac": __import__("hmac"), "json": json, "os": __import__("os"),
        "re": re, "shutil": __import__("shutil"), "sys": __import__("sys"),
        "stat": __import__("stat"), "zipfile": zipfile,
        "INPUT": tmp_path, "WORK": work, "BUNDLE_INPUT": archive,
        "EXPECTED_MANIFEST_SHA256": "0" * 64,
    }
    loader = secure_loader_source(
        package_id="test-package", identity_key="package_id", marker="manifest.json",
        prefix="repo/", zip_name="bundle.zip", runtime_prefix="runtime_",
        repository_name="repo",
    )

    with pytest.raises(RuntimeError, match="not the trusted release"):
        exec(loader, scope)

    assert not list(work.glob("runtime_*"))
    assert len(raw) < 4096


def test_notebook_loader_rejects_symlinked_runtime_root(tmp_path):
    archive = tmp_path / "bundle.zip"
    raw = write_archive(archive, {"repo/payload.txt": b"trusted"}, marker="manifest.json")
    digest = hashlib.sha256(raw).hexdigest()
    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    runtime = work / ("runtime_" + digest[:12])
    try:
        runtime.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    scope = {
        "Path": Path, "PurePosixPath": __import__("pathlib").PurePosixPath,
        "hashlib": hashlib, "hmac": __import__("hmac"), "json": json, "os": __import__("os"),
        "re": re, "shutil": __import__("shutil"), "sys": __import__("sys"),
        "stat": __import__("stat"), "zipfile": zipfile,
        "INPUT": tmp_path, "WORK": work, "BUNDLE_INPUT": archive,
        "EXPECTED_MANIFEST_SHA256": digest,
    }
    loader = secure_loader_source(
        package_id="test-package", identity_key="package_id", marker="manifest.json",
        prefix="repo/", zip_name="bundle.zip", runtime_prefix="runtime_",
        repository_name="repo",
    )

    with pytest.raises(RuntimeError, match="runtime root must not be a symlink"):
        exec(loader, scope)

    assert not any(outside.iterdir())


def test_public_tree_has_no_private_paths_or_executed_notebook_metadata():
    public_files = [
        PROJECT / "docs/data_provenance.md",
        PROJECT / "docs/post_training_backup_summary.md",
        PROJECT / "reports/data/preprocessing_smoke.md",
        PROJECT / "docs/cloud.md",
        PROJECT / "requirements/train_qwen35.lock.txt",
    ]
    kaggle_path_pattern = "/".join(("kaggle", "input", "datasets", r"[^/]+")) + "/"
    forbidden = re.compile(r"(?i)(?:[A-Z]:[\\/](?!/)|/" + kaggle_path_pattern + ")")
    assert all(not forbidden.search(path.read_text(encoding="utf-8-sig")) for path in public_files)

    notebook = json.loads((PROJECT / "notebooks/cuhk-x-base7b.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        assert cell.get("execution_count") is None
        assert cell.get("outputs", []) == []
        assert "execution" not in cell.get("metadata", {})
        assert "trusted" not in cell.get("metadata", {})


def test_uv_bootstrap_is_hash_locked_and_used_by_every_notebook_lane():
    lock = (PROJECT / "requirements/bootstrap.lock.txt").read_text(encoding="utf-8")
    assert "uv==0.12.6" in lock
    assert "8cb1c4af10a1037d2e875ac86da32ea0df20a9623e11769d33515d14158cd3c2" in lock
    for path in (
        PROJECT / "notebooks/cuhk-x-base7b.ipynb",
        PROJECT / "notebooks/cuhk-x-qlora-full-v3.ipynb",
        PROJECT / "notebooks/qwen35-4b-vllm.ipynb",
        PROJECT / "notebooks/qwen35-4b-qlora-vllm.ipynb",
        PROJECT / "scripts/build_qwen35_notebook.py",
        PROJECT / "scripts/build_training_notebook.py",
        PROJECT / "scripts/build_qwen35_training_notebook.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "bootstrap.lock.txt" in source
        assert "--require-hashes" in source
        notebook = json.loads(source) if path.suffix == ".ipynb" else None
        code = ("\n".join("".join(c.get("source", [])) if isinstance(c.get("source"), list)
                            else c.get("source", "") for c in notebook["cells"] if c["cell_type"] == "code")
                if notebook else source)
        assert "TemporaryDirectory(prefix=\"cuhkx_uv_bootstrap_\"" in code
        assert "BOOT = WORK /" not in code
