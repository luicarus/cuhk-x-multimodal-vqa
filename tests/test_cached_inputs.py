from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

from cuhkx.cli import main, prepare_run
from cuhkx.config import InputError, load_config
from cuhkx.data.inputs import QA_FIELDS, pending_targets, select_targets
from cuhkx.data.validate import check_inputs


PROJECT = Path(__file__).resolve().parents[1]
CACHE = "uniform_time_v1/ir/520837f5b798f45a"


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def seal(data):
    entries = []
    for path in sorted(data.rglob("*")):
        if path.is_file() and path.name != "asset_manifest.json":
            entries.append({"path": "data/" + path.relative_to(data).as_posix(), "bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (data / "asset_manifest.json").write_text(json.dumps({"schema_version": 1, "baseline_id": "ir4_7b_v1", "files": entries}), encoding="utf-8")


@pytest.fixture
def project(tmp_path):
    project = tmp_path / "project"
    config = project / "configs"
    config.mkdir(parents=True)
    for name in ("baseline", "datasets", "submission"):
        shutil.copy2(PROJECT / f"configs/{name}.yaml", config)
    baseline_path = config / "baseline.yaml"
    baseline = yaml.safe_load(baseline_path.read_text())
    baseline["model"]["revision"] = None  # Fixture is independent of the user's pinned production run.
    baseline_path.write_text(yaml.safe_dump(baseline))
    path = config / "datasets.yaml"
    datasets = yaml.safe_load(path.read_text())
    datasets["datasets"]["test"]["expected_qa"] = 2
    path.write_text(yaml.safe_dump(datasets))
    data = project / "data"
    qa = [dict(zip(QA_FIELDS, [f"q{i}", "HAU", "absent/video.mp4", "single", "Action?", "A text", "B text", "", ""])) for i in range(2)]
    write_csv(data / "qa/test.csv", QA_FIELDS, qa)
    write_csv(data / "qa/sample_submission.csv", ["qa_id", "prediction"], [{"qa_id": f"q{i}", "prediction": "A"} for i in range(2)])
    folder = data / "frames/test" / CACHE / "test/clip"
    folder.mkdir(parents=True)
    positions = [(i + 0.5) / 8 for i in range(8)]
    names = [f"frame_{i:04}.jpg" for i in range(8)]
    for i, name in enumerate(names):
        Image.new("RGB", (448, 448), (i * 30, 0, 0)).save(folder / name)
    meta = {"sample_id": "clip", "clip_key": "test:clip", "source": "HAU", "split": "test", "status": "ok",
            "modality": "IR", "config_hash": "520837f5b798f45a", "protocol_version": "uniform_time_v1",
            "qa_ids": ["q0", "q1"], "qa_count": 2, "requested_frames": 8, "actual_frames": 8,
            "config": {"num_frames": 8, "image_size": 448, "resize_mode": "letterbox"},
            "frame_files": names, "source_frame_indices": list(range(8)), "normalized_positions": positions,
            "target_timestamps_seconds": positions, "actual_timestamps_seconds": positions}
    (folder / "metadata.json").write_text(json.dumps(meta))
    row = {k: meta[k] for k in ("sample_id", "clip_key", "source", "split", "status", "modality", "config_hash", "protocol_version", "qa_count")}
    row.update(qa_ids="q0;q1", metadata_path=f"{CACHE}/test/clip/metadata.json",
               frame_paths=";".join(f"{CACHE}/test/clip/{name}" for name in names))
    for field in ("source_frame_indices", "normalized_positions", "target_timestamps_seconds", "actual_timestamps_seconds"):
        row[field] = ";".join(map(str, meta[field]))
    write_csv(data / f"frames/test/{CACHE}/frame_index.csv", list(row), [row])
    seal(data)
    return project


def change_index(project, mutate):
    path = project / f"data/frames/test/{CACHE}/frame_index.csv"
    with path.open() as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    mutate(rows)
    write_csv(path, fields, rows)
    seal(project / "data")


def test_valid_check_selection_and_idempotent_run(project):
    config = load_config(project)
    result = check_inputs(config, "test")
    assert result["target_ids"] == ["q0", "q1"]
    assert result["images_decoded"] == 8
    assert [Path(p).name for p in result["selected_frames"]["q0"]] == [f"frame_{i:04}.jpg" for i in (1, 3, 5, 7)]
    assert result["model_revision_pinned"] is False
    output = prepare_run(config, result, "smoke")
    before = (output / "resolved_config.json").read_bytes()
    prepare_run(config, result, "smoke")
    assert (output / "resolved_config.json").read_bytes() == before
    with pytest.raises(InputError, match="differ"):
        prepare_run(config, check_inputs(config, "test", 1), "smoke")


def test_subset_before_resume():
    rows = [{"qa_id": str(i)} for i in range(25)]
    targets = select_targets(rows, 16)
    assert [q["qa_id"] for q in pending_targets(targets, {str(i) for i in range(10)})] == [str(i) for i in range(10, 16)]
    assert pending_targets(targets, {str(i) for i in range(16)}) == []
    with pytest.raises(InputError, match="outside"):
        pending_targets(targets, {"24"})


@pytest.mark.parametrize("limit", [0, -1, 3])
def test_bad_limit(project, limit):
    with pytest.raises(InputError, match="limit"):
        check_inputs(load_config(project), "test", limit)


@pytest.mark.parametrize("mutation", [
    lambda rs: rs[0].update(qa_ids="q0"),
    lambda rs: rs.append({**rs[0], "sample_id": "another"}),
    lambda rs: rs[0].update(status="failed"),
    lambda rs: rs[0].update(modality="Depth"),
    lambda rs: rs[0].update(frame_paths=";".join(rs[0]["frame_paths"].split(";")[:7])),
    lambda rs: rs[0].update(metadata_path="../../outside.json"),
])
def test_bad_cache_is_not_silently_filtered(project, mutation):
    change_index(project, mutation)
    with pytest.raises(InputError):
        check_inputs(load_config(project), "test")


def test_no_answer_columns_even_when_manifest_matches(project):
    path = project / "data/qa/test.csv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    write_csv(path, [*QA_FIELDS, "answer"], [{**r, "answer": "A"} for r in rows])
    seal(project / "data")
    with pytest.raises(InputError, match="whitelist"):
        check_inputs(load_config(project), "test")


def test_tampered_image_and_corrupt_image(project):
    path = project / f"data/frames/test/{CACHE}/test/clip/frame_0000.jpg"
    path.write_bytes(b"not an image")
    with pytest.raises(InputError, match="asset"):
        check_inputs(load_config(project), "test")
    seal(project / "data")
    with pytest.raises(OSError):
        check_inputs(load_config(project), "test")


@pytest.mark.parametrize("field,value", [("revision", "main"), ("id", "Qwen/Qwen2.5-VL-3B-Instruct")])
def test_bad_model_config(project, field, value):
    path = project / "configs/baseline.yaml"
    config = yaml.safe_load(path.read_text())
    config["model"][field] = value
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(InputError):
        load_config(project)


def test_unpinned_model_cannot_start_inference(project):
    with pytest.raises(InputError, match="not pinned"):
        load_config(project, require_revision=True)


def test_relocation_and_no_gpu_imports(project, tmp_path):
    before = check_inputs(load_config(project), "test")["input_signature"]
    relocated = tmp_path / "elsewhere"
    shutil.move(str(project / "data"), relocated)
    config = load_config(project, relocated)
    assert check_inputs(config, "test")["input_signature"] == before
    assert main(["check", "--project-root", str(project), "--data-root", str(relocated), "--dataset", "test"]) == 0
    code = "import sys; from cuhkx.cli import main; result=main(sys.argv[1:]); assert not {'torch','transformers','bitsandbytes','av'} & set(sys.modules); raise SystemExit(result)"
    proc = subprocess.run([sys.executable, "-c", code, "check", "--project-root", str(project),
                           "--data-root", str(relocated), "--dataset", "test"], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_duplicate_yaml_keys_rejected(project):
    path = project / "configs/baseline.yaml"
    with path.open("a") as handle:
        handle.write("\nbaseline_id: ir4_7b_v1\n")
    with pytest.raises(InputError, match="duplicate"):
        load_config(project)


def test_malformed_yaml_returns_cli_error(project, capsys):
    (project / "configs/baseline.yaml").write_text("model: [unterminated")
    assert main(["check", "--project-root", str(project), "--dataset", "test"]) == 2
    assert "Input check failed" in capsys.readouterr().err


@pytest.mark.parametrize("field,value", [("cache_num_frames", 4), ("input_num_frames", 2),
                                        ("indices", [0, 2, 4, 6])])
def test_wrong_frame_protocol_rejected(project, field, value):
    path = project / "configs/baseline.yaml"
    config = yaml.safe_load(path.read_text())
    config["frames"][field] = value
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(InputError, match="frame protocol"):
        load_config(project)


def test_duplicate_qa_rejected(project):
    path = project / "data/qa/test.csv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    rows[1]["qa_id"] = rows[0]["qa_id"]
    write_csv(path, QA_FIELDS, rows)
    seal(project / "data")
    with pytest.raises(InputError, match="duplicate QA"):
        check_inputs(load_config(project), "test")
