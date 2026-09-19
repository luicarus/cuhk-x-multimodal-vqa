import csv
import json

import pytest
import yaml

from test_cached_inputs import project, seal, write_csv
from test_ir4_runner import setup_run, FakeBackend
from cuhkx.config import InputError, load_config
from cuhkx.data.inputs import QA_FIELDS
from cuhkx.inference.runner import run_predictions
from cuhkx.submission.export import evaluate_run, export_submission


def execute(config, source, dataset="test", limit=None, mode="cloud"):
    return run_predictions(config, dataset, limit, "export", source, lambda: FakeBackend(["A", "B"]), execution_mode=mode)


def test_export_template_order_and_idempotence(project):
    qa_path = project / "data/qa/test.csv"
    with qa_path.open() as handle:
        qa = list(csv.DictReader(handle))
    write_csv(qa_path, QA_FIELDS, [{**row, "C": "C text", "D": "D text"} for row in qa])
    write_csv(project / "data/qa/sample_submission.csv", ["qa_id", "prediction"],
              [{"qa_id": "q1", "prediction": "A"}, {"qa_id": "q0", "prediction": "A"}])
    seal(project / "data")
    config, source = setup_run(project)
    execute(config, source)
    result = export_submission(config, "export")
    assert result["valid"]
    path = project / "outputs/export/submission.csv"
    assert path.read_text().splitlines() == ["qa_id,prediction", "q1,B", "q0,A"]
    assert export_submission(config, "export") == result
    path.write_text("changed")
    with pytest.raises(InputError, match="overwrite"):
        export_submission(config, "export")


@pytest.mark.parametrize("mode,limit,error", [("simulation", None, "simulation"), ("cloud", 1, "complete test")])
def test_export_rejects_simulation_and_partial(project, mode, limit, error):
    config, source = setup_run(project)
    execute(config, source, limit=limit, mode=mode)
    with pytest.raises(InputError, match=error):
        export_submission(config, "export")
    assert not (project / "outputs/export/submission.csv").exists()


def test_evaluate_exact_pilot_subset_and_tamper_rejection(project):
    path = project / "configs/datasets.yaml"
    datasets = yaml.safe_load(path.read_text())
    binding = {**datasets["datasets"]["test"], "split": "train"}
    datasets["datasets"]["pilot"] = binding
    path.write_text(yaml.safe_dump(datasets))
    index_path = project / "data" / binding["frame_index"]
    with index_path.open() as handle:
        reader = csv.DictReader(handle)
        fields, entries = reader.fieldnames, list(reader)
    entries[0]["split"] = "train"
    write_csv(index_path, fields, entries)
    meta_path = project / "data" / binding["frames_root"] / entries[0]["metadata_path"]
    meta = json.loads(meta_path.read_text())
    meta["split"] = "train"
    meta_path.write_text(json.dumps(meta))
    with (project / "data/qa/test.csv").open() as handle:
        qa = list(csv.DictReader(handle))
    reference = project / "data/references/pilot_answers.csv"
    write_csv(reference, [*QA_FIELDS, "answer"], [{**row, "answer": "A"} for row in qa])
    seal(project / "data")
    config, source = setup_run(project)
    execute(config, source, dataset="pilot", limit=1, mode="simulation")
    metrics = evaluate_run(config, "export")
    assert metrics["execution_mode"] == "simulation"
    assert metrics["metrics"]["total"] == metrics["metrics"]["correct"] == 1
    assert evaluate_run(config, "export") == metrics
    (project / "outputs/export/predictions.csv").write_text("qa_id,prediction\nq0,B\n")
    with pytest.raises(InputError, match="hash"):
        evaluate_run(config, "export")
