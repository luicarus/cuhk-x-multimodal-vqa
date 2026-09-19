"""CPU evaluation and template-ordered submission export from verified runs."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from cuhkx.config import inside, require
from cuhkx.data.inputs import QA_FIELDS, read_csv
from cuhkx.data.validate import sha256
from cuhkx.evaluation.metric import evaluate_records
from cuhkx.inference.runner import output_directory, verify_completed_run
from cuhkx.inference.storage import atomic_write, json_bytes, run_lock
from cuhkx.submission.validator import validate_submission_records


def _verified_reference(config, relative):
    data = Path(config["data_root"])
    path = inside(data, relative)
    manifest = json.loads(inside(data, config["datasets"]["asset_manifest"]).read_text(encoding="utf-8"))
    entries = [e for e in manifest["files"] if e["path"] == "data/" + relative]
    require(len(entries) == 1 and path.stat().st_size == entries[0]["bytes"] and
            sha256(path) == entries[0]["sha256"], "reference file differs from asset manifest")
    return path


def _save_outputs(output, payloads):
    for name, payload in payloads.items():
        path = output / name
        require(not path.exists() or path.read_bytes() == payload, f"refusing to overwrite different {name}")
    for name, payload in payloads.items():
        if not (output / name).exists():
            atomic_write(output / name, payload)


def evaluate_run(config, run_id):
    with run_lock(output_directory(config, run_id)):
        output, targets, records, summary, contract = verify_completed_run(config, run_id)
        require(contract["dataset"] == "pilot", "only pilot has evaluation references")
        reference = _verified_reference(config, config["submission"]["references"]["pilot"])
        fields, all_refs = read_csv(reference)
        require(set(fields) == {*QA_FIELDS, "answer"}, "unexpected reference schema")
        refs = {row["qa_id"]: row for row in all_refs}
        require(len(refs) == len(all_refs), "duplicate reference QA IDs")
        chosen = []
        predictions = []
        for qa in targets:
            require(qa["qa_id"] in refs, "reference QA missing")
            row = refs[qa["qa_id"]]
            require({key: row[key] for key in QA_FIELDS} == qa, "reference question differs from inference input")
            chosen.append(row)
            predictions.append({"qa_id": qa["qa_id"], "prediction": records[qa["qa_id"]]["prediction"]})
        metrics = evaluate_records(chosen, predictions)
        result = {"status": "PASS", "execution_mode": contract["execution_mode"],
                  "run_signature": summary["signature"], "reference_sha256": sha256(reference),
                  "predictions_sha256": summary["output_sha256"]["predictions.csv"], "metrics": metrics}
        _save_outputs(output, {"metrics.json": json_bytes(result)})
        return result


def export_submission(config, run_id):
    with run_lock(output_directory(config, run_id)):
        output, targets, records, summary, contract = verify_completed_run(config, run_id)
        require(contract["execution_mode"] == "cloud", "simulation outputs cannot be exported as a submission")
        require(contract["model_source"].get("adapter", {}).get("purpose", "sft") == "sft", "smoke adapter outputs cannot be exported")
        require(contract["dataset"] == "test" and len(targets) == config["datasets"]["datasets"]["test"]["expected_qa"],
                "submission requires the complete test dataset")
        template_path = _verified_reference(config, config["submission"]["template"])
        fields, template = read_csv(template_path)
        require(set(records) == {row["qa_id"] for row in template} == {row["qa_id"] for row in targets}, "submission ID sets differ")
        rows = [{"qa_id": row["qa_id"], "prediction": records[row["qa_id"]]["prediction"]} for row in template]
        # The template defines export order; preserve the run's original target order in its contract.
        targets_by_id = {row["qa_id"]: row for row in targets}
        ordered_targets = [targets_by_id[row["qa_id"]] for row in template]
        report = validate_submission_records(ordered_targets, template, rows, test_fields=QA_FIELDS, template_fields=fields)
        require(report["valid"], f"submission validation failed: {report['checks']}")
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=config["submission"]["columns"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        payload = stream.getvalue().encode("utf-8")
        import hashlib
        report.update(run_signature=summary["signature"], template_sha256=sha256(template_path),
                      submission_sha256=hashlib.sha256(payload).hexdigest(), execution_mode="cloud")
        _save_outputs(output, {"submission.csv": payload, "submission_validation.json": json_bytes(report)})
        return report
