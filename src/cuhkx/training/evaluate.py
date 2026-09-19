"""Baseline/adapter evaluation on the same dev or confirmation targets."""
import json
from pathlib import Path

from cuhkx.config import require
from cuhkx.data.validate import fingerprint
from cuhkx.evaluation.metric import evaluate_records
from cuhkx.inference.runner import output_directory, run_predictions, verify_completed_run
from cuhkx.inference.storage import json_bytes, run_lock, atomic_write
from cuhkx.inference.weights import verify_weights
from cuhkx.training.adapter import verify_adapter
from cuhkx.training.dataset import prepare_data, require_ready


def evaluation_input(config, settings, split):
    require(split in ("dev", "confirm"), "only dev/confirm evaluation is allowed")
    prepared = prepare_data(config, settings)
    require_ready(prepared, (split,))
    samples = prepared["samples"][split]
    targets = [sample["qa"] for sample in samples]
    selected = {s["qa"]["qa_id"]:s["frame_paths"] for s in samples}
    checked = {"status":"PASS", "dataset":split, "target_ids":[q["qa_id"] for q in targets],
               "selected_frames":selected, "input_signature":fingerprint({"data":prepared["data_signature"],
               "baseline":config["baseline"], "split":split, "qa":targets, "frames":selected})}
    return (targets,checked), samples


def evaluate_training(config, settings, split, run_id, weights, adapter=None, resume=False):
    prepared, samples = evaluation_input(config, settings, split)
    qwen35 = config["baseline"]["model"]["id"] == "Qwen/Qwen3.5-4B"
    if qwen35:
        from cuhkx.inference.qwen35_weights import verify_qwen35_weights
        source = verify_qwen35_weights(weights, config["baseline"]["model"])
    else:
        from cuhkx.inference.weights import verify_weights
        source = verify_weights(weights, config["baseline"]["model"])
    if adapter is not None:
        source["adapter"] = verify_adapter(adapter, source)
    output = output_directory(config, run_id)
    if qwen35:
        from cuhkx.inference.qwen35 import Qwen35Backend
        backend_factory = lambda: Qwen35Backend(config["baseline"], weights, adapter=adapter)
    else:
        from cuhkx.inference.qwen import QwenBackend
        backend_factory = lambda: QwenBackend(config["baseline"], weights, output / "offload", adapter=adapter)
    result = run_predictions(config, split, None, run_id, source,
        backend_factory,
        resume=resume, prepared_input=prepared)
    if result["status"] != "PASS":
        return result
    with run_lock(output):
        _,targets,records,summary,_ = verify_completed_run(config,run_id,prepared_input=prepared)
        references = [{**s["qa"],"answer":s["answer"]} for s in samples]
        predictions = [{"qa_id":q["qa_id"],"prediction":records[q["qa_id"]]["prediction"]} for q in targets]
        metrics = {"status":"PASS","dataset":split,"run_signature":summary["signature"],
                   "metrics":evaluate_records(references,predictions)}
        path = output/"metrics.json"
        content = json_bytes(metrics)
        require(not path.exists() or path.read_bytes()==content, "refusing to overwrite different metrics")
        if not path.exists():atomic_write(path,content)
        return metrics
