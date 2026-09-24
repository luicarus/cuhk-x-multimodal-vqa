"""One signed target set, atomic per-QA checkpoints, and verified resume."""
from __future__ import annotations

import csv
import io
import json
import re
import time
from pathlib import Path

from PIL import Image

from cuhkx.config import inside, require
from cuhkx.data.inputs import load_qa, pending_targets, select_targets
from cuhkx.data.validate import check_inputs, fingerprint, sha256
from cuhkx.evaluation.metric import available_option_letters
from cuhkx.inference.profiling import (RequestTimer, gpu_memory_peak_snapshot,
                                       request_metrics, sample_gpu_memory)
from cuhkx.inference.prompt import (PROMPT_VERSION, allowed_answer_outputs, build_mcq_prompt,
                                    detect_prompt_leakage, parse_model_answer)
from cuhkx.inference.storage import atomic_write, run_lock, write_json


ARTIFACTS = ("checkpoint.jsonl", "predictions.csv", "audit.jsonl")


def profiling_torch():
    """Return torch if it is importable, else None.

    Memory sampling is best-effort: CPU test runs and simulation backends have no
    CUDA, and profiling must never be the reason a prediction run fails.
    """
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


def _run_one(backend, qa, prompts, config, checked, data, records, target_ids,
             signature, timer, output, targets_len):
    """Sequential path: one request at a time. Returns the request status."""
    qa_id = qa["qa_id"]
    images = []
    raw, prediction, error, status = "", None, None, "failed"
    request_started = time.perf_counter()
    image_ms = generate_ms = 0.0
    metrics = {}
    try:
        phase_started = time.perf_counter()
        for relative in checked["selected_frames"][qa_id]:
            with Image.open(inside(data, relative)) as image:
                image.load()
                images.append(image.copy())
        image_ms = (time.perf_counter() - phase_started) * 1000.0
        phase_started = time.perf_counter()
        raw = backend.generate(images, prompts[qa_id],
            allowed_outputs=allowed_answer_outputs(qa["category"], available_option_letters(qa)),
            max_new_tokens=config["baseline"]["generation"]["max_new_tokens"])
        generate_ms = (time.perf_counter() - phase_started) * 1000.0
        metrics = request_metrics(backend)
        require(isinstance(raw, str), "backend output must be text")
        parsed = parse_model_answer(raw, category=qa["category"], valid_options=available_option_letters(qa))
        if parsed.is_valid and raw in allowed_answer_outputs(qa["category"], available_option_letters(qa)):
            status, prediction = "valid", parsed.prediction
        else:
            status, error = "invalid", parsed.error or "output violates constrained answer space"
    except Exception as exc:
        raw, prediction, error = "", None, f"{type(exc).__name__}: {exc}"
    finally:
        for image in images:
            image.close()
    record = {"qa_id": qa_id, "status": status, "raw_output": raw, "prediction": prediction,
              "error": error, "attempts": records.get(qa_id, {}).get("attempts", 0) + 1,
              "signature": signature, "prompt_sha256": fingerprint(prompts[qa_id]),
              "selected_frames": checked["selected_frames"][qa_id]}
    record["record_sha256"] = fingerprint(record)
    records[qa_id] = record
    ordered = [records[q] for q in target_ids if q in records]
    atomic_write(output / "checkpoint.jsonl", _jsonl(ordered))
    # Recorded after the checkpoint write so total_ms covers the full
    # per-request cost, including the O(n) rewrite of checkpoint.jsonl.
    timer.record(qa_id, total_ms=(time.perf_counter() - request_started) * 1000.0,
                 image_ms=image_ms, generate_ms=generate_ms, status=status,
                 ttft_ms=metrics.get("ttft_ms"),
                 generation_tokens=metrics.get("generation_tokens"))
    print(f"{qa_id}: {status} ({sum(r['status'] == 'valid' for r in records.values())}/{targets_len} valid)", flush=True)
    return status


def verify_completed_run(config, run_id, prepared_input=None):
    """Read-only validation for evaluation/export. Caller holds the run lock."""
    output = output_directory(config, run_id)
    state = json.loads((output / "resume_state.json").read_text(encoding="utf-8"))
    contract = state["contract"]
    require(state["signature"] == fingerprint(contract), "run contract hash mismatch")
    # Runs produced before the generation engine was part of the signed
    # contract have no engine field. Read them as the Transformers engine they
    # necessarily used, so previously recorded results stay verifiable.
    contract.setdefault("engine", "transformers")
    contract.setdefault("engine_options", {})
    dataset = contract["dataset"]
    if prepared_input is None:
        binding = config["datasets"]["datasets"][dataset]
        targets = select_targets(load_qa(inside(Path(config["data_root"]), binding["qa"]),
                                        binding["expected_qa"]), len(contract["target_ids"]))
        checked = check_inputs(config, dataset, len(targets))
    else:
        require(dataset in ("dev", "confirm"), "custom inputs only supported for training evaluation")
        targets, checked = prepared_input
    prompts = {row["qa_id"]: build_mcq_prompt(row) for row in targets}
    for row in targets:
        require(not detect_prompt_leakage(prompts[row["qa_id"]], row), "prompt leakage in current input")
    expected = {"schema_version": 1, "runner_version": 1, "execution_mode": contract["execution_mode"],
                "baseline": config["baseline"], "model_source": contract["model_source"], "dataset": dataset,
                "target_ids": checked["target_ids"], "input_signature": checked["input_signature"],
                "engine": contract["engine"], "engine_options": contract["engine_options"],
                "prompt_hashes": {q: fingerprint(p) for q, p in prompts.items()}}
    require(contract == expected, "current config/input differs from finished run")
    require(contract["execution_mode"] in ("cloud", "simulation"), "unknown execution mode")
    require(contract["model_source"]["model_id"] == config["baseline"]["model"]["id"] and
            contract["model_source"]["revision"] == config["baseline"]["model"]["revision"], "recorded model differs")
    records = _load_checkpoint(output, targets, prompts, state["signature"])
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    require(summary["status"] == "PASS" and summary["execution_mode"] == contract["execution_mode"], "run is not a completed matching result")
    _verify_finished(output, summary, records, checked["target_ids"], state["signature"])
    return output, targets, records, summary, contract


def output_directory(config: dict, run_id: str) -> Path:
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id) is not None, "invalid run-id")
    return inside(Path(config["project_root"]) / "outputs", run_id)


def _jsonl(records):
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records).encode()


def _predictions(records):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["qa_id", "prediction"], lineterminator="\n")
    writer.writeheader()
    writer.writerows({"qa_id": row["qa_id"], "prediction": row["prediction"]}
                     for row in records if row["status"] == "valid")
    return stream.getvalue().encode()


def _load_checkpoint(output, targets, prompts, signature):
    path = output / "checkpoint.jsonl"
    if not path.exists():
        require(not any((output / name).exists() for name in ("predictions.csv", "audit.jsonl")), "checkpoint missing beside existing outputs")
        return {}
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    by_id = {row["qa_id"]: row for row in targets}
    result = {}
    for record in records:
        qa_id = record["qa_id"]
        require(qa_id in by_id and qa_id not in result, "checkpoint contains duplicate or out-of-scope QA")
        require(record.get("record_sha256") == fingerprint({k: v for k, v in record.items() if k != "record_sha256"}), "checkpoint record hash mismatch")
        require(record["signature"] == signature and record["prompt_sha256"] == fingerprint(prompts[qa_id]), "checkpoint protocol mismatch")
        require(type(record["attempts"]) is int and record["attempts"] >= 1, "invalid attempt count")
        qa = by_id[qa_id]
        require(record["status"] in ("valid", "invalid", "failed"), "invalid checkpoint status")
        parsed = parse_model_answer(record["raw_output"], category=qa["category"], valid_options=available_option_letters(qa))
        allowed = allowed_answer_outputs(qa["category"], available_option_letters(qa))
        valid = parsed.is_valid and record["raw_output"] in allowed
        if record["status"] == "valid":
            require(valid and parsed.prediction == record["prediction"] and not record["error"], "checkpoint answer is invalid")
        else:
            require(record["prediction"] is None and bool(record["error"]), "invalid failure record")
            require(not valid, "valid answer incorrectly marked as failed")
        result[qa_id] = record
    expected_order = [row["qa_id"] for row in targets if row["qa_id"] in result]
    require([row["qa_id"] for row in records] == expected_order, "checkpoint target order mismatch")
    return result


def _verify_finished(output, summary, records, target_ids, signature):
    require(summary["signature"] == signature and summary["target_ids"] == target_ids, "summary protocol/target mismatch")
    ordered = [records[q] for q in target_ids if q in records]
    for name in ARTIFACTS:
        path = output / name
        require(path.is_file() and sha256(path) == summary["output_sha256"][name], f"result hash mismatch: {name}")
    require((output / "checkpoint.jsonl").read_bytes() == _jsonl(ordered), "checkpoint serialization mismatch")
    require((output / "audit.jsonl").read_bytes() == _jsonl(ordered), "audit differs from checkpoint")
    require((output / "predictions.csv").read_bytes() == _predictions(ordered), "predictions differ from checkpoint")
    counts = {status: sum(row["status"] == status for row in ordered) for status in ("valid", "invalid", "failed")}
    counts["pending"] = len(target_ids) - len(ordered)
    counts["prompt_leakage"] = 0
    require(summary["counts"] == counts, "summary counts differ from checkpoint")
    complete = counts["valid"] == len(target_ids)
    require((summary["status"] == "PASS") == complete, "summary falsely claims completion")


def _release_backend(backend):
    """Release engine-held resources once a run is finished.

    Only backends that own something outside this process define close(); the
    single-engine paths need nothing. Cleanup never fails a run: the results are
    already written and verified by the time this is called.
    """
    close = getattr(backend, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception:                     # noqa: BLE001 - cleanup only
        pass


def run_predictions(config, dataset, limit, run_id, model_source, backend_factory, *, resume=False,
                    execution_mode="cloud", fail_fast=True, prepared_input=None,
                    engine="transformers", engine_options=None):
    """Factory is lazy: a complete verified run never loads a model.

    Production callers must verify_weights before passing model_source. CPU tests
    use execution_mode='simulation', which cannot resume as a cloud run.

    ``engine`` and ``engine_options`` are part of the signed contract. Two runs
    that differ only in generation engine -- for example Transformers versus
    vLLM -- must never share a run signature, or a resumed run could silently
    mix predictions produced by different engines.
    """
    require(execution_mode in ("cloud", "simulation"), "unknown execution mode")
    require(isinstance(engine, str) and engine, "generation engine must be named")
    require(config["baseline"]["model"]["revision"] is not None, "model revision is not pinned")
    require(model_source["model_id"] == config["baseline"]["model"]["id"] and
            model_source["revision"] == config["baseline"]["model"]["revision"], "model source mismatch")
    require(config["baseline"]["prompt_version"] == PROMPT_VERSION, "prompt implementation version mismatch")
    data = Path(config["data_root"])
    if prepared_input is None:
        checked = check_inputs(config, dataset, limit)
        binding = config["datasets"]["datasets"][dataset]
        targets = select_targets(load_qa(inside(data, binding["qa"]), binding["expected_qa"]), limit)
    else:
        from cuhkx.data.inputs import QA_FIELDS
        require(dataset in ("dev", "confirm") and limit is None, "invalid prepared evaluation request")
        targets, checked = prepared_input
        require(targets and all(tuple(row) == QA_FIELDS for row in targets), "prepared inputs must be answer-free")
        ids = [row["qa_id"] for row in targets]
        require(len(set(ids)) == len(ids) and ids == checked["target_ids"], "prepared target IDs differ")
    prompts = {row["qa_id"]: build_mcq_prompt(row) for row in targets}
    for row in targets:
        require(not detect_prompt_leakage(prompts[row["qa_id"]], row), f"prompt leakage detected: {row['qa_id']}")
    target_ids = checked["target_ids"]
    contract = {"schema_version": 1, "runner_version": 1, "execution_mode": execution_mode,
                "baseline": config["baseline"], "model_source": model_source, "dataset": dataset,
                "target_ids": target_ids, "input_signature": checked["input_signature"],
                "engine": engine, "engine_options": dict(engine_options or {}),
                "prompt_hashes": {q: fingerprint(p) for q, p in prompts.items()}}
    signature = fingerprint(contract)
    output = output_directory(config, run_id)
    with run_lock(output):
        state = output / "resume_state.json"
        if state.exists():
            require(resume, "run already exists; use --resume")
            require(json.loads(state.read_text(encoding="utf-8")) == {"signature": signature, "contract": contract}, "resume signature differs; use a new run-id")
        else:
            existing = {p.name for p in output.iterdir()} - {".run.lock"}
            if existing:
                require(existing == {"resolved_config.json", "input_check.json"}, "existing outputs have no resume state")
                prepared = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
                require(prepared.get("phase") == "prepared_only" and prepared["target_ids"] == target_ids and
                        prepared["input_signature"] == checked["input_signature"], "prepared input differs; use a new run-id")
            write_json(state, {"signature": signature, "contract": contract})
        records = _load_checkpoint(output, targets, prompts, signature)
        summary_path = output / "run_summary.json"
        if summary_path.exists():
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            require(previous["signature"] == signature and previous["target_ids"] == target_ids, "summary scope mismatch")
            require(previous["status"] in ("RUNNING", "PASS", "FAIL"), "unknown summary status")
            if previous["status"] != "RUNNING":
                _verify_finished(output, previous, records, target_ids, signature)
                if previous["status"] == "PASS":
                    return previous
        write_json(output / "resolved_config.json", {"phase": "inference", "config": config,
                   "signature": signature, "contract": contract})
        write_json(output / "input_check.json", checked)
        write_json(summary_path, {"status": "RUNNING", "signature": signature, "target_ids": target_ids})
        completed = {q for q, row in records.items() if row["status"] == "valid"}
        pending = pending_targets(targets, completed)
        started = time.perf_counter()
        backend = None
        run_error = None
        if pending:
            backend = backend_factory()
        timer = RequestTimer()
        memory_before = None
        memory_after = None
        memory_peaks = {}
        if pending:
            memory_before = sample_gpu_memory(profiling_torch())
        options = engine_options or {}
        engine_batch_limit = int(options.get("max_num_seqs") or 1)
        data_parallel_size = int(options.get("data_parallel_size") or 1)
        batch_size = int(options.get("runner_batch_size") or engine_batch_limit)
        require(batch_size >= 1, "runner batch size must allow at least one request")
        require(batch_size <= engine_batch_limit * data_parallel_size,
                "runner batch size exceeds aggregate per-replica scheduler capacity")
        if batch_size > 1:
            # A concurrent scheduler is pointless unless the runner submits more
            # than one request at a time; the engine setting selects the path.
            require(hasattr(backend, "generate_batch"),
                    f"engine {engine} does not support batched submission")
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            if batch_size == 1:
                status = _run_one(backend, batch[0], prompts, config, checked, data,
                                  records, target_ids, signature, timer, output,
                                  len(targets))
                if status != "valid" and fail_fast:
                    break
                continue

            # Load every frame set first: the decode is cheap (~8 ms per request)
            # and keeping it outside the timed region shows the batched call as
            # one engine operation, which is what the batch view wants to measure.
            batch_images = []
            image_started = time.perf_counter()
            try:
                for qa in batch:
                    images = []
                    for relative in checked["selected_frames"][qa["qa_id"]]:
                        with Image.open(inside(data, relative)) as image:
                            image.load()
                            images.append(image.copy())
                    batch_images.append(images)
                triples = [(batch_images[index], prompts[qa["qa_id"]],
                            allowed_answer_outputs(qa["category"], available_option_letters(qa)))
                           for index, qa in enumerate(batch)]
                raws = backend.generate_batch(
                    triples, max_new_tokens=config["baseline"]["generation"]["max_new_tokens"])
                batch_ms = (time.perf_counter() - image_started) * 1000.0
                metrics = request_metrics(backend)
                for qa, images, raw in zip(batch, batch_images, raws, strict=True):
                    qa_id = qa["qa_id"]
                    # Per-request state must reset inside the loop: a variable set
                    # on a failed request would otherwise leak into the next one.
                    status, prediction, error, raw_output = "failed", None, None, ""
                    try:
                        require(isinstance(raw, str), "backend output must be text")
                        parsed = parse_model_answer(raw, category=qa["category"],
                                                    valid_options=available_option_letters(qa))
                        if parsed.is_valid and raw in allowed_answer_outputs(
                                qa["category"], available_option_letters(qa)):
                            status, prediction = "valid", parsed.prediction
                        else:
                            status = "invalid"
                            prediction = None
                            error = parsed.error or "output violates constrained answer space"
                        raw_output = raw
                    except Exception as exc:            # noqa: BLE001 - per-request guard
                        status, prediction, error, raw_output = "failed", None, f"{type(exc).__name__}: {exc}", ""
                        run_error = error
                    record = {"qa_id": qa_id, "status": status, "raw_output": raw_output,
                              "prediction": prediction, "error": error,
                              "attempts": records.get(qa_id, {}).get("attempts", 0) + 1,
                              "signature": signature,
                              "prompt_sha256": fingerprint(prompts[qa_id]),
                              "selected_frames": checked["selected_frames"][qa_id]}
                    record["record_sha256"] = fingerprint(record)
                    records[qa_id] = record
                    print(f"{qa_id}: {status} "
                          f"({sum(r['status'] == 'valid' for r in records.values())}/{len(targets)} valid)",
                          flush=True)
            finally:
                for images in batch_images:
                    for image in images:
                        image.close()
            # The batch record lands after its checkpoint write, matching the
            # sequential path's convention that timing includes bookkeeping.
            ordered = [records[q] for q in target_ids if q in records]
            atomic_write(output / "checkpoint.jsonl", _jsonl(ordered))
            timer.record_batch(
                batch_ms=(time.perf_counter() - image_started) * 1000.0,
                image_ms=0.0,
                generate_ms=metrics.get("e2e_ms") or 0.0,
                size=len(batch),
                ttft_ms=metrics.get("ttft_ms"),
                generation_tokens=metrics.get("generation_tokens") or 0,
            )
            if fail_fast and any(records[q["qa_id"]]["status"] != "valid" for q in batch):
                break
        ordered = [records[q] for q in target_ids if q in records]
        atomic_write(output / "checkpoint.jsonl", _jsonl(ordered))
        atomic_write(output / "predictions.csv", _predictions(ordered))
        atomic_write(output / "audit.jsonl", _jsonl(ordered))
        if pending:
            # Sampled while the engine is still resident: this is the figure that
            # decides whether a larger batch fits.
            memory_after = sample_gpu_memory(profiling_torch())
            memory_peaks = gpu_memory_peak_snapshot(profiling_torch())
        counts = {status: sum(row["status"] == status for row in ordered) for status in ("valid", "invalid", "failed")}
        counts.update(pending=len(targets) - len(ordered), prompt_leakage=0)
        summary = {"status": "PASS" if counts["valid"] == len(targets) else "FAIL",
                   "execution_mode": execution_mode, "signature": signature, "target_ids": target_ids,
                   "counts": counts, "resumed_valid": len(completed), "elapsed_seconds": time.perf_counter() - started,
                   # Timing is advisory evidence, not part of the verified
                   # contract: _verify_finished never reads it, so a metric change
                   # cannot invalidate an already finished run.
                   "latency": timer.summary(),
                   "gpu_memory": {"before": memory_before or {}, "after": memory_after or {},
                                  "peaks": memory_peaks},
                   "backend": backend.metadata() if backend else {"loaded_this_run": False}, "error": run_error,
                   "output_sha256": {name: sha256(output / name) for name in ARTIFACTS}}
        write_json(summary_path, summary)
        _verify_finished(output, summary, records, target_ids, signature)
        # Release the engine after every figure has been sampled and written. In
        # data-parallel mode each replica is a separate process holding most of a
        # GPU, and they are not daemons (vLLM spawns workers beneath them), so
        # without this a smoke run would keep both cards reserved for the rest of
        # the session.
        _release_backend(backend)
        return summary
