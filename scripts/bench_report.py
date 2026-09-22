"""Compare inference runs on infrastructure metrics.

This lane is judged on serving efficiency, not answer accuracy, so the report
tracks throughput, latency distribution, where the time goes, and whether the
engine change altered the outputs at all.

Usage::

    python scripts/bench_report.py                     # every run under outputs/
    python scripts/bench_report.py --runs a b c        # specific run ids
    python scripts/bench_report.py --baseline a --candidate b

Reads only finished runs. A run without latency data (produced before the
profiling was added) still contributes end-to-end numbers, and the report says so
rather than printing blanks.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def read_contract(root: Path) -> dict:
    """Read the signed contract, which is where engine identity actually lives.

    run_summary.json carries results and backend metadata but not the engine
    name; that is in resume_state.contract. Reading only the summary silently
    labelled every vLLM run as "transformers".
    """
    state = root / "resume_state.json"
    if not state.is_file():
        return {}
    try:
        return json.loads(state.read_text(encoding="utf-8")).get("contract") or {}
    except (ValueError, OSError):
        return {}


def load_run(outputs: Path, run_id: str) -> dict:
    root = outputs / run_id
    summary_path = root / "run_summary.json"
    if not summary_path.is_file():
        raise ValueError(f"run has no run_summary.json: {run_id}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    backend = summary.get("backend") or {}
    contract = read_contract(root)
    targets = summary.get("target_ids") or []
    counts = summary.get("counts") or {}
    # Prefer predictions.csv, but a run that never exported it still has the same
    # answers in checkpoint.jsonl.
    source = prediction_source(root)
    predictions = read_predictions(root / "predictions.csv")
    if not predictions and source and source.name == "checkpoint.jsonl":
        predictions = read_targets(source)
    elapsed = float(summary.get("elapsed_seconds") or 0.0)
    load_seconds = float(backend.get("load_seconds") or 0.0)
    # Engine load is excluded from per-request cost: it is a one-off the batch
    # curve amortizes, and mixing it in would make a 16-request smoke look slow.
    inference_seconds = max(0.0, elapsed - load_seconds)
    valid = int(counts.get("valid") or 0)
    # A run resumed to completion executes no requests, so its wall clock is pure
    # bookkeeping. Reporting it as throughput would invent a huge fake number.
    executed = len(targets) - int(summary.get("resumed_valid") or 0)
    measured = executed > 0 and bool(backend) and backend.get("backend") != ""
    return {
        "run_id": run_id,
        "engine": contract.get("engine") or summary.get("engine") or "transformers",
        "engine_options": contract.get("engine_options") or summary.get("engine_options") or {},
        "backend": backend.get("backend", "?"),
        "backend_metadata": backend,
        "dataset": contract.get("dataset") or "?",
        "requests": len(targets),
        "executed": executed,
        "measured": measured,
        "valid": valid,
        "invalid": int(counts.get("invalid") or 0),
        "failed": int(counts.get("failed") or 0),
        "pending": int(counts.get("pending") or 0),
        "status": summary.get("status"),
        "elapsed_seconds": round(elapsed, 3),
        "load_seconds": round(load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "seconds_per_request": round(inference_seconds / len(targets), 4) if targets else None,
        "requests_per_second": round(len(targets) / inference_seconds, 4) if inference_seconds else None,
        "latency": summary.get("latency") or {},
        "gpu_memory": summary.get("gpu_memory") or {},
        "tensor_parallel_size": backend.get("tensor_parallel_size"),
        "attention_backend": backend.get("attention_backend"),
        "versions": backend.get("versions") or {},
        "prediction_source": source.name if source else None,
        "predictions": predictions,
    }


def read_predictions(path: Path) -> dict:
    """Read a run's predictions, or an empty map with a recorded reason.

    A finished run normally has predictions.csv, but an older or partially
    exported run may not. Returning {} silently made the whole comparison look
    like "no overlapping predictions", which hides the real cause.
    """
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["qa_id"]: row["prediction"] for row in csv.DictReader(handle)}


def prediction_source(root: Path) -> Path | None:
    for name in ("predictions.csv", "checkpoint.jsonl"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def read_targets(path: Path) -> dict:
    """Fall back to checkpoint.jsonl when predictions.csv is absent."""
    predictions = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("status") == "valid" and record.get("prediction"):
                predictions[record["qa_id"]] = record["prediction"]
    return predictions


def agreement(left: dict, right: dict) -> dict:
    """Compare two runs request by request on the shared target set."""
    shared = sorted(set(left["predictions"]) & set(right["predictions"]))
    if not shared:
        return {"shared": 0}
    same = sum(1 for qa_id in shared
               if left["predictions"][qa_id] == right["predictions"][qa_id])
    return {"shared": len(shared), "identical": same,
            "agreement": round(same / len(shared), 4),
            "differing": [qa_id for qa_id in shared
                          if left["predictions"][qa_id] != right["predictions"][qa_id]][:10]}


def _fmt(value, width, places=3):
    if value is None:
        return " " * max(0, width - 1) + "-"
    if isinstance(value, float):
        text = f"{value:.{places}f}"
    else:
        text = str(value)
    return text.rjust(width)


def discover_runs(outputs: Path) -> list:
    """Find every finished run at any depth, as paths relative to --outputs.

    Runs are frequently grouped into folders for tidiness (for example all
    qwen35_4b variants under one directory). A single-level glob silently missed
    those, so discovery recurses and identifies each run by its relative path --
    a bare directory name would collide when two groups contain the same run id.
    """
    found = []
    for summary in sorted(outputs.rglob("run_summary.json")):
        relative = summary.parent.relative_to(outputs)
        if str(relative) == ".":
            continue
        found.append(relative.as_posix())
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs", type=Path, default=PROJECT / "outputs")
    parser.add_argument("--runs", nargs="*",
                        help="run ids (relative to --outputs, e.g. group/run); "
                             "defaults to every run found at any depth")
    parser.add_argument("--baseline", help="run id to diff predictions against")
    parser.add_argument("--json", type=Path, help="also write the collected metrics here")
    args = parser.parse_args()

    outputs = args.outputs.resolve()
    run_ids = args.runs or discover_runs(outputs)
    if not run_ids:
        print(f"no finished runs under {outputs}")
        return 1

    runs = []
    for run_id in run_ids:
        try:
            runs.append(load_run(outputs, run_id))
        except ValueError as error:
            print(f"skipping {run_id}: {error}")

    print(f"{'run':<34} {'engine':<13} {'dataset':<8} {'N':>5} {'ran':>5} {'valid':>6} "
          f"{'load_s':>8} {'infer_s':>9} {'s/req':>8} {'req/s':>8}")
    print("-" * 120)
    for run in runs:
        if not run["measured"]:
            # Say why instead of printing a throughput computed from nothing.
            print(f"{run['run_id']:<34} {run['engine']:<13} {run['dataset']:<8} "
                  f"{run['requests']:>5} {run['executed']:>5} {run['valid']:>6} "
                  f"{'-':>8} {'-':>9} {'-':>8} {'-':>8}   (resumed; nothing executed)")
            continue
        print(f"{run['run_id']:<34} {run['engine']:<13} {run['dataset']:<8} "
              f"{run['requests']:>5} {run['executed']:>5} {run['valid']:>6} "
              f"{run['load_seconds']:>8.1f} {run['inference_seconds']:>9.1f} "
              f"{_fmt(run['seconds_per_request'], 8)} {_fmt(run['requests_per_second'], 8)}")

    with_latency = [run for run in runs if run["latency"].get("requests")]
    if with_latency:
        print("\nlatency per request (ms)")
        print(f"{'run':<34} {'warmup':>7} {'p50':>9} {'p90':>9} {'p95':>9} {'p99':>9} "
              f"{'max':>9} {'steady r/s':>11}")
        print("-" * 120)
        for run in with_latency:
            whole = run["latency"]["latency_ms"]["total_ms"]
            steady = run["latency"]["steady_state_ms"]["total_ms"]
            print(f"{run['run_id']:<34} {run['latency'].get('warmup_skipped', 0):>7} "
                  f"{_fmt(steady.get('p50'), 9)} {_fmt(steady.get('p90'), 9)} "
                  f"{_fmt(steady.get('p95'), 9)} {_fmt(steady.get('p99'), 9)} "
                  f"{_fmt(whole.get('max_ms'), 9)} "
                  f"{_fmt(run['latency'].get('steady_state_throughput_rps'), 11)}")

        print("\nphase breakdown, steady state (mean ms, share of request)")
        print(f"{'run':<34} {'image':>10} {'generate':>10} {'overhead':>10} {'total':>10}")
        print("-" * 120)
        for run in with_latency:
            steady = run["latency"]["steady_state_ms"]
            total = steady["total_ms"]
            share = total.get("share") or {}
            print(f"{run['run_id']:<34} "
                  f"{_fmt(steady['image_ms'].get('mean_ms'), 10)} "
                  f"{_fmt(steady['generate_ms'].get('mean_ms'), 10)} "
                  f"{_fmt(steady['overhead_ms'].get('mean_ms'), 10)} "
                  f"{_fmt(total.get('mean_ms'), 10)}")
            print(f"{'':<34} "
                  f"{_fmt(share.get('image_ms'), 10, 3)} "
                  f"{_fmt(share.get('generate_ms'), 10, 3)} "
                  f"{_fmt(share.get('overhead_ms'), 10, 3)}")

        print("\ntime to first token (ms)")
        print(f"{'run':<34} {'coverage':>9} {'p50':>9} {'p90':>9} {'p95':>9} {'p99':>9} "
              f"{'token/s':>9}")
        print("-" * 120)
        for run in with_latency:
            ttft = run["latency"].get("steady_state_ttft_ms") or {}
            if not ttft.get("reported_by_engine"):
                print(f"{run['run_id']:<34} {'-':>9} engine does not report a first-token "
                      f"timestamp (single blocking call)")
                continue
            print(f"{run['run_id']:<34} {_fmt(ttft.get('coverage'), 9)} "
                  f"{_fmt(ttft.get('p50'), 9)} {_fmt(ttft.get('p90'), 9)} "
                  f"{_fmt(ttft.get('p95'), 9)} {_fmt(ttft.get('p99'), 9)} "
                  f"{_fmt(run['latency'].get('steady_state_token_rate'), 9)}")

    memory_runs = [run for run in runs if (run.get("gpu_memory") or {}).get("after")]
    if memory_runs:
        print("\ngpu memory per device (MiB), sampled from the parent process")
        print(f"{'run':<34} {'dev':>4} {'name':<16} {'total':>9} {'used':>9} {'free':>9} "
              f"{'alloc':>9} {'reserved':>9} {'peak_alloc':>11}")
        print("-" * 120)
        for run in memory_runs:
            memory = run["gpu_memory"]
            for device, values in sorted(memory["after"].items()):
                peak = (memory.get("peaks") or {}).get(device, {})
                print(f"{run['run_id']:<34} {device:>4} {values.get('name', '?')[:16]:<16} "
                      f"{_fmt(values.get('total_mib'), 9, 1)} {_fmt(values.get('used_mib'), 9, 1)} "
                      f"{_fmt(values.get('free_mib'), 9, 1)} {_fmt(values.get('allocated_mib'), 9, 1)} "
                      f"{_fmt(values.get('reserved_mib'), 9, 1)} "
                      f"{_fmt(peak.get('peak_allocated_mib'), 11, 1)}")
            before = memory.get("before") or {}
            if before:
                deltas = [f"dev{d}: {before[d]['used_mib']:.0f}->{memory['after'][d]['used_mib']:.0f} MiB"
                          for d in sorted(before) if d in memory["after"]]
                print(f"{'':<34} growth during run: {'; '.join(deltas)}")

    worker_runs = [run for run in runs
                   if ((run.get("backend_metadata") or {}).get("workers") or {}).get("workers")]
    if worker_runs:
        print("\ngpu memory per vLLM worker process (MiB), sampled inside each worker")
        print(f"{'run':<34} {'dev':>4} {'used':>9} {'free':>9} {'alloc':>9} {'reserved':>9} "
              f"{'peak_alloc':>11} {'kv_cache':>10} {'kv_tokens':>10}")
        print("-" * 120)
        for run in worker_runs:
            for worker in run["backend_metadata"]["workers"]["workers"]:
                print(f"{run['run_id']:<34} {worker.get('device_index', '?'):>4} "
                      f"{_fmt(worker.get('used_mib'), 9, 1)} {_fmt(worker.get('free_mib'), 9, 1)} "
                      f"{_fmt(worker.get('allocated_mib'), 9, 1)} "
                      f"{_fmt(worker.get('reserved_mib'), 9, 1)} "
                      f"{_fmt(worker.get('peak_allocated_mib'), 11, 1)} "
                      f"{_fmt(worker.get('kv_cache_mib'), 10, 1)} "
                      f"{_fmt(worker.get('kv_cache_tokens'), 10)}")
            error = run["backend_metadata"]["workers"].get("error")
            if error:
                print(f"{'':<34} worker telemetry error: {error}")

    if args.baseline:
        base = next((run for run in runs if run["run_id"] == args.baseline), None)
        if base is None:
            print(f"\nbaseline run not found: {args.baseline}")
        else:
            print(f"\noutput agreement against {args.baseline}")
            print(f"{'run':<34} {'shared':>7} {'identical':>10} {'agreement':>10}")
            print("-" * 120)
            for run in runs:
                if run["run_id"] == args.baseline:
                    continue
                result = agreement(base, run)
                if not result.get("shared"):
                    missing = [name for name, value in
                               ((base["run_id"], base["predictions"]), (run["run_id"], run["predictions"]))
                               if not value]
                    detail = f" (no predictions in {', '.join(missing)})" if missing else ""
                    print(f"{run['run_id']:<34} {'-':>7} no overlapping predictions{detail}")
                    continue
                print(f"{run['run_id']:<34} {result['shared']:>7} {result['identical']:>10} "
                      f"{result['agreement']:>10.4f}")
                if result["differing"]:
                    print(f"    first differing: {', '.join(result['differing'])}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        skip = {"predictions", "backend_metadata"}
        payload = [{key: value for key, value in run.items() if key not in skip}
                   for run in runs]
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
