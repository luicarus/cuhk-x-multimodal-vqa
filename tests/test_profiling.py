"""CPU-only tests for the inference profiling and reporting added for the bench."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from cuhkx.inference.profiling import RequestTimer, percentiles


PROJECT = Path(__file__).resolve().parents[1]


def test_percentiles_use_nearest_rank():
    # 1..100 makes the expected rank obvious: p99 must be an observed value.
    assert percentiles(range(1, 101)) == {"p50": 50.0, "p90": 90.0, "p95": 95.0, "p99": 99.0}
    assert percentiles([7]) == {"p50": 7.0, "p90": 7.0, "p95": 7.0, "p99": 7.0}
    with pytest.raises(ValueError):
        percentiles([])


def test_request_timer_splits_phases_and_skips_warmup():
    timer = RequestTimer(warmup=2)
    # Two slow warmup requests, then three steady ones at 100 ms each.
    for index, total in enumerate((900.0, 800.0, 100.0, 100.0, 100.0)):
        timer.record(f"q{index}", total_ms=total, image_ms=10.0, generate_ms=80.0, status="valid")
    summary = timer.summary()

    assert summary["requests"] == 5
    assert summary["warmup_skipped"] == 2
    steady = summary["steady_state_ms"]["total_ms"]
    assert steady["mean_ms"] == 100.0
    assert steady["max_ms"] == 100.0
    # Warmup shows up only in the whole-run view.
    assert summary["latency_ms"]["total_ms"]["max_ms"] == 900.0
    # phase accounting must reconcile with the measured total
    assert summary["steady_state_ms"]["image_ms"]["mean_ms"] == 10.0
    assert summary["steady_state_ms"]["generate_ms"]["mean_ms"] == 80.0
    assert summary["steady_state_ms"]["overhead_ms"]["mean_ms"] == 10.0
    assert summary["steady_state_ms"]["total_ms"]["share"]["generate_ms"] == 0.8
    assert summary["steady_state_throughput_rps"] == 10.0


def test_request_timer_falls_back_when_run_is_all_warmup():
    """A resumed run can execute fewer requests than the warmup window."""
    timer = RequestTimer(warmup=10)
    timer.record("q1", total_ms=50.0, image_ms=5.0, generate_ms=40.0, status="valid")
    summary = timer.summary()
    assert summary["warmup_skipped"] == 0
    assert summary["steady_state_ms"]["total_ms"]["count"] == 1


def test_request_timer_rejects_negative_durations():
    timer = RequestTimer()
    with pytest.raises(ValueError):
        timer.record("q", total_ms=-1.0, image_ms=0.0, generate_ms=0.0, status="valid")
    with pytest.raises(ValueError):
        timer.record("q", total_ms=1.0, image_ms=0.0, generate_ms=0.0, status="valid",
                     ttft_ms=-0.5)


def test_ttft_is_only_averaged_over_requests_that_reported_it():
    """Counting silent backends as zero TTFT would report a flattering figure."""
    timer = RequestTimer(warmup=0)
    timer.record("q1", total_ms=100.0, image_ms=10.0, generate_ms=80.0, status="valid",
                 ttft_ms=60.0, generation_tokens=1)
    timer.record("q2", total_ms=100.0, image_ms=10.0, generate_ms=80.0, status="valid")
    summary = timer.summary()

    ttft = summary["ttft_ms"]
    assert ttft["reported_by_engine"] is True
    assert ttft["count"] == 1
    assert ttft["p50"] == 60.0        # not averaged down toward 0 by the silent request
    assert ttft["coverage"] == 0.5    # and the gap is visible
    assert summary["generation_tokens"] == 1


def test_ttft_is_absent_when_no_backend_reports_it():
    timer = RequestTimer(warmup=0)
    timer.record("q1", total_ms=100.0, image_ms=10.0, generate_ms=80.0, status="valid")
    ttft = timer.summary()["ttft_ms"]
    assert ttft == {"count": 0, "reported_by_engine": False}


def test_token_rate_uses_only_reported_tokens():
    timer = RequestTimer(warmup=0)
    timer.record("q1", total_ms=1000.0, image_ms=0.0, generate_ms=900.0, status="valid",
                 generation_tokens=2)
    timer.record("q2", total_ms=1000.0, image_ms=0.0, generate_ms=900.0, status="valid",
                 generation_tokens=2)
    # 4 tokens over 2 seconds.
    assert timer.summary()["steady_state_token_rate"] == 2.0


def test_request_metrics_handles_a_backend_without_engine_timing():
    from cuhkx.inference.profiling import request_metrics

    class Silent:
        pass

    class ExplicitlySilent:
        last_metrics = {"reported": False}

    assert request_metrics(Silent()) == {}
    assert request_metrics(ExplicitlySilent()) == {}

    class Reporting:
        last_metrics = {"reported": True, "first_token_latency": 0.25,
                        "num_generation_tokens": 1, "num_prompt_tokens": 909,
                        "e2e_latency": 0.5}

    metrics = request_metrics(Reporting())
    assert metrics["ttft_ms"] == 250.0          # seconds -> milliseconds
    assert metrics["generation_tokens"] == 1
    assert metrics["prompt_tokens"] == 909
    assert metrics["e2e_ms"] == 500.0


def test_request_metrics_batch_preserves_each_reported_ttft_and_missing_values():
    from cuhkx.inference.profiling import request_metrics_batch

    class Reporting:
        last_metrics = {"request_metrics": [
            {"reported": True, "first_token_latency": 0.025},
            {"reported": False, "first_token_latency": None},
            {"reported": True, "num_generation_tokens": 2},
        ]}

    assert request_metrics_batch(Reporting(), 3) == [
        {"ttft_ms": 25.0}, {"ttft_ms": None}, {"ttft_ms": None},
    ]
    # A mismatched engine response is missing telemetry, not a failed run.
    assert request_metrics_batch(Reporting(), 2) == [{"ttft_ms": None}] * 2


def test_memory_sampling_degrades_without_cuda():
    """Profiling must never be the reason a prediction run fails."""
    from cuhkx.inference.profiling import gpu_memory_peak_snapshot, sample_gpu_memory

    assert sample_gpu_memory(None) == {}
    assert gpu_memory_peak_snapshot(None) == {}

    class BrokenCuda:
        class cuda:
            @staticmethod
            def is_available():
                raise RuntimeError("no driver")

    assert sample_gpu_memory(BrokenCuda()) == {}


def test_sampled_gpu_memory_peak_tracks_high_water_and_low_free_memory():
    from cuhkx.inference.profiling import GpuMemoryPeakMonitor

    monitor = GpuMemoryPeakMonitor(interval_ms=100, scope="test")
    monitor.observe({"0": {"name": "Tesla T4", "total_mib": 15000,
                           "used_mib": 11000, "free_mib": 4000}})
    monitor.observe({"0": {"name": "Tesla T4", "total_mib": 15000,
                           "used_mib": 13200, "free_mib": 1800}})
    monitor.observe({"0": {"name": "Tesla T4", "total_mib": 15000,
                           "used_mib": 12500, "free_mib": 2500}})

    result = monitor.stop()
    assert result["source"] == "nvml_device_memory_poll"
    assert result["available"] is True
    assert result["scope"] == "test"
    assert result["interval_ms"] == 100
    assert result["sample_count"] == 3
    assert result["devices"]["0"]["sampled_peak_used_mib"] == 13200.0
    assert result["devices"]["0"]["sampled_min_free_mib"] == 1800.0
    assert result["devices"]["0"]["last_used_mib"] == 12500.0


def test_gpu_memory_peak_monitor_is_optional_without_nvml(monkeypatch):
    import sys
    from cuhkx.inference.profiling import GpuMemoryPeakMonitor

    monkeypatch.setitem(sys.modules, "pynvml", None)
    result = GpuMemoryPeakMonitor(interval_ms=100).start().stop()
    assert result["available"] is False
    assert result["sample_count"] == 0
    assert "pynvml" in result["error"]


def test_memory_snapshot_reports_each_device():
    from cuhkx.inference.profiling import gpu_memory_snapshot

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

        @staticmethod
        def get_device_name(index):
            return f"Tesla T4 #{index}"

        @staticmethod
        def mem_get_info(index):
            # device 1 already has more in use: an imbalance must be visible
            used = (2 * 1024**3, 6 * 1024**3)[index]
            return (16 * 1024**3 - used, 16 * 1024**3)

        @staticmethod
        def memory_allocated(index):
            return 1024**3

        @staticmethod
        def memory_reserved(index):
            return 2 * 1024**3

    class Torch:
        cuda = FakeCuda()

    snapshot = gpu_memory_snapshot(Torch())
    assert set(snapshot) == {"0", "1"}
    assert snapshot["0"]["total_mib"] == 16384.0
    assert snapshot["0"]["used_mib"] == 2048.0
    assert snapshot["1"]["used_mib"] == 6144.0
    assert snapshot["0"]["allocated_mib"] == 1024.0
    assert snapshot["0"]["reserved_mib"] == 2048.0


def test_summary_latency_is_not_part_of_the_verified_contract():
    """Timing is advisory: adding it must not invalidate previously verified runs.

    _verify_finished compares only signature, target_ids, output hashes, counts
    and status, so the latency block can change without breaking a finished run.
    """
    source = (PROJECT / "src/cuhkx/inference/runner.py").read_text(encoding="utf-8")
    verify = source[source.index("def _verify_finished"):source.index("def run_predictions")]
    assert "latency" not in verify
    assert '"latency": timer.summary()' in source


def _write_run(root: Path, run_id: str, *, elapsed: float, load: float, resumed: int,
               rows: list[tuple[str, str]], latency: dict | None = None,
               gpu_memory: dict | None = None, engine: str = "vllm",
               dataset: str = "pilot", engine_options: dict | None = None,
               signature: str = "s") -> None:
    directory = root / run_id
    directory.mkdir(parents=True)
    backend = {"backend": "test_backend", "load_seconds": load, "versions": {}}
    summary = {
        "status": "PASS", "signature": signature, "target_ids": [qa for qa, _ in rows],
        "counts": {"valid": len(rows), "invalid": 0, "failed": 0, "pending": 0},
        "resumed_valid": resumed, "elapsed_seconds": elapsed, "backend": backend,
    }
    if latency is not None:
        summary["latency"] = latency
    if gpu_memory is not None:
        summary["gpu_memory"] = gpu_memory
    (directory / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    # Engine identity lives in the signed contract, not in the summary.
    (directory / "resume_state.json").write_text(
        json.dumps({"contract": {"dataset": dataset, "engine": engine,
                                 "engine_options": engine_options or
                                 {"attention_backend": "TRITON_ATTN"}}}),
        encoding="utf-8")
    with (directory / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["qa_id", "prediction"])
        writer.writerows(rows)


def _report(tmp_path, *args):
    return subprocess.run(
        [sys.executable, "-B", str(PROJECT / "scripts/bench_report.py"),
         "--outputs", str(tmp_path), *args],
        capture_output=True, text=True, encoding="utf-8", cwd=PROJECT)


def test_report_reads_engine_identity_from_the_contract(tmp_path):
    """Engine identity lives in resume_state.contract, not run_summary.json.

    Reading only the summary silently labelled every vLLM run as "transformers",
    which is exactly the column the comparison turns on.
    """
    rows = [("p1", "A")]
    _write_run(tmp_path, "vllm_run", elapsed=10.0, load=0.0, resumed=0, rows=rows,
               engine="vllm")
    _write_run(tmp_path, "hf_run", elapsed=10.0, load=0.0, resumed=0, rows=rows,
               engine="transformers")

    result = _report(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "vllm_run" in result.stdout and "hf_run" in result.stdout
    vllm_line = next(l for l in result.stdout.splitlines() if l.startswith("vllm_run"))
    hf_line = next(l for l in result.stdout.splitlines() if l.startswith("hf_run"))
    assert "vllm" in vllm_line
    assert "transformers" in hf_line


def test_report_finds_runs_nested_in_group_folders(tmp_path):
    """Runs are often grouped into folders; a single-level glob missed them."""
    rows = [("p1", "A")]
    _write_run(tmp_path, "group/hf", elapsed=10.0, load=0.0, resumed=0, rows=rows,
               engine="transformers")
    _write_run(tmp_path, "group/vllm_v1", elapsed=5.0, load=1.0, resumed=0, rows=rows,
               engine="vllm")
    # A run at the top level must still be found alongside the grouped ones.
    _write_run(tmp_path, "standalone", elapsed=10.0, load=0.0, resumed=0, rows=rows)

    result = _report(tmp_path)
    assert result.returncode == 0, result.stderr
    # Identified by relative path, so same-named runs in different groups do not
    # collapse into one row.
    assert "group/hf" in result.stdout
    assert "group/vllm_v1" in result.stdout
    assert "standalone" in result.stdout


def test_report_baseline_accepts_a_nested_run_id(tmp_path):
    rows = [("p1", "A"), ("p2", "B"), ("p3", "C"), ("p4", "D")]
    _write_run(tmp_path, "group/hf", elapsed=10.0, load=0.0, resumed=0, rows=rows)
    _write_run(tmp_path, "group/vllm", elapsed=10.0, load=0.0, resumed=0, rows=rows)

    result = _report(tmp_path, "--baseline", "group/hf")
    assert result.returncode == 0, result.stderr
    assert "1.0000" in result.stdout


def test_report_measures_throughput_and_flags_fully_resumed_runs(tmp_path):
    rows = [("p1", "A"), ("p2", "B"), ("p3", "A"), ("p4", "C")]
    _write_run(tmp_path, "measured", elapsed=100.0, load=20.0, resumed=0, rows=rows)
    # Nothing executed: elapsed is bookkeeping only, so no throughput may be shown.
    _write_run(tmp_path, "resumed_only", elapsed=0.01, load=0.0, resumed=4, rows=rows)

    result = _report(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "resumed; nothing executed" in result.stdout
    # 80 s of inference over 4 requests, engine load excluded from the rate.
    assert "20.000" in result.stdout      # s/req
    assert "0.050" in result.stdout       # req/s
    assert "24462" not in result.stdout    # the fake figure must not appear


def test_report_reports_prediction_agreement(tmp_path):
    _write_run(tmp_path, "base", elapsed=10.0, load=0.0, resumed=0,
               rows=[("p1", "A"), ("p2", "B"), ("p3", "C"), ("p4", "D")])
    _write_run(tmp_path, "cand", elapsed=10.0, load=0.0, resumed=0,
               rows=[("p1", "A"), ("p2", "B"), ("p3", "D"), ("p4", "C")])

    result = _report(tmp_path, "--baseline", "base")
    assert result.returncode == 0, result.stderr
    assert "0.5000" in result.stdout
    assert "p3" in result.stdout


def test_report_falls_back_to_checkpoint_when_predictions_csv_is_absent(tmp_path):
    rows = [("p1", "A"), ("p2", "B")]
    _write_run(tmp_path, "base", elapsed=10.0, load=0.0, resumed=0, rows=rows)
    _write_run(tmp_path, "cand", elapsed=10.0, load=0.0, resumed=0, rows=rows)
    # Simulate a run that finished without exporting predictions.csv.
    (tmp_path / "cand/predictions.csv").unlink()
    with (tmp_path / "cand/checkpoint.jsonl").open("w", encoding="utf-8") as handle:
        for qa_id, prediction in rows:
            handle.write(json.dumps({"qa_id": qa_id, "status": "valid",
                                     "prediction": prediction}) + "\n")

    result = _report(tmp_path, "--baseline", "base")
    assert result.returncode == 0, result.stderr
    assert "1.0000" in result.stdout
    assert "no predictions in" not in result.stdout


def test_report_says_why_a_comparison_has_nothing_to_compare(tmp_path):
    _write_run(tmp_path, "base", elapsed=10.0, load=0.0, resumed=0, rows=[("p1", "A")])
    _write_run(tmp_path, "cand", elapsed=10.0, load=0.0, resumed=0, rows=[("p1", "A")])
    (tmp_path / "base/predictions.csv").unlink()

    result = _report(tmp_path, "--baseline", "base")
    assert result.returncode == 0, result.stderr
    assert "no predictions in base" in result.stdout


def test_report_prints_latency_table_when_timing_is_present(tmp_path):
    rows = [("p1", "A"), ("p2", "B")]
    latency = {
        "requests": 2, "warmup_skipped": 0,
        "latency_ms": {"total_ms": {"count": 2, "mean_ms": 50.0, "max_ms": 60.0,
                                    "min_ms": 40.0, "stdev_ms": 10.0, "total_ms": 100.0,
                                    "p50": 40.0, "p90": 60.0, "p95": 60.0, "p99": 60.0}},
        "steady_state_ms": {"total_ms": {"count": 2, "mean_ms": 50.0, "max_ms": 60.0,
                                         "min_ms": 40.0, "stdev_ms": 10.0, "total_ms": 100.0,
                                         "p50": 40.0, "p90": 60.0, "p95": 60.0, "p99": 60.0,
                                         "share": {"image_ms": 0.2, "generate_ms": 0.7,
                                                   "overhead_ms": 0.1}},
                            "image_ms": {"count": 2, "mean_ms": 10.0},
                            "generate_ms": {"count": 2, "mean_ms": 35.0},
                            "overhead_ms": {"count": 2, "mean_ms": 5.0}},
        "steady_state_throughput_rps": 20.0, "steady_state_valid_rate": 1.0,
        "run_throughput_rps": 20.0, "slowest_requests": [],
    }
    _write_run(tmp_path, "timed", elapsed=10.0, load=0.0, resumed=0, rows=rows, latency=latency)

    result = _report(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "latency per request" in result.stdout
    assert "phase breakdown" in result.stdout
    assert "40.000" in result.stdout


def test_report_shows_the_batched_view(tmp_path):
    """A batched run must be readable without fabricated per-request numbers."""
    rows = [("p1", "A"), ("p2", "B"), ("p3", "C")]
    latency = {
        "requests": 0, "warmup_skipped": 0, "batched": True,
        "batch": {"batches": 1, "requests": 3, "batch_sizes": [3],
                  "total_ms": 900.0, "mean_batch_ms": 900.0, "mean_batch_size": 3.0,
                  "throughput_rps": 3.3333, "token_rate": 3.3333,
                  "amortized_ms_per_request": 300.0},
        "ttft_ms": {"count": 2, "requests": 3, "coverage": 0.6667,
                    "reported_by_engine": True, "p50": 60.0, "p90": 80.0,
                    "p95": 80.0, "p99": 80.0},
    }
    _write_run(tmp_path, "batched", elapsed=1.0, load=0.0, resumed=0, rows=rows,
               latency=latency)

    result = _report(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "batched execution" in result.stdout
    assert "300.000" in result.stdout
    assert "time to first token" in result.stdout
    assert "60.000" in result.stdout


def test_report_shows_nvml_sampled_gpu_memory_peaks(tmp_path):
    gpu_memory = {
        "after": {"0": {"name": "Tesla T4", "total_mib": 15000.0,
                         "used_mib": 12000.0, "free_mib": 3000.0}},
        "peaks": {"device_polling": {
            "available": True, "scope": "prediction_after_model_load",
            "interval_ms": 100, "sample_count": 12,
            "devices": {"0": {"sampled_peak_used_mib": 13500.0,
                               "sampled_min_free_mib": 1500.0}},
        }},
    }
    _write_run(tmp_path, "nvml", elapsed=1.0, load=0.0, resumed=0,
               rows=[("p1", "A")], gpu_memory=gpu_memory)

    result = _report(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "sample peak" in result.stdout
    assert "13500.0" in result.stdout
    assert "100 ms interval" in result.stdout


def test_report_highlights_best_single_run_for_repeated_batch_config(tmp_path):
    options = {"data_parallel_size": 2, "tensor_parallel_size": 1,
               "runner_batch_size": 32, "max_num_seqs": 16,
               "attention_backend": "TRITON_ATTN"}
    for run_id, throughput, ttft in (("rep1", 1.39, 12000.0),
                                     ("rep2", 1.41, 11800.0),
                                     ("rep3", 1.52, 11200.0)):
        latency = {
            "requests": 0, "warmup_skipped": 0, "batched": True,
            "batch": {"batches": 1, "requests": 3, "batch_sizes": [3],
                      "total_ms": 2000.0, "mean_batch_ms": 2000.0,
                      "mean_batch_size": 3.0, "throughput_rps": throughput,
                      "token_rate": 3.0, "amortized_ms_per_request": 2000 / 3},
            "ttft_ms": {"count": 3, "requests": 3, "coverage": 1.0,
                        "reported_by_engine": True, "p50": ttft,
                        "p95": ttft + 1000, "p99": ttft + 2000},
        }
        _write_run(tmp_path, run_id, elapsed=3.0, load=0.0, resumed=0,
                   rows=[("p1", "A"), ("p2", "B"), ("p3", "C")],
                   latency=latency, engine_options=options, signature="same-input")

    json_path = tmp_path / "summary.json"
    result = _report(tmp_path, "--json", str(json_path))
    assert result.returncode == 0, result.stderr
    assert "best single run per repeated configuration" in result.stdout
    assert "rep3" in result.stdout
    assert "1.5200" in result.stdout
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    winner = next(run for run in payload if run["run_id"] == "rep3")
    assert winner["best_in_repeat_group"] is True
    assert winner["repeat_count"] == 3


def test_timer_batch_view_amortizes_over_the_whole_run():
    from cuhkx.inference.profiling import RequestTimer

    timer = RequestTimer()
    timer.record_batch(batch_ms=1000.0, image_ms=50.0, generate_ms=900.0,
                       size=8, ttft_ms=120.0, generation_tokens=8)
    timer.record_batch(batch_ms=500.0, image_ms=25.0, generate_ms=450.0,
                       size=4, ttft_ms=130.0, generation_tokens=4)
    view = timer.summary()["batch"]
    assert view["batches"] == 2
    assert view["requests"] == 12
    assert view["batch_sizes"] == [8, 4]
    # 12 requests over 1.5 s.
    assert view["throughput_rps"] == 8.0
    assert view["amortized_ms_per_request"] == 125.0
    assert view["mean_batch_size"] == 6.0
    with pytest.raises(ValueError):
        timer.record_batch(batch_ms=1.0, image_ms=0.0, generate_ms=0.0, size=0)


def test_batched_ttft_summary_uses_per_request_engine_samples_and_reports_coverage():
    timer = RequestTimer()
    timer.record_batch(
        batch_ms=300.0, image_ms=0.0, generate_ms=250.0, size=3,
        request_ttft=[
            {"qa_id": "q1", "ttft_ms": 40.0},
            {"qa_id": "q2", "ttft_ms": 80.0},
            {"qa_id": "q3", "ttft_ms": None},
        ],
    )

    ttft = timer.summary()["ttft_ms"]
    assert ttft["reported_by_engine"] is True
    assert ttft["requests"] == 3
    assert ttft["count"] == 2
    assert ttft["coverage"] == 0.6667
    assert ttft["mean_ms"] == 60.0
    assert ttft["p50"] == 40.0
    assert ttft["p95"] == 80.0
    assert ttft["p99"] == 80.0
    assert ttft["samples"] == [
        {"qa_id": "q1", "ttft_ms": 40.0},
        {"qa_id": "q2", "ttft_ms": 80.0},
        {"qa_id": "q3", "ttft_ms": None},
    ]
