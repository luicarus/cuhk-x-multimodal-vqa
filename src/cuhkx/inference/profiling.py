"""Per-request timing for the inference runner, kept out of the signed contract.

The runner's correctness contract is built from file hashes and answer sets; wall
clock is not part of it. Timing therefore lives beside the run rather than inside
it, so adding or changing a metric can never invalidate a previously verified run.

What is measured, and why each one matters when the question is "where does the
time actually go":

- ``image_ms``      frame decode and load. Four 448x448 IR JPEGs per request come
                    off disk every time; on a slow filesystem this can rival the
                    forward pass and it is invisible in end-to-end numbers.
- ``generate_ms``   the model call itself, including vLLM's own preprocessing.
- ``overhead_ms``   bookkeeping, answer parsing and the atomic checkpoint write.
                    Each request rewrites checkpoint.jsonl, which is O(n) in the
                    number of completed records -- a real cost at 682 requests.
- ``total_ms``      wall clock per request, so the three parts can be reconciled.

Percentiles are computed over the whole run and over a warm subset. The first few
requests include lazy kernel/JIT work and one-off allocations, so a steady-state
figure that skips them is the honest throughput number, while the full
distribution shows the cold-start tail that a user actually experiences.
"""
from __future__ import annotations

import math
import statistics
import threading
import time

from cuhkx.config import require


# Requests excluded from the steady-state view. vLLM's first call still triggers
# per-shape kernel selection and the vision tower's first allocation; the runner
# also pays a one-off cost on the very first checkpoint write.
WARMUP_REQUESTS = 3


def gpu_memory_snapshot(torch_module):
    """Per-device memory in MiB, from the driver and from torch's allocator.

    Both views are needed and they answer different questions. The driver figure
    (`mem_get_info`) is what the process has taken from the card, so it is the
    number that decides whether a larger batch will fit. The allocator figures
    separate memory torch holds for live tensors from memory it reserved and is
    keeping, and a large gap between the two means fragmentation rather than a
    real shortage -- which changes the fix.

    Reported per device because under tensor parallelism the two GPUs are not
    guaranteed to be balanced, and an imbalance is invisible in a single total.
    """
    require(torch_module is not None, "torch is required for memory sampling")
    require(torch_module.cuda.is_available(), "CUDA is required for memory sampling")
    devices = {}
    for index in range(torch_module.cuda.device_count()):
        free_bytes, total_bytes = torch_module.cuda.mem_get_info(index)
        devices[str(index)] = {
            "name": torch_module.cuda.get_device_name(index),
            "total_mib": round(total_bytes / 1024**2, 1),
            "free_mib": round(free_bytes / 1024**2, 1),
            "used_mib": round((total_bytes - free_bytes) / 1024**2, 1),
            "allocated_mib": round(torch_module.cuda.memory_allocated(index) / 1024**2, 1),
            "reserved_mib": round(torch_module.cuda.memory_reserved(index) / 1024**2, 1),
        }
    return devices


def gpu_memory_peaks(torch_module):
    """Peak allocated/reserved per device since the last reset.

    Used after a warmup so the peak reflects steady-state serving rather than
    one-off model loading, which would otherwise dominate and say nothing about
    whether a larger batch fits.
    """
    peaks = {}
    for index in range(torch_module.cuda.device_count()):
        peaks[str(index)] = {
            "peak_allocated_mib": round(torch_module.cuda.max_memory_allocated(index) / 1024**2, 1),
            "peak_reserved_mib": round(torch_module.cuda.max_memory_reserved(index) / 1024**2, 1),
        }
    return peaks


def reset_gpu_peaks(torch_module):
    for index in range(torch_module.cuda.device_count()):
        torch_module.cuda.reset_peak_memory_stats(index)


def percentiles(values, points=(50, 90, 95, 99)):
    """Nearest-rank percentiles; no interpolation, so every value is observed."""
    ordered = sorted(float(value) for value in values)
    require(bool(ordered), "percentiles require at least one sample")
    result = {}
    for point in points:
        require(0 < point <= 100, "percentile must be within (0, 100]")
        # Nearest-rank: ceil(p/100 * n) with a 1-based rank.
        rank = max(1, -(-len(ordered) * point // 100))
        result[f"p{point}"] = round(ordered[rank - 1], 3)
    return result


def _describe(values):
    ordered = [float(value) for value in values]
    if not ordered:
        return {"count": 0}
    total = sum(ordered)
    return {
        "count": len(ordered),
        "total_ms": round(total, 3),
        "mean_ms": round(total / len(ordered), 3),
        "min_ms": round(min(ordered), 3),
        "max_ms": round(max(ordered), 3),
        "stdev_ms": round(statistics.pstdev(ordered), 3) if len(ordered) > 1 else 0.0,
        **percentiles(ordered),
    }


def request_metrics(backend):
    """Read the last request's engine-reported metrics, whatever the backend.

    Backends are not required to expose timing. The Transformers path returns
    no first-token timestamp because it generates in one blocking call, so the
    runner must treat "absent" as a real answer rather than an error.
    """
    values = getattr(backend, "last_metrics", None)
    if not isinstance(values, dict) or not values.get("reported"):
        return {}
    ttft = values.get("first_token_latency")
    return {
        "ttft_ms": None if ttft is None else float(ttft) * 1000.0,
        "generation_tokens": values.get("num_generation_tokens"),
        "prompt_tokens": values.get("num_prompt_tokens"),
        "e2e_ms": None if values.get("e2e_latency") is None else float(values["e2e_latency"]) * 1000.0,
    }


def request_metrics_batch(backend, expected_requests):
    """Return per-request TTFT values for the last batched engine call.

    vLLM returns one ``RequestOutput.metrics`` object per prompt. The batch
    backend preserves those objects as plain dictionaries in
    ``last_metrics.request_metrics``; missing or malformed metrics remain
    absent and never fail inference.
    """
    require(expected_requests >= 0, "expected request count cannot be negative")
    missing = [{"ttft_ms": None} for _ in range(expected_requests)]
    values = getattr(backend, "last_metrics", None)
    if not isinstance(values, dict):
        return missing
    per_request = values.get("request_metrics")
    if not isinstance(per_request, list) or len(per_request) != expected_requests:
        return missing
    result = []
    for metrics in per_request:
        ttft = (metrics.get("first_token_latency")
                if isinstance(metrics, dict) and metrics.get("reported") else None)
        if (not isinstance(ttft, (int, float)) or isinstance(ttft, bool)
                or not math.isfinite(float(ttft)) or ttft < 0):
            result.append({"ttft_ms": None})
        else:
            result.append({"ttft_ms": float(ttft) * 1000.0})
    return result


def sample_gpu_memory(torch_module):
    """Memory snapshot that degrades to an empty dict instead of failing a run."""
    if torch_module is None:
        return {}
    try:
        return gpu_memory_snapshot(torch_module)
    except Exception:
        # Profiling must never be the reason a prediction run fails.
        return {}


def gpu_memory_peak_snapshot(torch_module):
    """Peak snapshot that degrades to an empty dict."""
    if torch_module is None:
        return {}
    try:
        return gpu_memory_peaks(torch_module)
    except Exception:
        return {}


class GpuMemoryPeakMonitor:
    """Poll device-wide NVML memory and retain a sampled high-water mark.

    The parent runner cannot see vLLM's CUDA allocator peaks when engines live
    in child processes. NVML reports total device use across processes, so this
    monitor samples each visible physical GPU while prediction is running. It
    is best-effort: driver/library failures are returned as metadata and never
    fail inference.
    """

    def __init__(self, *, interval_ms=100, scope="prediction_after_model_load"):
        require(interval_ms >= 10, "GPU memory polling interval must be at least 10 ms")
        self.interval_ms = int(interval_ms)
        self.scope = scope
        self.sample_count = 0
        self.devices = {}
        self.error = None
        self._nvml = None
        self._handles = {}
        self._stop = threading.Event()
        self._thread = None
        self._initialized = False

    def start(self):
        try:
            import pynvml

            self._nvml = pynvml
            pynvml.nvmlInit()
            self._initialized = True
            for index in range(pynvml.nvmlDeviceGetCount()):
                self._handles[str(index)] = pynvml.nvmlDeviceGetHandleByIndex(index)
            if not self._handles:
                raise RuntimeError("NVML reported no visible GPU devices")
            self._sample_once()
            self._thread = threading.Thread(
                target=self._poll, name="cuhkx-gpu-memory-monitor", daemon=True)
            self._thread.start()
        except Exception as error:  # noqa: BLE001 - telemetry must not block inference
            self.error = f"{type(error).__name__}: {error}"
            self._shutdown_nvml()
        return self

    def _read_devices(self):
        readings = {}
        for index, handle in self._handles.items():
            memory = self._nvml.nvmlDeviceGetMemoryInfo(handle)
            name = self._nvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            readings[index] = {
                "name": str(name),
                "total_mib": memory.total / 1024**2,
                "used_mib": memory.used / 1024**2,
                "free_mib": memory.free / 1024**2,
            }
        return readings

    def observe(self, readings):
        """Add one set of device readings; public for deterministic CPU tests."""
        self.sample_count += 1
        for index, reading in readings.items():
            current = self.devices.get(index)
            used = float(reading["used_mib"])
            free = float(reading["free_mib"])
            if current is None:
                self.devices[index] = {
                    "name": reading.get("name", "?"),
                    "total_mib": round(float(reading["total_mib"]), 1),
                    "first_used_mib": round(used, 1),
                    "last_used_mib": round(used, 1),
                    "sampled_peak_used_mib": round(used, 1),
                    "sampled_min_free_mib": round(free, 1),
                }
            else:
                current["last_used_mib"] = round(used, 1)
                current["sampled_peak_used_mib"] = round(
                    max(current["sampled_peak_used_mib"], used), 1)
                current["sampled_min_free_mib"] = round(
                    min(current["sampled_min_free_mib"], free), 1)

    def _sample_once(self):
        try:
            self.observe(self._read_devices())
        except Exception as error:  # noqa: BLE001 - telemetry must not block inference
            if self.error is None:
                self.error = f"{type(error).__name__}: {error}"
            self._stop.set()

    def _poll(self):
        interval = self.interval_ms / 1000.0
        while not self._stop.wait(interval):
            self._sample_once()

    def _shutdown_nvml(self):
        if self._initialized:
            try:
                self._nvml.nvmlShutdown()
            except Exception:  # noqa: BLE001 - teardown is best-effort
                pass
            self._initialized = False

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_ms / 1000.0 * 4))
            self._thread = None
        if self._initialized:
            self._sample_once()
        self._shutdown_nvml()
        return {
            "source": "nvml_device_memory_poll",
            "available": self.sample_count > 0,
            "scope": self.scope,
            "interval_ms": self.interval_ms,
            "sample_count": self.sample_count,
            "devices": self.devices,
            "error": self.error,
        }


class RequestTimer:
    """Collect one timing sample per request and summarize the run."""

    def __init__(self, warmup=WARMUP_REQUESTS):
        self.warmup = warmup
        self.samples = []
        # Batches are the unit of scheduling in the batched runner. Per-request
        # latencies lose their meaning when N requests finish together, so the
        # batch view is what actually shows the throughput mechanism.
        self.batches = []

    def record(self, qa_id, *, total_ms, image_ms, generate_ms, status,
               ttft_ms=None, generation_tokens=None):
        require(total_ms >= 0 and image_ms >= 0 and generate_ms >= 0, "negative duration")
        require(ttft_ms is None or ttft_ms >= 0, "negative time to first token")
        # Python's own bookkeeping is the remainder; it cannot be measured
        # directly without distorting what is being measured.
        overhead_ms = max(0.0, total_ms - image_ms - generate_ms)
        self.samples.append({
            "qa_id": qa_id, "status": status,
            "total_ms": round(total_ms, 3), "image_ms": round(image_ms, 3),
            "generate_ms": round(generate_ms, 3), "overhead_ms": round(overhead_ms, 3),
            "ttft_ms": None if ttft_ms is None else round(ttft_ms, 3),
            "generation_tokens": generation_tokens,
        })

    def record_batch(self, *, batch_ms, image_ms, generate_ms, size,
                     ttft_ms=None, generation_tokens=0, request_ttft=None):
        """One timing sample for a whole submitted batch.

        Per-request end-to-end latency is not derivable from a batched call: the
        engine interleaves requests, so each answer does not own a measurable
        slice of the wall clock. vLLM's engine-reported TTFT is retained per
        request separately; batch throughput still comes from size over batch
        wall time.
        """
        require(batch_ms >= 0 and image_ms >= 0 and generate_ms >= 0, "negative duration")
        require(size >= 1, "an empty batch has no timing")
        if request_ttft is None:
            request_ttft = [{"qa_id": None, "ttft_ms": None} for _ in range(size)]
        require(len(request_ttft) == size, "request TTFT count must match batch size")
        normalized_ttft = []
        for item in request_ttft:
            require(isinstance(item, dict), "request TTFT entries must be mappings")
            value = item.get("ttft_ms")
            require(value is None or (isinstance(value, (int, float))
                    and not isinstance(value, bool) and math.isfinite(value) and value >= 0),
                    "invalid time to first token")
            normalized_ttft.append({
                "qa_id": item.get("qa_id"),
                "ttft_ms": None if value is None else round(float(value), 3),
            })
        self.batches.append({
            "size": size,
            "batch_ms": round(batch_ms, 3),
            "image_ms": round(image_ms, 3),
            "generate_ms": round(generate_ms, 3),
            "overhead_ms": round(max(0.0, batch_ms - image_ms - generate_ms), 3),
            "ttft_ms": None if ttft_ms is None else round(ttft_ms, 3),
            "generation_tokens": generation_tokens,
            "request_ttft": normalized_ttft,
        })

    def _batch_view(self):
        if not self.batches:
            return None
        total_ms = sum(batch["batch_ms"] for batch in self.batches)
        requests = sum(batch["size"] for batch in self.batches)
        tokens = sum(batch["generation_tokens"] for batch in self.batches)
        return {
            "batches": len(self.batches),
            "requests": requests,
            "batch_sizes": [batch["size"] for batch in self.batches],
            "total_ms": round(total_ms, 3),
            "mean_batch_ms": round(total_ms / len(self.batches), 3),
            "mean_batch_size": round(requests / len(self.batches), 3),
            "throughput_rps": round(requests / (total_ms / 1000.0), 4) if total_ms else 0.0,
            "token_rate": round(tokens / (total_ms / 1000.0), 4) if total_ms and tokens else None,
            # Amortized per-request cost is the number to compare against the
            # sequential runner's s/req; it is not a latency measurement.
            "amortized_ms_per_request": round(total_ms / requests, 3) if requests else None,
        }

    def _batch_ttft(self):
        samples = [sample for batch in self.batches for sample in batch["request_ttft"]]
        observed = [sample["ttft_ms"] for sample in samples if sample["ttft_ms"] is not None]
        coverage = round(len(observed) / len(samples), 4) if samples else 0.0
        if not observed:
            return {"count": 0, "requests": len(samples), "coverage": coverage,
                    "reported_by_engine": False, "samples": samples}
        return {**_describe(observed), "requests": len(samples), "coverage": coverage,
                "reported_by_engine": True, "samples": samples}

    def _phase(self, name, samples):
        if not samples:
            return {"count": 0}
        values = [sample[name] for sample in samples]
        described = _describe(values)
        mean = described["mean_ms"]
        if name == "total_ms" and mean:
            described["share"] = {
                phase: round(
                    sum(sample[phase] for sample in samples) / len(samples) / mean, 4)
                for phase in ("image_ms", "generate_ms", "overhead_ms")
            }
        return described

    def _ttft(self, samples):
        """TTFT over only the requests whose engine actually reported it.

        Backends differ: vLLM exposes a first-token timestamp, the Transformers
        path does not. Counting the missing ones as zero would report a
        flattering average, so they are excluded and the coverage is stated.
        """
        reported = [sample["ttft_ms"] for sample in samples if sample["ttft_ms"] is not None]
        if not reported:
            return {"count": 0, "reported_by_engine": False}
        return {**_describe(reported), "reported_by_engine": True,
                "coverage": round(len(reported) / len(samples), 4)}

    def _token_rate(self, samples):
        tokens = sum(sample["generation_tokens"] or 0 for sample in samples)
        if not tokens:
            return None
        steady_ms = sum(sample["total_ms"] for sample in samples)
        return round(tokens / (steady_ms / 1000.0), 4) if steady_ms else None

    def summary(self):
        batch_view = self._batch_view()
        if not self.samples and not self.batches:
            return {"requests": 0, "warmup_skipped": 0}
        if not self.samples:
            # Batched run: per-request timing is not observable, so the summary
            # stays batch-level except for vLLM's independently reported TTFT.
            return {"requests": 0, "warmup_skipped": 0, "batched": True,
                    "batch": batch_view, "ttft_ms": self._batch_ttft()}
        warm = self.samples[self.warmup:]
        # A resumed run may legitimately finish in fewer requests than the warmup
        # window; fall back to the full set rather than reporting nothing.
        steady = warm or self.samples
        whole_total = sum(sample["total_ms"] for sample in self.samples)
        steady_total = sum(sample["total_ms"] for sample in steady)
        valid = sum(1 for sample in steady if sample["status"] == "valid")
        return {
            "requests": len(self.samples),
            "warmup_skipped": len(self.samples) - len(steady),
            "latency_ms": {name: self._phase(name, self.samples)
                           for name in ("total_ms", "image_ms", "generate_ms", "overhead_ms")},
            "steady_state_ms": {name: self._phase(name, steady)
                                for name in ("total_ms", "image_ms", "generate_ms", "overhead_ms")},
            "steady_state_throughput_rps": round(len(steady) / (steady_total / 1000.0), 4) if steady_total else 0.0,
            "steady_state_valid_rate": round(valid / len(steady), 4) if steady else 0.0,
            "run_throughput_rps": round(len(self.samples) / (whole_total / 1000.0), 4) if whole_total else 0.0,
            # TTFT answers a different question from throughput: with an 8-token
            # answer budget, prefill dominates and decode is a handful of steps,
            # so TTFT is expected to sit close to total latency here. Reporting
            # both makes that explicit instead of implying a chat-style profile.
            "ttft_ms": self._ttft(self.samples),
            "steady_state_ttft_ms": self._ttft(steady),
            "steady_state_token_rate": self._token_rate(steady),
            "generation_tokens": sum(sample["generation_tokens"] or 0 for sample in self.samples),
            "batched": bool(self.batches),
            "batch": batch_view,
            "slowest_requests": sorted(
                ({"qa_id": s["qa_id"], "total_ms": s["total_ms"], "status": s["status"]}
                 for s in self.samples), key=lambda item: item["total_ms"], reverse=True)[:5],
        }
