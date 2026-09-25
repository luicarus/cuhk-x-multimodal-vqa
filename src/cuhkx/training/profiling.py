"""Per-step training telemetry for the QLoRA lanes, kept out of the contract.

The training run's correctness contract is built from data fingerprints, model
provenance and software versions; wall clock is not part of it. Timing therefore
lives beside the run rather than inside it, so adding or changing a metric can
never invalidate a previously verified run. This mirrors the inference-side
``profiling`` module on purpose.

What is measured, and why each one matters when the question is "why is training
slow":

- ``dataloader_ms``  time spent waiting for the next batch. Collation here decodes
                     four JPEGs and runs the image processor twice per sample, so
                     at ``dataloader_num_workers=0`` this work sits on the
                     critical path and a small worker count should remove most of
                     it. Without this number the change is unfalsifiable.
- ``step_ms``        wall clock per optimizer step, i.e. the forward/backward the
                     checkpointing recomputes plus the optimizer update.
- ``tokens_per_s``   throughput over the samples actually processed, excluding
                     warmup steps whose cost is dominated by lazy kernel
                     compilation and one-off allocations.
- ``peak_mib``       device-wide high-water mark from NVML. The paged optimizer
                     is reported to *reduce* allocator state while sometimes
                     *increasing* device usage, and an OOM on a 16 GB T4 has to be
                     attributable rather than guessed.

A limitation worth stating rather than hiding: with
``gradient_checkpointing=True`` the forward pass is re-executed inside the
backward pass, so ``step_ms`` cannot be split into separate forward and backward
figures. This module does not invent that split.
"""
from __future__ import annotations

import statistics
import time

from cuhkx.config import require

# Steps excluded from the steady-state view. The first steps include lazy kernel
# selection, the first allocator growth, and the first checkpoint write.
WARMUP_STEPS = 3


def percentiles(values, points=(50, 90, 95, 99)):
    """Nearest-rank percentiles; no interpolation, so every value is observed.

    Shared shape with the inference profiler: a reported percentile is always a
    measurement that actually happened.
    """
    ordered = sorted(float(value) for value in values)
    require(bool(ordered), "percentiles require at least one sample")
    result = {}
    for point in points:
        require(0 < point <= 100, "percentile must be within (0, 100]")
        rank = max(1, -(-len(ordered) * point // 100))
        result[f"p{point}"] = round(ordered[rank - 1], 3)
    return result


def describe(values):
    """Count/total/mean/percentiles for one timing series, in milliseconds."""
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


class TrainingProfiler:
    """Collect one timing sample per optimizer step and summarize the run.

    Driven from Trainer callbacks rather than from the training loop itself, so
    the measurement cannot change what is being measured. Every method is
    best-effort: telemetry must never fail a training run, which is the same rule
    the inference profiler follows.
    """

    def __init__(self, *, warmup=WARMUP_STEPS, samples_per_step=None,
                 interval_ms=200, scope="training_after_model_load"):
        require(warmup >= 0, "warmup cannot be negative")
        self.warmup = int(warmup)
        self.samples_per_step = samples_per_step
        self.interval_ms = int(interval_ms)
        self.scope = scope
        self.steps = []
        self.batches_waited = []
        self.monitor = None
        self.monitor_result = None
        self.error = None
        self._step_started = None
        self._batch_started = None

    # ---- lifecycle -------------------------------------------------------

    def start(self):
        """Begin GPU polling; a failure is recorded, never raised."""
        try:
            from cuhkx.inference.profiling import GpuMemoryPeakMonitor

            self.monitor = GpuMemoryPeakMonitor(
                interval_ms=self.interval_ms, scope=self.scope).start()
        except Exception as error:                # noqa: BLE001 - telemetry only
            self.error = f"{type(error).__name__}: {error}"
            self.monitor = None
        return self

    def stop(self):
        if self.monitor is not None:
            try:
                self.monitor_result = self.monitor.stop()
            except Exception as error:            # noqa: BLE001 - telemetry only
                self.monitor_result = {"available": False,
                                       "error": f"{type(error).__name__}: {error}"}
            self.monitor = None
        return self.summary()

    # ---- sampling --------------------------------------------------------

    def begin_batch(self):
        """Mark the moment a batch was requested, to time the wait for it."""
        self._batch_started = time.perf_counter()

    def note_batch(self):
        """Record how long the next batch took to arrive, if a wait is open.

        Called once the batch has been handed to the model, so the measured span
        covers exactly the dataloader wait and not the forward pass.
        """
        if self._batch_started is None:
            return
        elapsed = (time.perf_counter() - self._batch_started) * 1000.0
        self._batch_started = None
        self.batches_waited.append(elapsed)

    def begin_step(self):
        self._step_started = time.perf_counter()
        self.note_batch()

    def end_step(self, *, samples=None):
        """Close one optimizer step and append its timing sample."""
        if self._step_started is None:
            return
        elapsed = (time.perf_counter() - self._step_started) * 1000.0
        self._step_started = None
        self.steps.append({
            "step_ms": round(elapsed, 3),
            "samples": samples,
            "dataloader_ms": round(self.batches_waited[-1], 3)
            if self.batches_waited else None,
        })

    # ---- summary ---------------------------------------------------------

    def _throughput(self, steady):
        if self.samples_per_step is None or not steady:
            return None
        seconds = sum(step["step_ms"] for step in steady) / 1000.0
        if seconds <= 0:
            return None
        return round(len(steady) * self.samples_per_step / seconds, 4)

    def summary(self):
        if not self.steps:
            return {
                "available": False,
                "reason": "no optimizer step was observed",
                "gpu_memory": self.monitor_result,
                "error": self.error,
            }
        steady = self.steps[self.warmup:] or self.steps
        step_values = [step["step_ms"] for step in self.steps]
        steady_values = [step["step_ms"] for step in steady]
        waits = [step["dataloader_ms"] for step in self.steps
                 if step["dataloader_ms"] is not None]
        total_ms = sum(step_values)
        wait_total = sum(waits)
        summary = {
            "available": True,
            "warmup_skipped": len(self.steps) - len(steady),
            "steps": len(self.steps),
            "step_ms": describe(step_values),
            "steady_state_step_ms": describe(steady_values),
            "samples_per_step": self.samples_per_step,
            "steady_state_samples_per_s": self._throughput(steady),
            # A share only means something if something was actually measured.
            "dataloader_share": (round(wait_total / total_ms, 4) if total_ms and waits
                                 else None),
            "gpu_memory": self.monitor_result,
            "error": self.error,
            "note": ("forward and backward are not separable under gradient "
                     "checkpointing; step_ms covers both plus the optimizer update"),
        }
        if waits:
            summary["dataloader_ms"] = describe(waits)
        return summary
