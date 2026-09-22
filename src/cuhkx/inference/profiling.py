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

import statistics

from cuhkx.config import require


# Requests excluded from the steady-state view. vLLM's first call still triggers
# per-shape kernel selection and the vision tower's first allocation; the runner
# also pays a one-off cost on the very first checkpoint write.
WARMUP_REQUESTS = 3


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


class RequestTimer:
    """Collect one timing sample per request and summarize the run."""

    def __init__(self, warmup=WARMUP_REQUESTS):
        self.warmup = warmup
        self.samples = []

    def record(self, qa_id, *, total_ms, image_ms, generate_ms, status):
        require(total_ms >= 0 and image_ms >= 0 and generate_ms >= 0, "negative duration")
        # Python's own bookkeeping is the remainder; it cannot be measured
        # directly without distorting what is being measured.
        overhead_ms = max(0.0, total_ms - image_ms - generate_ms)
        self.samples.append({
            "qa_id": qa_id, "status": status,
            "total_ms": round(total_ms, 3), "image_ms": round(image_ms, 3),
            "generate_ms": round(generate_ms, 3), "overhead_ms": round(overhead_ms, 3),
        })

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

    def summary(self):
        if not self.samples:
            return {"requests": 0, "warmup_skipped": 0}
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
            "slowest_requests": sorted(
                ({"qa_id": s["qa_id"], "total_ms": s["total_ms"], "status": s["status"]}
                 for s in self.samples), key=lambda item: item["total_ms"], reverse=True)[:5],
        }
