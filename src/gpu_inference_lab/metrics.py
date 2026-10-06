"""Explicit summary definitions, independent of GPU execution."""

import math
import statistics


def summarize(latencies: list[float], tokens_per_run: int) -> dict:
    if not latencies or any(not math.isfinite(x) or x <= 0 for x in latencies):
        raise ValueError("Latencies must be nonempty, positive, and finite")
    if tokens_per_run <= 0:
        raise ValueError("Token count must be positive")
    ordered = sorted(latencies)
    return {
        "median_latency_s": statistics.median(ordered),
        "p95_latency_s": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "mean_latency_s": statistics.mean(ordered),
        "aggregate_generated_tokens_per_s": tokens_per_run * len(ordered) / sum(ordered),
    }
