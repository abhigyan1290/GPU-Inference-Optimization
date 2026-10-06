import math

import pytest

from gpu_inference_lab.metrics import summarize


def test_summary_definitions():
    result = summarize([4.0, 1.0, 3.0, 2.0], 10)
    assert result["median_latency_s"] == 2.5
    assert result["p95_latency_s"] == 4.0
    assert result["aggregate_generated_tokens_per_s"] == 4.0


def test_p95_nearest_rank():
    assert summarize(list(range(1, 21)), 1)["p95_latency_s"] == 19


@pytest.mark.parametrize("values", [[], [0], [-1], [math.nan], [math.inf]])
def test_invalid_latencies(values):
    with pytest.raises(ValueError):
        summarize(values, 1)


def test_invalid_token_count():
    with pytest.raises(ValueError):
        summarize([1], 0)
