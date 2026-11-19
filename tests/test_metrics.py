import time

import pytest

from llmprobe.metrics import latency_stats, percentile, summarize_results
from llmprobe.models import Cell, RequestResult


def result(
    *,
    started_at: float,
    stream_end_s: float,
    success: bool = True,
    ttft_s: float | None = 0.1,
    e2el_s: float | None = 0.5,
    output_tokens: int = 5,
    input_tokens: int | None = 10,
) -> RequestResult:
    return RequestResult(
        target="mock",
        cell_id="in4-out2-conc2",
        repeat=0,
        request_index=0,
        concurrency=2,
        input_length_target=4,
        output_length_target=2,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_tokens_source="usage" if input_tokens is not None else "unavailable",
        output_tokens_source="usage",
        input_tokens_estimated=False,
        output_tokens_estimated=False,
        client_queue_s=0.0,
        started_at=started_at,
        ttft_s=ttft_s,
        e2el_s=e2el_s,
        stream_end_s=stream_end_s,
        tpot_s=(e2el_s - ttft_s) / (output_tokens - 1)
        if e2el_s and ttft_s and output_tokens > 1
        else None,
        itl_s=[0.1, 0.1, 0.1, 0.1],
        token_event_count=output_tokens,
        finish_reason="stop",
        success=success,
        error_kind=None if success else "http",
        error=None if success else "failed",
        http_status=200 if success else 500,
        generated_text=None,
    )


def test_percentile_uses_linear_interpolation() -> None:
    assert percentile([10.0, 20.0, 30.0, 40.0], 50) == 25.0
    assert percentile([10.0, 20.0], 0) == 10.0
    assert percentile([10.0, 20.0], 100) == 20.0
    assert percentile([], 50) is None


def test_latency_stats_reports_population_stddev() -> None:
    stats = latency_stats([0.1, 0.2, 0.3])
    assert stats.count == 3
    assert stats.mean_ms == pytest.approx(200.0)
    assert stats.std_ms == pytest.approx(81.649658, rel=1e-5)
    assert stats.p50_ms == pytest.approx(200.0)


def test_summarize_results_calculates_throughput() -> None:
    cell = Cell(id="in4-out2-conc2", concurrency=2, input_length=4, output_length=2)
    results = [
        result(started_at=100.0, stream_end_s=0.6),
        result(started_at=100.5, stream_end_s=0.5, output_tokens=5, input_tokens=20),
    ]
    summary = summarize_results(results, cell, "mock", 0)
    assert summary.completed == 2
    assert summary.failed == 0
    assert summary.duration_s == pytest.approx(1.0)
    assert summary.request_throughput == pytest.approx(2.0)
    assert summary.output_token_throughput == pytest.approx(10.0)
    assert summary.total_token_throughput == pytest.approx(40.0)
    assert summary.peak_concurrency == 2
    assert summary.ttft.p50_ms == pytest.approx(100.0)
    assert summary.decode_tokens_per_s == pytest.approx(10.0)
    assert summary.decode_tokens_per_s_per_stream == pytest.approx(10.0)


def test_duration_uses_stream_end_instead_of_last_token() -> None:
    cell = Cell(id="in4-out2-conc1", concurrency=1, input_length=4, output_length=2)
    summary = summarize_results(
        [result(started_at=100.0, stream_end_s=0.8, e2el_s=0.5)], cell, "mock", 0
    )
    assert summary.duration_s == pytest.approx(0.8)


def test_throughput_counts_only_successful_requests() -> None:
    cell = Cell(id="in4-out2-conc1", concurrency=1, input_length=4, output_length=2)
    results = [
        result(started_at=100.0, stream_end_s=0.5, input_tokens=10),
        result(
            started_at=100.5,
            stream_end_s=0.5,
            success=False,
            input_tokens=None,
            output_tokens=0,
        ),
    ]
    summary = summarize_results(results, cell, "mock", 0)
    assert summary.duration_s == pytest.approx(1.0)
    assert summary.request_throughput == pytest.approx(1.0)
    assert summary.output_token_throughput == pytest.approx(5.0)
    assert summary.total_token_throughput == pytest.approx(15.0)


def test_peak_concurrency_uses_half_open_intervals() -> None:
    cell = Cell(id="in4-out2-conc2", concurrency=2, input_length=4, output_length=2)
    results = [
        result(started_at=100.0, stream_end_s=0.5),
        result(started_at=100.5, stream_end_s=0.5),
    ]
    summary = summarize_results(results, cell, "mock", 0)
    assert summary.peak_concurrency == 1


def test_total_throughput_is_none_when_input_tokens_are_unknown() -> None:
    cell = Cell(id="in4-out2-conc1", concurrency=1, input_length=4, output_length=2)
    summary = summarize_results(
        [result(started_at=time.perf_counter(), stream_end_s=0.5, input_tokens=None)],
        cell,
        "mock",
        0,
    )
    assert summary.total_token_throughput is None
    assert summary.output_token_throughput is not None


def test_all_failed_requests_do_not_produce_latency_stats() -> None:
    cell = Cell(id="in4-out2-conc1", concurrency=1, input_length=4, output_length=2)
    summary = summarize_results(
        [result(started_at=100.0, stream_end_s=0.2, success=False)], cell, "mock", 0
    )
    assert summary.completed == 0
    assert summary.failed == 1
    assert summary.ttft.count == 0
    assert summary.ttft.mean_ms is None
