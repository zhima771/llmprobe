from __future__ import annotations

import math
import statistics
from collections import Counter

from llmbench.models import Cell, CellSummary, LatencyStats, RequestResult


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if not 0 <= p <= 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(values)
    position = (len(ordered) - 1) * (p / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def latency_stats(values_s: list[float]) -> LatencyStats:
    if not values_s:
        return LatencyStats()
    values_ms = [value * 1000.0 for value in values_s]
    return LatencyStats(
        count=len(values_ms),
        mean_ms=statistics.fmean(values_ms),
        std_ms=statistics.pstdev(values_ms),
        p50_ms=percentile(values_ms, 50),
        p90_ms=percentile(values_ms, 90),
        p95_ms=percentile(values_ms, 95),
        p99_ms=percentile(values_ms, 99),
    )


def _peak_concurrency(results: list[RequestResult]) -> int:
    events: list[tuple[float, int]] = []
    for result in results:
        events.append((result.started_at, 1))
        events.append((result.started_at + result.stream_end_s, -1))

    active = 0
    peak = 0
    # End events sort before start events at the same timestamp, matching
    # half-open intervals: [started_at, started_at + stream_end_s).
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        peak = max(peak, active)
    return peak


def summarize_results(
    results: list[RequestResult], cell: Cell, target: str, repeat: int
) -> CellSummary:
    successful = [result for result in results if result.success]
    failed = len(results) - len(successful)

    if results:
        start = min(result.started_at for result in results)
        end = max(result.started_at + result.stream_end_s for result in results)
        duration_s = max(0.0, end - start)
    else:
        duration_s = 0.0

    ttfts = [result.ttft_s for result in successful if result.ttft_s is not None]
    e2els = [result.e2el_s for result in successful if result.e2el_s is not None]
    tpots = [result.tpot_s for result in successful if result.tpot_s is not None]
    itls = [value for result in successful for value in result.itl_s]

    output_total = sum(result.output_tokens for result in successful)
    input_known = bool(successful) and all(
        result.input_tokens is not None for result in successful
    )
    input_total = sum(
        result.input_tokens for result in successful if result.input_tokens is not None
    )

    if duration_s > 0:
        request_throughput = len(successful) / duration_s
        output_throughput: float | None = output_total / duration_s
        total_throughput: float | None = (
            (input_total + output_total) / duration_s if input_known else None
        )
    else:
        request_throughput = 0.0
        output_throughput = None
        total_throughput = None

    tpot_stats = latency_stats(tpots)
    decode_tps = (
        1000.0 / tpot_stats.mean_ms
        if tpot_stats.mean_ms is not None and tpot_stats.mean_ms > 0
        else None
    )

    input_values = [
        result.input_tokens for result in successful if result.input_tokens is not None
    ]
    output_values = [result.output_tokens for result in successful]
    token_sources = Counter(
        f"{result.input_tokens_source}/{result.output_tokens_source}"
        for result in successful
    )

    return CellSummary(
        target=target,
        cell_id=cell.id,
        repeat=repeat,
        concurrency=cell.concurrency,
        input_length=cell.input_length,
        output_length=cell.output_length,
        completed=len(successful),
        failed=failed,
        error_rate=(failed / len(results)) if results else 0.0,
        duration_s=duration_s,
        request_throughput=request_throughput,
        output_token_throughput=output_throughput,
        total_token_throughput=total_throughput,
        ttft=latency_stats(ttfts),
        tpot=tpot_stats,
        itl=latency_stats(itls),
        e2el=latency_stats(e2els),
        decode_tokens_per_s_per_stream=decode_tps,
        input_tokens_mean=statistics.fmean(input_values) if input_values else None,
        output_tokens_mean=statistics.fmean(output_values) if output_values else None,
        peak_concurrency=_peak_concurrency(results),
        token_sources=dict(token_sources),
    )
