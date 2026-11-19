import asyncio

import httpx
import pytest

from llmprobe.vllm_metrics import (
    VLLMMetricsError,
    diff_spec_decode,
    fetch_spec_decode_snapshot,
    parse_spec_decode_metrics,
)

METRICS_TEXT = """
# HELP vllm:spec_decode_num_drafts Number of speculative decoding drafts.
# TYPE vllm:spec_decode_num_drafts counter
vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 10
vllm:spec_decode_num_drafts_total{engine="1",model_name="m"} 5
# HELP vllm:spec_decode_num_draft_tokens Number of draft tokens.
# TYPE vllm:spec_decode_num_draft_tokens counter
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="m"} 50
vllm:spec_decode_num_draft_tokens_total{engine="1",model_name="m"} 25
# HELP vllm:spec_decode_num_accepted_tokens Number of accepted tokens.
# TYPE vllm:spec_decode_num_accepted_tokens counter
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m"} 10
vllm:spec_decode_num_accepted_tokens_total{engine="1",model_name="m"} 3
# HELP vllm:spec_decode_num_accepted_tokens_per_pos Accepted tokens per position.
# TYPE vllm:spec_decode_num_accepted_tokens_per_pos counter
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="0"} 8
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="1",model_name="m",position="0"} 3
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="1"} 2
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="1",model_name="m",position="1"} 0
# unrelated metric
vllm:num_requests_running{model_name="m"} 2
"""


def test_parse_spec_decode_metrics_aggregates_label_sets() -> None:
    snapshot = parse_spec_decode_metrics(METRICS_TEXT)

    assert snapshot.found is True
    assert snapshot.drafts == 15
    assert snapshot.draft_tokens == 75
    assert snapshot.accepted_tokens == 13
    assert snapshot.accepted_tokens_per_pos == {0: 11, 1: 2}


def test_diff_spec_decode_calculates_acceptance_metrics() -> None:
    before = parse_spec_decode_metrics(
        """
        vllm:spec_decode_num_drafts_total 10
        vllm:spec_decode_num_draft_tokens_total 50
        vllm:spec_decode_num_accepted_tokens_total 10
        vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 8
        vllm:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 2
        """
    )
    after = parse_spec_decode_metrics(
        """
        vllm:spec_decode_num_drafts_total 20
        vllm:spec_decode_num_draft_tokens_total 100
        vllm:spec_decode_num_accepted_tokens_total 23
        vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 16
        vllm:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 7
        """
    )

    metrics = diff_spec_decode(before, after)

    assert metrics.available is True
    assert metrics.drafts == 10
    assert metrics.draft_tokens == 50
    assert metrics.accepted_tokens == 13
    assert metrics.acceptance_rate == pytest.approx(0.26)
    assert metrics.mean_acceptance_length == pytest.approx(2.3)
    assert metrics.per_position_acceptance_rate == pytest.approx((0.8, 0.5))
    assert metrics.error is None


def test_diff_spec_decode_reports_missing_metrics_and_counter_reset() -> None:
    empty = parse_spec_decode_metrics("vllm:num_requests_running 1\n")
    metrics = diff_spec_decode(empty, empty)
    assert metrics.available is False
    assert "no vLLM" in (metrics.error or "")

    before = parse_spec_decode_metrics(
        "vllm:spec_decode_num_drafts_total 10\n"
        "vllm:spec_decode_num_draft_tokens_total 50\n"
        "vllm:spec_decode_num_accepted_tokens_total 10\n"
    )
    after = parse_spec_decode_metrics(
        "vllm:spec_decode_num_drafts_total 5\n"
        "vllm:spec_decode_num_draft_tokens_total 20\n"
        "vllm:spec_decode_num_accepted_tokens_total 5\n"
    )
    metrics = diff_spec_decode(before, after)
    assert metrics.available is False
    assert "reset" in (metrics.error or "")


def test_fetch_spec_decode_snapshot_uses_explicit_headers() -> None:
    seen_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update(request.headers)
        return httpx.Response(200, text=METRICS_TEXT)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            snapshot = await fetch_spec_decode_snapshot(
                client,
                "http://metrics.test/metrics",
                {"Authorization": "Bearer metrics-token"},
                timeout_s=1,
            )
        assert snapshot.drafts == 15

    asyncio.run(run())
    assert seen_headers["authorization"] == "Bearer metrics-token"


def test_fetch_spec_decode_snapshot_does_not_leak_url_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(VLLMMetricsError, match="HTTP 401") as exc_info:
                await fetch_spec_decode_snapshot(
                    client,
                    "http://metrics.test/metrics?token=super-secret",
                    {},
                    timeout_s=1,
                )
        assert "super-secret" not in str(exc_info.value)

    asyncio.run(run())

