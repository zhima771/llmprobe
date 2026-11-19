from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class Cell:
    id: str
    concurrency: int
    input_length: int
    output_length: int


@dataclass(frozen=True)
class GeneratedPrompt:
    system: str
    user: str
    target_tokens: int
    estimated_tokens: int
    tokenizer_used: bool

    @property
    def completions_text(self) -> str:
        if self.system:
            return f"{self.system}\n\n{self.user}"
        return self.user


@dataclass
class RequestResult:
    target: str
    cell_id: str
    repeat: int
    request_index: int
    concurrency: int
    input_length_target: int
    output_length_target: int
    input_tokens: int | None
    output_tokens: int
    input_tokens_source: str
    output_tokens_source: str
    input_tokens_estimated: bool
    output_tokens_estimated: bool
    client_queue_s: float
    started_at: float
    ttft_s: float | None
    e2el_s: float | None
    stream_end_s: float
    tpot_s: float | None
    itl_s: list[float]
    token_event_count: int
    finish_reason: str | None
    success: bool
    error_kind: str | None
    error: str | None
    http_status: int | None
    generated_text: str | None


@dataclass(frozen=True)
class LatencyStats:
    count: int = 0
    mean_ms: float | None = None
    std_ms: float | None = None
    p50_ms: float | None = None
    p90_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None


@dataclass(frozen=True)
class CellSummary:
    target: str
    cell_id: str
    repeat: int
    concurrency: int
    input_length: int
    output_length: int
    completed: int
    failed: int
    error_rate: float
    duration_s: float
    request_throughput: float
    output_token_throughput: float | None
    total_token_throughput: float | None
    ttft: LatencyStats
    tpot: LatencyStats
    itl: LatencyStats
    e2el: LatencyStats
    decode_tokens_per_s: float | None
    input_tokens_mean: float | None
    output_tokens_mean: float | None
    peak_concurrency: int
    token_sources: dict[str, int] = field(default_factory=dict)

    @property
    def decode_tokens_per_s_per_stream(self) -> float | None:
        """Backward-compatible alias for decode_tokens_per_s."""
        return self.decode_tokens_per_s


@dataclass(frozen=True)
class TargetCapability:
    name: str
    available: bool
    stream_options_sent: bool
    usage_stream_supported: bool | None
    error: str | None = None


@dataclass(frozen=True)
class RunResult:
    run_id: str
    started_at: str
    ended_at: str
    summaries: list[CellSummary]
    requests: list[RequestResult]
    capabilities: list[TargetCapability]
    interrupted: bool = False
    tool_version: str = "0.1.0"
    schema_version: int = 1


def make_run_id(output_dir: Path, name: str) -> str:
    del output_dir
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-").lower() or "run"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{slug}"
