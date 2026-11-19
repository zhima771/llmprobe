from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

import httpx

from llmprobe.models import SpecDecodeMetrics

_METRIC_LINE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)"
)
_LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:\\.|[^"\\])*)"')
_METRIC_NAMES = {
    "vllm:spec_decode_num_drafts": "drafts",
    "vllm:spec_decode_num_draft_tokens": "draft_tokens",
    "vllm:spec_decode_num_accepted_tokens": "accepted_tokens",
    "vllm:spec_decode_num_accepted_tokens_per_pos": "accepted_tokens_per_pos",
}


class VLLMMetricsError(RuntimeError):
    """Raised when a vLLM metrics endpoint cannot be read."""


@dataclass(frozen=True)
class SpecDecodeSnapshot:
    drafts: float | None = None
    draft_tokens: float | None = None
    accepted_tokens: float | None = None
    accepted_tokens_per_pos: dict[int, float] = field(default_factory=dict)
    found: bool = False


def _canonical_metric_name(name: str) -> str:
    if name.endswith("_total"):
        return name[:-6]
    return name


def _parse_labels(text: str) -> dict[str, str]:
    return {key: value for key, value in _LABEL.findall(text)}


def parse_spec_decode_metrics(text: str) -> SpecDecodeSnapshot:
    """Parse the speculative-decoding subset of a Prometheus exposition."""
    values: dict[str, float] = {}
    per_position: dict[int, float] = {}
    found = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE.match(line)
        if match is None:
            continue
        metric_name = _canonical_metric_name(match.group("name"))
        field_name = _METRIC_NAMES.get(metric_name)
        if field_name is None:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        found = True
        labels = _parse_labels(match.group("labels") or "")
        if field_name == "accepted_tokens_per_pos":
            try:
                position = int(labels["position"])
            except (KeyError, ValueError):
                continue
            per_position[position] = per_position.get(position, 0.0) + value
            continue
        values[field_name] = values.get(field_name, 0.0) + value

    return SpecDecodeSnapshot(
        drafts=values.get("drafts"),
        draft_tokens=values.get("draft_tokens"),
        accepted_tokens=values.get("accepted_tokens"),
        accepted_tokens_per_pos=per_position,
        found=found,
    )


def _delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return after - before


def diff_spec_decode(
    before: SpecDecodeSnapshot, after: SpecDecodeSnapshot
) -> SpecDecodeMetrics:
    """Calculate one cell's spec-decoding metrics from two counter snapshots."""
    if not before.found or not after.found:
        return SpecDecodeMetrics(
            available=False,
            error="no vLLM speculative decoding metrics found",
        )

    drafts = _delta(before.drafts, after.drafts)
    draft_tokens = _delta(before.draft_tokens, after.draft_tokens)
    accepted_tokens = _delta(before.accepted_tokens, after.accepted_tokens)
    if any(value is not None and value < 0 for value in (drafts, draft_tokens, accepted_tokens)):
        return SpecDecodeMetrics(
            available=False,
            error="vLLM spec decoding counters reset during the cell",
        )

    per_position: list[float] = []
    positions = sorted(
        set(before.accepted_tokens_per_pos) | set(after.accepted_tokens_per_pos)
    )
    for position in positions:
        delta = _delta(
            before.accepted_tokens_per_pos.get(position, 0.0),
            after.accepted_tokens_per_pos.get(position, 0.0),
        )
        if delta is not None and delta < 0:
            return SpecDecodeMetrics(
                available=False,
                error="vLLM spec decoding per-position counters reset during the cell",
            )
        if drafts and delta is not None:
            per_position.append(delta / drafts)

    if draft_tokens is None or accepted_tokens is None:
        return SpecDecodeMetrics(
            available=False,
            drafts=drafts,
            draft_tokens=draft_tokens,
            accepted_tokens=accepted_tokens,
            error="missing vLLM draft/accepted token counters",
        )

    acceptance_rate = accepted_tokens / draft_tokens if draft_tokens > 0 else None
    acceptance_length = 1.0 + accepted_tokens / drafts if drafts and drafts > 0 else None
    return SpecDecodeMetrics(
        available=True,
        drafts=drafts,
        draft_tokens=draft_tokens,
        accepted_tokens=accepted_tokens,
        acceptance_rate=acceptance_rate,
        mean_acceptance_length=acceptance_length,
        per_position_acceptance_rate=tuple(per_position),
    )


async def fetch_spec_decode_snapshot(
    client: httpx.AsyncClient,
    url: str,
    headers: Mapping[str, str],
    *,
    timeout_s: float,
) -> SpecDecodeSnapshot:
    """Fetch and parse a vLLM Prometheus endpoint without leaking its URL."""
    try:
        response = await client.get(url, headers=dict(headers), timeout=timeout_s)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise VLLMMetricsError(f"metrics endpoint returned HTTP {exc.response.status_code}") from None
    except httpx.TimeoutException:
        raise VLLMMetricsError("metrics endpoint timed out") from None
    except httpx.HTTPError as exc:
        raise VLLMMetricsError(f"metrics request failed: {type(exc).__name__}") from None
    except ValueError as exc:
        raise VLLMMetricsError(f"invalid metrics response: {exc}") from None
    return parse_spec_decode_metrics(response.text)

