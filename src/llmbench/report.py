from __future__ import annotations

import csv
import json
import re
import statistics
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

from llmbench.config import AppConfig
from llmbench.models import CellSummary, RequestResult, RunResult

_REDACTED = "<redacted>"
_CAMEL_BOUNDARY = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_SENSITIVE_TOKENS = {
    "authorization",
    "password",
    "passwords",
    "passwd",
    "secret",
    "secrets",
    "cookie",
    "cookies",
    "credential",
    "credentials",
}
_EXACT_SENSITIVE_KEYS = {"key"}
_BENIGN_COMPACT_KEYS = {
    "maxtokens",
    "maxtoken",
    "stoptoken",
    "stoptokens",
    "inputtokens",
    "outputtokens",
    "totaltokens",
    "prompttokens",
    "completiontokens",
    "maxcompletiontokens",
    "maxoutputtokens",
    "tokencount",
    "apikeyenv",
    "authmethod",
    "cookiesenabled",
}
_COMPACT_CREDENTIAL_FRAGMENTS = (
    "apikey",
    "password",
    "secret",
    "privatekey",
    "signingkey",
    "setcookie",
)
_TOKEN_QUALIFIERS = {
    "access",
    "refresh",
    "api",
    "auth",
    "bearer",
    "github",
    "jwt",
    "oauth",
    "oauth2",
    "session",
    "id",
    "client",
}
_TOKEN_NAMES = {"token", "tokens"}
_CSV_FIELDS = [
    "target",
    "cell_id",
    "repeat",
    "concurrency",
    "input_length",
    "output_length",
    "completed",
    "failed",
    "error_rate",
    "duration_s",
    "request_throughput",
    "output_token_throughput",
    "total_token_throughput",
    "ttft_mean_ms",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "ttft_p99_ms",
    "tpot_mean_ms",
    "tpot_p50_ms",
    "tpot_p95_ms",
    "itl_p95_ms",
    "e2el_p50_ms",
    "e2el_p95_ms",
    "decode_tokens_per_s_per_stream",
    "input_tokens_mean",
    "output_tokens_mean",
    "peak_concurrency",
    "token_sources",
]
_METRIC_NAMES = [
    "request_throughput",
    "output_token_throughput",
    "total_token_throughput",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "ttft_p99_ms",
    "tpot_p50_ms",
    "tpot_p95_ms",
    "itl_p95_ms",
    "e2el_p50_ms",
    "e2el_p95_ms",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _normalized_tokens(key: Any) -> tuple[str, ...]:
    with_camel_boundaries = _CAMEL_BOUNDARY.sub("_", str(key))
    normalized = _NON_ALNUM.sub("_", with_camel_boundaries.lower()).strip("_")
    return tuple(part for part in normalized.split("_") if part)


def _has_adjacent_tokens(
    tokens: tuple[str, ...], first: set[str], second: set[str]
) -> bool:
    return any(
        left in first and right in second
        for left, right in pairwise(tokens)
    )


def _has_compact_qualified_token(compact: str) -> bool:
    return any(
        compact.endswith(qualifier + name)
        for qualifier in _TOKEN_QUALIFIERS
        for name in _TOKEN_NAMES
    )


def _is_sensitive_key(key: Any) -> bool:
    tokens = _normalized_tokens(key)
    if not tokens:
        return False
    compact = "".join(tokens)
    if compact in _BENIGN_COMPACT_KEYS:
        return False
    if compact in _EXACT_SENSITIVE_KEYS:
        return True
    if any(fragment in compact for fragment in _COMPACT_CREDENTIAL_FRAGMENTS):
        return True
    if _SENSITIVE_TOKENS.intersection(tokens):
        return True
    if _has_adjacent_tokens(tokens, _TOKEN_QUALIFIERS, _TOKEN_NAMES):
        return True
    if _has_compact_qualified_token(compact):
        return True
    return tokens == ("token",)


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key)
            if _is_sensitive_key(normalized_key):
                result[normalized_key] = _REDACTED
            else:
                result[normalized_key] = _redact(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    return _redact(_jsonable(config))


def request_to_dict(
    result: RequestResult, *, save_itl: bool, save_generated_text: bool
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "target": result.target,
        "cell_id": result.cell_id,
        "repeat": result.repeat,
        "request_index": result.request_index,
        "concurrency": result.concurrency,
        "input_length_target": result.input_length_target,
        "output_length_target": result.output_length_target,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "input_tokens_source": result.input_tokens_source,
        "output_tokens_source": result.output_tokens_source,
        "input_tokens_estimated": result.input_tokens_estimated,
        "output_tokens_estimated": result.output_tokens_estimated,
        "client_queue_ms": result.client_queue_s * 1000.0,
        "ttft_ms": result.ttft_s * 1000.0 if result.ttft_s is not None else None,
        "e2el_ms": result.e2el_s * 1000.0 if result.e2el_s is not None else None,
        "stream_end_ms": result.stream_end_s * 1000.0,
        "tpot_ms": result.tpot_s * 1000.0 if result.tpot_s is not None else None,
        "itl_count": len(result.itl_s),
        "itl_mean_ms": (
            statistics.fmean(result.itl_s) * 1000.0 if result.itl_s else None
        ),
        "token_event_count": result.token_event_count,
        "finish_reason": result.finish_reason,
        "success": result.success,
        "error_kind": result.error_kind,
        "error": result.error,
        "http_status": result.http_status,
    }
    if save_itl:
        data["itl_ms"] = [value * 1000.0 for value in result.itl_s]
    if save_generated_text:
        data["generated_text"] = result.generated_text
    return data


def summary_to_dict(summary: CellSummary) -> dict[str, Any]:
    return _jsonable(summary)


def _metric_value(summary: CellSummary, name: str) -> float | None:
    mapping: dict[str, Callable[[CellSummary], float | None]] = {
        "request_throughput": lambda item: item.request_throughput,
        "output_token_throughput": lambda item: item.output_token_throughput,
        "total_token_throughput": lambda item: item.total_token_throughput,
        "ttft_p50_ms": lambda item: item.ttft.p50_ms,
        "ttft_p95_ms": lambda item: item.ttft.p95_ms,
        "ttft_p99_ms": lambda item: item.ttft.p99_ms,
        "tpot_p50_ms": lambda item: item.tpot.p50_ms,
        "tpot_p95_ms": lambda item: item.tpot.p95_ms,
        "itl_p95_ms": lambda item: item.itl.p95_ms,
        "e2el_p50_ms": lambda item: item.e2el.p50_ms,
        "e2el_p95_ms": lambda item: item.e2el.p95_ms,
    }
    return mapping[name](summary)


def cross_repeat_aggregates(summaries: list[CellSummary]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[CellSummary]] = {}
    for summary in summaries:
        grouped.setdefault((summary.target, summary.cell_id), []).append(summary)

    output: list[dict[str, Any]] = []
    for (target, cell_id), items in sorted(grouped.items()):
        metrics: dict[str, dict[str, float | None]] = {}
        for metric_name in _METRIC_NAMES:
            values = [
                value
                for item in items
                if (value := _metric_value(item, metric_name)) is not None
            ]
            metrics[metric_name] = {
                "mean": statistics.fmean(values) if values else None,
                "std": (
                    statistics.pstdev(values)
                    if len(values) > 1
                    else 0.0 if values else None
                ),
            }
        output.append(
            {
                "target": target,
                "cell_id": cell_id,
                "repeats": len(items),
                "metrics": metrics,
            }
        )
    return output


def _flatten_summary(summary: CellSummary) -> dict[str, Any]:
    return {
        "target": summary.target,
        "cell_id": summary.cell_id,
        "repeat": summary.repeat,
        "concurrency": summary.concurrency,
        "input_length": summary.input_length,
        "output_length": summary.output_length,
        "completed": summary.completed,
        "failed": summary.failed,
        "error_rate": summary.error_rate,
        "duration_s": summary.duration_s,
        "request_throughput": summary.request_throughput,
        "output_token_throughput": summary.output_token_throughput,
        "total_token_throughput": summary.total_token_throughput,
        "ttft_mean_ms": summary.ttft.mean_ms,
        "ttft_p50_ms": summary.ttft.p50_ms,
        "ttft_p95_ms": summary.ttft.p95_ms,
        "ttft_p99_ms": summary.ttft.p99_ms,
        "tpot_mean_ms": summary.tpot.mean_ms,
        "tpot_p50_ms": summary.tpot.p50_ms,
        "tpot_p95_ms": summary.tpot.p95_ms,
        "itl_p95_ms": summary.itl.p95_ms,
        "e2el_p50_ms": summary.e2el.p50_ms,
        "e2el_p95_ms": summary.e2el.p95_ms,
        "decode_tokens_per_s_per_stream": summary.decode_tokens_per_s_per_stream,
        "input_tokens_mean": summary.input_tokens_mean,
        "output_tokens_mean": summary.output_tokens_mean,
        "peak_concurrency": summary.peak_concurrency,
        "token_sources": json.dumps(summary.token_sources, sort_keys=True),
    }


def _format_number(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _escape_line_breaks(value: str) -> str:
    return value.replace("\r", "\\r").replace("\n", "\\n")


def _markdown(result: RunResult) -> str:
    lines = [
        f"# Benchmark Report: {result.run_id}",
        "",
        f"- Started: {result.started_at}",
        f"- Ended: {result.ended_at}",
        f"- Interrupted: {result.interrupted}",
        "",
        (
            "| Target | Cell | Repeat | OK/Err | Req/s | Out tok/s | TTFT p50 | "
            "TTFT p95 | TPOT p50 | ITL p95 | E2EL p95 |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in result.summaries:
        lines.append(
            "| {target} | {cell} | {repeat} | {ok}/{failed} | {rps} | {out_tps} | "
            "{ttft50} | {ttft95} | {tpot50} | {itl95} | {e2el95} |".format(
                target=summary.target,
                cell=summary.cell_id,
                repeat=summary.repeat,
                ok=summary.completed,
                failed=summary.failed,
                rps=_format_number(summary.request_throughput),
                out_tps=_format_number(summary.output_token_throughput),
                ttft50=_format_number(summary.ttft.p50_ms) + " ms",
                ttft95=_format_number(summary.ttft.p95_ms) + " ms",
                tpot50=_format_number(summary.tpot.p50_ms) + " ms",
                itl95=_format_number(summary.itl.p95_ms) + " ms",
                e2el95=_format_number(summary.e2el.p95_ms) + " ms",
            )
        )
    return "\n".join(lines) + "\n"


def write_report(result: RunResult, config: AppConfig) -> Path:
    run_dir = config.run.output_dir / result.run_id
    suffix = 1
    while run_dir.exists():
        run_dir = config.run.output_dir / f"{result.run_id}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)

    config_data = config_to_dict(config)
    (run_dir / "config.effective.yaml").write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    run_data = {
        "schema_version": result.schema_version,
        "tool_version": result.tool_version,
        "run_id": result.run_id,
        "started_at": result.started_at,
        "ended_at": result.ended_at,
        "interrupted": result.interrupted,
        "config": config_data,
        "capabilities": _jsonable(result.capabilities),
    }
    (run_dir / "run.json").write_text(
        json.dumps(run_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    request_rows = [
        request_to_dict(
            item,
            save_itl=config.run.save_itl,
            save_generated_text=config.run.save_generated_text,
        )
        for item in result.requests
    ]
    (run_dir / "requests.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in request_rows),
        encoding="utf-8",
    )

    summary_data = {
        "schema_version": result.schema_version,
        "summaries": [summary_to_dict(item) for item in result.summaries],
        "cross_repeat": cross_repeat_aggregates(result.summaries),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for summary in result.summaries:
            writer.writerow(_flatten_summary(summary))
    (run_dir / "summary.md").write_text(_markdown(result), encoding="utf-8")

    errors = [
        _escape_line_breaks(
            f"{item.target} {item.cell_id} repeat={item.repeat} "
            f"request={item.request_index}: {item.error_kind or 'error'}: {item.error}"
        )
        for item in result.requests
        if not item.success
    ]
    (run_dir / "errors.log").write_text(
        "\n".join(errors) + ("\n" if errors else ""),
        encoding="utf-8",
    )
    return run_dir


def print_summary(result: RunResult, console: Console | None = None) -> None:
    output = console or Console()
    table = Table(title=f"llmbench results: {result.run_id}")
    for column in (
        "Target",
        "Cell",
        "Repeat",
        "OK/Err",
        "Req/s",
        "Out tok/s",
        "TTFT p50",
        "TTFT p95",
        "TPOT p50",
        "E2EL p95",
    ):
        table.add_column(
            column,
            justify="right" if column not in {"Target", "Cell"} else "left",
        )
    for summary in result.summaries:
        table.add_row(
            summary.target,
            summary.cell_id,
            str(summary.repeat),
            f"{summary.completed}/{summary.failed}",
            _format_number(summary.request_throughput),
            _format_number(summary.output_token_throughput),
            _format_number(summary.ttft.p50_ms) + " ms",
            _format_number(summary.ttft.p95_ms) + " ms",
            _format_number(summary.tpot.p50_ms) + " ms",
            _format_number(summary.e2el.p95_ms) + " ms",
        )
    output.print(table)
