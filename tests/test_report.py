import csv
import io
import json
import time
from dataclasses import replace
from pathlib import Path

from rich.console import Console

from llmbench.config import AppConfig, RunConfig, TargetConfig, WorkloadConfig
from llmbench.metrics import summarize_results
from llmbench.models import Cell, CellSummary, RequestResult, RunResult, TargetCapability
from llmbench.report import (
    config_to_dict,
    cross_repeat_aggregates,
    print_summary,
    request_to_dict,
    summary_to_dict,
    write_report,
)

REQUIRED_REPORT_FILES = {
    "config.effective.yaml",
    "run.json",
    "requests.jsonl",
    "summary.csv",
    "summary.json",
    "summary.md",
    "errors.log",
}


def make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        run=RunConfig(name="report-test", output_dir=tmp_path / "results"),
        workload=WorkloadConfig(
            requests_per_cell=1,
            warmup_requests=0,
            concurrencies=(1,),
            input_lengths=(8,),
            output_lengths=(2,),
        ),
        targets=(
            TargetConfig(
                name="mock",
                base_url="http://mock/v1",
                model="m",
                headers={"Authorization": "Bearer super-secret"},
            ),
        ),
    )


def make_result() -> RequestResult:
    return RequestResult(
        target="mock",
        cell_id="i8-o2-c1",
        repeat=0,
        request_index=0,
        concurrency=1,
        input_length_target=8,
        output_length_target=2,
        input_tokens=8,
        output_tokens=2,
        input_tokens_source="usage",
        output_tokens_source="usage",
        input_tokens_estimated=False,
        output_tokens_estimated=False,
        client_queue_s=0.0,
        started_at=time.perf_counter(),
        ttft_s=0.01,
        e2el_s=0.02,
        stream_end_s=0.03,
        tpot_s=0.01,
        itl_s=[0.01],
        token_event_count=2,
        finish_reason="stop",
        success=True,
        error_kind=None,
        error=None,
        http_status=200,
        generated_text="secret output text",
    )


def make_summary(item: RequestResult, repeat: int = 0) -> CellSummary:
    cell = Cell(id="i8-o2-c1", concurrency=1, input_length=8, output_length=2)
    repeated = replace(item, repeat=repeat)
    return summarize_results([repeated], cell, "mock", repeat)


def make_run(
    summaries: list[CellSummary],
    requests: list[RequestResult],
    *,
    capabilities: list[TargetCapability] | None = None,
) -> RunResult:
    return RunResult(
        run_id="20260101T000000Z-report-test",
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:00:02+00:00",
        summaries=summaries,
        requests=requests,
        capabilities=capabilities or [],
    )


def test_config_to_dict_serializes_paths_and_redacts_nested_secrets(tmp_path: Path) -> None:
    config = AppConfig(
        run=RunConfig(name="redaction", output_dir=tmp_path / "results"),
        workload=WorkloadConfig(requests_per_cell=1),
        targets=(
            TargetConfig(
                name="mock",
                base_url="http://mock/v1",
                model="m",
                api_key_env="PROVIDER_API_KEY",
                headers={
                    "Authorization": "Bearer super-secret",
                    "X-API-Key": "header-api-secret",
                    "X-Tenant": "visible-tenant",
                },
                request_params={
                    "nested": {
                        "client_secret": "nested-secret",
                        "clientSecret": "camel-secret",
                        "plain": "visible",
                    },
                    "items": [{"access-token": "list-secret"}],
                },
            ),
        ),
    )

    data = config_to_dict(config)
    target = data["targets"][0]

    assert data["run"]["output_dir"] == str(tmp_path / "results")
    assert target["api_key_env"] == "PROVIDER_API_KEY"
    assert target["headers"]["Authorization"] == "<redacted>"
    assert target["headers"]["X-API-Key"] == "<redacted>"
    assert target["headers"]["X-Tenant"] == "visible-tenant"
    assert target["request_params"]["nested"]["client_secret"] == "<redacted>"
    assert target["request_params"]["nested"]["clientSecret"] == "<redacted>"
    assert target["request_params"]["nested"]["plain"] == "visible"
    assert target["request_params"]["items"][0]["access-token"] == "<redacted>"
    rendered = json.dumps(data)
    for secret in (
        "super-secret",
        "header-api-secret",
        "nested-secret",
        "camel-secret",
        "list-secret",
    ):
        assert secret not in rendered


def test_request_serialization_respects_optional_payloads() -> None:
    item = make_result()

    compact = request_to_dict(item, save_itl=False, save_generated_text=False)
    assert "itl_ms" not in compact
    assert "generated_text" not in compact
    assert compact["ttft_ms"] == 10.0
    assert compact["e2el_ms"] == 20.0
    assert compact["stream_end_ms"] == 30.0
    assert compact["itl_count"] == 1
    assert compact["itl_mean_ms"] == 10.0

    full = request_to_dict(item, save_itl=True, save_generated_text=True)
    assert full["itl_ms"] == [10.0]
    assert full["generated_text"] == "secret output text"


def test_request_serialization_preserves_unavailable_latencies() -> None:
    item = replace(make_result(), ttft_s=None, e2el_s=None, tpot_s=None, itl_s=[])

    data = request_to_dict(item, save_itl=False, save_generated_text=False)

    assert data["ttft_ms"] is None
    assert data["e2el_ms"] is None
    assert data["tpot_ms"] is None
    assert data["stream_end_ms"] == 30.0
    assert data["itl_count"] == 0
    assert data["itl_mean_ms"] is None


def test_write_report_creates_all_files_and_redacts_secret(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    result_item = make_result()
    summary = make_summary(result_item)
    run = make_run(
        [summary],
        [result_item],
        capabilities=[
            TargetCapability(
                name="mock",
                available=True,
                stream_options_sent=True,
                usage_stream_supported=True,
            )
        ],
    )

    output = write_report(run, config)

    assert {path.name for path in output.iterdir()} == REQUIRED_REPORT_FILES
    for path in output.iterdir():
        assert "super-secret" not in path.read_text(encoding="utf-8")
    with (output / "summary.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["target"] == "mock"
    assert rows[0]["cell_id"] == "i8-o2-c1"
    assert rows[0]["repeat"] == "0"
    assert json.loads(rows[0]["token_sources"]) == {"usage/usage": 1}
    assert all(value is not None for value in rows[0].values())
    payload = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["capabilities"][0]["name"] == "mock"
    request_row = json.loads(
        (output / "requests.jsonl").read_text(encoding="utf-8").strip()
    )
    assert "itl_ms" not in request_row
    assert "generated_text" not in request_row


def test_write_report_honors_save_flags_and_writes_errors(tmp_path: Path) -> None:
    config = replace(
        make_config(tmp_path),
        run=replace(
            make_config(tmp_path).run,
            save_itl=True,
            save_generated_text=True,
        ),
    )
    failed = replace(
        make_result(),
        success=False,
        error_kind="http",
        error="upstream failed",
        generated_text="failed output",
    )
    summary = make_summary(failed)
    run = make_run([summary], [failed])

    output = write_report(run, config)

    request_row = json.loads(
        (output / "requests.jsonl").read_text(encoding="utf-8").strip()
    )
    assert request_row["itl_ms"] == [10.0]
    assert request_row["generated_text"] == "failed output"
    assert (
        (output / "errors.log").read_text(encoding="utf-8")
        == "mock i8-o2-c1 repeat=0 request=0: http: upstream failed\n"
    )


def test_summary_outputs_include_repeat_rows_and_cross_repeat_mean_std(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    first_request = replace(make_result(), ttft_s=0.01, repeat=0)
    second_request = replace(make_result(), ttft_s=0.02, repeat=1)
    first_summary = make_summary(first_request, repeat=0)
    second_summary = make_summary(second_request, repeat=1)
    run = make_run([first_summary, second_summary], [first_request, second_request])

    output = write_report(run, config)
    payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert [item["repeat"] for item in payload["summaries"]] == [0, 1]
    assert payload["summaries"][0]["ttft"]["p50_ms"] == 10.0
    assert payload["cross_repeat"] == [
        {
            "target": "mock",
            "cell_id": "i8-o2-c1",
            "repeats": 2,
            "metrics": {
                "request_throughput": {
                    "mean": payload["cross_repeat"][0]["metrics"]["request_throughput"][
                        "mean"
                    ],
                    "std": payload["cross_repeat"][0]["metrics"]["request_throughput"][
                        "std"
                    ],
                },
                "output_token_throughput": payload["cross_repeat"][0]["metrics"][
                    "output_token_throughput"
                ],
                "total_token_throughput": payload["cross_repeat"][0]["metrics"][
                    "total_token_throughput"
                ],
                "ttft_p50_ms": {"mean": 15.0, "std": 5.0},
                "ttft_p95_ms": {"mean": 15.0, "std": 5.0},
                "ttft_p99_ms": {"mean": 15.0, "std": 5.0},
                "tpot_p50_ms": payload["cross_repeat"][0]["metrics"]["tpot_p50_ms"],
                "tpot_p95_ms": payload["cross_repeat"][0]["metrics"]["tpot_p95_ms"],
                "itl_p95_ms": payload["cross_repeat"][0]["metrics"]["itl_p95_ms"],
                "e2el_p50_ms": payload["cross_repeat"][0]["metrics"]["e2el_p50_ms"],
                "e2el_p95_ms": payload["cross_repeat"][0]["metrics"]["e2el_p95_ms"],
            },
        }
    ]
    with (output / "summary.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(isinstance(value, str) for row in rows for value in row.values())


def test_cross_repeat_aggregates_and_summary_to_dict_are_public() -> None:
    first = make_summary(replace(make_result(), ttft_s=0.01), repeat=0)
    second = make_summary(replace(make_result(), ttft_s=0.03), repeat=1)

    aggregates = cross_repeat_aggregates([second, first])

    assert len(aggregates) == 1
    assert aggregates[0]["repeats"] == 2
    assert aggregates[0]["metrics"]["ttft_p50_ms"] == {"mean": 20.0, "std": 10.0}
    assert summary_to_dict(first)["repeat"] == 0
    assert summary_to_dict(first)["target"] == "mock"


def test_write_report_does_not_overwrite_an_existing_run(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    item = make_result()
    run = make_run([make_summary(item)], [item])

    first = write_report(run, config)
    second = write_report(run, config)

    assert first != second
    assert second.name == f"{run.run_id}-1"
    assert (first / "run.json").exists()
    assert (second / "run.json").exists()


def test_print_summary_renders_the_run_table() -> None:
    item = make_result()
    run = make_run([make_summary(item)], [item])
    stream = io.StringIO()

    print_summary(run, Console(file=stream, force_terminal=False, width=160))

    rendered = stream.getvalue()
    assert "llmbench results: 20260101T000000Z-report-test" in rendered
    assert "mock" in rendered
    assert "i8-o2-c1" in rendered
