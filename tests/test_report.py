import csv
import io
import json
import time
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from rich.console import Console

from llmprobe.config import AppConfig, CellSpec, RunConfig, TargetConfig, WorkloadConfig
from llmprobe.metrics import summarize_results
from llmprobe.models import Cell, CellSummary, RequestResult, RunResult, TargetCapability
from llmprobe.report import (
    config_to_dict,
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
            cells=(CellSpec(8, 2, 1),),
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
        cell_id="in8-out2-conc1",
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
    cell = Cell(id="in8-out2-conc1", concurrency=1, input_length=8, output_length=2)
    repeated = replace(item, repeat=repeat)
    return summarize_results([repeated], cell, "mock", repeat)


def make_config_with_request_params(
    params: dict[str, object], *, api_key_env: str | None = None
) -> AppConfig:
    return AppConfig(
        run=RunConfig(),
        workload=WorkloadConfig(requests_per_cell=1),
        targets=(
            TargetConfig(
                name="mock",
                base_url="http://mock/v1",
                model="m",
                api_key_env=api_key_env,
                request_params=params,
            ),
        ),
    )


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
                    "X-Subscription-Key": "subscription-secret",
                    "X-Auth-Key": "auth-key-secret",
                    "X-Functions-Key": "functions-secret",
                    "X-Amz-Security-Token": "amz-security-secret",
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
    for header_name in (
        "Authorization",
        "X-API-Key",
        "X-Subscription-Key",
        "X-Auth-Key",
        "X-Functions-Key",
        "X-Amz-Security-Token",
        "X-Tenant",
    ):
        assert target["headers"][header_name] == "<redacted>"
    assert target["request_params"]["nested"]["client_secret"] == "<redacted>"
    assert target["request_params"]["nested"]["clientSecret"] == "<redacted>"
    assert target["request_params"]["nested"]["plain"] == "visible"
    assert target["request_params"]["items"][0]["access-token"] == "<redacted>"
    rendered = json.dumps(data)
    for secret in (
        "super-secret",
        "header-api-secret",
        "subscription-secret",
        "auth-key-secret",
        "functions-secret",
        "amz-security-secret",
        "visible-tenant",
        "nested-secret",
        "camel-secret",
        "list-secret",
    ):
        assert secret not in rendered


def test_config_to_dict_preserves_only_unresolved_header_placeholders(tmp_path: Path) -> None:
    base = make_config(tmp_path)
    config = replace(
        base,
        targets=(
            replace(
                base.targets[0],
                headers={
                    "X-Tenant": "${TENANT}",
                    "X-Auth-Key": "${AUTH_KEY}",
                    "Authorization": "Bearer ${TOKEN}",
                    "X-Benign": "provider-value",
                },
            ),
        ),
    )

    headers = config_to_dict(config)["targets"][0]["headers"]

    assert headers["X-Tenant"] == "${TENANT}"
    assert headers["X-Auth-Key"] == "${AUTH_KEY}"
    assert headers["Authorization"] == "<redacted>"
    assert headers["X-Benign"] == "<redacted>"


@pytest.mark.parametrize(
    "key",
    [
        "Authorization",
        "key",
        "api_key",
        "api_keys",
        "apikey",
        "APIKEY",
        "x_apikey",
        "xapikey",
        "apiKey",
        "apiKeyValue",
        "X-API-Key",
        "apiToken",
        "api_token",
        "githubToken",
        "github_token",
        "jwtToken",
        "jwt_token",
        "oauth2_token",
        "access_token",
        "access_tokens",
        "refresh_token",
        "session_token",
        "auth_token",
        "id_token",
        "client_token",
        "password",
        "passwords",
        "password_hash",
        "dbpassword",
        "secret",
        "secrets",
        "client_secret",
        "client_secrets",
        "clientsecret",
        "private_key",
        "privatekey",
        "secretkey",
        "signingkey",
        "cookie",
        "cookies",
        "setcookie",
        "credential",
        "credentials",
    ],
)
def test_sensitive_key_variants_are_redacted(key: str) -> None:
    config = make_config_with_request_params({key: "do-not-emit"})

    target = config_to_dict(config)["targets"][0]

    assert target["request_params"][key] == "<redacted>"
    assert "do-not-emit" not in json.dumps(target)


@pytest.mark.parametrize(
    "key",
    [
        "max_tokens",
        "max_token",
        "stop_token",
        "stop_tokens",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "max_completion_tokens",
        "max_output_tokens",
        "token_count",
        "api_key_env",
        "auth_method",
        "cookies_enabled",
    ],
)
def test_benign_token_keys_are_not_redacted(key: str) -> None:
    config = make_config_with_request_params(
        {key: "benchmark-value"}, api_key_env="PROVIDER_API_KEY"
    )

    target = config_to_dict(config)["targets"][0]

    assert target["api_key_env"] == "PROVIDER_API_KEY"
    assert target["request_params"][key] == "benchmark-value"


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
    result_item = replace(make_result(), started_at=0.0, stream_end_s=2.0)
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
    expected_csv = {
        "target": "mock",
        "cell_id": "in8-out2-conc1",
        "concurrency": "1",
        "input_length": "8",
        "output_length": "2",
        "completed": "1",
        "failed": "0",
        "error_rate": "0.0",
        "duration_s": "2.0",
        "request_throughput": "0.5",
        "output_token_throughput": "1.0",
        "total_token_throughput": "5.0",
        "ttft_mean_ms": "10.0",
        "ttft_p50_ms": "10.0",
        "ttft_p95_ms": "10.0",
        "ttft_p99_ms": "10.0",
        "tpot_mean_ms": "10.0",
        "tpot_p50_ms": "10.0",
        "tpot_p95_ms": "10.0",
        "itl_p95_ms": "10.0",
        "e2el_p50_ms": "20.0",
        "e2el_p95_ms": "20.0",
        "decode_tokens_per_s": "100.0",
        "input_tokens_mean": "8.0",
        "output_tokens_mean": "2.0",
        "peak_concurrency": "1",
        "token_sources": '{"usage/usage": 1}',
    }
    for field, expected in expected_csv.items():
        assert rows[0][field] == expected
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
        == "mock in8-out2-conc1 request=0: http: upstream failed\n"
    )


def test_errors_log_escapes_embedded_line_breaks(tmp_path: Path) -> None:
    failed = replace(
        make_result(),
        success=False,
        error_kind="stream",
        error="first\r\nsecond\nthird\rfourth",
    )
    run = make_run([make_summary(failed)], [failed])

    output = write_report(run, make_config(tmp_path))
    log = (output / "errors.log").read_text(encoding="utf-8")

    assert log == "mock in8-out2-conc1 request=0: stream: first\\r\\nsecond\\nthird\\rfourth\n"
    assert log.count("\n") == 1


def test_summary_markdown_escapes_table_delimiters_and_line_breaks(tmp_path: Path) -> None:
    item = make_result()
    summary = replace(make_summary(item), target="a|b", cell_id="cell\r\nid")
    run = make_run([summary], [item])

    output = write_report(run, make_config(tmp_path))
    markdown = (output / "summary.md").read_text(encoding="utf-8")

    assert "| a\\|b | cell\\r\\nid |" in markdown
    assert "| TTFT avg/p95 | TPOT avg/p95 | ITL avg/p95 | E2EL avg/p95 |" in markdown
    assert "\r" not in markdown


def test_summary_json_omits_repeat_and_cross_repeat_fields(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    request = replace(
        make_result(),
        started_at=0.0,
        stream_end_s=1.0,
        ttft_s=0.01,
        e2el_s=0.02,
        tpot_s=0.01,
        itl_s=[0.01],
    )
    summary = make_summary(request, repeat=0)
    run = make_run([summary], [request])

    output = write_report(run, config)
    payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))

    assert "cross_repeat" not in payload
    assert len(payload["summaries"]) == 1
    assert "repeat" not in payload["summaries"][0]
    assert payload["summaries"][0]["cell_id"] == "in8-out2-conc1"
    with (output / "summary.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert "repeat" not in rows[0]
    assert all(isinstance(value, str) for row in rows for value in row.values())


def test_summary_to_dict_omits_repeat_and_exposes_metrics() -> None:
    summary = make_summary(replace(make_result(), tpot_s=0.01), repeat=0)

    data = summary_to_dict(summary)

    assert "repeat" not in data
    assert data["target"] == "mock"
    assert data["decode_tokens_per_s"] == 100.0
    assert data["decode_tokens_per_s_per_stream"] == 100.0


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

    print_summary(run, Console(file=stream, force_terminal=False, width=240))

    rendered = stream.getvalue()
    assert "llmprobe results: 20260101T000000Z-report-test" in rendered
    assert "mock" in rendered
    assert "in8-out2-conc1" in rendered
    assert "Decode tok/s" in rendered
    assert "TTFT avg/p95" in rendered
    assert "TPOT avg/p95" in rendered
    assert "ITL avg/p95" in rendered
    assert "E2EL avg/p95" in rendered
    assert "Repeat" not in rendered
    assert "10.00 / 10.00 ms" in rendered
    assert "20.00 / 20.00 ms" in rendered
    assert "100.00" in rendered


def test_config_serialization_sanitizes_base_url_credentials_and_query(
    tmp_path: Path,
) -> None:
    config = replace(
        make_config(tmp_path),
        targets=(
            replace(
                make_config(tmp_path).targets[0],
                base_url=(
                    "https://alice:url-secret@mock.example/v1"
                    "?api-version=query-secret&mode=bench"
                ),
            ),
        ),
    )

    target = config_to_dict(config)["targets"][0]
    sanitized = target["base_url"]

    assert sanitized.startswith("https://mock.example/v1?")
    assert "alice" not in sanitized
    assert "url-secret" not in sanitized
    assert "query-secret" not in sanitized
    assert "api-version=%3Credacted%3E" in sanitized
    assert "mode=%3Credacted%3E" in sanitized


def test_report_files_persist_sanitized_base_url_only(tmp_path: Path) -> None:
    config = replace(
        make_config(tmp_path),
        targets=(
            replace(
                make_config(tmp_path).targets[0],
                base_url=(
                    "https://alice:url-secret@mock.example/v1"
                    "?api-version=query-secret&mode=bench"
                ),
            ),
        ),
    )
    item = make_result()
    output = write_report(make_run([make_summary(item)], [item]), config)

    config_payload = yaml.safe_load(
        (output / "config.effective.yaml").read_text(encoding="utf-8")
    )
    run_payload = json.loads((output / "run.json").read_text(encoding="utf-8"))
    for base_url in (
        config_payload["targets"][0]["base_url"],
        run_payload["config"]["targets"][0]["base_url"],
    ):
        assert base_url.startswith("https://mock.example/v1?")
        assert "api-version=%3Credacted%3E" in base_url
        assert "mode=%3Credacted%3E" in base_url
        assert "alice" not in base_url
        assert "url-secret" not in base_url
        assert "query-secret" not in base_url


def test_direct_api_key_is_redacted_from_effective_config() -> None:
    config = AppConfig(
        run=RunConfig(),
        workload=WorkloadConfig(requests_per_cell=1),
        targets=(
            TargetConfig(
                name="mock",
                base_url="http://mock/v1",
                model="m",
                api_key="sk-direct-secret",
            ),
        ),
    )

    serialized = json.dumps(config_to_dict(config))
    assert "sk-direct-secret" not in serialized
    assert config_to_dict(config)["targets"][0]["api_key"] == "<redacted>"
