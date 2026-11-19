import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import yaml

from llmbench.cli import main
from llmbench.config import load_config
from llmbench.models import RunResult, make_run_id
from tests.mock_openai import MockOpenAIHandler, MockOpenAIServer


@contextmanager
def running_server(*, reject_stream_options: bool = False) -> Iterator[MockOpenAIServer]:
    server = MockOpenAIServer(("127.0.0.1", 0), MockOpenAIHandler)  # type: ignore[arg-type]
    server.reject_stream_options = reject_stream_options
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def write_config(
    tmp_path: Path, port: int, *, name: str = "e2e", base_url: str | None = None
) -> Path:
    config = {
        "run": {
            "name": name,
            "seed": 42,
            "output_dir": str(tmp_path / "results"),
            "repeats": 1,
            "cooldown_s": 0,
        },
        "workload": {
            "requests_per_cell": 4,
            "warmup_requests": 0,
            "concurrencies": [2],
            "input_lengths": [8],
            "output_lengths": [4],
            "stream_options": "auto",
        },
        "targets": [
            {
                "name": "mock",
                "backend": "openai-chat",
                "base_url": base_url or f"http://127.0.0.1:{port}/v1",
                "model": "mock-model",
            }
        ],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_dry_run_does_not_need_a_server(tmp_path: Path) -> None:
    config = write_config(tmp_path, 9)
    assert main(["run", str(config), "--dry-run"]) == 0


def test_dry_run_masks_base_url_query_values(tmp_path: Path, capsys) -> None:
    config = write_config(
        tmp_path,
        9,
        base_url="http://127.0.0.1:9/v1?api-version=query-secret&mode=bench",
    )
    assert main(["run", str(config), "--dry-run"]) == 0
    rendered = "".join(capsys.readouterr().out.split())
    assert "api-version=%3Credacted%3E" in rendered
    assert "mode=%3Credacted%3E" in rendered
    assert "query-secret" not in rendered


def test_invalid_config_returns_two(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("targets: []\n", encoding="utf-8")
    assert main(["validate", str(path)]) == 2


def test_end_to_end_chat_benchmark(tmp_path: Path, capsys) -> None:
    with running_server() as server:
        config = write_config(tmp_path, server.server_address[1])
        assert main(["run", str(config)]) == 0
    output_dirs = list((tmp_path / "results").iterdir())
    assert len(output_dirs) == 1
    output = output_dirs[0]
    request_lines = (output / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(request_lines) == 4
    request_rows = [json.loads(line) for line in request_lines]
    assert {item["input_tokens_source"] for item in request_rows} == {"usage"}
    assert {item["output_tokens_source"] for item in request_rows} == {"usage"}
    assert {item["input_tokens"] for item in request_rows} == {8}
    assert {item["output_tokens"] for item in request_rows} == {3}
    row = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert row["capabilities"][0]["usage_stream_supported"] is True
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["summaries"][0]["completed"] == 4


def test_stream_options_fallback_end_to_end(tmp_path: Path) -> None:
    with running_server() as server:
        server.reject_stream_options = True
        config = write_config(tmp_path, server.server_address[1], name="fallback")
        assert main(["run", str(config)]) == 0
    output = next((tmp_path / "results").iterdir())
    request_rows = [
        json.loads(line)
        for line in (output / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {item["input_tokens_source"] for item in request_rows} == {"unavailable"}
    assert {item["output_tokens_source"] for item in request_rows} == {"events"}
    assert {item["output_tokens"] for item in request_rows} == {3}
    row = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert row["capabilities"][0]["stream_options_sent"] is False
    assert row["capabilities"][0]["usage_stream_supported"] is False
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["summaries"][0]["completed"] == 4


def test_validate_valid_config_returns_zero(tmp_path: Path) -> None:
    config = write_config(tmp_path, 9)
    assert main(["validate", str(config)]) == 0


def test_run_with_no_successful_requests_returns_one(
    tmp_path: Path, monkeypatch
) -> None:
    config = write_config(tmp_path, 9)
    loaded = load_config(config)
    result = RunResult(
        run_id=make_run_id(loaded.run.output_dir, loaded.run.name),
        started_at="2026-11-19T00:00:00+00:00",
        ended_at="2026-11-19T00:00:01+00:00",
        summaries=[],
        requests=[],
        capabilities=[],
    )

    class FailedRunner:
        def __init__(self, config: object, progress: object) -> None:
            self.config = config
            self.progress = progress

        async def run(self) -> RunResult:
            return result

    monkeypatch.setattr("llmbench.cli.BenchmarkRunner", FailedRunner)
    assert main(["run", str(config)]) == 1


def test_unexpected_benchmark_error_returns_one(tmp_path: Path, monkeypatch) -> None:
    config = write_config(tmp_path, 9)

    class BrokenRunner:
        def __init__(self, config: object, progress: object) -> None:
            self.config = config
            self.progress = progress

        async def run(self) -> RunResult:
            raise RuntimeError("boom")

    monkeypatch.setattr("llmbench.cli.BenchmarkRunner", BrokenRunner)
    assert main(["run", str(config)]) == 1


def test_keyboard_interrupt_writes_partial_report_and_returns_130(
    tmp_path: Path, monkeypatch
) -> None:
    config = write_config(tmp_path, 9)
    loaded = load_config(config)
    run_id = make_run_id(loaded.run.output_dir, loaded.run.name)
    partial = RunResult(
        run_id=run_id,
        started_at="2026-11-19T00:00:00+00:00",
        ended_at="2026-11-19T00:00:01+00:00",
        summaries=[],
        requests=[],
        capabilities=[],
        interrupted=True,
    )

    class InterruptedRunner:
        def __init__(self, config: object, progress: object) -> None:
            self.config = config
            self.progress = progress

        async def run(self) -> RunResult:
            raise KeyboardInterrupt

        def partial_result(self) -> RunResult:
            return partial

    monkeypatch.setattr("llmbench.cli.BenchmarkRunner", InterruptedRunner)
    assert main(["run", str(config)]) == 130
    run_data = json.loads(
        (tmp_path / "results" / run_id / "run.json").read_text(encoding="utf-8")
    )
    assert run_data["interrupted"] is True


def test_report_write_failure_returns_one(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = write_config(tmp_path, 9)
    loaded = load_config(config)
    result = RunResult(
        run_id=make_run_id(loaded.run.output_dir, loaded.run.name),
        started_at="2026-11-19T00:00:00+00:00",
        ended_at="2026-11-19T00:00:01+00:00",
        summaries=[],
        requests=[],
        capabilities=[],
    )

    class CompletedRunner:
        def __init__(self, config: object, progress: object) -> None:
            self.config = config
            self.progress = progress

        async def run(self) -> RunResult:
            return result

    def fail_write(result: RunResult, config: object) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr("llmbench.cli.BenchmarkRunner", CompletedRunner)
    monkeypatch.setattr("llmbench.cli.write_report", fail_write)
    assert main(["run", str(config)]) == 1
    stderr = capsys.readouterr().err
    assert "report failed" in stderr
    assert "disk full" in stderr


def test_print_summary_failure_returns_one(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = write_config(tmp_path, 9)
    loaded = load_config(config)
    result = RunResult(
        run_id=make_run_id(loaded.run.output_dir, loaded.run.name),
        started_at="2026-11-19T00:00:00+00:00",
        ended_at="2026-11-19T00:00:01+00:00",
        summaries=[],
        requests=[],
        capabilities=[],
    )

    class CompletedRunner:
        def __init__(self, config: object, progress: object) -> None:
            self.config = config
            self.progress = progress

        async def run(self) -> RunResult:
            return result

    def fail_summary(result: RunResult, console: object) -> None:
        raise RuntimeError("render failed")

    monkeypatch.setattr("llmbench.cli.BenchmarkRunner", CompletedRunner)
    monkeypatch.setattr(
        "llmbench.cli.write_report", lambda result, config: tmp_path / "report"
    )
    monkeypatch.setattr("llmbench.cli.print_summary", fail_summary)
    assert main(["run", str(config)]) == 1
    assert "report failed" in capsys.readouterr().err


def test_interrupt_with_report_write_failure_still_returns_130(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = write_config(tmp_path, 9)
    loaded = load_config(config)
    partial = RunResult(
        run_id=make_run_id(loaded.run.output_dir, loaded.run.name),
        started_at="2026-11-19T00:00:00+00:00",
        ended_at="2026-11-19T00:00:01+00:00",
        summaries=[],
        requests=[],
        capabilities=[],
        interrupted=True,
    )

    class InterruptedRunner:
        def __init__(self, config: object, progress: object) -> None:
            self.config = config
            self.progress = progress

        async def run(self) -> RunResult:
            raise KeyboardInterrupt

        def partial_result(self) -> RunResult:
            return partial

    def fail_write(result: RunResult, config: object) -> Path:
        raise OSError("partial disk full")

    monkeypatch.setattr("llmbench.cli.BenchmarkRunner", InterruptedRunner)
    monkeypatch.setattr("llmbench.cli.write_report", fail_write)
    assert main(["run", str(config)]) == 130
    stderr = capsys.readouterr().err
    assert "partial report failed" in stderr
    assert "partial disk full" in stderr
