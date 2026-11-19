import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from llmbench.client import TargetProbeError
from llmbench.config import AppConfig, ConfigError, RunConfig, TargetConfig, WorkloadConfig
from llmbench.models import Cell, GeneratedPrompt, RequestResult
from llmbench.runner import BenchmarkRunner, NullProgress


class FakeClient:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def fake_http_clients(monkeypatch: pytest.MonkeyPatch) -> list[FakeClient]:
    clients: list[FakeClient] = []

    def create_client(target: TargetConfig, max_connections: int) -> FakeClient:
        assert max_connections >= 1
        client = FakeClient()
        clients.append(client)
        return client

    monkeypatch.setattr("llmbench.runner.create_http_client", create_client)
    return clients


def config(
    tmp_path: Path,
    *,
    stream_options: str = "auto",
    repeats: int = 1,
    cooldown_s: float = 0.0,
    warmup_requests: int = 1,
    requests_per_cell: int = 2,
    concurrencies: tuple[int, ...] = (1,),
    input_lengths: tuple[int, ...] = (8,),
    output_lengths: tuple[int, ...] = (2,),
    system_prompt: str = "",
    tokenizer: str | None = None,
) -> AppConfig:
    return AppConfig(
        run=RunConfig(
            name="test",
            output_dir=tmp_path,
            repeats=repeats,
            cooldown_s=cooldown_s,
        ),
        workload=WorkloadConfig(
            requests_per_cell=requests_per_cell,
            warmup_requests=warmup_requests,
            concurrencies=concurrencies,
            input_lengths=input_lengths,
            output_lengths=output_lengths,
            system_prompt=system_prompt,
            tokenizer=tokenizer,
            stream_options=stream_options,  # type: ignore[arg-type]
        ),
        targets=(
            TargetConfig(name="a", base_url="http://a/v1", model="m"),
            TargetConfig(name="b", base_url="http://b/v1", model="m"),
        ),
    )


def make_result(
    target: str,
    cell: Cell,
    repeat: int,
    index: int,
    success: bool = True,
    *,
    client_queue_s: float = 0.0,
    output_tokens_source: str = "usage",
) -> RequestResult:
    return RequestResult(
        target=target,
        cell_id=cell.id,
        repeat=repeat,
        request_index=index,
        concurrency=cell.concurrency,
        input_length_target=cell.input_length,
        output_length_target=cell.output_length,
        input_tokens=8,
        output_tokens=2,
        input_tokens_source="usage",
        output_tokens_source=output_tokens_source,
        input_tokens_estimated=False,
        output_tokens_estimated=False,
        client_queue_s=client_queue_s,
        started_at=time.perf_counter(),
        ttft_s=0.01,
        e2el_s=0.02,
        stream_end_s=0.02,
        tpot_s=0.01,
        itl_s=[0.01],
        token_event_count=2,
        finish_reason="stop",
        success=success,
        error_kind=None if success else "http",
        error=None if success else "failed",
        http_status=200 if success else 500,
        generated_text=None,
    )


def test_runner_executes_probe_warmup_and_measurement_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str, str, int, int]] = []

    async def fake_probe(
        client: Any,
        target: TargetConfig,
        prompt: GeneratedPrompt,
        **kwargs: Any,
    ) -> tuple[bool, RequestResult]:
        cell = kwargs["cell"]
        calls.append((target.name, "probe", cell.id, -1, 0))
        return True, make_result(target.name, cell, -1, 0)

    async def fake_request(
        client: Any,
        target: TargetConfig,
        prompt: GeneratedPrompt,
        **kwargs: Any,
    ) -> RequestResult:
        cell = kwargs["cell"]
        repeat = kwargs["repeat"]
        index = kwargs["request_index"]
        calls.append(
            (target.name, "warmup" if repeat == -1 else "measure", cell.id, repeat, index)
        )
        return make_result(target.name, cell, repeat, index)

    monkeypatch.setattr("llmbench.runner.probe_usage", fake_probe)
    monkeypatch.setattr("llmbench.runner.stream_request", fake_request)
    runner = BenchmarkRunner(config(tmp_path), progress=NullProgress(), environ={})
    result = asyncio.run(runner.run())

    assert calls == [
        ("a", "probe", "probe", -1, 0),
        ("b", "probe", "probe", -1, 0),
        ("a", "warmup", "i8-o2-c1", -1, 0),
        ("b", "warmup", "i8-o2-c1", -1, 0),
        ("a", "measure", "i8-o2-c1", 0, 0),
        ("a", "measure", "i8-o2-c1", 0, 1),
        ("b", "measure", "i8-o2-c1", 0, 0),
        ("b", "measure", "i8-o2-c1", 0, 1),
    ]
    assert len(result.summaries) == 2
    assert len(result.requests) == 4
    assert all(item.repeat == 0 for item in result.requests)
    assert {item.name: item.usage_stream_supported for item in result.capabilities} == {
        "a": True,
        "b": True,
    }


def test_probe_runs_once_per_target_only_for_auto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe_calls: list[tuple[str, Cell]] = []

    async def fake_probe(
        client: Any,
        target: TargetConfig,
        prompt: GeneratedPrompt,
        **kwargs: Any,
    ) -> tuple[bool, RequestResult]:
        cell = kwargs["cell"]
        probe_calls.append((target.name, cell))
        return True, make_result(target.name, cell, -1, 0)

    async def fake_request(
        client: Any,
        target: TargetConfig,
        prompt: GeneratedPrompt,
        **kwargs: Any,
    ) -> RequestResult:
        return make_result(target.name, kwargs["cell"], kwargs["repeat"], kwargs["request_index"])

    monkeypatch.setattr("llmbench.runner.probe_usage", fake_probe)
    monkeypatch.setattr("llmbench.runner.stream_request", fake_request)
    runner = BenchmarkRunner(
        config(
            tmp_path,
            repeats=2,
            warmup_requests=0,
            requests_per_cell=1,
            input_lengths=(8, 16),
        ),
        progress=NullProgress(),
        environ={},
    )
    asyncio.run(runner.run())

    assert [name for name, _ in probe_calls] == ["a", "b"]
    assert all(cell.id == "probe" for _, cell in probe_calls)
    assert all(cell.concurrency == 1 for _, cell in probe_calls)
    assert all(cell.output_length == 2 for _, cell in probe_calls)


@pytest.mark.parametrize(
    ("mode", "probe_supported", "expected_probes", "expected_sent", "expected_usage"),
    [
        ("none", None, 0, False, None),
        ("include_usage", None, 0, True, None),
        ("auto", True, 2, True, True),
        ("auto", False, 2, False, False),
    ],
)
def test_capabilities_and_include_usage_follow_stream_option_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    probe_supported: bool | None,
    expected_probes: int,
    expected_sent: bool,
    expected_usage: bool | None,
) -> None:
    probe_calls = 0
    request_usage_flags: list[bool] = []

    async def fake_probe(
        client: Any,
        target: TargetConfig,
        prompt: GeneratedPrompt,
        **kwargs: Any,
    ) -> tuple[bool, RequestResult]:
        nonlocal probe_calls
        probe_calls += 1
        assert probe_supported is not None
        source = "usage" if probe_supported else "events"
        return probe_supported, make_result(
            target.name,
            kwargs["cell"],
            -1,
            0,
            output_tokens_source=source,
        )

    async def fake_request(
        client: Any,
        target: TargetConfig,
        prompt: GeneratedPrompt,
        **kwargs: Any,
    ) -> RequestResult:
        request_usage_flags.append(kwargs["include_usage"])
        return make_result(target.name, kwargs["cell"], kwargs["repeat"], kwargs["request_index"])

    monkeypatch.setattr("llmbench.runner.probe_usage", fake_probe)
    monkeypatch.setattr("llmbench.runner.stream_request", fake_request)
    runner = BenchmarkRunner(
        config(tmp_path, stream_options=mode, warmup_requests=0, requests_per_cell=1),
        progress=NullProgress(),
        environ={},
    )
    result = asyncio.run(runner.run())

    assert probe_calls == expected_probes
    assert request_usage_flags == [expected_sent, expected_sent]
    assert len(result.capabilities) == 2
    for capability in result.capabilities:
        assert capability.available is True
        assert capability.stream_options_sent is expected_sent
        assert capability.usage_stream_supported is expected_usage


class LongSystemTokenizer:
    is_tokenizer = True
    vocab_size = 100

    def count(self, text: str) -> int:
        return 100 if text else 0

    def encode_ids(self, text: str) -> list[int]:
        return list(range(len(text)))

    def decode_ids(self, token_ids: list[int]) -> str:
        return "x" * len(token_ids)

    def truncate(self, text: str, max_tokens: int) -> str:
        return text[:max_tokens]


def test_preflight_rejects_too_long_system_prompt_before_client_creation(
    tmp_path: Path, fake_http_clients: list[FakeClient]
) -> None:
    runner = BenchmarkRunner(
        config(
            tmp_path,
            stream_options="none",
            input_lengths=(32, 8),
            system_prompt="system",
        ),
        progress=NullProgress(),
        environ={},
        tokenizer=LongSystemTokenizer(),
    )

    with pytest.raises(ConfigError, match="too long"):
        asyncio.run(runner.run())

    assert fake_http_clients == []


def test_preflight_tokenizer_load_failure_happens_before_client_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_http_clients: list[FakeClient],
) -> None:
    def fail_load(spec: str | None) -> None:
        raise ConfigError(f"failed to load tokenizer {spec!r}")

    monkeypatch.setattr("llmbench.runner.load_tokenizer", fail_load)
    runner = BenchmarkRunner(
        config(tmp_path, stream_options="none", tokenizer="missing-tokenizer"),
        progress=NullProgress(),
        environ={},
    )

    with pytest.raises(ConfigError, match="tokenizer"):
        asyncio.run(runner.run())

    assert fake_http_clients == []


def test_prompts_are_shared_by_targets_and_fresh_per_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[tuple[str, int, int], GeneratedPrompt] = {}

    async def fake_probe(*args: Any, **kwargs: Any) -> tuple[bool, RequestResult]:
        raise AssertionError("probe must not run when stream_options is none")

    async def fake_request(
        client: Any,
        target: TargetConfig,
        prompt: GeneratedPrompt,
        **kwargs: Any,
    ) -> RequestResult:
        seen[(target.name, kwargs["repeat"], kwargs["request_index"])] = prompt
        return make_result(target.name, kwargs["cell"], kwargs["repeat"], kwargs["request_index"])

    monkeypatch.setattr("llmbench.runner.probe_usage", fake_probe)
    monkeypatch.setattr("llmbench.runner.stream_request", fake_request)
    runner = BenchmarkRunner(
        config(
            tmp_path,
            stream_options="none",
            repeats=2,
            warmup_requests=1,
            requests_per_cell=1,
        ),
        progress=NullProgress(),
        environ={},
    )
    asyncio.run(runner.run())

    assert seen[("a", 0, 0)] is seen[("b", 0, 0)]
    assert seen[("a", 1, 0)] is seen[("b", 1, 0)]
    assert seen[("a", 0, 0)] != seen[("a", 1, 0)]
    assert seen[("a", -1, 0)] != seen[("a", 0, 0)]


def test_warmup_all_failure_disables_only_that_target_for_remaining_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str, int]] = []

    async def fake_request(
        client: Any,
        target: TargetConfig,
        prompt: GeneratedPrompt,
        **kwargs: Any,
    ) -> RequestResult:
        cell = kwargs["cell"]
        repeat = kwargs["repeat"]
        calls.append((target.name, cell.id, repeat))
        success = not (target.name == "a" and repeat == -1)
        return make_result(
            target.name,
            cell,
            repeat,
            kwargs["request_index"],
            success=success,
        )

    monkeypatch.setattr("llmbench.runner.stream_request", fake_request)
    runner = BenchmarkRunner(
        config(
            tmp_path,
            stream_options="none",
            warmup_requests=1,
            requests_per_cell=1,
            input_lengths=(8, 16),
        ),
        progress=NullProgress(),
        environ={},
    )
    result = asyncio.run(runner.run())

    assert [item for item in calls if item[0] == "a"] == [("a", "i8-o2-c1", -1)]
    assert [item for item in calls if item[0] == "b"] == [
        ("b", "i8-o2-c1", -1),
        ("b", "i8-o2-c1", 0),
        ("b", "i16-o2-c1", -1),
        ("b", "i16-o2-c1", 0),
    ]
    capabilities = {item.name: item for item in result.capabilities}
    assert capabilities["a"].available is False
    assert capabilities["a"].error is not None
    assert capabilities["b"].available is True
    assert [(item.target, item.cell_id) for item in result.requests] == [
        ("b", "i8-o2-c1"),
        ("b", "i16-o2-c1"),
    ]


def test_probe_failure_disables_only_that_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    measured: list[str] = []

    async def fake_probe(
        client: Any,
        target: TargetConfig,
        prompt: GeneratedPrompt,
        **kwargs: Any,
    ) -> tuple[bool, RequestResult]:
        if target.name == "a":
            raise TargetProbeError("usage probe unsupported")
        return True, make_result(target.name, kwargs["cell"], -1, 0)

    async def fake_request(
        client: Any,
        target: TargetConfig,
        prompt: GeneratedPrompt,
        **kwargs: Any,
    ) -> RequestResult:
        measured.append(target.name)
        return make_result(target.name, kwargs["cell"], kwargs["repeat"], kwargs["request_index"])

    monkeypatch.setattr("llmbench.runner.probe_usage", fake_probe)
    monkeypatch.setattr("llmbench.runner.stream_request", fake_request)
    runner = BenchmarkRunner(
        config(tmp_path, warmup_requests=0, requests_per_cell=1),
        progress=NullProgress(),
        environ={},
    )
    result = asyncio.run(runner.run())

    assert measured == ["b"]
    capabilities = {item.name: item for item in result.capabilities}
    assert capabilities["a"].available is False
    assert capabilities["a"].error == "usage probe unsupported"
    assert capabilities["b"].available is True


def test_requests_use_closed_loop_concurrency_and_measure_queue_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    active = 0
    max_active = 0
    queue_waits: list[float] = []

    async def fake_request(
        client: Any,
        target: TargetConfig,
        prompt: GeneratedPrompt,
        **kwargs: Any,
    ) -> RequestResult:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        queue_waits.append(kwargs["client_queue_s"])
        await asyncio.sleep(0.01)
        active -= 1
        return make_result(
            target.name,
            kwargs["cell"],
            kwargs["repeat"],
            kwargs["request_index"],
            client_queue_s=kwargs["client_queue_s"],
        )

    monkeypatch.setattr("llmbench.runner.stream_request", fake_request)
    runner = BenchmarkRunner(
        config(
            tmp_path,
            stream_options="none",
            warmup_requests=0,
            requests_per_cell=3,
            concurrencies=(2,),
        ),
        progress=NullProgress(),
        environ={},
    )
    result = asyncio.run(runner.run())

    assert max_active == 2
    assert len(queue_waits) == 6
    assert max(queue_waits) > 0.005
    assert all(value >= 0.0 for value in queue_waits)
    assert [item.client_queue_s for item in result.requests] == queue_waits


def test_cooldown_occurs_only_after_the_full_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    async def fake_request(
        client: Any,
        target: TargetConfig,
        prompt: GeneratedPrompt,
        **kwargs: Any,
    ) -> RequestResult:
        cell = kwargs["cell"]
        repeat = kwargs["repeat"]
        events.append(f"request:{target.name}:{cell.id}:{repeat}")
        return make_result(target.name, cell, repeat, kwargs["request_index"])

    async def fake_sleep(delay: float) -> None:
        events.append(f"cooldown:{delay}")

    monkeypatch.setattr("llmbench.runner.stream_request", fake_request)
    monkeypatch.setattr("llmbench.runner.asyncio.sleep", fake_sleep)
    runner = BenchmarkRunner(
        config(
            tmp_path,
            stream_options="none",
            repeats=2,
            cooldown_s=0.25,
            warmup_requests=1,
            requests_per_cell=1,
            input_lengths=(8, 16),
        ),
        progress=NullProgress(),
        environ={},
    )
    asyncio.run(runner.run())

    assert events == [
        "request:a:i8-o2-c1:-1",
        "request:b:i8-o2-c1:-1",
        "request:a:i8-o2-c1:0",
        "request:b:i8-o2-c1:0",
        "request:a:i8-o2-c1:1",
        "request:b:i8-o2-c1:1",
        "cooldown:0.25",
        "request:a:i16-o2-c1:-1",
        "request:b:i16-o2-c1:-1",
        "request:a:i16-o2-c1:0",
        "request:b:i16-o2-c1:0",
        "request:a:i16-o2-c1:1",
        "request:b:i16-o2-c1:1",
    ]


def test_partial_result_is_available_while_run_is_still_in_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocked = asyncio.Event()
    release = asyncio.Event()

    async def fake_request(
        client: Any,
        target: TargetConfig,
        prompt: GeneratedPrompt,
        **kwargs: Any,
    ) -> RequestResult:
        if target.name == "b" and kwargs["repeat"] == 0:
            blocked.set()
            await release.wait()
        return make_result(target.name, kwargs["cell"], kwargs["repeat"], kwargs["request_index"])

    monkeypatch.setattr("llmbench.runner.stream_request", fake_request)
    runner = BenchmarkRunner(
        config(tmp_path, stream_options="none", warmup_requests=1),
        progress=NullProgress(),
        environ={},
    )

    async def scenario() -> None:
        run_task = asyncio.create_task(runner.run())
        await asyncio.wait_for(blocked.wait(), timeout=1.0)
        partial = runner.partial_result()
        assert partial.interrupted is True
        assert len(partial.requests) == 2
        assert all(item.target == "a" for item in partial.requests)
        assert len(partial.summaries) == 1
        assert partial.summaries[0].target == "a"
        release.set()
        result = await run_task
        assert len(result.requests) == 4
        assert len(result.summaries) == 2

    asyncio.run(scenario())


def test_partial_result_summarizes_completed_requests_after_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_request(
        client: Any,
        target: TargetConfig,
        prompt: GeneratedPrompt,
        **kwargs: Any,
    ) -> RequestResult:
        return make_result(target.name, kwargs["cell"], kwargs["repeat"], kwargs["request_index"])

    monkeypatch.setattr("llmbench.runner.stream_request", fake_request)
    runner = BenchmarkRunner(
        config(tmp_path, stream_options="none"),
        progress=NullProgress(),
        environ={},
    )
    result = asyncio.run(runner.run())
    partial = runner.partial_result()

    assert result.interrupted is False
    assert partial.interrupted is True
    assert len(partial.requests) == 4
    assert len(partial.summaries) == 2
    assert partial.run_id == result.run_id
