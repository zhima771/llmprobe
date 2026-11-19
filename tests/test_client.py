import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest

from llmbench.client import (
    TargetProbeError,
    build_payload,
    create_http_client,
    probe_usage,
    stream_request,
)
from llmbench.config import TargetConfig
from llmbench.models import Cell, GeneratedPrompt


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], delay: float = 0.0) -> None:
        self._chunks = chunks
        self._delay = delay

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            if self._delay:
                await asyncio.sleep(self._delay)
            yield chunk


def sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def target(**overrides: object) -> TargetConfig:
    values = {
        "name": "mock",
        "base_url": "http://mock/v1",
        "model": "mock-model",
        "request_params": {"temperature": 0},
    }
    values.update(overrides)
    return TargetConfig(**values)  # type: ignore[arg-type]


def prompt() -> GeneratedPrompt:
    return GeneratedPrompt(
        system="",
        user="hello",
        target_tokens=5,
        estimated_tokens=1,
        tokenizer_used=False,
    )


def cell() -> Cell:
    return Cell(id="i5-o3-c1", concurrency=1, input_length=5, output_length=3)


def test_build_payload_uses_target_token_field() -> None:
    payload = build_payload(
        target(max_tokens_field="max_completion_tokens"),
        prompt(),
        output_length=12,
        include_usage=True,
    )
    assert payload["max_completion_tokens"] == 12
    assert "max_tokens" not in payload
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["temperature"] == 0


def test_stream_request_supports_completions_backend() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["prompt"] == "hello"
        chunks = [
            sse({"choices": [{"text": "A"}]}),
            sse({"choices": [{"text": "B", "finish_reason": "stop"}]}),
        ]
        return httpx.Response(200, stream=ChunkStream(chunks))

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await stream_request(
                client,
                target(backend="openai-completions"),
                prompt(),
                tokenizer=None,
                include_usage=False,
                request_index=0,
                repeat=0,
                cell=cell(),
                client_queue_s=0.0,
                headers={},
                save_generated_text=True,
            )
        assert result.success is True
        assert result.generated_text == "AB"
        assert result.output_tokens == 2

    asyncio.run(scenario())


def test_stream_request_measures_usage_ttft_tpot_and_itl() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        chunks = [
            sse({"choices": [{"delta": {"role": "assistant"}}]}),
            sse({"choices": [{"delta": {"content": "A"}}]}),
            sse({"choices": [{"delta": {"reasoning_content": "B"}}]}),
            sse({"choices": [{"delta": {"content": "C"}, "finish_reason": "stop"}]}),
            sse({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}),
            b"data: [DONE]\n\n",
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=ChunkStream(chunks, delay=0.005),
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await stream_request(
                client,
                target(),
                prompt(),
                tokenizer=CountingTokenizer(),
                include_usage=True,
                request_index=0,
                repeat=0,
                cell=cell(),
                client_queue_s=0.0,
                headers={},
                save_generated_text=True,
            )
        assert result.success is True
        assert result.http_status == 200
        assert result.input_tokens == 5
        assert result.output_tokens == 3
        assert result.input_tokens_source == "usage"
        assert result.output_tokens_source == "usage"
        assert result.input_tokens_estimated is False
        assert result.output_tokens_estimated is False
        assert 0.003 < result.ttft_s < 0.2
        assert result.e2el_s > result.ttft_s
        assert len(result.itl_s) == 2
        assert result.tpot_s == pytest.approx((result.e2el_s - result.ttft_s) / 2)
        assert result.generated_text == "AC"
        assert result.finish_reason == "stop"

    asyncio.run(scenario())


def test_stream_request_falls_back_to_event_count_without_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        chunks = [
            sse({"choices": [{"delta": {"content": "A"}}]}),
            sse({"choices": [{"delta": {"content": "B"}}]}),
            sse({"choices": [{"delta": {"content": "C"}}]}),
        ]
        return httpx.Response(200, stream=ChunkStream(chunks))

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await stream_request(
                client,
                target(),
                prompt(),
                tokenizer=None,
                include_usage=False,
                request_index=0,
                repeat=0,
                cell=cell(),
                client_queue_s=0.0,
                headers={},
                save_generated_text=False,
            )
        assert result.success is True
        assert result.input_tokens is None
        assert result.input_tokens_source == "unavailable"
        assert result.output_tokens == 3
        assert result.output_tokens_source == "events"
        assert result.output_tokens_estimated is True

    asyncio.run(scenario())


def test_stream_request_records_http_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await stream_request(
                client,
                target(),
                prompt(),
                tokenizer=None,
                include_usage=True,
                request_index=1,
                repeat=0,
                cell=cell(),
                client_queue_s=0.25,
                headers={},
                save_generated_text=False,
            )
        assert result.success is False
        assert result.http_status == 429
        assert result.error_kind == "http"
        assert result.client_queue_s == 0.25

    asyncio.run(scenario())


def test_probe_usage_retries_without_stream_options() -> None:
    seen: list[bool] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent = "stream_options" in body
        seen.append(sent)
        if sent:
            return httpx.Response(400, text="unknown field stream_options")
        return httpx.Response(
            200,
            stream=ChunkStream([sse({"choices": [{"delta": {"content": "x"}}]})]),
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            include_usage, result = await probe_usage(
                client,
                target(),
                prompt(),
                tokenizer=None,
                headers={},
                cell=cell(),
            )
        assert include_usage is False
        assert result.success is True
        assert seen == [True, False]

    asyncio.run(scenario())


def test_probe_usage_raises_on_unrelated_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(TargetProbeError, match="401"):
                await probe_usage(
                    client,
                    target(),
                    prompt(),
                    tokenizer=None,
                    headers={},
                    cell=cell(),
                )

    asyncio.run(scenario())


class CountingTokenizer:
    is_tokenizer = True
    vocab_size = 100

    def count(self, text: str) -> int:
        return len(text)

    def encode_ids(self, text: str) -> list[int]:
        return list(range(len(text)))

    def decode_ids(self, token_ids: list[int]) -> str:
        return "x" * len(token_ids)

    def truncate(self, text: str, max_tokens: int) -> str:
        return text[:max_tokens]


def test_create_http_client_configures_connection_limits_and_tls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    client = create_http_client(
        target(http2=True, verify_tls=False, ca_bundle="/tmp/ca.pem", timeout_s=7.5),
        max_connections=9,
    )

    assert isinstance(client, FakeAsyncClient)
    assert captured["http2"] is True
    assert captured["verify"] == "/tmp/ca.pem"
    limits = captured["limits"]
    assert isinstance(limits, httpx.Limits)
    assert limits.max_connections == 9
    assert limits.max_keepalive_connections == 9
    timeout = captured["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 7.5


def test_build_payload_chat_includes_system_message_without_usage_options() -> None:
    generated = GeneratedPrompt(
        system="be concise",
        user="hello",
        target_tokens=5,
        estimated_tokens=3,
        tokenizer_used=False,
    )

    payload = build_payload(target(), generated, output_length=8, include_usage=False)

    assert payload["messages"] == [
        {"role": "system", "content": "be concise"},
        {"role": "user", "content": "hello"},
    ]
    assert "stream_options" not in payload


def test_stream_request_counts_nonempty_token_deltas_only() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        chunks = [
            sse({"choices": [{"delta": {"role": "assistant", "content": ""}}]}),
            sse({"choices": [{"delta": {"content": ""}}]}),
            sse({"choices": [{"delta": {"tool_calls": []}}]}),
            sse({"choices": [{"delta": {"reasoning_content": "think"}}]}),
            sse(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"name": "lookup", "arguments": "{}"},
                                    }
                                ]
                            }
                        }
                    ]
                }
            ),
            sse({"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]}),
        ]
        return httpx.Response(200, stream=ChunkStream(chunks))

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await stream_request(
                client,
                target(),
                prompt(),
                tokenizer=None,
                include_usage=False,
                request_index=0,
                repeat=0,
                cell=cell(),
                client_queue_s=0.0,
                headers={},
                save_generated_text=True,
            )
        assert result.success is True
        assert result.token_event_count == 3
        assert result.output_tokens == 3
        assert result.output_tokens_source == "events"
        assert len(result.itl_s) == 2
        assert result.generated_text == "answer"
        assert result.finish_reason == "stop"

    asyncio.run(scenario())


def test_stream_request_tokenizer_fallback_marks_counts_estimated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        chunks = [sse({"choices": [{"delta": {"content": "abc"}}]})]
        return httpx.Response(200, stream=ChunkStream(chunks))

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await stream_request(
                client,
                target(),
                prompt(),
                tokenizer=CountingTokenizer(),
                include_usage=False,
                request_index=0,
                repeat=0,
                cell=cell(),
                client_queue_s=0.0,
                headers={},
                save_generated_text=True,
            )
        assert result.input_tokens == 5
        assert result.input_tokens_source == "tokenizer"
        assert result.input_tokens_estimated is True
        assert result.output_tokens == 3
        assert result.output_tokens_source == "tokenizer"
        assert result.output_tokens_estimated is True

    asyncio.run(scenario())


def test_stream_request_redacts_header_secrets_from_http_errors() -> None:
    secret = "sk-client-secret"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"invalid {secret}; Authorization: Bearer {secret}")

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await stream_request(
                client,
                target(),
                prompt(),
                tokenizer=None,
                include_usage=False,
                request_index=0,
                repeat=0,
                cell=cell(),
                client_queue_s=0.0,
                headers={
                    "Authorization": f"Bearer {secret}",
                    "X-Api-Key": secret,
                },
                save_generated_text=False,
            )
        assert result.success is False
        assert result.error is not None
        assert secret not in result.error
        assert "REDACTED" in result.error

    asyncio.run(scenario())


def test_stream_request_completion_ignores_empty_text_deltas() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        chunks = [
            sse({"choices": [{"text": ""}]}),
            sse({"choices": [{"text": "A", "finish_reason": "length"}]}),
        ]
        return httpx.Response(200, stream=ChunkStream(chunks))

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await stream_request(
                client,
                target(backend="openai-completions"),
                prompt(),
                tokenizer=None,
                include_usage=False,
                request_index=0,
                repeat=0,
                cell=cell(),
                client_queue_s=0.0,
                headers={},
                save_generated_text=True,
            )
        assert result.success is True
        assert result.token_event_count == 1
        assert result.output_tokens == 1
        assert result.generated_text == "A"
        assert result.finish_reason == "length"

    asyncio.run(scenario())


def test_probe_usage_only_retries_explicit_stream_options_400() -> None:
    secret = "sk-probe-secret"
    seen: list[bool] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append("stream_options" in body)
        return httpx.Response(500, text=f"stream_options failure for Bearer {secret}")

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(TargetProbeError) as exc_info:
                await probe_usage(
                    client,
                    target(),
                    prompt(),
                    tokenizer=None,
                    headers={"Authorization": f"Bearer {secret}"},
                    cell=cell(),
                )
        message = str(exc_info.value)
        assert seen == [True]
        assert "500" in message
        assert secret not in message
        assert "REDACTED" in message

    asyncio.run(scenario())


class SlowCloseStream(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield sse({"choices": [{"delta": {"content": "A"}}]})

    async def aclose(self) -> None:
        await asyncio.sleep(0.03)


def test_stream_end_includes_response_close_time() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=SlowCloseStream())

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await stream_request(
                client,
                target(),
                prompt(),
                tokenizer=None,
                include_usage=False,
                request_index=0,
                repeat=0,
                cell=cell(),
                client_queue_s=0.0,
                headers={},
                save_generated_text=True,
            )
        assert result.success is True
        assert result.e2el_s is not None
        assert result.stream_end_s >= result.e2el_s + 0.02

    asyncio.run(scenario())
