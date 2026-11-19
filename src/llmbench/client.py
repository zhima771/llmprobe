from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from typing import Any

import httpx

from llmbench.config import TargetConfig, target_url
from llmbench.models import Cell, GeneratedPrompt, RequestResult
from llmbench.sse import SSEParser
from llmbench.tokenization import TextTokenizer


class TargetProbeError(RuntimeError):
    """Raised when a target cannot execute a stream-options capability probe."""


_MANAGED_PAYLOAD_KEYS = frozenset(
    {
        "model",
        "messages",
        "prompt",
        "stream",
        "stream_options",
        "max_tokens",
        "max_completion_tokens",
    }
)


def _redact_secrets(text: str, headers: Mapping[str, str]) -> str:
    secrets: set[str] = set()
    for value in headers.values():
        if not value:
            continue
        secrets.add(value)
        scheme, separator, credential = value.partition(" ")
        if separator and credential and scheme.lower() in {"bearer", "basic"}:
            secrets.add(credential)
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    return text


def create_http_client(target: TargetConfig, max_connections: int) -> httpx.AsyncClient:
    verify: bool | str = target.ca_bundle if target.ca_bundle else target.verify_tls
    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_connections,
    )
    return httpx.AsyncClient(
        http2=target.http2,
        verify=verify,
        limits=limits,
        timeout=httpx.Timeout(target.timeout_s),
    )


def build_payload(
    target: TargetConfig,
    prompt: GeneratedPrompt,
    output_length: int,
    include_usage: bool,
) -> dict[str, Any]:
    if target.backend == "openai-chat":
        messages: list[dict[str, str]] = []
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.user})
        payload: dict[str, Any] = {
            "model": target.model,
            "messages": messages,
            "stream": True,
            target.max_tokens_field: output_length,
        }
    else:
        payload = {
            "model": target.model,
            "prompt": prompt.completions_text,
            "stream": True,
            target.max_tokens_field: output_length,
        }
    for key, value in target.request_params.items():
        if key not in _MANAGED_PAYLOAD_KEYS:
            payload[key] = value
    payload["stream"] = True
    if include_usage:
        payload["stream_options"] = {"include_usage": True}
    else:
        payload.pop("stream_options", None)
    return payload


def _extract_usage(data: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None, None
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    prompt_value = prompt_tokens if type(prompt_tokens) is int and prompt_tokens >= 0 else None
    completion_value = (
        completion_tokens
        if type(completion_tokens) is int and completion_tokens >= 0
        else None
    )
    return prompt_value, completion_value


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _tool_call_has_content(call: Any) -> bool:
    if not isinstance(call, dict):
        return False
    function = call.get("function")
    if not isinstance(function, dict):
        return False
    return _nonempty_string(function.get("name")) or _nonempty_string(
        function.get("arguments")
    )


def _function_call_has_content(function_call: Any) -> bool:
    if not isinstance(function_call, dict):
        return False
    return _nonempty_string(function_call.get("name")) or _nonempty_string(
        function_call.get("arguments")
    )


def _extract_chat_token(data: dict[str, Any]) -> tuple[bool, str, str, str | None]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return False, "", "", None
    choice = choices[0]
    if not isinstance(choice, dict):
        return False, "", "", None
    finish_reason = choice.get("finish_reason")
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        return False, "", "", finish_reason
    counted_parts: list[str] = []
    displayed_parts: list[str] = []
    for key in ("content", "reasoning", "reasoning_content"):
        value = delta.get(key)
        if isinstance(value, str) and value:
            counted_parts.append(value)
            if key == "content":
                displayed_parts.append(value)
    tool_calls = delta.get("tool_calls")
    if isinstance(tool_calls, list):
        meaningful_calls = [call for call in tool_calls if _tool_call_has_content(call)]
        if meaningful_calls:
            counted_parts.append(
                json.dumps(meaningful_calls, ensure_ascii=False, sort_keys=True)
            )
    function_call = delta.get("function_call")
    if _function_call_has_content(function_call):
        counted_parts.append(json.dumps(function_call, ensure_ascii=False, sort_keys=True))
    if not counted_parts:
        return False, "", "", finish_reason
    return True, "".join(counted_parts), "".join(displayed_parts), finish_reason


def _extract_completion_token(data: dict[str, Any]) -> tuple[bool, str, str, str | None]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return False, "", "", None
    choice = choices[0]
    if not isinstance(choice, dict):
        return False, "", "", None
    finish_reason = choice.get("finish_reason")
    if "text" not in choice:
        return False, "", "", finish_reason
    value = choice.get("text")
    if not isinstance(value, str) or not value:
        return False, "", "", finish_reason
    return True, value, value, finish_reason


def _extract_token(
    backend: str, data: dict[str, Any]
) -> tuple[bool, str, str, str | None]:
    if backend == "openai-chat":
        return _extract_chat_token(data)
    return _extract_completion_token(data)


async def stream_request(
    client: httpx.AsyncClient,
    target: TargetConfig,
    prompt: GeneratedPrompt,
    *,
    tokenizer: TextTokenizer | None,
    include_usage: bool,
    request_index: int,
    repeat: int,
    cell: Cell,
    client_queue_s: float,
    headers: Mapping[str, str],
    save_generated_text: bool,
) -> RequestResult:
    started_at = time.perf_counter()
    first_token_at: float | None = None
    last_token_at: float | None = None
    itl_s: list[float] = []
    token_events = 0
    counted_parts: list[str] = []
    visible_parts: list[str] = []
    finish_reason: str | None = None
    http_status: int | None = None
    usage_input: int | None = None
    usage_output: int | None = None
    error: str | None = None
    error_kind: str | None = None
    parser = SSEParser()

    def observe_payload(data: dict[str, Any]) -> None:
        nonlocal first_token_at, last_token_at, token_events, finish_reason
        nonlocal usage_input, usage_output
        if data.get("error"):
            message = json.dumps(data["error"], ensure_ascii=False)
            raise ValueError(_redact_secrets(message, headers)[:2000])
        prompt_tokens, completion_tokens = _extract_usage(data)
        if prompt_tokens is not None:
            usage_input = prompt_tokens
        if completion_tokens is not None:
            usage_output = completion_tokens
            counted_parts.clear()
        token_bearing, counted_text, displayed_text, reason = _extract_token(
            target.backend, data
        )
        if reason is not None:
            finish_reason = reason
        if not token_bearing:
            return
        now = time.perf_counter()
        if first_token_at is None:
            first_token_at = now
        else:
            itl_s.append(now - (last_token_at if last_token_at is not None else now))
        last_token_at = now
        token_events += 1
        if tokenizer is not None and usage_output is None:
            counted_parts.append(counted_text)
        if save_generated_text:
            visible_parts.append(displayed_text)

    try:
        async with asyncio.timeout(target.timeout_s):
            async with client.stream(
                "POST",
                target_url(target),
                json=build_payload(target, prompt, cell.output_length, include_usage),
                headers=dict(headers),
            ) as response:
                http_status = response.status_code
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    error = f"HTTP {response.status_code}: {_redact_secrets(body, headers)[:2000]}"
                    error_kind = "http"
                else:
                    done = False
                    async for chunk in response.aiter_bytes():
                        for data_text in parser.feed(chunk):
                            if data_text == "[DONE]":
                                done = True
                                break
                            try:
                                payload = json.loads(data_text)
                                if not isinstance(payload, dict):
                                    raise ValueError(  # noqa: TRY004 - stream decoding failure
                                        "SSE data is not a JSON object"
                                    )
                                observe_payload(payload)
                            except (json.JSONDecodeError, ValueError) as exc:
                                error = f"invalid stream event: {exc}"
                                error_kind = "stream"
                                done = True
                                break
                        if done:
                            break
                    if error is None:
                        for data_text in parser.close():
                            if data_text == "[DONE]":
                                continue
                            payload = json.loads(data_text)
                            if not isinstance(payload, dict):
                                raise ValueError(  # noqa: TRY004 - stream decoding failure
                                    "SSE data is not a JSON object"
                                )
                            observe_payload(payload)
    except TimeoutError:
        error = f"request exceeded timeout of {target.timeout_s:g}s"
        error_kind = "timeout"
    except httpx.TimeoutException as exc:
        error = f"HTTP timeout: {exc}"
        error_kind = "timeout"
    except httpx.HTTPError as exc:
        error = f"HTTP error: {type(exc).__name__}: {exc}"
        error_kind = "connection"
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        error = f"stream error: {exc}"
        error_kind = "stream"
    except Exception as exc:  # noqa: BLE001 - benchmark failures must be recorded, not crash the run
        error = f"request error: {type(exc).__name__}: {exc}"
        error_kind = "request"

    stream_end_s = time.perf_counter() - started_at
    if error is not None:
        error = _redact_secrets(error, headers)
    if http_status == 200 and token_events == 0 and error is None:
        error = "stream completed without a token-bearing delta"
        error_kind = "no_tokens"

    success = http_status == 200 and token_events > 0 and error is None
    counted_text = "".join(counted_parts)
    visible_text = "".join(visible_parts)
    if usage_input is not None:
        input_tokens: int | None = usage_input
        input_source = "usage"
        input_estimated = False
    elif tokenizer is not None:
        input_tokens = tokenizer.count(prompt.system) + tokenizer.count(prompt.user)
        input_source = "tokenizer"
        input_estimated = True
    else:
        input_tokens = None
        input_source = "unavailable"
        input_estimated = False

    if usage_output is not None:
        output_tokens = usage_output
        output_source = "usage"
        output_estimated = False
    elif tokenizer is not None and counted_text:
        output_tokens = tokenizer.count(counted_text)
        output_source = "tokenizer"
        output_estimated = True
    else:
        output_tokens = token_events
        output_source = "events" if token_events else "unavailable"
        output_estimated = True

    ttft_s = first_token_at - started_at if first_token_at is not None else None
    e2el_s = last_token_at - started_at if last_token_at is not None else None
    tpot_s: float | None = None
    if e2el_s is not None and ttft_s is not None and output_tokens > 1:
        tpot_s = (e2el_s - ttft_s) / (output_tokens - 1)

    return RequestResult(
        target=target.name,
        cell_id=cell.id,
        repeat=repeat,
        request_index=request_index,
        concurrency=cell.concurrency,
        input_length_target=cell.input_length,
        output_length_target=cell.output_length,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_tokens_source=input_source,
        output_tokens_source=output_source,
        input_tokens_estimated=input_estimated,
        output_tokens_estimated=output_estimated,
        client_queue_s=client_queue_s,
        started_at=started_at,
        ttft_s=ttft_s,
        e2el_s=e2el_s,
        stream_end_s=stream_end_s,
        tpot_s=tpot_s,
        itl_s=itl_s,
        token_event_count=token_events,
        finish_reason=finish_reason,
        success=success,
        error_kind=error_kind,
        error=error,
        http_status=http_status,
        generated_text=visible_text if save_generated_text and visible_text else None,
    )


async def probe_usage(
    client: httpx.AsyncClient,
    target: TargetConfig,
    prompt: GeneratedPrompt,
    *,
    tokenizer: TextTokenizer | None,
    headers: Mapping[str, str],
    cell: Cell,
) -> tuple[bool, RequestResult]:
    result = await stream_request(
        client,
        target,
        prompt,
        tokenizer=tokenizer,
        include_usage=True,
        request_index=0,
        repeat=-1,
        cell=cell,
        client_queue_s=0.0,
        headers=headers,
        save_generated_text=False,
    )
    if result.success:
        return True, result
    error_text = (result.error or "").lower()
    if result.http_status == 400 and "stream_options" in error_text:
        retry = await stream_request(
            client,
            target,
            prompt,
            tokenizer=tokenizer,
            include_usage=False,
            request_index=0,
            repeat=-1,
            cell=cell,
            client_queue_s=0.0,
            headers=headers,
            save_generated_text=False,
        )
        if retry.success:
            return False, retry
        raise TargetProbeError(
            f"target {target.name} failed after disabling stream_options: {retry.error}"
        )
    raise TargetProbeError(
        f"target {target.name} capability probe failed: {result.error or result.http_status}"
    )
