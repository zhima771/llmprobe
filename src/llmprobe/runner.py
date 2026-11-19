from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, Protocol

from llmprobe.client import (
    TargetProbeError,
    create_http_client,
    probe_usage,
    stream_request,
)
from llmprobe.config import AppConfig, TargetConfig, resolve_headers
from llmprobe.metrics import summarize_results
from llmprobe.models import (
    Cell,
    CellSummary,
    GeneratedPrompt,
    RequestResult,
    RunResult,
    TargetCapability,
    make_run_id,
)
from llmprobe.promptgen import generate_prompt
from llmprobe.tokenization import TextTokenizer, canonical_tokenizer_key, load_tokenizer
from llmprobe.workload import build_cells


class ProgressSink(Protocol):
    def cell_started(self, target: str, cell: Cell, repeat: int) -> None: ...

    def request_finished(self, result: RequestResult) -> None: ...

    def cell_finished(self, summary: CellSummary) -> None: ...

    def warning(self, message: str) -> None: ...


class NullProgress:
    def cell_started(self, target: str, cell: Cell, repeat: int) -> None:
        return None

    def request_finished(self, result: RequestResult) -> None:
        return None

    def cell_finished(self, summary: CellSummary) -> None:
        return None

    def warning(self, message: str) -> None:
        return None


class TargetUnavailable(RuntimeError):
    """Raised internally when a target cannot continue the run."""


def _effective_tokenizer_spec(config: AppConfig, target: TargetConfig) -> str | None:
    return target.tokenizer if target.tokenizer is not None else config.workload.tokenizer


def _tokenizer_cache_key(
    config: AppConfig, target: TargetConfig, injected: TextTokenizer | None
) -> str | None:
    if target.tokenizer is None and injected is not None:
        return "__injected__"
    spec = _effective_tokenizer_spec(config, target)
    return canonical_tokenizer_key(spec)


def preflight_workload(
    config: AppConfig, tokenizer: TextTokenizer | None = None
) -> dict[str, TextTokenizer | None]:
    cells = build_cells(config.workload)
    minimum_cell = min(cells, key=lambda item: item.input_length)
    tokenizers: dict[str, TextTokenizer | None] = {}
    cache: dict[str | None, TextTokenizer | None] = {}
    validated: set[str | None] = set()
    for target in config.targets:
        cache_key = _tokenizer_cache_key(config, target, tokenizer)
        if cache_key == "__injected__" and tokenizer is not None:
            active_tokenizer: TextTokenizer | None = tokenizer
        else:
            if cache_key not in cache:
                cache[cache_key] = load_tokenizer(_effective_tokenizer_spec(config, target))
            active_tokenizer = cache[cache_key]
        tokenizers[target.name] = active_tokenizer
        if cache_key not in validated:
            generate_prompt(
                config.run.seed,
                minimum_cell,
                -1,
                0,
                config.workload,
                active_tokenizer,
            )
            validated.add(cache_key)
    return tokenizers


class BenchmarkRunner:
    def __init__(
        self,
        config: AppConfig,
        progress: ProgressSink | None = None,
        environ: Mapping[str, str] | None = None,
        tokenizer: TextTokenizer | None = None,
    ) -> None:
        self.config = config
        self.progress = progress if progress is not None else NullProgress()
        self.environ = environ
        self._tokenizer = tokenizer
        self.run_id = make_run_id(config.run.output_dir, config.run.name)
        self.results: list[RequestResult] = []
        self.summaries: list[CellSummary] = []
        self.capabilities: dict[str, TargetCapability] = {}
        self.started_at: str | None = None
        self.ended_at: str | None = None

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()

    async def run(self) -> RunResult:
        self.started_at = self._now()
        self.ended_at = None
        target_tokenizers = preflight_workload(self.config, self._tokenizer)
        cells = build_cells(self.config.workload)
        clients: dict[str, Any] = {}
        headers: dict[str, dict[str, str]] = {}
        include_usage: dict[str, bool] = {}
        max_concurrency = max(cell.concurrency for cell in cells)

        try:
            for target in self.config.targets:
                headers[target.name] = resolve_headers(target, self.environ)
                clients[target.name] = create_http_client(target, max_concurrency)

            for target in self.config.targets:
                available, stream_options_sent, usage_supported, error = (
                    await self._probe_target(
                        target,
                        clients[target.name],
                        headers[target.name],
                        target_tokenizers[target.name],
                        cells,
                    )
                )
                self.capabilities[target.name] = TargetCapability(
                    name=target.name,
                    available=available,
                    stream_options_sent=stream_options_sent,
                    usage_stream_supported=usage_supported,
                    error=error,
                )
                include_usage[target.name] = stream_options_sent
                if not available:
                    self.progress.warning(f"target {target.name} disabled after probe: {error}")

            for cell_index, cell in enumerate(cells):
                for target in self.config.targets:
                    capability = self.capabilities[target.name]
                    if not capability.available:
                        continue
                    try:
                        await self._warmup_target(
                            target,
                            clients[target.name],
                            headers[target.name],
                            cell,
                            include_usage[target.name],
                            target_tokenizers[target.name],
                        )
                    except TargetUnavailable as exc:
                        self.capabilities[target.name] = replace(
                            capability,
                            available=False,
                            error=str(exc),
                        )
                        self.progress.warning(f"target {target.name} disabled: {exc}")

                prompt_cache: dict[str | None, list[GeneratedPrompt]] = {}
                for target in self.config.targets:
                    capability = self.capabilities[target.name]
                    if not capability.available:
                        continue
                    active_tokenizer = target_tokenizers[target.name]
                    cache_key = _tokenizer_cache_key(
                        self.config, target, self._tokenizer
                    )
                    if cache_key not in prompt_cache:
                        prompt_cache[cache_key] = [
                            generate_prompt(
                                self.config.run.seed,
                                cell,
                                0,
                                index,
                                self.config.workload,
                                active_tokenizer,
                            )
                            for index in range(self.config.workload.requests_per_cell)
                        ]
                    self.progress.cell_started(target.name, cell, 0)
                    results = await self._execute_requests(
                        target=target,
                        client=clients[target.name],
                        headers=headers[target.name],
                        cell=cell,
                        prompts=prompt_cache[cache_key],
                        include_usage=include_usage[target.name],
                        tokenizer=active_tokenizer,
                        repeat=0,
                        record=True,
                    )
                    summary = summarize_results(results, cell, target.name, 0)
                    self.summaries.append(summary)
                    self.progress.cell_finished(summary)

                if cell_index < len(cells) - 1 and self.config.run.cooldown_s > 0:
                    await asyncio.sleep(self.config.run.cooldown_s)
        finally:
            for client in clients.values():
                await client.aclose()
            self.ended_at = self._now()

        return self._to_result(interrupted=False)

    async def _probe_target(
        self,
        target: TargetConfig,
        client: Any,
        headers: Mapping[str, str],
        tokenizer: TextTokenizer | None,
        cells: list[Cell],
    ) -> tuple[bool, bool, bool | None, str | None]:
        mode = self.config.workload.stream_options
        if mode == "none":
            return True, False, None, None
        if mode == "include_usage":
            return True, True, None, None

        probe_cell = Cell(
            id="probe",
            concurrency=1,
            input_length=min((cell.input_length for cell in cells), default=1),
            output_length=min((cell.output_length for cell in cells), default=1),
        )
        probe_prompt = generate_prompt(
            self.config.run.seed,
            probe_cell,
            -1,
            0,
            self.config.workload,
            tokenizer,
        )
        try:
            include_usage, result = await probe_usage(
                client,
                target,
                probe_prompt,
                tokenizer=tokenizer,
                headers=headers,
                cell=probe_cell,
            )
        except TargetProbeError as exc:
            return False, True, False, str(exc)
        return True, include_usage, result.output_tokens_source == "usage", None

    async def _warmup_target(
        self,
        target: TargetConfig,
        client: Any,
        headers: Mapping[str, str],
        cell: Cell,
        include_usage: bool,
        tokenizer: TextTokenizer | None,
    ) -> None:
        if self.config.workload.warmup_requests == 0:
            return
        prompts = [
            generate_prompt(
                self.config.run.seed,
                cell,
                -1,
                index,
                self.config.workload,
                tokenizer,
            )
            for index in range(self.config.workload.warmup_requests)
        ]
        results = await self._execute_requests(
            target=target,
            client=client,
            headers=headers,
            cell=cell,
            prompts=prompts,
            include_usage=include_usage,
            tokenizer=tokenizer,
            repeat=-1,
            record=False,
        )
        failures = sum(not result.success for result in results)
        if failures == len(results):
            raise TargetUnavailable(
                f"all {failures} warmup requests failed for cell {cell.id}"
            )
        if failures:
            self.progress.warning(
                f"target {target.name} had {failures}/{len(results)} warmup failures "
                f"for cell {cell.id}"
            )

    async def _execute_requests(
        self,
        *,
        target: TargetConfig,
        client: Any,
        headers: Mapping[str, str],
        cell: Cell,
        prompts: list[GeneratedPrompt],
        include_usage: bool,
        tokenizer: TextTokenizer | None,
        repeat: int,
        record: bool,
    ) -> list[RequestResult]:
        semaphore = asyncio.Semaphore(cell.concurrency)
        collected: list[tuple[int, RequestResult]] = []

        async def one(index: int, prompt: GeneratedPrompt) -> RequestResult:
            scheduled_at = time.perf_counter()
            async with semaphore:
                queue_s = time.perf_counter() - scheduled_at
                result = await stream_request(
                    client,
                    target,
                    prompt,
                    tokenizer=tokenizer,
                    include_usage=include_usage,
                    request_index=index,
                    repeat=repeat,
                    cell=cell,
                    client_queue_s=queue_s,
                    headers=headers,
                    save_generated_text=self.config.run.save_generated_text,
                )
                collected.append((index, result))
                if record:
                    self.results.append(result)
                    self.progress.request_finished(result)
            return result

        await asyncio.gather(*(one(index, prompt) for index, prompt in enumerate(prompts)))
        return [result for _, result in sorted(collected, key=lambda item: item[0])]

    def _to_result(self, *, interrupted: bool) -> RunResult:
        ended_at = self.ended_at or self._now()
        return RunResult(
            run_id=self.run_id,
            started_at=self.started_at or ended_at,
            ended_at=ended_at,
            summaries=list(self.summaries),
            requests=list(self.results),
            capabilities=list(self.capabilities.values()),
            interrupted=interrupted,
        )

    def partial_result(self) -> RunResult:
        cells = {cell.id: cell for cell in build_cells(self.config.workload)}
        grouped: dict[tuple[str, str], list[RequestResult]] = defaultdict(list)
        for result in self.results:
            grouped[(result.target, result.cell_id)].append(result)
        summaries = [
            summarize_results(results, cells[cell_id], target, 0)
            for (target, cell_id), results in sorted(
                grouped.items(), key=lambda item: item[0]
            )
            if cell_id in cells
        ]
        return RunResult(
            run_id=self.run_id,
            started_at=self.started_at or self._now(),
            ended_at=self._now(),
            summaries=summaries,
            requests=list(self.results),
            capabilities=list(self.capabilities.values()),
            interrupted=True,
        )
