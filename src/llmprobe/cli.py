from __future__ import annotations

import argparse
import asyncio
from typing import Self

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from llmprobe import __version__
from llmprobe.config import AppConfig, ConfigError, load_config, sanitize_url
from llmprobe.models import Cell, CellSummary, RequestResult
from llmprobe.report import print_summary, write_report
from llmprobe.runner import BenchmarkRunner, preflight_workload
from llmprobe.workload import build_cells


class RichProgressSink:
    def __init__(self, console: Console, total_requests: int) -> None:
        self.console = console
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
            transient=True,
        )
        self.task_id = self.progress.add_task("starting", total=total_requests)

    def __enter__(self) -> Self:
        self.progress.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.progress.stop()

    def cell_started(self, target: str, cell: Cell, repeat: int) -> None:
        self.progress.update(
            self.task_id,
            description=f"{target} {cell.id}",
        )

    def request_finished(self, result: RequestResult) -> None:
        self.progress.advance(self.task_id)

    def cell_finished(self, summary: CellSummary) -> None:
        self.console.print(
            f"[cyan]{summary.target}[/cyan] {summary.cell_id}: "
            f"{summary.completed} ok / {summary.failed} failed, "
            f"{summary.request_throughput:.2f} req/s, "
            f"{summary.output_token_throughput or 0:.2f} out tok/s"
        )

    def warning(self, message: str) -> None:
        self.console.print(f"[yellow]warning:[/yellow] {message}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmprobe", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="execute a benchmark config")
    run_parser.add_argument("config", help="path to YAML config")
    run_parser.add_argument("--dry-run", action="store_true", help="show matrix and exit")

    validate_parser = subparsers.add_parser("validate", help="validate a benchmark config")
    validate_parser.add_argument("config", help="path to YAML config")
    return parser


def _formal_total(config: AppConfig) -> int:
    cells = build_cells(config.workload)
    return (
        len(cells)
        * len(config.targets)
        * config.workload.requests_per_cell
    )


def _print_plan(config: AppConfig, console: Console) -> None:
    cells = build_cells(config.workload)
    warmups = len(cells) * len(config.targets) * config.workload.warmup_requests
    console.print(f"targets: {', '.join(target.name for target in config.targets)}")
    console.print(f"cells: {len(cells)}")
    console.print(f"formal requests: {_formal_total(config)}")
    console.print(f"warmup requests: {warmups}")
    for target in config.targets:
        tokenizer = target.tokenizer or config.workload.tokenizer or "none"
        console.print(
            f"  - {target.name}: {target.backend} {sanitize_url(target.base_url)} "
            f"model={target.model} tokenizer={tokenizer}"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    console = Console()
    error_console = Console(stderr=True)
    try:
        config = load_config(args.config)
        if args.command == "validate" or args.dry_run:
            preflight_workload(config)
    except ConfigError as exc:
        error_console.print(f"[red]configuration error:[/red] {exc}")
        return 2

    if args.command == "validate":
        console.print(
            f"configuration valid: {len(config.targets)} targets, "
            f"{_formal_total(config)} formal requests"
        )
        return 0

    if args.dry_run:
        _print_plan(config, console)
        return 0

    total = _formal_total(config)
    runner: BenchmarkRunner | None = None
    try:
        with RichProgressSink(console, total) as progress:
            runner = BenchmarkRunner(config, progress=progress)
            try:
                result = asyncio.run(runner.run())
            except KeyboardInterrupt:
                try:
                    result = runner.partial_result()
                    output = write_report(result, config)
                except Exception as exc:  # noqa: BLE001 - preserve interrupt exit code
                    error_console.print(
                        f"[red]partial report failed:[/red] {type(exc).__name__}: {exc}"
                    )
                else:
                    console.print(
                        f"[yellow]interrupted; partial results:[/yellow] {output}"
                    )
                return 130
    except ConfigError as exc:
        error_console.print(f"[red]configuration error:[/red] {exc}")
        if runner is not None and getattr(runner, "results", None):
            try:
                partial = runner.partial_result()
                output = write_report(partial, config)
            except Exception as report_exc:  # noqa: BLE001 - preserve config exit code
                error_console.print(
                    "[red]partial report failed:[/red] "
                    f"{type(report_exc).__name__}: {report_exc}"
                )
            else:
                console.print(f"[yellow]partial results:[/yellow] {output}")
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI must report failures cleanly
        error_console.print(f"[red]benchmark failed:[/red] {type(exc).__name__}: {exc}")
        return 1

    try:
        output = write_report(result, config)
        print_summary(result, console)
    except Exception as exc:  # noqa: BLE001 - reporting failures must be controlled
        error_console.print(f"[red]report failed:[/red] {type(exc).__name__}: {exc}")
        return 1
    console.print(f"results: {output}")
    return 0 if any(item.success for item in result.requests) else 1
