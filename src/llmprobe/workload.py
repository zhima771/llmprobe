from __future__ import annotations

import hashlib

from llmprobe.config import WorkloadConfig
from llmprobe.models import Cell


def build_cells(workload: WorkloadConfig) -> list[Cell]:
    cells: list[Cell] = []
    for input_length in workload.input_lengths:
        for output_length in workload.output_lengths:
            for concurrency in workload.concurrency:
                cells.append(
                    Cell(
                        id=f"in{input_length}-out{output_length}-conc{concurrency}",
                        concurrency=concurrency,
                        input_length=input_length,
                        output_length=output_length,
                    )
                )
    return cells


def cell_seed(global_seed: int, cell: Cell, repeat: int) -> int:
    payload = f"{global_seed}|{cell.id}|{repeat}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
