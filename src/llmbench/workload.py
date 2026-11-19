from __future__ import annotations

import hashlib

from llmbench.config import WorkloadConfig
from llmbench.models import Cell


def build_cells(workload: WorkloadConfig) -> list[Cell]:
    cells: list[Cell] = []
    for input_length in workload.input_lengths:
        for output_length in workload.output_lengths:
            for concurrency in workload.concurrencies:
                cells.append(
                    Cell(
                        id=f"i{input_length}-o{output_length}-c{concurrency}",
                        concurrency=concurrency,
                        input_length=input_length,
                        output_length=output_length,
                    )
                )
    return cells


def cell_seed(global_seed: int, cell: Cell, repeat: int) -> int:
    payload = f"{global_seed}|{cell.id}|{repeat}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
