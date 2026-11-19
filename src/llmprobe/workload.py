from __future__ import annotations

import hashlib

from llmprobe.config import WorkloadConfig
from llmprobe.models import Cell


def build_cells(workload: WorkloadConfig) -> list[Cell]:
    return [
        Cell(
            id=f"in{spec.input_length}-out{spec.output_length}-conc{spec.concurrency}",
            concurrency=spec.concurrency,
            input_length=spec.input_length,
            output_length=spec.output_length,
        )
        for spec in workload.cells
    ]


def cell_seed(global_seed: int, cell: Cell, repeat: int) -> int:
    payload = f"{global_seed}|{cell.id}|{repeat}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
