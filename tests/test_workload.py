from pathlib import Path

from llmbench.config import WorkloadConfig
from llmbench.models import make_run_id
from llmbench.workload import build_cells, cell_seed


def test_build_cells_preserves_config_order() -> None:
    workload = WorkloadConfig(
        concurrencies=(8, 1),
        input_lengths=(128, 16),
        output_lengths=(32, 64),
    )
    cells = build_cells(workload)
    assert [cell.id for cell in cells] == [
        "i128-o32-c8",
        "i128-o32-c1",
        "i128-o64-c8",
        "i128-o64-c1",
        "i16-o32-c8",
        "i16-o32-c1",
        "i16-o64-c8",
        "i16-o64-c1",
    ]


def test_cell_seed_is_stable_and_repeat_sensitive() -> None:
    workload = WorkloadConfig(concurrencies=(1,), input_lengths=(10,), output_lengths=(5,))
    cell = build_cells(workload)[0]
    assert cell_seed(42, cell, 0) == cell_seed(42, cell, 0)
    assert cell_seed(42, cell, 0) != cell_seed(42, cell, 1)
    assert cell_seed(42, cell, 0) != cell_seed(43, cell, 0)


def test_make_run_id_uses_a_timestamp_and_slug() -> None:
    run_id = make_run_id(Path("ignored"), " Local Smoke / Test ")
    stamp, slug = run_id.split("-", 1)
    assert stamp.endswith("Z")
    assert len(stamp) == 22
    assert slug == "local-smoke-test"
