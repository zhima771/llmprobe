from pathlib import Path

from llmprobe.config import WorkloadConfig
from llmprobe.models import make_run_id
from llmprobe.workload import build_cells, cell_seed


def test_build_cells_preserves_config_order() -> None:
    workload = WorkloadConfig(
        concurrency=(8, 1),
        input_lengths=(128, 16),
        output_lengths=(32, 64),
    )
    cells = build_cells(workload)
    assert [cell.id for cell in cells] == [
        "in128-out32-conc8",
        "in128-out32-conc1",
        "in128-out64-conc8",
        "in128-out64-conc1",
        "in16-out32-conc8",
        "in16-out32-conc1",
        "in16-out64-conc8",
        "in16-out64-conc1",
    ]


def test_cell_seed_is_stable_and_repeat_sensitive() -> None:
    workload = WorkloadConfig(concurrency=(1,), input_lengths=(10,), output_lengths=(5,))
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
