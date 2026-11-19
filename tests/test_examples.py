from pathlib import Path

import pytest

from llmprobe.config import load_config

CONFIG_DIR = Path(__file__).parents[1] / "configs"


@pytest.mark.parametrize(
    ("filename", "target_count", "requests_per_cell"),
    [
        ("vllm-chat.yaml", 1, 60),
        ("compare.yaml", 2, 40),
        ("per-target-tokenizers.yaml", 3, 16),
    ],
)
def test_example_configs_parse(
    filename: str, target_count: int, requests_per_cell: int
) -> None:
    environ = {"DEEPSEEK_API_KEY": "test-key"} if filename == "compare.yaml" else {}
    config = load_config(CONFIG_DIR / filename, environ=environ)
    assert len(config.targets) == target_count
    assert config.workload.requests_per_cell == requests_per_cell
