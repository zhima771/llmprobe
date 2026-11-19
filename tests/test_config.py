from pathlib import Path

import pytest

from llmbench.config import (
    ConfigError,
    load_config,
    resolve_headers,
    target_url,
)

MINIMAL = """
run:
  name: smoke
  output_dir: results
workload:
  requests_per_cell: 4
  concurrencies: [1, 2]
  input_lengths: [16]
  output_lengths: [8]
targets:
  - name: local
    base_url: http://127.0.0.1:8000/v1
    model: test-model
"""


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_minimal_config_applies_defaults(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, MINIMAL), environ={})
    assert config.run.name == "smoke"
    assert config.run.repeats == 1
    assert config.workload.stream_options == "auto"
    assert config.workload.dataset == "random"
    assert config.targets[0].backend == "openai-chat"
    assert config.targets[0].max_tokens_field == "max_tokens"


def test_resolve_headers_expands_env_and_sets_bearer_token(tmp_path: Path) -> None:
    text = MINIMAL.replace(
        "    model: test-model\n",
        "    model: test-model\n    api_key_env: TEST_KEY\n    headers:\n      X-Tenant: ${TENANT}\n",
    )
    config = load_config(write_config(tmp_path, text), environ={"TEST_KEY": "secret", "TENANT": "acme"})
    headers = resolve_headers(config.targets[0], environ={"TEST_KEY": "secret", "TENANT": "acme"})
    assert headers == {"Authorization": "Bearer secret", "X-Tenant": "acme"}


def test_missing_env_is_a_config_error(tmp_path: Path) -> None:
    text = MINIMAL.replace("    model: test-model\n", "    model: test-model\n    api_key_env: MISSING\n")
    with pytest.raises(ConfigError, match="MISSING"):
        load_config(write_config(tmp_path, text), environ={})


def test_reserved_request_param_is_rejected(tmp_path: Path) -> None:
    text = MINIMAL.replace("    model: test-model\n", "    model: test-model\n    request_params:\n      stream: true\n")
    with pytest.raises(ConfigError, match="reserved"):
        load_config(write_config(tmp_path, text), environ={})


def test_max_concurrency_cannot_exceed_requests_per_cell(tmp_path: Path) -> None:
    text = MINIMAL.replace("concurrencies: [1, 2]", "concurrencies: [8]")
    with pytest.raises(ConfigError, match="requests_per_cell"):
        load_config(write_config(tmp_path, text), environ={})


def test_target_url_uses_backend_specific_endpoint(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, MINIMAL), environ={})
    assert target_url(config.targets[0]) == "http://127.0.0.1:8000/v1/chat/completions"


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    text = MINIMAL + "\nunknown: true\n"
    with pytest.raises(ConfigError, match="unknown"):
        load_config(write_config(tmp_path, text), environ={})
