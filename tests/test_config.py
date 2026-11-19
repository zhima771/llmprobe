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


def test_yaml_parser_error_does_not_leak_source_or_chain_exception(tmp_path: Path) -> None:
    secret = "sk-super-secret"
    text = MINIMAL.replace("model: test-model", f"model: [{secret}")
    with pytest.raises(ConfigError) as exc_info:
        load_config(write_config(tmp_path, text), environ={})
    message = str(exc_info.value)
    assert secret not in message
    assert "line " in message
    assert "column " in message
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__ is True


def test_non_string_mapping_key_is_a_config_error(tmp_path: Path) -> None:
    text = MINIMAL + "\n1: true\n"
    with pytest.raises(ConfigError, match="keys must be strings"):
        load_config(write_config(tmp_path, text), environ={})


@pytest.mark.parametrize("value", [".nan", ".inf", "-.inf"])
def test_non_finite_number_is_rejected(tmp_path: Path, value: str) -> None:
    text = MINIMAL.replace(
        "    model: test-model\n",
        f"    model: test-model\n    timeout_s: {value}\n",
    )
    with pytest.raises(ConfigError, match="finite"):
        load_config(write_config(tmp_path, text), environ={})


def test_authorization_header_conflict_is_case_insensitive(tmp_path: Path) -> None:
    text = MINIMAL.replace(
        "    model: test-model\n",
        "    model: test-model\n"
        "    api_key_env: TEST_KEY\n"
        "    headers:\n"
        "      authorization: Bearer literal\n",
    )
    with pytest.raises(ConfigError, match="Authorization"):
        load_config(write_config(tmp_path, text), environ={"TEST_KEY": "secret"})


def test_undecodable_config_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(ConfigError, match="UTF-8"):
        load_config(path, environ={})


def test_missing_header_env_is_a_config_error(tmp_path: Path) -> None:
    text = MINIMAL.replace(
        "    model: test-model\n",
        "    model: test-model\n    headers:\n      X-Tenant: ${MISSING_TENANT}\n",
    )
    with pytest.raises(ConfigError, match="MISSING_TENANT"):
        load_config(write_config(tmp_path, text), environ={})


def test_duplicate_target_names_are_rejected(tmp_path: Path) -> None:
    text = MINIMAL + """
  - name: local
    base_url: http://127.0.0.1:8001/v1
    model: another-model
"""
    with pytest.raises(ConfigError, match="unique"):
        load_config(write_config(tmp_path, text), environ={})


def test_completions_target_url_uses_backend_specific_endpoint(tmp_path: Path) -> None:
    text = MINIMAL.replace(
        "    model: test-model\n",
        "    model: test-model\n    backend: openai-completions\n",
    )
    config = load_config(write_config(tmp_path, text), environ={})
    assert target_url(config.targets[0]) == "http://127.0.0.1:8000/v1/completions"
