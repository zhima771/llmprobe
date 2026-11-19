from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import pytest

from llmprobe.config import (
    ConfigError,
    load_config,
    resolve_headers,
    resolve_metrics_headers,
    sanitize_url,
    target_url,
)

MINIMAL = """
run:
  name: smoke
  output_dir: results
workload:
  requests_per_cell: 4
  cells:
    - 16-8-1
    - 16-8-2
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
    text = MINIMAL.replace("    - 16-8-2", "    - 16-8-8")
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


@pytest.mark.parametrize(
    "huge_integer",
    [pytest.param(10**1000, id="positive"), pytest.param(-(10**1000), id="negative")],
)
def test_huge_integer_is_a_config_error(tmp_path: Path, huge_integer: int) -> None:
    text = MINIMAL.replace(
        "    model: test-model\n",
        f"    model: test-model\n    timeout_s: {huge_integer}\n",
    )
    with pytest.raises(ConfigError, match="finite"):
        load_config(write_config(tmp_path, text), environ={})


def test_target_url_preserves_query_when_appending_endpoint(tmp_path: Path) -> None:
    text = MINIMAL.replace(
        "base_url: http://127.0.0.1:8000/v1",
        "base_url: http://127.0.0.1:8000/v1?api-version=2024-01-01",
    )
    config = load_config(write_config(tmp_path, text), environ={})
    assert (
        target_url(config.targets[0])
        == "http://127.0.0.1:8000/v1/chat/completions?api-version=2024-01-01"
    )


def test_base_url_userinfo_is_rejected_without_echoing_credentials(tmp_path: Path) -> None:
    text = MINIMAL.replace(
        "base_url: http://127.0.0.1:8000/v1",
        "base_url: http://alice:super-secret@127.0.0.1:8000/v1",
    )
    with pytest.raises(ConfigError) as exc_info:
        load_config(write_config(tmp_path, text), environ={})
    message = str(exc_info.value)
    assert "username or password" in message
    assert "alice" not in message
    assert "super-secret" not in message


def test_sanitize_url_removes_userinfo_and_masks_query_values() -> None:
    sanitized = sanitize_url(
        "https://alice:super-secret@example.com:8443/v1"
        "?api-version=query-secret&empty=&flag#fragment"
    )

    parsed = urlsplit(sanitized)
    assert parsed.scheme == "https"
    assert parsed.hostname == "example.com"
    assert parsed.port == 8443
    assert parsed.username is None
    assert parsed.password is None
    assert parsed.path == "/v1"
    assert parsed.fragment == ""
    assert parse_qsl(parsed.query, keep_blank_values=True) == [
        ("api-version", "<redacted>"),
        ("empty", "<redacted>"),
        ("flag", "<redacted>"),
    ]
    assert "alice" not in sanitized
    assert "super-secret" not in sanitized
    assert "query-secret" not in sanitized
    assert "fragment" not in sanitized


def test_target_url_preserves_query_value_with_trailing_slash(tmp_path: Path) -> None:
    text = MINIMAL.replace(
        "base_url: http://127.0.0.1:8000/v1",
        "base_url: http://127.0.0.1:8000/v1?api-version=2024-01-01&route=/v1/",
    )
    config = load_config(write_config(tmp_path, text), environ={})
    assert (
        target_url(config.targets[0])
        == "http://127.0.0.1:8000/v1/chat/completions"
        "?api-version=2024-01-01&route=/v1/"
    )


def test_target_tokenizer_override_is_parsed(tmp_path: Path) -> None:
    text = MINIMAL.replace(
        "    model: test-model\n",
        "    model: test-model\n    tokenizer: qwen3.6-35b\n",
    )
    config = load_config(write_config(tmp_path, text), environ={})
    assert config.targets[0].tokenizer == "qwen3.6-35b"


def test_repeats_is_no_longer_supported(tmp_path: Path) -> None:
    text = MINIMAL.replace("run:\n  name: smoke\n", "run:\n  name: smoke\n  repeats: 2\n")
    with pytest.raises(ConfigError, match="repeats"):
        load_config(write_config(tmp_path, text), environ={})


def test_direct_api_key_is_supported_and_env_conflict_is_rejected(tmp_path: Path) -> None:
    text = MINIMAL.replace(
        "    model: test-model\n",
        "    model: test-model\n    api_key: sk-direct\n",
    )
    config = load_config(write_config(tmp_path, text), environ={})
    assert config.targets[0].api_key == "sk-direct"
    assert resolve_headers(config.targets[0], environ={}) == {
        "Authorization": "Bearer sk-direct"
    }

    conflict = text.replace("    api_key: sk-direct\n", "    api_key: sk-direct\n    api_key_env: TEST_KEY\n")
    with pytest.raises(ConfigError, match="api_key"):
        load_config(write_config(tmp_path, conflict), environ={"TEST_KEY": "env-key"})


def test_explicit_cell_formats_and_duplicate_rejection(tmp_path: Path) -> None:
    text = MINIMAL.replace(
        "    - 16-8-1\n    - 16-8-2",
        "    - 4096-512-2\n    - [8192, 1024, 4]\n"
        "    - {input_length: 2048, output_length: 256, concurrency: 1}",
    )
    config = load_config(write_config(tmp_path, text), environ={})
    assert [
        (cell.input_length, cell.output_length, cell.concurrency)
        for cell in config.workload.cells
    ] == [(4096, 512, 2), (8192, 1024, 4), (2048, 256, 1)]

    duplicate = MINIMAL.replace("    - 16-8-2", "    - 16-8-1")
    with pytest.raises(ConfigError, match="duplicate"):
        load_config(write_config(tmp_path, duplicate), environ={})

@pytest.mark.parametrize(
    "cell",
    ["16-8", "16-8-2-1", "16-8-0", "0-8-2", "16-0-2", "16-8-two"],
)
def test_invalid_cell_format_is_rejected(tmp_path: Path, cell: str) -> None:
    text = MINIMAL.replace("    - 16-8-1\n    - 16-8-2", f"    - {cell}")
    with pytest.raises(ConfigError, match="workload.cells"):
        load_config(write_config(tmp_path, text), environ={})


def test_empty_cells_are_rejected(tmp_path: Path) -> None:
    text = MINIMAL.replace("  cells:\n    - 16-8-1\n    - 16-8-2", "  cells: []")
    with pytest.raises(ConfigError, match="non-empty"):
        load_config(write_config(tmp_path, text), environ={})


def test_legacy_scan_fields_point_to_explicit_cells(tmp_path: Path) -> None:
    text = MINIMAL.replace(
        "    - 16-8-1\n    - 16-8-2",
        "    concurrency: [1, 2]\n    input_lengths: [16]\n    output_lengths: [8]",
    )
    with pytest.raises(ConfigError, match="workload.cells"):
        load_config(write_config(tmp_path, text), environ={})


def test_metrics_endpoint_and_headers_are_parsed(tmp_path: Path) -> None:
    text = MINIMAL.replace(
        "    model: test-model\n",
        "    model: test-model\n"
        "    metrics_url: http://127.0.0.1:8000/metrics\n"
        "    metrics_headers:\n"
        "      Authorization: Bearer ${METRICS_TOKEN}\n",
    )
    environment = {"METRICS_TOKEN": "metrics-secret"}
    config = load_config(write_config(tmp_path, text), environ=environment)

    target = config.targets[0]
    assert target.metrics_url == "http://127.0.0.1:8000/metrics"
    assert resolve_metrics_headers(target, environ=environment) == {
        "Authorization": "Bearer metrics-secret"
    }


def test_missing_metrics_header_env_is_a_config_error(tmp_path: Path) -> None:
    text = MINIMAL.replace(
        "    model: test-model\n",
        "    model: test-model\n"
        "    metrics_url: http://127.0.0.1:8000/metrics\n"
        "    metrics_headers:\n"
        "      Authorization: Bearer ${MISSING_METRICS_TOKEN}\n",
    )
    with pytest.raises(ConfigError, match="MISSING_METRICS_TOKEN"):
        load_config(write_config(tmp_path, text), environ={})


@pytest.mark.parametrize(
    "metrics_url",
    ["localhost:8000/metrics", "ftp://127.0.0.1/metrics"],
)
def test_invalid_metrics_url_is_rejected(tmp_path: Path, metrics_url: str) -> None:
    text = MINIMAL.replace(
        "    model: test-model\n",
        f"    model: test-model\n    metrics_url: {metrics_url}\n",
    )
    with pytest.raises(ConfigError, match="metrics_url"):
        load_config(write_config(tmp_path, text), environ={})
