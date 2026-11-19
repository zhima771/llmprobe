from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_RESERVED_REQUEST_KEYS = {
    "model",
    "messages",
    "prompt",
    "stream",
    "stream_options",
    "max_tokens",
    "max_completion_tokens",
}
_BACKENDS = {"openai-chat", "openai-completions"}
_STREAM_OPTIONS = {"auto", "include_usage", "none"}
_MAX_TOKEN_FIELDS = {"max_tokens", "max_completion_tokens"}


class ConfigError(ValueError):
    """Raised when a benchmark configuration is invalid."""


@dataclass(frozen=True)
class RunConfig:
    name: str = "run"
    seed: int = 42
    output_dir: Path = Path("results")
    repeats: int = 1
    cooldown_s: float = 0.0
    save_generated_text: bool = False
    save_itl: bool = False


@dataclass(frozen=True)
class WorkloadConfig:
    dataset: Literal["random"] = "random"
    requests_per_cell: int = 50
    warmup_requests: int = 2
    concurrencies: tuple[int, ...] = (1,)
    input_lengths: tuple[int, ...] = (512,)
    output_lengths: tuple[int, ...] = (128,)
    system_prompt: str = ""
    tokenizer: str | None = None
    stream_options: Literal["auto", "include_usage", "none"] = "auto"


@dataclass(frozen=True)
class TargetConfig:
    name: str
    base_url: str
    model: str
    backend: Literal["openai-chat", "openai-completions"] = "openai-chat"
    api_key_env: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 300.0
    max_tokens_field: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    request_params: dict[str, Any] = field(default_factory=dict)
    http2: bool = False
    verify_tls: bool = True
    ca_bundle: str | None = None


@dataclass(frozen=True)
class AppConfig:
    run: RunConfig
    workload: WorkloadConfig
    targets: tuple[TargetConfig, ...]


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be a mapping")
    return value


def _unknown_keys(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"unknown keys in {path}: {', '.join(unknown)}")


def _string(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ConfigError(f"{path} must be a non-empty string")
    return value


def _integer(value: Any, path: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{path} must be an integer >= {minimum}")
    return value


def _number(value: Any, path: str, *, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < minimum:
        raise ConfigError(f"{path} must be a number >= {minimum}")
    return float(value)


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be a boolean")
    return value


def _string_map(value: Any, path: str) -> dict[str, str]:
    data = _mapping(value, path)
    result: dict[str, str] = {}
    for key, item in data.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ConfigError(f"{path} keys and values must be strings")
        result[key] = item
    return result


def _int_tuple(value: Any, path: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{path} must be a non-empty list")
    return tuple(_integer(item, f"{path}[{index}]", minimum=1) for index, item in enumerate(value))


def _parse_run(raw: Any) -> RunConfig:
    data = _mapping(raw, "run")
    _unknown_keys(
        data,
        {"name", "seed", "output_dir", "repeats", "cooldown_s", "save_generated_text", "save_itl"},
        "run",
    )
    return RunConfig(
        name=_string(data.get("name", "run"), "run.name"),
        seed=_integer(data.get("seed", 42), "run.seed", minimum=0),
        output_dir=Path(_string(data.get("output_dir", "results"), "run.output_dir")),
        repeats=_integer(data.get("repeats", 1), "run.repeats", minimum=1),
        cooldown_s=_number(data.get("cooldown_s", 0.0), "run.cooldown_s", minimum=0.0),
        save_generated_text=_boolean(
            data.get("save_generated_text", False), "run.save_generated_text"
        ),
        save_itl=_boolean(data.get("save_itl", False), "run.save_itl"),
    )


def _parse_workload(raw: Any) -> WorkloadConfig:
    data = _mapping(raw, "workload")
    _unknown_keys(
        data,
        {
            "dataset",
            "requests_per_cell",
            "warmup_requests",
            "concurrencies",
            "input_lengths",
            "output_lengths",
            "system_prompt",
            "tokenizer",
            "stream_options",
        },
        "workload",
    )
    dataset = _string(data.get("dataset", "random"), "workload.dataset")
    if dataset != "random":
        raise ConfigError("workload.dataset must be random in v1")
    stream_options = _string(data.get("stream_options", "auto"), "workload.stream_options")
    if stream_options not in _STREAM_OPTIONS:
        raise ConfigError("workload.stream_options must be auto, include_usage, or none")
    tokenizer = data.get("tokenizer")
    if tokenizer is not None:
        tokenizer = _string(tokenizer, "workload.tokenizer")
    return WorkloadConfig(
        dataset=dataset,  # type: ignore[arg-type]
        requests_per_cell=_integer(
            data.get("requests_per_cell", 50), "workload.requests_per_cell", minimum=1
        ),
        warmup_requests=_integer(
            data.get("warmup_requests", 2), "workload.warmup_requests", minimum=0
        ),
        concurrencies=_int_tuple(data.get("concurrencies", [1]), "workload.concurrencies"),
        input_lengths=_int_tuple(data.get("input_lengths", [512]), "workload.input_lengths"),
        output_lengths=_int_tuple(data.get("output_lengths", [128]), "workload.output_lengths"),
        system_prompt=_string(
            data.get("system_prompt", ""), "workload.system_prompt", allow_empty=True
        ),
        tokenizer=tokenizer,
        stream_options=stream_options,  # type: ignore[arg-type]
    )


def _parse_target(raw: Any, index: int) -> TargetConfig:
    path = f"targets[{index}]"
    data = _mapping(raw, path)
    _unknown_keys(
        data,
        {
            "name",
            "backend",
            "base_url",
            "model",
            "api_key_env",
            "headers",
            "timeout_s",
            "max_tokens_field",
            "request_params",
            "http2",
            "verify_tls",
            "ca_bundle",
        },
        path,
    )
    backend = _string(data.get("backend", "openai-chat"), f"{path}.backend")
    if backend not in _BACKENDS:
        raise ConfigError(f"{path}.backend must be openai-chat or openai-completions")
    max_tokens_field = _string(
        data.get("max_tokens_field", "max_tokens"), f"{path}.max_tokens_field"
    )
    if max_tokens_field not in _MAX_TOKEN_FIELDS:
        raise ConfigError(
            f"{path}.max_tokens_field must be max_tokens or max_completion_tokens"
        )
    base_url = _string(data.get("base_url"), f"{path}.base_url").rstrip("/")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigError(f"{path}.base_url must be an absolute http(s) URL")
    api_key_env = data.get("api_key_env")
    if api_key_env is not None:
        api_key_env = _string(api_key_env, f"{path}.api_key_env")
    headers = _string_map(data.get("headers", {}), f"{path}.headers")
    if api_key_env and "Authorization" in headers:
        raise ConfigError(f"{path} cannot set both api_key_env and Authorization header")
    request_params = dict(_mapping(data.get("request_params", {}), f"{path}.request_params"))
    reserved = sorted(_RESERVED_REQUEST_KEYS.intersection(request_params))
    if reserved:
        raise ConfigError(f"{path}.request_params contains reserved keys: {', '.join(reserved)}")
    ca_bundle = data.get("ca_bundle")
    if ca_bundle is not None:
        ca_bundle = _string(ca_bundle, f"{path}.ca_bundle")
    return TargetConfig(
        name=_string(data.get("name"), f"{path}.name"),
        backend=backend,  # type: ignore[arg-type]
        base_url=base_url,
        model=_string(data.get("model"), f"{path}.model"),
        api_key_env=api_key_env,
        headers=headers,
        timeout_s=_number(data.get("timeout_s", 300.0), f"{path}.timeout_s", minimum=0.001),
        max_tokens_field=max_tokens_field,  # type: ignore[arg-type]
        request_params=request_params,
        http2=_boolean(data.get("http2", False), f"{path}.http2"),
        verify_tls=_boolean(data.get("verify_tls", True), f"{path}.verify_tls"),
        ca_bundle=ca_bundle,
    )


def load_config(
    path: str | Path, environ: Mapping[str, str] | None = None
) -> AppConfig:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    data = _mapping(raw, "config")
    _unknown_keys(data, {"run", "workload", "targets"}, "config")
    run = _parse_run(data.get("run", {}))
    workload = _parse_workload(data.get("workload", {}))
    raw_targets = data.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ConfigError("targets must be a non-empty list")
    targets = tuple(_parse_target(item, index) for index, item in enumerate(raw_targets))
    names = [target.name for target in targets]
    if len(names) != len(set(names)):
        raise ConfigError("target names must be unique")
    max_concurrency = max(workload.concurrencies)
    if max_concurrency > workload.requests_per_cell:
        raise ConfigError(
            "workload.requests_per_cell must be >= the largest concurrency "
            f"({workload.requests_per_cell} < {max_concurrency})"
        )
    config = AppConfig(run=run, workload=workload, targets=targets)
    validate_environment(config, environ)
    return config


def _expand_env(value: str, environ: Mapping[str, str], path: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in environ or not environ[name]:
            raise ConfigError(f"environment variable {name} is required by {path}")
        return environ[name]

    return _ENV_PATTERN.sub(replace, value)


def resolve_headers(
    target: TargetConfig, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    env = os.environ if environ is None else environ
    headers = {
        key: _expand_env(value, env, f"target {target.name} headers.{key}")
        for key, value in target.headers.items()
    }
    if target.api_key_env:
        value = env.get(target.api_key_env)
        if not value:
            raise ConfigError(
                f"environment variable {target.api_key_env} is required by target {target.name}"
            )
        headers["Authorization"] = f"Bearer {value}"
    return headers


def validate_environment(
    config: AppConfig, environ: Mapping[str, str] | None = None
) -> None:
    env = os.environ if environ is None else environ
    for target in config.targets:
        resolve_headers(target, env)


def target_url(target: TargetConfig) -> str:
    endpoint = "chat/completions" if target.backend == "openai-chat" else "completions"
    return f"{target.base_url.rstrip('/')}/{endpoint}"
