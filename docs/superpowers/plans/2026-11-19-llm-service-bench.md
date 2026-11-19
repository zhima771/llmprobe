# llmprobe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-target, OpenAI-compatible streaming LLM benchmark CLI that measures TTFT, TPOT, ITL, E2EL, request throughput and token throughput across concurrency and input/output-length sweeps.

**Architecture:** An asyncio runner drives one shared request matrix against multiple targets, using httpx for streaming HTTP, a resilient incremental SSE parser, deterministic prompt generation, per-request token accounting with usage fallback, and stable JSONL/CSV/JSON/Markdown reports. Configuration is parsed into immutable stdlib dataclasses and validated before any network call.

**Tech Stack:** Python >= 3.11, `argparse`, `asyncio`, `dataclasses`, `httpx`, `PyYAML`, `rich`; tests use `pytest` and a stdlib `http.server` mock OpenAI-compatible service.

**Spec:** `docs/superpowers/specs/2026-11-19-llm-service-bench-design.md`

## Global Constraints

- Package name is `llmprobe`; console entry point is `llmprobe`; module entry point is `python -m llmprobe`.
- Python floor is `>=3.11`.
- Runtime dependencies are exactly: `httpx>=0.27,<1`, `PyYAML>=6`, `rich>=13`.
- Optional tokenizer dependencies are `tokenizers>=0.20` and `huggingface-hub>=0.23`.
- Supported backends in v1: `openai-chat` and `openai-completions`.
- Supported workload mode in v1: closed-loop fixed concurrency only.
- Supported dataset in v1: `dataset: random` (default), using deterministic random token-ID generation when a tokenizer is configured.
- No Web UI, database, distributed load generation, embeddings, multimodal input, retries, or open-loop request-rate mode.
- Metrics: TTFT, TPOT, ITL, E2EL, request throughput, output-token throughput, total-token throughput, peak concurrency, error counts.
- E2EL ends at the last token-bearing delta; benchmark duration ends when all HTTP streams close.
- API keys are read only from environment variables and must never appear in logs or result files.
- Use `time.perf_counter()` for all request timing.
- Unknown YAML keys and reserved request-body overrides are configuration errors.
- Never expose a generated API key, `Authorization` value, or expanded secret header in output.
- Tests must not require public network access.
- The current `.git` directory is an empty read-only placeholder. Task 1 initializes a real repository only after confirming the directory is empty.
- In this sandbox, dependency installation is unavailable. Until a normal `uv sync` can run, execute tests with the local pure-Python harness below:

```bash
export PYTHONPATH="src:/mnt/4t/code/total_bench/harbor/.venv/lib/python3.13/site-packages"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python3 -m pytest -q
```

- Every task ends with a commit. Use Conventional Commit messages shown in the task.

---

### Task 1: Repository and Package Scaffold

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `src/llmprobe/__init__.py`
- Test: `tests/test_scaffold.py`

**Interfaces:**
- Consumes: nothing.
- Produces: importable package `llmprobe` with `__version__ = "0.1.0"`.

- [ ] **Step 1: Initialize the repository if `.git` is still the empty placeholder**

Run:

```bash
if [ -d .git ] && [ -z "$(find .git -mindepth 1 -print -quit)" ]; then
  chmod u+w .git
  git init
fi
git status --short --branch
```

Expected: a valid git repository. Do not delete or overwrite a `.git` directory that contains files.

- [ ] **Step 2: Write the failing scaffold test**

```python
# tests/test_scaffold.py
import llmprobe


def test_package_version_is_defined() -> None:
    assert llmprobe.__version__ == "0.1.0"
```

- [ ] **Step 3: Run the test and verify it fails**

Run:

```bash
export PYTHONPATH="src:/mnt/4t/code/total_bench/harbor/.venv/lib/python3.13/site-packages"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python3 -m pytest tests/test_scaffold.py -q
```

Expected: FAIL because `llmprobe` cannot be imported.

- [ ] **Step 4: Create project metadata and package init**

`.gitignore`:

```gitignore
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
results/
*.egg-info/
dist/
build/
```

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "llmprobe"
version = "0.1.0"
description = "Multi-target OpenAI-compatible LLM serving benchmark"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "httpx>=0.27,<1",
  "PyYAML>=6",
  "rich>=13",
]

[project.optional-dependencies]
tokenizer = ["tokenizers>=0.20", "huggingface-hub>=0.23"]
dev = ["pytest>=8", "ruff>=0.8"]

[project.scripts]
llmprobe = "llmprobe.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
addopts = "-ra"

[tool.ruff]
line-length = 100
target-version = "py311"
```

`src/llmprobe/__init__.py`:

```python
"""llmprobe package."""

__version__ = "0.1.0"
```

- [ ] **Step 5: Run the test and verify it passes**

Run:

```bash
export PYTHONPATH="src:/mnt/4t/code/total_bench/harbor/.venv/lib/python3.13/site-packages"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python3 -m pytest tests/test_scaffold.py -q
```

Expected: `1 passed`.

- [ ] **Step 6: Commit**

```bash
git add .gitignore pyproject.toml src/llmprobe/__init__.py tests/test_scaffold.py
git commit -m "chore: scaffold llmprobe package"
```

---

### Task 2: Configuration Parsing and Secret Resolution

**Files:**
- Create: `src/llmprobe/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `llmprobe` package.
- Produces:
  - `ConfigError(ValueError)`
  - `RunConfig(name, seed, output_dir, repeats, cooldown_s, save_generated_text, save_itl)`
  - `WorkloadConfig(requests_per_cell, warmup_requests, concurrencies, input_lengths, output_lengths, system_prompt, tokenizer, stream_options)`
  - `TargetConfig(name, backend, base_url, model, api_key_env, headers, timeout_s, max_tokens_field, request_params, http2, verify_tls, ca_bundle)`
  - `AppConfig(run, workload, targets)`
  - `load_config(path: str | Path, environ: Mapping[str, str] | None = None) -> AppConfig`
  - `resolve_headers(target: TargetConfig, environ: Mapping[str, str] | None = None) -> dict[str, str]`
  - `target_url(target: TargetConfig) -> str`
  - `validate_environment(config: AppConfig, environ: Mapping[str, str] | None = None) -> None`

- [ ] **Step 1: Write failing configuration tests**

```python
# tests/test_config.py
from pathlib import Path

import pytest

from llmprobe.config import (
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
```

- [ ] **Step 2: Run tests and verify failure**

```bash
export PYTHONPATH="src:/mnt/4t/code/total_bench/harbor/.venv/lib/python3.13/site-packages"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python3 -m pytest tests/test_config.py -q
```

Expected: FAIL because `llmprobe.config` does not exist.

- [ ] **Step 3: Implement configuration parsing**

Create `src/llmprobe/config.py`:

```python
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping
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
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
export PYTHONPATH="src:/mnt/4t/code/total_bench/harbor/.venv/lib/python3.13/site-packages"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python3 -m pytest tests/test_config.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/llmprobe/config.py tests/test_config.py
git commit -m "feat: parse and validate benchmark config"
```

---

### Task 3: Models, Matrix Expansion, Tokenization, and Deterministic Prompts

**Files:**
- Create: `src/llmprobe/models.py`
- Create: `src/llmprobe/workload.py`
- Create: `src/llmprobe/tokenization.py`
- Create: `src/llmprobe/promptgen.py`
- Test: `tests/test_workload.py`
- Test: `tests/test_promptgen.py`

**Interfaces:**
- Consumes: `RunConfig`, `WorkloadConfig`, `TargetConfig`.
- Produces:
  - `Cell(id, concurrency, input_length, output_length)`
  - `GeneratedPrompt(system, user, target_tokens, estimated_tokens, tokenizer_used)`
  - `RequestResult`, `LatencyStats`, `CellSummary`, `TargetCapability`, `RunResult`
  - `make_run_id(output_dir: Path, name: str) -> str`
  - `build_cells(workload: WorkloadConfig) -> list[Cell]`
  - `cell_seed(global_seed: int, cell: Cell, repeat: int) -> int`
  - `load_tokenizer(spec: str | None) -> TextTokenizer | None`
  - `generate_prompt(global_seed, cell, repeat, request_index, workload, tokenizer) -> GeneratedPrompt`

- [ ] **Step 1: Write failing matrix and prompt tests**

```python
# tests/test_workload.py
from llmprobe.config import WorkloadConfig
from llmprobe.workload import build_cells, cell_seed


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
```

```python
# tests/test_promptgen.py
from llmprobe.config import WorkloadConfig
from llmprobe.promptgen import generate_prompt
from llmprobe.workload import build_cells


class WordTokenizer:
    @property
    def is_tokenizer(self) -> bool:
        return True

    @property
    def vocab_size(self) -> int:
        return 10_000

    def decode_ids(self, token_ids: list[int]) -> str:
        return " ".join(f"t{token_id}" for token_id in token_ids)

    def encode_ids(self, text: str) -> list[int]:
        return [int(word[1:]) for word in text.split() if word.startswith("t")]

    def count(self, text: str) -> int:
        return len(text.split())

    def truncate(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])


def test_prompt_is_deterministic_and_varies_by_request() -> None:
    workload = WorkloadConfig(
        concurrencies=(2,), input_lengths=(64,), output_lengths=(8,), system_prompt="be terse"
    )
    cell = build_cells(workload)[0]
    first = generate_prompt(7, cell, 0, 0, workload, None)
    same = generate_prompt(7, cell, 0, 0, workload, None)
    other = generate_prompt(7, cell, 0, 1, workload, None)
    assert first == same
    assert first.user != other.user
    assert first.system == "be terse"
    assert first.tokenizer_used is False


def test_tokenizer_prompt_targets_total_content_tokens() -> None:
    workload = WorkloadConfig(
        concurrencies=(1,),
        input_lengths=(40,),
        output_lengths=(8,),
        system_prompt="system instructions",
    )
    cell = build_cells(workload)[0]
    tokenizer = WordTokenizer()
    prompt = generate_prompt(9, cell, 0, 3, workload, tokenizer)
    assert prompt.tokenizer_used is True
    assert tokenizer.count(prompt.system) + tokenizer.count(prompt.user) == 40
    assert prompt.estimated_tokens == 40
    assert prompt.user.startswith("t")


def test_completions_text_concatenates_system_and_user() -> None:
    workload = WorkloadConfig(concurrencies=(1,), input_lengths=(8,), output_lengths=(2,), system_prompt="sys")
    cell = build_cells(workload)[0]
    prompt = generate_prompt(1, cell, 0, 0, workload, None)
    assert prompt.completions_text.startswith("sys\n\n")
```

- [ ] **Step 2: Run tests and verify failure**

```bash
export PYTHONPATH="src:/mnt/4t/code/total_bench/harbor/.venv/lib/python3.13/site-packages"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python3 -m pytest tests/test_workload.py tests/test_promptgen.py -q
```

Expected: FAIL because the modules do not exist.

- [ ] **Step 3: Implement shared models**

Create `src/llmprobe/models.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Cell:
    id: str
    concurrency: int
    input_length: int
    output_length: int


@dataclass(frozen=True)
class GeneratedPrompt:
    system: str
    user: str
    target_tokens: int
    estimated_tokens: int
    tokenizer_used: bool

    @property
    def completions_text(self) -> str:
        if self.system:
            return f"{self.system}\n\n{self.user}"
        return self.user


@dataclass
class RequestResult:
    target: str
    cell_id: str
    repeat: int
    request_index: int
    concurrency: int
    input_length_target: int
    output_length_target: int
    input_tokens: int | None
    output_tokens: int
    input_tokens_source: str
    output_tokens_source: str
    input_tokens_estimated: bool
    output_tokens_estimated: bool
    client_queue_s: float
    started_at: float
    ttft_s: float | None
    e2el_s: float | None
    stream_end_s: float
    tpot_s: float | None
    itl_s: list[float]
    token_event_count: int
    finish_reason: str | None
    success: bool
    error_kind: str | None
    error: str | None
    http_status: int | None
    generated_text: str | None


@dataclass(frozen=True)
class LatencyStats:
    count: int = 0
    mean_ms: float | None = None
    std_ms: float | None = None
    p50_ms: float | None = None
    p90_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None


@dataclass(frozen=True)
class CellSummary:
    target: str
    cell_id: str
    repeat: int
    concurrency: int
    input_length: int
    output_length: int
    completed: int
    failed: int
    error_rate: float
    duration_s: float
    request_throughput: float
    output_token_throughput: float | None
    total_token_throughput: float | None
    ttft: LatencyStats
    tpot: LatencyStats
    itl: LatencyStats
    e2el: LatencyStats
    decode_tokens_per_s_per_stream: float | None
    input_tokens_mean: float | None
    output_tokens_mean: float | None
    peak_concurrency: int
    token_sources: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetCapability:
    name: str
    available: bool
    stream_options_sent: bool
    usage_stream_supported: bool | None
    error: str | None = None


@dataclass(frozen=True)
class RunResult:
    run_id: str
    started_at: str
    ended_at: str
    summaries: list[CellSummary]
    requests: list[RequestResult]
    capabilities: list[TargetCapability]
    interrupted: bool = False
    tool_version: str = "0.1.0"
    schema_version: int = 1


def make_run_id(output_dir: Path, name: str) -> str:
    del output_dir
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-").lower() or "run"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{slug}"
```

- [ ] **Step 4: Implement workload and tokenizer modules**

Create `src/llmprobe/workload.py`:

```python
from __future__ import annotations

import hashlib

from llmprobe.config import WorkloadConfig
from llmprobe.models import Cell


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
    payload = f"{global_seed}|{cell.id}|{repeat}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
```

Create `src/llmprobe/tokenization.py`:

```python
from __future__ import annotations

from typing import Protocol

from llmprobe.config import ConfigError


class TextTokenizer(Protocol):
    @property
    def is_tokenizer(self) -> bool: ...

    @property
    def vocab_size(self) -> int: ...

    def count(self, text: str) -> int: ...

    def encode_ids(self, text: str) -> list[int]: ...

    def decode_ids(self, token_ids: list[int]) -> str: ...

    def truncate(self, text: str, max_tokens: int) -> str: ...


class LocalTokenizer:
    is_tokenizer = True

    def __init__(self, tokenizer: object) -> None:
        self._tokenizer = tokenizer

    @property
    def vocab_size(self) -> int:
        return int(self._tokenizer.get_vocab_size())  # type: ignore[attr-defined]

    def count(self, text: str) -> int:
        return len(self.encode_ids(text))

    def encode_ids(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False).ids)  # type: ignore[attr-defined]

    def decode_ids(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=False)  # type: ignore[attr-defined]

    def truncate(self, text: str, max_tokens: int) -> str:
        encoding = self._tokenizer.encode(text, add_special_tokens=False)  # type: ignore[attr-defined]
        if len(encoding.ids) <= max_tokens:
            return text
        return self.decode_ids(encoding.ids[:max_tokens])


def load_tokenizer(spec: str | None) -> TextTokenizer | None:
    if not spec:
        return None
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise ConfigError(
            "tokenizer support requires the optional dependencies: "
            "pip install 'llmprobe[tokenizer]'"
        ) from exc
    if "/" in spec and not spec.endswith(".json"):
        tokenizer = Tokenizer.from_pretrained(spec)
    else:
        tokenizer = Tokenizer.from_file(spec)
    return LocalTokenizer(tokenizer)
```

- [ ] **Step 5: Implement deterministic prompt generation**

Create `src/llmprobe/promptgen.py`:

```python
from __future__ import annotations

import math
import random

from llmprobe.config import ConfigError, WorkloadConfig
from llmprobe.models import Cell, GeneratedPrompt
from llmprobe.tokenization import TextTokenizer
from llmprobe.workload import cell_seed

_WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
    "network kernel tensor token stream prompt decode prefill latency "
    "throughput request response cache queue server client model inference "
    "benchmark measurement interval concurrency workload synthetic data"
).split()


def _word_blob(rng: random.Random, approx_chars: int) -> str:
    words: list[str] = []
    size = 0
    while size < approx_chars:
        word = _WORDS[rng.randrange(len(_WORDS))]
        words.append(word)
        size += len(word) + 1
    return " ".join(words)


def _make_user_text(rng: random.Random, request_index: int, approx_chars: int) -> str:
    marker = f"request-{request_index}"
    return f"{marker} {_word_blob(rng, max(0, approx_chars - len(marker) - 1))}".strip()


def generate_prompt(
    global_seed: int,
    cell: Cell,
    repeat: int,
    request_index: int,
    workload: WorkloadConfig,
    tokenizer: TextTokenizer | None,
) -> GeneratedPrompt:
    seed = cell_seed(global_seed, cell, repeat) ^ (request_index * 0x9E3779B97F4A7C15)
    rng = random.Random(seed)
    system = workload.system_prompt

    if tokenizer is not None:
        system_tokens = tokenizer.count(system) if system else 0
        if system_tokens >= cell.input_length:
            raise ConfigError(
                f"system_prompt is too long for target input length {cell.input_length}"
            )
        remaining = cell.input_length - system_tokens
        vocab_size = max(1, tokenizer.vocab_size)
        offset = rng.randrange(vocab_size)
        token_ids = [
            (offset + request_index + index) % vocab_size for index in range(remaining)
        ]
        user = tokenizer.decode_ids(token_ids)
        for _ in range(5):
            owned = tokenizer.encode_ids(user)
            if len(owned) == remaining:
                break
            if len(owned) > remaining:
                user = tokenizer.decode_ids(owned[:remaining])
            else:
                extra = [
                    (offset + request_index + len(owned) + index) % vocab_size
                    for index in range(remaining - len(owned))
                ]
                user = tokenizer.decode_ids(owned + extra)
        estimated = (tokenizer.count(system) if system else 0) + len(
            tokenizer.encode_ids(user)
        )
        return GeneratedPrompt(
            system=system,
            user=user,
            target_tokens=cell.input_length,
            estimated_tokens=estimated,
            tokenizer_used=True,
        )

    approx_system_tokens = math.ceil(len(system) / 4) if system else 0
    if approx_system_tokens >= cell.input_length:
        raise ConfigError(f"system_prompt is too long for target input length {cell.input_length}")
    remaining_tokens = cell.input_length - approx_system_tokens
    user = _make_user_text(rng, request_index, remaining_tokens * 4)
    estimated = math.ceil((len(system) + len(user)) / 4)
    return GeneratedPrompt(
        system=system,
        user=user,
        target_tokens=cell.input_length,
        estimated_tokens=estimated,
        tokenizer_used=False,
    )
```

- [ ] **Step 6: Run tests and verify they pass**

```bash
export PYTHONPATH="src:/mnt/4t/code/total_bench/harbor/.venv/lib/python3.13/site-packages"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python3 -m pytest tests/test_workload.py tests/test_promptgen.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/llmprobe/models.py src/llmprobe/workload.py src/llmprobe/tokenization.py src/llmprobe/promptgen.py tests/test_workload.py tests/test_promptgen.py
git commit -m "feat: add workload matrix and synthetic prompt generation"
```

---

### Task 4: Incremental SSE Parser

**Files:**
- Create: `src/llmprobe/sse.py`
- Test: `tests/test_sse.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `SSEParser.feed(chunk: bytes) -> list[str]` and `SSEParser.close() -> list[str]`. Each returned string is the joined `data:` payload of one SSE event.

- [ ] **Step 1: Write failing SSE tests**

```python
# tests/test_sse.py
from llmprobe.sse import SSEParser


def test_parses_fragmented_utf8_and_crlf() -> None:
    parser = SSEParser()
    raw = (
        b'data: {"choices":[{"delta":{"content":"\xe4\xbd\xa0"}}]}\r\n\r\n'
        b"data: [DONE]\r\n\r\n"
    )
    events: list[str] = []
    for index in range(len(raw)):
        events.extend(parser.feed(raw[index : index + 1]))
    assert events == ['{"choices":[{"delta":{"content":"你"}}]}', "[DONE]"]


def test_parses_multiline_data_and_ignores_comments() -> None:
    parser = SSEParser()
    events = parser.feed(b": keepalive\n\ndata: first\ndata: second\n\n")
    assert events == ["first\nsecond"]


def test_close_flushes_event_without_trailing_blank_line() -> None:
    parser = SSEParser()
    assert parser.feed(b"data: tail") == []
    assert parser.close() == ["tail"]


def test_ignores_non_data_fields() -> None:
    parser = SSEParser()
    assert parser.feed(b"event: message\nid: 3\nretry: 1000\n\ndata: x\n\n") == ["x"]
```

- [ ] **Step 2: Run tests and verify failure**

```bash
export PYTHONPATH="src:/mnt/4t/code/total_bench/harbor/.venv/lib/python3.13/site-packages"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python3 -m pytest tests/test_sse.py -q
```

Expected: FAIL because `llmprobe.sse` does not exist.

- [ ] **Step 3: Implement the parser**

Create `src/llmprobe/sse.py`:

```python
from __future__ import annotations

import codecs


class SSEParser:
    """Incrementally parse an SSE byte stream into data payloads."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._buffer = ""
        self._data_lines: list[str] = []

    def feed(self, chunk: bytes) -> list[str]:
        self._buffer += self._decoder.decode(chunk)
        return self._drain(final=False)

    def close(self) -> list[str]:
        self._buffer += self._decoder.decode(b"", final=True)
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> list[str]:
        events: list[str] = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.endswith("\r"):
                line = line[:-1]
            if line == "":
                if self._data_lines:
                    events.append("\n".join(self._data_lines))
                    self._data_lines = []
            elif line.startswith(":"):
                continue
            elif line.startswith("data:"):
                value = line[5:]
                if value.startswith(" "):
                    value = value[1:]
                self._data_lines.append(value)
        if final and self._buffer:
            line = self._buffer[:-1] if self._buffer.endswith("\r") else self._buffer
            if line.startswith("data:"):
                value = line[5:]
                if value.startswith(" "):
                    value = value[1:]
                self._data_lines.append(value)
            self._buffer = ""
        if final and self._data_lines:
            events.append("\n".join(self._data_lines))
            self._data_lines = []
        return events
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
export PYTHONPATH="src:/mnt/4t/code/total_bench/harbor/.venv/lib/python3.13/site-packages"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python3 -m pytest tests/test_sse.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/llmprobe/sse.py tests/test_sse.py
git commit -m "feat: add incremental SSE parser"
```

---

### Task 5: Streaming OpenAI-Compatible Client and Measurement

**Files:**
- Create: `src/llmprobe/client.py`
- Test: `tests/test_client.py`

**Interfaces:**
- Consumes: `TargetConfig`, `Cell`, `GeneratedPrompt`, `RequestResult`, `TextTokenizer`, `SSEParser`, `target_url`.
- Produces:
  - `create_http_client(target: TargetConfig, max_connections: int) -> httpx.AsyncClient`
  - `build_payload(target, prompt, output_length, include_usage) -> dict`
  - `stream_request(client, target, prompt, *, tokenizer, include_usage, request_index, repeat, cell, client_queue_s, headers, save_generated_text) -> RequestResult`
  - `probe_usage(client, target, prompt, *, tokenizer, headers, cell) -> tuple[bool, RequestResult]`
  - `TargetProbeError(RuntimeError)`

- [ ] **Step 1: Write failing client tests**

```python
# tests/test_client.py
import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest

from llmprobe.client import (
    TargetProbeError,
    build_payload,
    probe_usage,
    stream_request,
)
from llmprobe.config import TargetConfig
from llmprobe.models import Cell, GeneratedPrompt


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], delay: float = 0.0) -> None:
        self._chunks = chunks
        self._delay = delay

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            if self._delay:
                await asyncio.sleep(self._delay)
            yield chunk


def sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def target(**overrides: object) -> TargetConfig:
    values = {
        "name": "mock",
        "base_url": "http://mock/v1",
        "model": "mock-model",
        "request_params": {"temperature": 0},
    }
    values.update(overrides)
    return TargetConfig(**values)  # type: ignore[arg-type]


def prompt() -> GeneratedPrompt:
    return GeneratedPrompt(
        system="",
        user="hello",
        target_tokens=5,
        estimated_tokens=1,
        tokenizer_used=False,
    )


def cell() -> Cell:
    return Cell(id="i5-o3-c1", concurrency=1, input_length=5, output_length=3)


def test_build_payload_uses_target_token_field() -> None:
    payload = build_payload(
        target(max_tokens_field="max_completion_tokens"),
        prompt(),
        output_length=12,
        include_usage=True,
    )
    assert payload["max_completion_tokens"] == 12
    assert "max_tokens" not in payload
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["temperature"] == 0


def test_stream_request_supports_completions_backend() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["prompt"] == "hello"
        chunks = [
            sse({"choices": [{"text": "A"}]}),
            sse({"choices": [{"text": "B", "finish_reason": "stop"}]}),
        ]
        return httpx.Response(200, stream=ChunkStream(chunks))

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await stream_request(
                client,
                target(backend="openai-completions"),
                prompt(),
                tokenizer=None,
                include_usage=False,
                request_index=0,
                repeat=0,
                cell=cell(),
                client_queue_s=0.0,
                headers={},
                save_generated_text=True,
            )
        assert result.success is True
        assert result.generated_text == "AB"
        assert result.output_tokens == 2

    asyncio.run(scenario())


def test_stream_request_measures_usage_ttft_tpot_and_itl() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        chunks = [
            sse({"choices": [{"delta": {"role": "assistant"}}]}),
            sse({"choices": [{"delta": {"content": "A"}}]}),
            sse({"choices": [{"delta": {"reasoning_content": "B"}}]}),
            sse({"choices": [{"delta": {"content": "C"}, "finish_reason": "stop"}]}),
            sse({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}),
            b"data: [DONE]\n\n",
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=ChunkStream(chunks, delay=0.005),
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await stream_request(
                client,
                target(),
                prompt(),
                tokenizer=None,
                include_usage=True,
                request_index=0,
                repeat=0,
                cell=cell(),
                client_queue_s=0.0,
                headers={},
                save_generated_text=True,
            )
        assert result.success is True
        assert result.http_status == 200
        assert result.input_tokens == 5
        assert result.output_tokens == 3
        assert result.input_tokens_source == "usage"
        assert result.output_tokens_source == "usage"
        assert 0.003 < result.ttft_s < 0.2
        assert result.e2el_s > result.ttft_s
        assert len(result.itl_s) == 2
        assert result.tpot_s == pytest.approx((result.e2el_s - result.ttft_s) / 2)
        assert result.generated_text == "AC"
        assert result.finish_reason == "stop"

    asyncio.run(scenario())


def test_stream_request_falls_back_to_event_count_without_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        chunks = [
            sse({"choices": [{"delta": {"content": "A"}}]}),
            sse({"choices": [{"delta": {"content": "B"}}]}),
            sse({"choices": [{"delta": {"content": "C"}}]}),
        ]
        return httpx.Response(200, stream=ChunkStream(chunks))

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await stream_request(
                client,
                target(),
                prompt(),
                tokenizer=None,
                include_usage=False,
                request_index=0,
                repeat=0,
                cell=cell(),
                client_queue_s=0.0,
                headers={},
                save_generated_text=False,
            )
        assert result.success is True
        assert result.input_tokens is None
        assert result.input_tokens_source == "unavailable"
        assert result.output_tokens == 3
        assert result.output_tokens_source == "events"
        assert result.output_tokens_estimated is True

    asyncio.run(scenario())


def test_stream_request_records_http_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await stream_request(
                client,
                target(),
                prompt(),
                tokenizer=None,
                include_usage=True,
                request_index=1,
                repeat=0,
                cell=cell(),
                client_queue_s=0.25,
                headers={},
                save_generated_text=False,
            )
        assert result.success is False
        assert result.http_status == 429
        assert result.error_kind == "http"
        assert result.client_queue_s == 0.25

    asyncio.run(scenario())


def test_probe_usage_retries_without_stream_options() -> None:
    seen: list[bool] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent = "stream_options" in body
        seen.append(sent)
        if sent:
            return httpx.Response(400, text="unknown field stream_options")
        return httpx.Response(
            200,
            stream=ChunkStream([sse({"choices": [{"delta": {"content": "x"}}]})]),
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            include_usage, result = await probe_usage(
                client,
                target(),
                prompt(),
                tokenizer=None,
                headers={},
                cell=cell(),
            )
        assert include_usage is False
        assert result.success is True
        assert seen == [True, False]

    asyncio.run(scenario())


def test_probe_usage_raises_on_unrelated_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(TargetProbeError, match="401"):
                await probe_usage(
                    client,
                    target(),
                    prompt(),
                    tokenizer=None,
                    headers={},
                    cell=cell(),
                )

    asyncio.run(scenario())
```

- [ ] **Step 2: Run tests and verify failure**

```bash
export PYTHONPATH="src:/mnt/4t/code/total_bench/harbor/.venv/lib/python3.13/site-packages"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python3 -m pytest tests/test_client.py -q
```

Expected: FAIL because `llmprobe.client` does not exist.

- [ ] **Step 3: Implement the client and stream measurement**

Create `src/llmprobe/client.py`:

```python
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from typing import Any

import httpx

from llmprobe.config import TargetConfig, target_url
from llmprobe.models import Cell, GeneratedPrompt, RequestResult
from llmprobe.sse import SSEParser
from llmprobe.tokenization import TextTokenizer


class TargetProbeError(RuntimeError):
    """Raised when a target cannot execute a stream-options capability probe."""


def create_http_client(target: TargetConfig, max_connections: int) -> httpx.AsyncClient:
    verify: bool | str = target.ca_bundle if target.ca_bundle else target.verify_tls
    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_connections,
    )
    return httpx.AsyncClient(
        http2=target.http2,
        verify=verify,
        limits=limits,
        timeout=httpx.Timeout(target.timeout_s),
    )


def build_payload(
    target: TargetConfig,
    prompt: GeneratedPrompt,
    output_length: int,
    include_usage: bool,
) -> dict[str, Any]:
    if target.backend == "openai-chat":
        messages: list[dict[str, str]] = []
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.user})
        payload: dict[str, Any] = {
            "model": target.model,
            "messages": messages,
            "stream": True,
            target.max_tokens_field: output_length,
        }
    else:
        payload = {
            "model": target.model,
            "prompt": prompt.completions_text,
            "stream": True,
            target.max_tokens_field: output_length,
        }
    payload.update(target.request_params)
    if include_usage:
        payload["stream_options"] = {"include_usage": True}
    return payload


def _extract_usage(data: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None, None
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    prompt_value = prompt_tokens if isinstance(prompt_tokens, int) else None
    completion_value = completion_tokens if isinstance(completion_tokens, int) else None
    return prompt_value, completion_value


def _extract_chat_token(data: dict[str, Any]) -> tuple[bool, str, str, str | None]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return False, "", "", None
    choice = choices[0]
    if not isinstance(choice, dict):
        return False, "", "", None
    finish_reason = choice.get("finish_reason")
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        return False, "", "", finish_reason
    token_keys = {"content", "reasoning", "reasoning_content", "tool_calls", "function_call"}
    if not token_keys.intersection(delta):
        return False, "", "", finish_reason
    if delta.get("content") == "" and "role" in delta and not (
        token_keys - {"content"}
    ).intersection(delta):
        return False, "", "", finish_reason
    counted_parts: list[str] = []
    displayed_parts: list[str] = []
    for key in ("content", "reasoning", "reasoning_content"):
        value = delta.get(key)
        if isinstance(value, str):
            counted_parts.append(value)
            if key == "content":
                displayed_parts.append(value)
    tool_calls = delta.get("tool_calls")
    if isinstance(tool_calls, list):
        counted_parts.append(json.dumps(tool_calls, ensure_ascii=False, sort_keys=True))
    function_call = delta.get("function_call")
    if isinstance(function_call, dict):
        counted_parts.append(json.dumps(function_call, ensure_ascii=False, sort_keys=True))
    return True, "".join(counted_parts), "".join(displayed_parts), finish_reason


def _extract_completion_token(data: dict[str, Any]) -> tuple[bool, str, str, str | None]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return False, "", "", None
    choice = choices[0]
    if not isinstance(choice, dict):
        return False, "", "", None
    finish_reason = choice.get("finish_reason")
    if "text" not in choice:
        return False, "", "", finish_reason
    value = choice.get("text")
    text = value if isinstance(value, str) else ""
    return True, text, text, finish_reason


def _extract_token(
    backend: str, data: dict[str, Any]
) -> tuple[bool, str, str, str | None]:
    if backend == "openai-chat":
        return _extract_chat_token(data)
    return _extract_completion_token(data)


async def stream_request(
    client: httpx.AsyncClient,
    target: TargetConfig,
    prompt: GeneratedPrompt,
    *,
    tokenizer: TextTokenizer | None,
    include_usage: bool,
    request_index: int,
    repeat: int,
    cell: Cell,
    client_queue_s: float,
    headers: Mapping[str, str],
    save_generated_text: bool,
) -> RequestResult:
    started_at = time.perf_counter()
    first_token_at: float | None = None
    last_token_at: float | None = None
    itl_s: list[float] = []
    token_events = 0
    visible_parts: list[str] = []
    output_parts: list[str] = []
    finish_reason: str | None = None
    http_status: int | None = None
    usage_input: int | None = None
    usage_output: int | None = None
    error: str | None = None
    error_kind: str | None = None
    parser = SSEParser()

    def observe_payload(data: dict[str, Any]) -> None:
        nonlocal first_token_at, last_token_at, token_events, finish_reason
        nonlocal usage_input, usage_output
        if data.get("error"):
            raise ValueError(json.dumps(data["error"], ensure_ascii=False)[:2000])
        prompt_tokens, completion_tokens = _extract_usage(data)
        if prompt_tokens is not None:
            usage_input = prompt_tokens
        if completion_tokens is not None:
            usage_output = completion_tokens
        token_bearing, counted_text, displayed_text, reason = _extract_token(
            target.backend, data
        )
        if reason is not None:
            finish_reason = reason
        if not token_bearing:
            return
        now = time.perf_counter()
        if first_token_at is None:
            first_token_at = now
        else:
            itl_s.append(now - (last_token_at if last_token_at is not None else now))
        last_token_at = now
        token_events += 1
        visible_parts.append(counted_text)
        output_parts.append(displayed_text)

    try:
        async with asyncio.timeout(target.timeout_s):
            async with client.stream(
                "POST",
                target_url(target),
                json=build_payload(target, prompt, cell.output_length, include_usage),
                headers=dict(headers),
            ) as response:
                http_status = response.status_code
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    error = f"HTTP {response.status_code}: {body[:2000]}"
                    error_kind = "http"
                else:
                    done = False
                    async for chunk in response.aiter_bytes():
                        for data_text in parser.feed(chunk):
                            if data_text == "[DONE]":
                                done = True
                                break
                            try:
                                payload = json.loads(data_text)
                                if not isinstance(payload, dict):
                                    raise ValueError("SSE data is not a JSON object")
                                observe_payload(payload)
                            except (json.JSONDecodeError, ValueError) as exc:
                                error = f"invalid stream event: {exc}"
                                error_kind = "stream"
                                done = True
                                break
                        if done:
                            break
                    if error is None:
                        for data_text in parser.close():
                            if data_text == "[DONE]":
                                continue
                            payload = json.loads(data_text)
                            if not isinstance(payload, dict):
                                raise ValueError("SSE data is not a JSON object")
                            observe_payload(payload)
    except TimeoutError:
        error = f"request exceeded timeout of {target.timeout_s:g}s"
        error_kind = "timeout"
    except httpx.TimeoutException as exc:
        error = f"HTTP timeout: {exc}"
        error_kind = "timeout"
    except httpx.HTTPError as exc:
        error = f"HTTP error: {type(exc).__name__}: {exc}"
        error_kind = "connection"
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        error = f"stream error: {exc}"
        error_kind = "stream"
    except Exception as exc:  # noqa: BLE001 - benchmark failures must be recorded, not crash the run
        error = f"request error: {type(exc).__name__}: {exc}"
        error_kind = "request"

    stream_end_s = time.perf_counter() - started_at
    if http_status == 200 and token_events == 0 and error is None:
        error = "stream completed without a token-bearing delta"
        error_kind = "no_tokens"

    success = http_status == 200 and token_events > 0 and error is None
    visible_text = "".join(visible_parts)
    output_text = "".join(output_parts)
    if usage_input is not None:
        input_tokens: int | None = usage_input
        input_source = "usage"
        input_estimated = False
    elif tokenizer is not None:
        input_tokens = tokenizer.count(prompt.system) + tokenizer.count(prompt.user)
        input_source = "tokenizer"
        input_estimated = True
    else:
        input_tokens = None
        input_source = "unavailable"
        input_estimated = False

    if usage_output is not None:
        output_tokens = usage_output
        output_source = "usage"
        output_estimated = False
    elif tokenizer is not None and visible_text:
        output_tokens = tokenizer.count(visible_text)
        output_source = "tokenizer"
        output_estimated = True
    else:
        output_tokens = token_events
        output_source = "events" if token_events else "unavailable"
        output_estimated = True

    ttft_s = first_token_at - started_at if first_token_at is not None else None
    e2el_s = last_token_at - started_at if last_token_at is not None else None
    tpot_s: float | None = None
    if e2el_s is not None and ttft_s is not None and output_tokens > 1:
        tpot_s = (e2el_s - ttft_s) / (output_tokens - 1)

    return RequestResult(
        target=target.name,
        cell_id=cell.id,
        repeat=repeat,
        request_index=request_index,
        concurrency=cell.concurrency,
        input_length_target=cell.input_length,
        output_length_target=cell.output_length,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_tokens_source=input_source,
        output_tokens_source=output_source,
        input_tokens_estimated=input_estimated,
        output_tokens_estimated=output_estimated,
        client_queue_s=client_queue_s,
        started_at=started_at,
        ttft_s=ttft_s,
        e2el_s=e2el_s,
        stream_end_s=stream_end_s,
        tpot_s=tpot_s,
        itl_s=itl_s,
        token_event_count=token_events,
        finish_reason=finish_reason,
        success=success,
        error_kind=error_kind,
        error=error,
        http_status=http_status,
        generated_text=output_text if save_generated_text and output_text else None,
    )


async def probe_usage(
    client: httpx.AsyncClient,
    target: TargetConfig,
    prompt: GeneratedPrompt,
    *,
    tokenizer: TextTokenizer | None,
    headers: Mapping[str, str],
    cell: Cell,
) -> tuple[bool, RequestResult]:
    result = await stream_request(
        client,
        target,
        prompt,
        tokenizer=tokenizer,
        include_usage=True,
        request_index=0,
        repeat=-1,
        cell=cell,
        client_queue_s=0.0,
        headers=headers,
        save_generated_text=False,
    )
    if result.success:
        return True, result
    error_text = (result.error or "").lower()
    if result.http_status == 400 and "stream_options" in error_text:
        retry = await stream_request(
            client,
            target,
            prompt,
            tokenizer=tokenizer,
            include_usage=False,
            request_index=0,
            repeat=-1,
            cell=cell,
            client_queue_s=0.0,
            headers=headers,
            save_generated_text=False,
        )
        if retry.success:
            return False, retry
        raise TargetProbeError(
            f"target {target.name} failed after disabling stream_options: {retry.error}"
        )
    raise TargetProbeError(
        f"target {target.name} capability probe failed: {result.error or result.http_status}"
    )
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
export PYTHONPATH="src:/mnt/4t/code/total_bench/harbor/.venv/lib/python3.13/site-packages"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python3 -m pytest tests/test_client.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/llmprobe/client.py tests/test_client.py
git commit -m "feat: add streaming benchmark client"
```

---

### Task 6: Request and Cell Metrics

**Files:**
- Create: `src/llmprobe/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `RequestResult`, `Cell`, `LatencyStats`, `CellSummary`.
- Produces:
  - `percentile(values: list[float], p: float) -> float | None`
  - `latency_stats(values_s: list[float]) -> LatencyStats`
  - `summarize_results(results: list[RequestResult], cell: Cell, target: str, repeat: int) -> CellSummary`

- [ ] **Step 1: Write failing metrics tests**

```python
# tests/test_metrics.py
import time

import pytest

from llmprobe.metrics import latency_stats, percentile, summarize_results
from llmprobe.models import Cell, RequestResult


def result(
    *,
    started_at: float,
    stream_end_s: float,
    success: bool = True,
    ttft_s: float | None = 0.1,
    e2el_s: float | None = 0.5,
    output_tokens: int = 5,
    input_tokens: int | None = 10,
) -> RequestResult:
    return RequestResult(
        target="mock",
        cell_id="i4-o2-c2",
        repeat=0,
        request_index=0,
        concurrency=2,
        input_length_target=4,
        output_length_target=2,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_tokens_source="usage" if input_tokens is not None else "unavailable",
        output_tokens_source="usage",
        input_tokens_estimated=False,
        output_tokens_estimated=False,
        client_queue_s=0.0,
        started_at=started_at,
        ttft_s=ttft_s,
        e2el_s=e2el_s,
        stream_end_s=stream_end_s,
        tpot_s=(e2el_s - ttft_s) / (output_tokens - 1) if e2el_s and ttft_s and output_tokens > 1 else None,
        itl_s=[0.1, 0.1, 0.1, 0.1],
        token_event_count=output_tokens,
        finish_reason="stop",
        success=success,
        error_kind=None if success else "http",
        error=None if success else "failed",
        http_status=200 if success else 500,
        generated_text=None,
    )


def test_percentile_uses_linear_interpolation() -> None:
    assert percentile([10.0, 20.0, 30.0, 40.0], 50) == 25.0
    assert percentile([10.0, 20.0], 0) == 10.0
    assert percentile([10.0, 20.0], 100) == 20.0
    assert percentile([], 50) is None


def test_latency_stats_reports_population_stddev() -> None:
    stats = latency_stats([0.1, 0.2, 0.3])
    assert stats.count == 3
    assert stats.mean_ms == pytest.approx(200.0)
    assert stats.std_ms == pytest.approx(81.649658, rel=1e-5)
    assert stats.p50_ms == pytest.approx(200.0)


def test_summarize_results_calculates_throughput() -> None:
    cell = Cell(id="i4-o2-c2", concurrency=2, input_length=4, output_length=2)
    results = [
        result(started_at=100.0, stream_end_s=0.5),
        result(started_at=100.5, stream_end_s=0.5, output_tokens=5, input_tokens=20),
    ]
    summary = summarize_results(results, cell, "mock", 0)
    assert summary.completed == 2
    assert summary.failed == 0
    assert summary.duration_s == pytest.approx(1.0)
    assert summary.request_throughput == pytest.approx(2.0)
    assert summary.output_token_throughput == pytest.approx(10.0)
    assert summary.total_token_throughput == pytest.approx(40.0)
    assert summary.peak_concurrency == 2
    assert summary.ttft.p50_ms == pytest.approx(100.0)


def test_total_throughput_is_none_when_input_tokens_are_unknown() -> None:
    cell = Cell(id="i4-o2-c1", concurrency=1, input_length=4, output_length=2)
    summary = summarize_results(
        [result(started_at=time.perf_counter(), stream_end_s=0.5, input_tokens=None)],
        cell,
        "mock",
        0,
    )
    assert summary.total_token_throughput is None
    assert summary.output_token_throughput is not None


def test_all_failed_requests_do_not_produce_latency_stats() -> None:
    cell = Cell(id="i4-o2-c1", concurrency=1, input_length=4, output_length=2)
    summary = summarize_results(
        [result(started_at=100.0, stream_end_s=0.2, success=False)], cell, "mock", 0
    )
    assert summary.completed == 0
    assert summary.failed == 1
    assert summary.ttft.count == 0
    assert summary.ttft.mean_ms is None
```

- [ ] **Step 2: Run tests and verify failure**

```bash
export PYTHONPATH="src:/mnt/4t/code/total_bench/harbor/.venv/lib/python3.13/site-packages"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python3 -m pytest tests/test_metrics.py -q
```

Expected: FAIL because `llmprobe.metrics` does not exist.

- [ ] **Step 3: Implement metrics**

Create `src/llmprobe/metrics.py`:

```python
from __future__ import annotations

import math
import statistics
from collections import Counter

from llmprobe.models import Cell, CellSummary, LatencyStats, RequestResult


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if not 0 <= p <= 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(values)
    position = (len(ordered) - 1) * (p / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def latency_stats(values_s: list[float]) -> LatencyStats:
    if not values_s:
        return LatencyStats()
    values_ms = [value * 1000.0 for value in values_s]
    return LatencyStats(
        count=len(values_ms),
        mean_ms=statistics.fmean(values_ms),
        std_ms=statistics.pstdev(values_ms),
        p50_ms=percentile(values_ms, 50),
        p90_ms=percentile(values_ms, 90),
        p95_ms=percentile(values_ms, 95),
        p99_ms=percentile(values_ms, 99),
    )


def _peak_concurrency(results: list[RequestResult]) -> int:
    events: list[tuple[float, int]] = []
    for result in results:
        events.append((result.started_at, 1))
        events.append((result.started_at + result.stream_end_s, -1))
    active = 0
    peak = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        peak = max(peak, active)
    return peak


def summarize_results(
    results: list[RequestResult], cell: Cell, target: str, repeat: int
) -> CellSummary:
    successful = [result for result in results if result.success]
    failed = len(results) - len(successful)
    if results:
        start = min(result.started_at for result in results)
        end = max(result.started_at + result.stream_end_s for result in results)
        duration_s = max(0.0, end - start)
    else:
        duration_s = 0.0

    ttfts = [result.ttft_s for result in successful if result.ttft_s is not None]
    e2els = [result.e2el_s for result in successful if result.e2el_s is not None]
    tpots = [result.tpot_s for result in successful if result.tpot_s is not None]
    itls = [value for result in successful for value in result.itl_s]
    ttft_stats = latency_stats(ttfts)
    output_total = sum(result.output_tokens for result in successful)
    input_known = bool(successful) and all(result.input_tokens is not None for result in successful)
    input_total = sum(result.input_tokens or 0 for result in successful)

    if duration_s > 0:
        request_throughput = len(successful) / duration_s
        output_throughput: float | None = output_total / duration_s
        total_throughput: float | None = (
            (input_total + output_total) / duration_s if input_known else None
        )
    else:
        request_throughput = 0.0
        output_throughput = None
        total_throughput = None

    input_values = [result.input_tokens for result in successful if result.input_tokens is not None]
    output_values = [result.output_tokens for result in successful]
    token_sources = Counter(
        f"{result.input_tokens_source}/{result.output_tokens_source}" for result in successful
    )
    tpot_stats = latency_stats(tpots)
    decode_tps = (
        1000.0 / tpot_stats.mean_ms
        if tpot_stats.mean_ms and tpot_stats.mean_ms > 0
        else None
    )

    return CellSummary(
        target=target,
        cell_id=cell.id,
        repeat=repeat,
        concurrency=cell.concurrency,
        input_length=cell.input_length,
        output_length=cell.output_length,
        completed=len(successful),
        failed=failed,
        error_rate=(failed / len(results)) if results else 0.0,
        duration_s=duration_s,
        request_throughput=request_throughput,
        output_token_throughput=output_throughput,
        total_token_throughput=total_throughput,
        ttft=ttft_stats,
        tpot=tpot_stats,
        itl=latency_stats(itls),
        e2el=latency_stats(e2els),
        decode_tokens_per_s_per_stream=decode_tps,
        input_tokens_mean=statistics.fmean(input_values) if input_values else None,
        output_tokens_mean=statistics.fmean(output_values) if output_values else None,
        peak_concurrency=_peak_concurrency(results),
        token_sources=dict(token_sources),
    )
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
export PYTHONPATH="src:/mnt/4t/code/total_bench/harbor/.venv/lib/python3.13/site-packages"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python3 -m pytest tests/test_metrics.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/llmprobe/metrics.py tests/test_metrics.py
git commit -m "feat: calculate serving benchmark metrics"
```

---

### Task 7: Benchmark Orchestration

**Files:**
- Create: `src/llmprobe/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: configuration, workload, prompt generation, client, metrics, models.
- Produces:
  - `ProgressSink` protocol and `NullProgress`
  - `TargetUnavailable(RuntimeError)`
  - `BenchmarkRunner(config, progress=None, environ=None, tokenizer=None)`
  - `BenchmarkRunner.run() -> RunResult`
  - `BenchmarkRunner.partial_result() -> RunResult`
  - Public attributes `run_id`, `results`, `summaries`, `capabilities`, `started_at`, `ended_at`

- [ ] **Step 1: Write failing runner tests**

```python
# tests/test_runner.py
import asyncio
import time
from pathlib import Path

import pytest

from llmprobe.config import AppConfig, RunConfig, TargetConfig, WorkloadConfig
from llmprobe.models import RequestResult
from llmprobe.runner import BenchmarkRunner, NullProgress


def config(tmp_path: Path, *, stream_options: str = "auto") -> AppConfig:
    return AppConfig(
        run=RunConfig(name="test", output_dir=tmp_path, repeats=1, cooldown_s=0.0),
        workload=WorkloadConfig(
            requests_per_cell=2,
            warmup_requests=1,
            concurrencies=(1,),
            input_lengths=(8,),
            output_lengths=(2,),
            stream_options=stream_options,  # type: ignore[arg-type]
        ),
        targets=(
            TargetConfig(name="a", base_url="http://a/v1", model="m"),
            TargetConfig(name="b", base_url="http://b/v1", model="m"),
        ),
    )


def make_result(target: str, cell: object, repeat: int, index: int, success: bool = True) -> RequestResult:
    return RequestResult(
        target=target,
        cell_id=cell.id,  # type: ignore[attr-defined]
        repeat=repeat,
        request_index=index,
        concurrency=cell.concurrency,  # type: ignore[attr-defined]
        input_length_target=cell.input_length,  # type: ignore[attr-defined]
        output_length_target=cell.output_length,  # type: ignore[attr-defined]
        input_tokens=8,
        output_tokens=2,
        input_tokens_source="usage",
        output_tokens_source="usage",
        input_tokens_estimated=False,
        output_tokens_estimated=False,
        client_queue_s=0.0,
        started_at=time.perf_counter(),
        ttft_s=0.01,
        e2el_s=0.02,
        stream_end_s=0.02,
        tpot_s=0.01,
        itl_s=[0.01],
        token_event_count=2,
        finish_reason="stop",
        success=success,
        error_kind=None if success else "http",
        error=None if success else "failed",
        http_status=200 if success else 500,
        generated_text=None,
    )


def test_runner_executes_probe_warmup_and_measurement_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []

    async def fake_probe(client, target, prompt, **kwargs):
        calls.append((target.name, "probe"))
        return True, make_result(target.name, kwargs["cell"], -1, 0)

    async def fake_request(client, target, prompt, **kwargs):
        calls.append((target.name, kwargs["cell"].id))
        return make_result(
            target.name,
            kwargs["cell"],
            kwargs["repeat"],
            kwargs["request_index"],
        )

    monkeypatch.setattr("llmprobe.runner.probe_usage", fake_probe)
    monkeypatch.setattr("llmprobe.runner.stream_request", fake_request)
    runner = BenchmarkRunner(config(tmp_path), progress=NullProgress(), environ={})
    result = asyncio.run(runner.run())
    assert len(result.summaries) == 2
    assert len(result.requests) == 4
    assert calls[:4] == [
        ("a", "probe"),
        ("b", "probe"),
        ("a", "i8-o2-c1"),
        ("b", "i8-o2-c1"),
    ]
    assert calls[4:] == [
        ("a", "i8-o2-c1"),
        ("a", "i8-o2-c1"),
        ("b", "i8-o2-c1"),
        ("b", "i8-o2-c1"),
    ]


def test_target_is_skipped_when_all_warmups_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    measured_targets: list[str] = []

    async def fake_request(client, target, prompt, **kwargs):
        if kwargs["repeat"] == -1:
            return make_result(target.name, kwargs["cell"], -1, 0, success=False)
        measured_targets.append(target.name)
        return make_result(target.name, kwargs["cell"], kwargs["repeat"], kwargs["request_index"])

    monkeypatch.setattr("llmprobe.runner.stream_request", fake_request)
    runner = BenchmarkRunner(
        config(tmp_path, stream_options="none"), progress=NullProgress(), environ={}
    )
    result = asyncio.run(runner.run())
    assert measured_targets == []
    assert all(capability.available is False for capability in result.capabilities)


def test_partial_result_summarizes_completed_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_request(client, target, prompt, **kwargs):
        return make_result(target.name, kwargs["cell"], kwargs["repeat"], kwargs["request_index"])

    monkeypatch.setattr("llmprobe.runner.stream_request", fake_request)
    runner = BenchmarkRunner(
        config(tmp_path, stream_options="none"), progress=NullProgress(), environ={}
    )
    asyncio.run(runner.run())
    partial = runner.partial_result()
    assert partial.interrupted is True
    assert len(partial.requests) == 4
    assert len(partial.summaries) == 2
```

- [ ] **Step 2: Run tests and verify failure**

```bash
export PYTHONPATH="src:/mnt/4t/code/total_bench/harbor/.venv/lib/python3.13/site-packages"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python3 -m pytest tests/test_runner.py -q
```

Expected: FAIL because `llmprobe.runner` does not exist.

- [ ] **Step 3: Implement the runner**

Create `src/llmprobe/runner.py`:

```python
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from typing import Protocol

from llmprobe.client import (
    TargetProbeError,
    create_http_client,
    probe_usage,
    stream_request,
)
from llmprobe.config import (
    AppConfig,
    TargetConfig,
    resolve_headers,
)
from llmprobe.metrics import summarize_results
from llmprobe.models import (
    Cell,
    CellSummary,
    RequestResult,
    RunResult,
    TargetCapability,
    make_run_id,
)
from llmprobe.promptgen import generate_prompt
from llmprobe.tokenization import TextTokenizer, load_tokenizer
from llmprobe.workload import build_cells


class ProgressSink(Protocol):
    def cell_started(self, target: str, cell: Cell, repeat: int) -> None: ...

    def request_finished(self, result: RequestResult) -> None: ...

    def cell_finished(self, summary: CellSummary) -> None: ...

    def warning(self, message: str) -> None: ...


class NullProgress:
    def cell_started(self, target: str, cell: Cell, repeat: int) -> None:
        return None

    def request_finished(self, result: RequestResult) -> None:
        return None

    def cell_finished(self, summary: CellSummary) -> None:
        return None

    def warning(self, message: str) -> None:
        return None


class TargetUnavailable(RuntimeError):
    """Raised internally when a target cannot continue the run."""


class BenchmarkRunner:
    def __init__(
        self,
        config: AppConfig,
        progress: ProgressSink | None = None,
        environ: Mapping[str, str] | None = None,
        tokenizer: TextTokenizer | None = None,
    ) -> None:
        self.config = config
        self.progress = progress or NullProgress()
        self.environ = environ
        self._tokenizer = tokenizer
        self.run_id = make_run_id(config.run.output_dir, config.run.name)
        self.results: list[RequestResult] = []
        self.summaries: list[CellSummary] = []
        self.capabilities: dict[str, TargetCapability] = {}
        self.started_at: str | None = None
        self.ended_at: str | None = None

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    async def run(self) -> RunResult:
        self.started_at = self._now()
        tokenizer = self._tokenizer or load_tokenizer(self.config.workload.tokenizer)
        cells = build_cells(self.config.workload)
        clients = {}
        headers: dict[str, dict[str, str]] = {}
        include_usage: dict[str, bool] = {}
        max_concurrency = max(self.config.workload.concurrencies)
        try:
            for target in self.config.targets:
                headers[target.name] = resolve_headers(target, self.environ)
                clients[target.name] = create_http_client(target, max_concurrency)

            for target in self.config.targets:
                available, stream_options_sent, usage_supported, error = await self._probe_target(
                    target, clients[target.name], headers[target.name], tokenizer, cells
                )
                self.capabilities[target.name] = TargetCapability(
                    name=target.name,
                    available=available,
                    stream_options_sent=stream_options_sent,
                    usage_stream_supported=usage_supported,
                    error=error,
                )
                include_usage[target.name] = stream_options_sent
                if not available:
                    self.progress.warning(
                        f"target {target.name} disabled after probe: {error}"
                    )

            for cell_index, cell in enumerate(cells):
                for target in self.config.targets:
                    capability = self.capabilities[target.name]
                    if not capability.available:
                        continue
                    try:
                        await self._warmup_target(
                            target,
                            clients[target.name],
                            headers[target.name],
                            cell,
                            include_usage[target.name],
                            tokenizer,
                        )
                    except TargetUnavailable as exc:
                        self.capabilities[target.name] = replace(
                            capability, available=False, error=str(exc)
                        )
                        self.progress.warning(f"target {target.name} disabled: {exc}")

                for repeat in range(self.config.run.repeats):
                    prompts = [
                        generate_prompt(
                            self.config.run.seed,
                            cell,
                            repeat,
                            index,
                            self.config.workload,
                            tokenizer,
                        )
                        for index in range(self.config.workload.requests_per_cell)
                    ]
                    for target in self.config.targets:
                        capability = self.capabilities[target.name]
                        if not capability.available:
                            continue
                        self.progress.cell_started(target.name, cell, repeat)
                        results = await self._execute_requests(
                            target=target,
                            client=clients[target.name],
                            headers=headers[target.name],
                            cell=cell,
                            prompts=prompts,
                            include_usage=include_usage[target.name],
                            tokenizer=tokenizer,
                            repeat=repeat,
                            record=True,
                        )
                        summary = summarize_results(results, cell, target.name, repeat)
                        self.summaries.append(summary)
                        self.progress.cell_finished(summary)
                if cell_index < len(cells) - 1 and self.config.run.cooldown_s > 0:
                    await asyncio.sleep(self.config.run.cooldown_s)
        finally:
            for client in clients.values():
                await client.aclose()
        self.ended_at = self._now()
        return self._to_result(interrupted=False)

    async def _probe_target(
        self,
        target: TargetConfig,
        client,
        headers: Mapping[str, str],
        tokenizer: TextTokenizer | None,
        cells: list[Cell],
    ) -> tuple[bool, bool, bool | None, str | None]:
        mode = self.config.workload.stream_options
        if mode == "none":
            return True, False, False, None
        if mode == "include_usage":
            return True, True, None, None
        probe_cell = Cell(
            id="probe",
            concurrency=1,
            input_length=min(cell.input_length for cell in cells),
            output_length=min(cell.output_length for cell in cells),
        )
        probe_prompt = generate_prompt(
            self.config.run.seed,
            probe_cell,
            -1,
            0,
            self.config.workload,
            tokenizer,
        )
        try:
            include_usage, result = await probe_usage(
                client,
                target,
                probe_prompt,
                tokenizer=tokenizer,
                headers=headers,
                cell=probe_cell,
            )
        except TargetProbeError as exc:
            return False, True, False, str(exc)
        return True, include_usage, result.output_tokens_source == "usage", None

    async def _warmup_target(
        self,
        target: TargetConfig,
        client,
        headers: Mapping[str, str],
        cell: Cell,
        include_usage: bool,
        tokenizer: TextTokenizer | None,
    ) -> None:
        if self.config.workload.warmup_requests == 0:
            return
        prompts = [
            generate_prompt(
                self.config.run.seed,
                cell,
                -1,
                index,
                self.config.workload,
                tokenizer,
            )
            for index in range(self.config.workload.warmup_requests)
        ]
        results = await self._execute_requests(
            target=target,
            client=client,
            headers=headers,
            cell=cell,
            prompts=prompts,
            include_usage=include_usage,
            tokenizer=tokenizer,
            repeat=-1,
            record=False,
        )
        failures = sum(not result.success for result in results)
        if failures == len(results):
            raise TargetUnavailable(
                f"all {failures} warmup requests failed for cell {cell.id}"
            )
        if failures:
            self.progress.warning(
                f"target {target.name} had {failures}/{len(results)} warmup failures "
                f"for cell {cell.id}"
            )

    async def _execute_requests(
        self,
        *,
        target: TargetConfig,
        client,
        headers: Mapping[str, str],
        cell: Cell,
        prompts: list,
        include_usage: bool,
        tokenizer: TextTokenizer | None,
        repeat: int,
        record: bool,
    ) -> list[RequestResult]:
        semaphore = asyncio.Semaphore(cell.concurrency)
        collected: list[tuple[int, RequestResult]] = []

        async def one(index: int, prompt) -> RequestResult:
            scheduled_at = time.perf_counter()
            async with semaphore:
                queue_s = time.perf_counter() - scheduled_at
                result = await stream_request(
                    client,
                    target,
                    prompt,
                    tokenizer=tokenizer,
                    include_usage=include_usage,
                    request_index=index,
                    repeat=repeat,
                    cell=cell,
                    client_queue_s=queue_s,
                    headers=headers,
                    save_generated_text=self.config.run.save_generated_text,
                )
            collected.append((index, result))
            if record:
                self.results.append(result)
                self.progress.request_finished(result)
            return result

        await asyncio.gather(*(one(index, prompt) for index, prompt in enumerate(prompts)))
        return [result for _, result in sorted(collected, key=lambda item: item[0])]

    def _to_result(self, *, interrupted: bool) -> RunResult:
        ended_at = self.ended_at or self._now()
        return RunResult(
            run_id=self.run_id,
            started_at=self.started_at or ended_at,
            ended_at=ended_at,
            summaries=list(self.summaries),
            requests=list(self.results),
            capabilities=list(self.capabilities.values()),
            interrupted=interrupted,
        )

    def partial_result(self) -> RunResult:
        cells = {cell.id: cell for cell in build_cells(self.config.workload)}
        grouped: dict[tuple[str, str, int], list[RequestResult]] = defaultdict(list)
        for result in self.results:
            grouped[(result.target, result.cell_id, result.repeat)].append(result)
        summaries = [
            summarize_results(results, cells[cell_id], target, repeat)
            for (target, cell_id, repeat), results in sorted(grouped.items())
            if cell_id in cells
        ]
        return RunResult(
            run_id=self.run_id,
            started_at=self.started_at or self._now(),
            ended_at=self._now(),
            summaries=summaries,
            requests=list(self.results),
            capabilities=list(self.capabilities.values()),
            interrupted=True,
        )
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
export PYTHONPATH="src:/mnt/4t/code/total_bench/harbor/.venv/lib/python3.13/site-packages"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python3 -m pytest tests/test_runner.py -q
```

Expected: all tests pass. The order test expects probe calls for both targets, followed by two warmup calls and four measured calls.

- [ ] **Step 5: Commit**

```bash
git add src/llmprobe/runner.py tests/test_runner.py
git commit -m "feat: orchestrate multi-target benchmark runs"
```

---

### Task 8: Result Serialization and Reports

**Files:**
- Create: `src/llmprobe/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `AppConfig`, `RunResult`, `CellSummary`, `RequestResult`.
- Produces:
  - `config_to_dict(config: AppConfig) -> dict`
  - `request_to_dict(result: RequestResult, *, save_itl: bool, save_generated_text: bool) -> dict`
  - `summary_to_dict(summary: CellSummary) -> dict`
  - `cross_repeat_aggregates(summaries: list[CellSummary]) -> list[dict]`
  - `write_report(result: RunResult, config: AppConfig) -> Path`
  - `print_summary(result: RunResult, console: Console | None = None) -> None`

- [ ] **Step 1: Write failing report tests**

```python
# tests/test_report.py
import csv
import io
import json
import time
from pathlib import Path

from rich.console import Console

from llmprobe.config import AppConfig, RunConfig, TargetConfig, WorkloadConfig
from llmprobe.metrics import summarize_results
from llmprobe.models import Cell, RequestResult, RunResult, TargetCapability
from llmprobe.report import print_summary, request_to_dict, write_report


def make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        run=RunConfig(name="report-test", output_dir=tmp_path / "results"),
        workload=WorkloadConfig(
            requests_per_cell=1,
            warmup_requests=0,
            concurrencies=(1,),
            input_lengths=(8,),
            output_lengths=(2,),
        ),
        targets=(
            TargetConfig(
                name="mock",
                base_url="http://mock/v1",
                model="m",
                headers={"Authorization": "Bearer super-secret"},
            ),
        ),
    )


def make_result() -> RequestResult:
    return RequestResult(
        target="mock",
        cell_id="i8-o2-c1",
        repeat=0,
        request_index=0,
        concurrency=1,
        input_length_target=8,
        output_length_target=2,
        input_tokens=8,
        output_tokens=2,
        input_tokens_source="usage",
        output_tokens_source="usage",
        input_tokens_estimated=False,
        output_tokens_estimated=False,
        client_queue_s=0.0,
        started_at=time.perf_counter(),
        ttft_s=0.01,
        e2el_s=0.02,
        stream_end_s=0.02,
        tpot_s=0.01,
        itl_s=[0.01],
        token_event_count=2,
        finish_reason="stop",
        success=True,
        error_kind=None,
        error=None,
        http_status=200,
        generated_text="secret output text",
    )


def test_request_serialization_respects_optional_payloads() -> None:
    data = request_to_dict(make_result(), save_itl=False, save_generated_text=False)
    assert "itl_ms" not in data
    assert "generated_text" not in data
    assert data["ttft_ms"] == 10.0
    assert data["itl_count"] == 1


def test_write_report_creates_all_files_and_redacts_secret(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    result_item = make_result()
    cell = Cell(id="i8-o2-c1", concurrency=1, input_length=8, output_length=2)
    summary = summarize_results([result_item], cell, "mock", 0)
    run = RunResult(
        run_id="20260101T000000Z-report-test",
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:00:02+00:00",
        summaries=[summary],
        requests=[result_item],
        capabilities=[
            TargetCapability(
                name="mock",
                available=True,
                stream_options_sent=True,
                usage_stream_supported=True,
            )
        ],
    )
    output = write_report(run, config)
    assert (output / "config.effective.yaml").exists()
    assert (output / "run.json").exists()
    assert (output / "requests.jsonl").exists()
    assert (output / "summary.csv").exists()
    assert (output / "summary.json").exists()
    assert (output / "summary.md").exists()
    assert (output / "errors.log").exists()
    for path in output.iterdir():
        assert "super-secret" not in path.read_text(encoding="utf-8")
    with (output / "summary.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["target"] == "mock"
    payload = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["capabilities"][0]["name"] == "mock"


def test_print_summary_does_not_fail(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    item = make_result()
    cell = Cell(id="i8-o2-c1", concurrency=1, input_length=8, output_length=2)
    summary = summarize_results([item], cell, "mock", 0)
    run = RunResult(
        run_id="r",
        started_at="a",
        ended_at="b",
        summaries=[summary],
        requests=[item],
        capabilities=[],
    )
    print_summary(run, Console(file=io.StringIO()))
```

- [ ] **Step 2: Run tests and verify failure**

```bash
export PYTHONPATH="src:/mnt/4t/code/total_bench/harbor/.venv/lib/python3.13/site-packages"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python3 -m pytest tests/test_report.py -q
```

Expected: FAIL because `llmprobe.report` does not exist.

- [ ] **Step 3: Implement report generation**

Create `src/llmprobe/report.py`:

```python
from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable

import yaml
from rich.console import Console
from rich.table import Table

from llmprobe.config import AppConfig
from llmprobe.models import CellSummary, RequestResult, RunResult

_SENSITIVE_KEYS = {
    "authorization",
    "api_key",
    "apikey",
    "access_token",
    "token",
    "secret",
    "client_secret",
}
_CSV_FIELDS = [
    "target",
    "cell_id",
    "repeat",
    "concurrency",
    "input_length",
    "output_length",
    "completed",
    "failed",
    "error_rate",
    "duration_s",
    "request_throughput",
    "output_token_throughput",
    "total_token_throughput",
    "ttft_mean_ms",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "ttft_p99_ms",
    "tpot_mean_ms",
    "tpot_p50_ms",
    "tpot_p95_ms",
    "itl_p95_ms",
    "e2el_p50_ms",
    "e2el_p95_ms",
    "decode_tokens_per_s_per_stream",
    "input_tokens_mean",
    "output_tokens_mean",
    "peak_concurrency",
    "token_sources",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in _SENSITIVE_KEYS or normalized.endswith("_secret"):
                result[key] = "<redacted>"
            else:
                result[key] = _redact(item)
        return result
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    return _redact(_jsonable(config))


def request_to_dict(
    result: RequestResult, *, save_itl: bool, save_generated_text: bool
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "target": result.target,
        "cell_id": result.cell_id,
        "repeat": result.repeat,
        "request_index": result.request_index,
        "concurrency": result.concurrency,
        "input_length_target": result.input_length_target,
        "output_length_target": result.output_length_target,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "input_tokens_source": result.input_tokens_source,
        "output_tokens_source": result.output_tokens_source,
        "input_tokens_estimated": result.input_tokens_estimated,
        "output_tokens_estimated": result.output_tokens_estimated,
        "client_queue_ms": result.client_queue_s * 1000.0,
        "ttft_ms": result.ttft_s * 1000.0 if result.ttft_s is not None else None,
        "e2el_ms": result.e2el_s * 1000.0 if result.e2el_s is not None else None,
        "stream_end_ms": result.stream_end_s * 1000.0,
        "tpot_ms": result.tpot_s * 1000.0 if result.tpot_s is not None else None,
        "itl_count": len(result.itl_s),
        "token_event_count": result.token_event_count,
        "finish_reason": result.finish_reason,
        "success": result.success,
        "error_kind": result.error_kind,
        "error": result.error,
        "http_status": result.http_status,
    }
    if result.itl_s:
        data["itl_mean_ms"] = statistics.fmean(result.itl_s) * 1000.0
    else:
        data["itl_mean_ms"] = None
    if save_itl:
        data["itl_ms"] = [value * 1000.0 for value in result.itl_s]
    if save_generated_text:
        data["generated_text"] = result.generated_text
    return data


def summary_to_dict(summary: CellSummary) -> dict[str, Any]:
    return _jsonable(summary)


def _metric_value(summary: CellSummary, name: str) -> float | None:
    mapping: dict[str, Callable[[CellSummary], float | None]] = {
        "request_throughput": lambda item: item.request_throughput,
        "output_token_throughput": lambda item: item.output_token_throughput,
        "total_token_throughput": lambda item: item.total_token_throughput,
        "ttft_p50_ms": lambda item: item.ttft.p50_ms,
        "ttft_p95_ms": lambda item: item.ttft.p95_ms,
        "ttft_p99_ms": lambda item: item.ttft.p99_ms,
        "tpot_p50_ms": lambda item: item.tpot.p50_ms,
        "tpot_p95_ms": lambda item: item.tpot.p95_ms,
        "itl_p95_ms": lambda item: item.itl.p95_ms,
        "e2el_p50_ms": lambda item: item.e2el.p50_ms,
        "e2el_p95_ms": lambda item: item.e2el.p95_ms,
    }
    return mapping[name](summary)


def cross_repeat_aggregates(summaries: list[CellSummary]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[CellSummary]] = {}
    for summary in summaries:
        grouped.setdefault((summary.target, summary.cell_id), []).append(summary)
    metric_names = [
        "request_throughput",
        "output_token_throughput",
        "total_token_throughput",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "ttft_p99_ms",
        "tpot_p50_ms",
        "tpot_p95_ms",
        "itl_p95_ms",
        "e2el_p50_ms",
        "e2el_p95_ms",
    ]
    output: list[dict[str, Any]] = []
    for (target, cell_id), items in sorted(grouped.items()):
        metrics: dict[str, dict[str, float | None]] = {}
        for metric_name in metric_names:
            values = [
                value
                for item in items
                if (value := _metric_value(item, metric_name)) is not None
            ]
            metrics[metric_name] = {
                "mean": statistics.fmean(values) if values else None,
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None,
            }
        output.append(
            {
                "target": target,
                "cell_id": cell_id,
                "repeats": len(items),
                "metrics": metrics,
            }
        )
    return output


def _flatten_summary(summary: CellSummary) -> dict[str, Any]:
    return {
        "target": summary.target,
        "cell_id": summary.cell_id,
        "repeat": summary.repeat,
        "concurrency": summary.concurrency,
        "input_length": summary.input_length,
        "output_length": summary.output_length,
        "completed": summary.completed,
        "failed": summary.failed,
        "error_rate": summary.error_rate,
        "duration_s": summary.duration_s,
        "request_throughput": summary.request_throughput,
        "output_token_throughput": summary.output_token_throughput,
        "total_token_throughput": summary.total_token_throughput,
        "ttft_mean_ms": summary.ttft.mean_ms,
        "ttft_p50_ms": summary.ttft.p50_ms,
        "ttft_p95_ms": summary.ttft.p95_ms,
        "ttft_p99_ms": summary.ttft.p99_ms,
        "tpot_mean_ms": summary.tpot.mean_ms,
        "tpot_p50_ms": summary.tpot.p50_ms,
        "tpot_p95_ms": summary.tpot.p95_ms,
        "itl_p95_ms": summary.itl.p95_ms,
        "e2el_p50_ms": summary.e2el.p50_ms,
        "e2el_p95_ms": summary.e2el.p95_ms,
        "decode_tokens_per_s_per_stream": summary.decode_tokens_per_s_per_stream,
        "input_tokens_mean": summary.input_tokens_mean,
        "output_tokens_mean": summary.output_tokens_mean,
        "peak_concurrency": summary.peak_concurrency,
        "token_sources": json.dumps(summary.token_sources, sort_keys=True),
    }


def _format_number(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _markdown(result: RunResult) -> str:
    lines = [
        f"# Benchmark Report: {result.run_id}",
        "",
        f"- Started: {result.started_at}",
        f"- Ended: {result.ended_at}",
        f"- Interrupted: {result.interrupted}",
        "",
        "| Target | Cell | Repeat | OK/Err | Req/s | Out tok/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2EL p95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in result.summaries:
        lines.append(
            "| {target} | {cell} | {repeat} | {ok}/{failed} | {rps} | {out_tps} | "
            "{ttft50} | {ttft95} | {tpot50} | {itl95} | {e2el95} |".format(
                target=summary.target,
                cell=summary.cell_id,
                repeat=summary.repeat,
                ok=summary.completed,
                failed=summary.failed,
                rps=_format_number(summary.request_throughput),
                out_tps=_format_number(summary.output_token_throughput),
                ttft50=_format_number(summary.ttft.p50_ms) + " ms",
                ttft95=_format_number(summary.ttft.p95_ms) + " ms",
                tpot50=_format_number(summary.tpot.p50_ms) + " ms",
                itl95=_format_number(summary.itl.p95_ms) + " ms",
                e2el95=_format_number(summary.e2el.p95_ms) + " ms",
            )
        )
    return "\n".join(lines) + "\n"


def write_report(result: RunResult, config: AppConfig) -> Path:
    run_dir = config.run.output_dir / result.run_id
    suffix = 1
    while run_dir.exists():
        run_dir = config.run.output_dir / f"{result.run_id}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)

    config_data = config_to_dict(config)
    (run_dir / "config.effective.yaml").write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    run_data = {
        "schema_version": result.schema_version,
        "tool_version": result.tool_version,
        "run_id": result.run_id,
        "started_at": result.started_at,
        "ended_at": result.ended_at,
        "interrupted": result.interrupted,
        "config": config_data,
        "capabilities": _jsonable(result.capabilities),
    }
    (run_dir / "run.json").write_text(
        json.dumps(run_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    request_rows = [
        request_to_dict(
            item,
            save_itl=config.run.save_itl,
            save_generated_text=config.run.save_generated_text,
        )
        for item in result.requests
    ]
    (run_dir / "requests.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in request_rows),
        encoding="utf-8",
    )

    summary_data = {
        "schema_version": result.schema_version,
        "summaries": [summary_to_dict(item) for item in result.summaries],
        "cross_repeat": cross_repeat_aggregates(result.summaries),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for summary in result.summaries:
            writer.writerow(_flatten_summary(summary))
    (run_dir / "summary.md").write_text(_markdown(result), encoding="utf-8")

    errors = [
        f"{item.target} {item.cell_id} repeat={item.repeat} request={item.request_index}: "
        f"{item.error_kind}: {item.error}"
        for item in result.requests
        if not item.success
    ]
    (run_dir / "errors.log").write_text(
        "\n".join(errors) + ("\n" if errors else ""), encoding="utf-8"
    )
    return run_dir


def print_summary(result: RunResult, console: Console | None = None) -> None:
    output = console or Console()
    table = Table(title=f"llmprobe results: {result.run_id}")
    for column in (
        "Target",
        "Cell",
        "Repeat",
        "OK/Err",
        "Req/s",
        "Out tok/s",
        "TTFT p50",
        "TTFT p95",
        "TPOT p50",
        "E2EL p95",
    ):
        table.add_column(column, justify="right" if column not in {"Target", "Cell"} else "left")
    for summary in result.summaries:
        table.add_row(
            summary.target,
            summary.cell_id,
            str(summary.repeat),
            f"{summary.completed}/{summary.failed}",
            _format_number(summary.request_throughput),
            _format_number(summary.output_token_throughput),
            _format_number(summary.ttft.p50_ms) + " ms",
            _format_number(summary.ttft.p95_ms) + " ms",
            _format_number(summary.tpot.p50_ms) + " ms",
            _format_number(summary.e2el.p95_ms) + " ms",
        )
    output.print(table)
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
export PYTHONPATH="src:/mnt/4t/code/total_bench/harbor/.venv/lib/python3.13/site-packages"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python3 -m pytest tests/test_report.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/llmprobe/report.py tests/test_report.py
git commit -m "feat: write benchmark result reports"
```

---

### Task 9: CLI, Mock Server, and End-to-End Flow

**Files:**
- Create: `src/llmprobe/cli.py`
- Create: `src/llmprobe/__main__.py`
- Create: `tests/mock_openai.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: all prior modules.
- Produces: `build_parser()`, `main(argv: list[str] | None = None) -> int`, `RichProgressSink`, and module entry point.

- [ ] **Step 1: Write the mock OpenAI-compatible server**

Create `tests/mock_openai.py`:

```python
from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class MockOpenAIServer(ThreadingHTTPServer):
    daemon_threads = True
    reject_stream_options = False
    token_delay_s = 0.002
    token_count = 3


class MockOpenAIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        return None

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path != "/v1/chat/completions":
            self._write_json(404, {"error": {"message": "not found"}})
            return
        server = self.server
        assert isinstance(server, MockOpenAIServer)
        include_usage = "stream_options" in body
        if include_usage and server.reject_stream_options:
            self._write_json(400, {"error": {"message": "unknown field stream_options"}})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()

        def event(payload: dict[str, Any]) -> None:
            data = f"data: {json.dumps(payload)}\n\n".encode("utf-8")
            self.wfile.write(data)
            self.wfile.flush()

        event({"choices": [{"delta": {"role": "assistant"}}]})
        for index in range(server.token_count):
            time.sleep(server.token_delay_s)
            event({"choices": [{"delta": {"content": f"t{index}"}}]})
        event({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        if include_usage:
            event(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": server.token_count,
                    },
                }
            )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
```

- [ ] **Step 2: Write failing CLI and end-to-end tests**

```python
# tests/test_cli.py
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import yaml

from llmprobe.cli import main
from tests.mock_openai import MockOpenAIHandler, MockOpenAIServer


@contextmanager
def running_server(*, reject_stream_options: bool = False) -> Iterator[MockOpenAIServer]:
    server = MockOpenAIServer(("127.0.0.1", 0), MockOpenAIHandler)  # type: ignore[arg-type]
    server.reject_stream_options = reject_stream_options
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def write_config(tmp_path: Path, port: int, *, name: str = "e2e") -> Path:
    config = {
        "run": {
            "name": name,
            "seed": 42,
            "output_dir": str(tmp_path / "results"),
            "repeats": 1,
            "cooldown_s": 0,
        },
        "workload": {
            "requests_per_cell": 4,
            "warmup_requests": 0,
            "concurrencies": [2],
            "input_lengths": [8],
            "output_lengths": [4],
            "stream_options": "auto",
        },
        "targets": [
            {
                "name": "mock",
                "backend": "openai-chat",
                "base_url": f"http://127.0.0.1:{port}/v1",
                "model": "mock-model",
            }
        ],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_dry_run_does_not_need_a_server(tmp_path: Path) -> None:
    config = write_config(tmp_path, 9)
    assert main(["run", str(config), "--dry-run"]) == 0


def test_invalid_config_returns_two(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("targets: []\n", encoding="utf-8")
    assert main(["validate", str(path)]) == 2


def test_end_to_end_chat_benchmark(tmp_path: Path, capsys) -> None:
    with running_server() as server:
        config = write_config(tmp_path, server.server_address[1])
        assert main(["run", str(config)]) == 0
    output_dirs = list((tmp_path / "results").iterdir())
    assert len(output_dirs) == 1
    output = output_dirs[0]
    requests = (output / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(requests) == 4
    row = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert row["capabilities"][0]["usage_stream_supported"] is True
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["summaries"][0]["completed"] == 4


def test_stream_options_fallback_end_to_end(tmp_path: Path) -> None:
    with running_server() as server:
        server.reject_stream_options = True
        config = write_config(tmp_path, server.server_address[1], name="fallback")
        assert main(["run", str(config)]) == 0
    output = next((tmp_path / "results").iterdir())
    row = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert row["capabilities"][0]["stream_options_sent"] is False
    assert row["capabilities"][0]["usage_stream_supported"] is False
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["summaries"][0]["completed"] == 4
```

- [ ] **Step 3: Run tests and verify failure**

```bash
export PYTHONPATH="src:/mnt/4t/code/total_bench/harbor/.venv/lib/python3.13/site-packages"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python3 -m pytest tests/test_cli.py -q
```

Expected: FAIL because `llmprobe.cli` does not exist.

- [ ] **Step 4: Implement the CLI and progress display**

Create `src/llmprobe/cli.py`:

```python
from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from llmprobe import __version__
from llmprobe.config import AppConfig, ConfigError, load_config
from llmprobe.models import Cell, CellSummary, RequestResult
from llmprobe.report import print_summary, write_report
from llmprobe.runner import BenchmarkRunner
from llmprobe.workload import build_cells


class RichProgressSink:
    def __init__(self, console: Console, total_requests: int) -> None:
        self.console = console
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
            transient=True,
        )
        self.task_id = self.progress.add_task("starting", total=total_requests)

    def __enter__(self) -> "RichProgressSink":
        self.progress.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.progress.stop()

    def cell_started(self, target: str, cell: Cell, repeat: int) -> None:
        self.progress.update(
            self.task_id,
            description=f"{target} {cell.id} repeat={repeat} c={cell.concurrency}",
        )

    def request_finished(self, result: RequestResult) -> None:
        self.progress.advance(self.task_id)

    def cell_finished(self, summary: CellSummary) -> None:
        self.console.print(
            f"[cyan]{summary.target}[/cyan] {summary.cell_id} r={summary.repeat}: "
            f"{summary.completed} ok / {summary.failed} failed, "
            f"{summary.request_throughput:.2f} req/s, "
            f"{summary.output_token_throughput or 0:.2f} out tok/s"
        )

    def warning(self, message: str) -> None:
        self.console.print(f"[yellow]warning:[/yellow] {message}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmprobe", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="execute a benchmark config")
    run_parser.add_argument("config", help="path to YAML config")
    run_parser.add_argument("--dry-run", action="store_true", help="show matrix and exit")

    validate_parser = subparsers.add_parser("validate", help="validate a benchmark config")
    validate_parser.add_argument("config", help="path to YAML config")
    return parser


def _formal_total(config: AppConfig) -> int:
    cells = build_cells(config.workload)
    return (
        len(cells)
        * len(config.targets)
        * config.run.repeats
        * config.workload.requests_per_cell
    )


def _print_plan(config: AppConfig, console: Console) -> None:
    cells = build_cells(config.workload)
    warmups = len(cells) * len(config.targets) * config.workload.warmup_requests
    console.print(f"targets: {', '.join(target.name for target in config.targets)}")
    console.print(f"cells: {len(cells)}")
    console.print(f"formal requests: {_formal_total(config)}")
    console.print(f"warmup requests: {warmups}")
    for target in config.targets:
        console.print(f"  - {target.name}: {target.backend} {target.base_url} model={target.model}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    console = Console()
    error_console = Console(stderr=True)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        error_console.print(f"[red]configuration error:[/red] {exc}")
        return 2

    if args.command == "validate":
        console.print(
            f"configuration valid: {len(config.targets)} targets, "
            f"{_formal_total(config)} formal requests"
        )
        return 0

    if args.dry_run:
        _print_plan(config, console)
        return 0

    total = _formal_total(config)
    runner: BenchmarkRunner | None = None
    try:
        with RichProgressSink(console, total) as progress:
            runner = BenchmarkRunner(config, progress=progress)
            try:
                result = asyncio.run(runner.run())
            except KeyboardInterrupt:
                result = runner.partial_result()
                output = write_report(result, config)
                console.print(f"[yellow]interrupted; partial results:[/yellow] {output}")
                return 130
    except Exception as exc:  # noqa: BLE001 - CLI must report failures cleanly
        error_console.print(f"[red]benchmark failed:[/red] {type(exc).__name__}: {exc}")
        return 1

    output = write_report(result, config)
    print_summary(result, console)
    console.print(f"results: {output}")
    return 0 if any(item.success for item in result.requests) else 1
```

Create `src/llmprobe/__main__.py`:

```python
from llmprobe.cli import main

raise SystemExit(main())
```

- [ ] **Step 5: Run tests and verify they pass**

```bash
export PYTHONPATH=".:src:/mnt/4t/code/total_bench/harbor/.venv/lib/python3.13/site-packages"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python3 -m pytest tests/test_cli.py -q
```

Expected: all tests pass. `PYTHONPATH` now includes `.` so `tests.mock_openai` imports correctly.

- [ ] **Step 6: Commit**

```bash
git add src/llmprobe/cli.py src/llmprobe/__main__.py tests/mock_openai.py tests/test_cli.py
git commit -m "feat: add CLI and end-to-end benchmark flow"
```

---

### Task 10: Documentation, Example Configs, and Full Verification

**Files:**
- Create: `README.md`
- Create: `configs/vllm-chat.yaml`
- Create: `configs/compare.yaml`
- Modify: `docs/superpowers/specs/2026-11-19-llm-service-bench-design.md` only if implementation reveals a factual mismatch.

**Interfaces:**
- Consumes: the complete CLI and report formats.
- Produces: installable, documented v1 with runnable examples.

- [ ] **Step 1: Write the README**

Create `README.md` with these sections and exact commands:

````markdown
# llmprobe

多 target、OpenAI-compatible 的 LLM 推理服务性能测试 CLI。用于比较本地 vLLM、llama.cpp、Qwen/DeepSeek 官方 API 和第三方网关的流式服务性能。

## 安装

```bash
uv sync --extra dev
```

需要 tokenizer 精确构造 prompt 时：

```bash
uv sync --extra dev --extra tokenizer
```

## 快速开始

```bash
cp configs/vllm-chat.yaml config.yaml
uv run llmprobe validate config.yaml
uv run llmprobe run config.yaml --dry-run
uv run llmprobe run config.yaml
```

结果写入 `results/<run_id>/`：

- `summary.csv`：每个 target/cell/repeat 一行
- `summary.json`：明细与跨 repeat 聚合
- `requests.jsonl`：请求级数据
- `summary.md`：可读报告
- `run.json`：配置能力、运行元数据
- `errors.log`：失败请求

## 指标

- **TTFT**：请求发送到第一个 token-bearing delta 的时间。
- **TPOT**：`(E2EL - TTFT) / (output_tokens - 1)`。
- **ITL**：相邻 token-bearing chunk 的间隔，属于事件级指标。
- **E2EL**：请求发送到最后一个 token-bearing delta。
- **Request throughput**：成功请求数 / benchmark duration。
- **Output token throughput**：成功请求输出 token 总数 / benchmark duration。
- **Total token throughput**：输入和输出 token 总数 / benchmark duration；输入 token 不可知时为 null。
- **Decode tok/s per stream**：`1000 / mean TPOT ms`。

## 配置要点

- `base_url` 通常应包含 `/v1`。
- `dataset: random` 使用 vLLM 风格确定性随机 token 序列生成 prompt。
- `concurrencies`、`input_lengths`、`output_lengths` 是扫描列表。
- `requests_per_cell` 必须不小于最大并发。
- `warmup_requests` 按 target × cell 执行，不计入结果。
- `stream_options: auto` 会用一个排除在正式结果外的 probe 探测 usage 支持。
- `max_tokens_field` 是 target 级配置，可选 `max_tokens` 或 `max_completion_tokens`。
- API key 通过环境变量设置，不要写入配置文件：

```bash
export DEEPSEEK_API_KEY=...
```

## 常见限制

- 公网 API 的 TTFT 包含网络延迟，适合同环境相对比较。
- 服务端可能一次返回多个 token，因此 ITL 是 chunk 级间隔；有 usage 时 TPOT 更准确。
- 无 usage 且无 tokenizer 时无法得到输入 token 数，total token throughput 为 null。
- 第一版只支持 closed-loop 固定并发，不支持固定 QPS。
- 不自动重试；429/5xx 会记录为失败。
- 极高并发下单进程客户端可能成为瓶颈，应观察 client queue time。

## 开发测试

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev
uv run pytest -q
uv run ruff check src tests
```
````

- [ ] **Step 2: Add example configurations**

Create `configs/vllm-chat.yaml`:

```yaml
run:
  name: vllm-local
  seed: 42
  output_dir: results
  repeats: 1
  cooldown_s: 2
  save_generated_text: false
  save_itl: false

workload:
  dataset: random
  requests_per_cell: 60
  warmup_requests: 3
  concurrencies: [1, 2, 4, 8]
  input_lengths: [128, 1024]
  output_lengths: [128]
  system_prompt: ""
  stream_options: auto

targets:
  - name: vllm-qwen
    backend: openai-chat
    base_url: http://127.0.0.1:8000/v1
    model: Qwen/Qwen3-8B
    timeout_s: 300
    max_tokens_field: max_tokens
    request_params:
      temperature: 0
      ignore_eos: true
```

Create `configs/compare.yaml`:

```yaml
run:
  name: vllm-vs-deepseek
  seed: 42
  output_dir: results
  repeats: 2
  cooldown_s: 5

workload:
  dataset: random
  requests_per_cell: 40
  warmup_requests: 2
  concurrencies: [1, 4, 8]
  input_lengths: [512]
  output_lengths: [128]
  stream_options: auto

targets:
  - name: vllm-qwen
    backend: openai-chat
    base_url: http://127.0.0.1:8000/v1
    model: Qwen/Qwen3-8B
    request_params:
      temperature: 0
      ignore_eos: true

  - name: deepseek
    backend: openai-chat
    base_url: https://api.deepseek.com/v1
    model: deepseek-chat
    api_key_env: DEEPSEEK_API_KEY
    timeout_s: 300
    request_params:
      temperature: 0
```

- [ ] **Step 3: Run the complete offline-compatible test suite**

```bash
export PYTHONPATH=".:src:/mnt/4t/code/total_bench/harbor/.venv/lib/python3.13/site-packages"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python3 -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 4: Verify bytecode compilation**

```bash
python3 -m compileall -q src tests
```

Expected: exit code 0 with no output.

- [ ] **Step 5: Verify the documented entry point**

```bash
export PYTHONPATH=".:src:/mnt/4t/code/total_bench/harbor/.venv/lib/python3.13/site-packages"
python3 -m llmprobe --version
python3 -m llmprobe run configs/vllm-chat.yaml --dry-run
```

Expected: version `0.1.0`, then a target/cell/request-count summary without network access.

- [ ] **Step 6: Commit**

```bash
git add README.md configs/vllm-chat.yaml configs/compare.yaml
git commit -m "docs: add usage guide and example configs"
```

- [ ] **Step 7: Inspect final repository state**

```bash
git status --short
git log --oneline -10
```

Expected: clean working tree and all task commits present.

