from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from llmbench.config import ConfigError

_BUNDLED_TOKENIZER_ROOT = Path(__file__).with_name("tokenizers")
_BUNDLED_MANIFEST = _BUNDLED_TOKENIZER_ROOT / "manifest.json"
_BUNDLED_PREFIXES = ("bundled:", "builtin:")


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
        encoding = self._tokenizer.encode(text, add_special_tokens=False)  # type: ignore[attr-defined]
        return list(encoding.ids)

    def decode_ids(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(  # type: ignore[attr-defined,no-any-return]
            token_ids, skip_special_tokens=False
        )

    def truncate(self, text: str, max_tokens: int) -> str:
        encoding = self._tokenizer.encode(text, add_special_tokens=False)  # type: ignore[attr-defined]
        if len(encoding.ids) <= max_tokens:
            return text
        return self.decode_ids(encoding.ids[:max_tokens])


@lru_cache(maxsize=1)
def _bundled_manifest() -> dict:
    if not _BUNDLED_MANIFEST.is_file():
        return {"tokenizers": {}}
    try:
        data = json.loads(_BUNDLED_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"failed to read bundled tokenizer manifest: {exc}") from exc
    entries = data.get("tokenizers")
    if not isinstance(entries, dict):
        raise ConfigError("bundled tokenizer manifest is missing tokenizers")
    return data


def available_bundled_tokenizers() -> tuple[str, ...]:
    return tuple(sorted(_bundled_manifest()["tokenizers"]))


def _split_bundled_spec(spec: str) -> tuple[str, bool]:
    for prefix in _BUNDLED_PREFIXES:
        if spec.startswith(prefix):
            return spec[len(prefix) :], True
    return spec, False


def _bundled_tokenizer_path(spec: str) -> Path | None:
    alias, explicit = _split_bundled_spec(spec)
    if (
        not alias
        or alias in {".", ".."}
        or Path(alias).name != alias
        or "/" in alias
        or "\\" in alias
    ):
        if explicit:
            raise ConfigError(f"invalid bundled tokenizer alias {alias!r}")
        return None
    entries = _bundled_manifest()["tokenizers"]
    if alias not in entries:
        if explicit:
            available = ", ".join(available_bundled_tokenizers()) or "(none)"
            raise ConfigError(
                f"unknown bundled tokenizer {alias!r}; available: {available}"
            )
        return None
    entry = entries[alias]
    if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
        raise ConfigError(f"invalid manifest entry for bundled tokenizer {alias!r}")
    path = _BUNDLED_TOKENIZER_ROOT / entry["file"]
    root = _BUNDLED_TOKENIZER_ROOT.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ConfigError(f"bundled tokenizer path escapes tokenizers root: {alias!r}")
    if not resolved.is_file():
        raise ConfigError(f"bundled tokenizer file is missing: {alias!r}")
    return resolved


def canonical_tokenizer_key(spec: str | None) -> str | None:
    if not spec:
        return None
    alias, explicit = _split_bundled_spec(spec)
    if explicit or alias in available_bundled_tokenizers():
        _bundled_tokenizer_path(spec)
        return f"bundled:{alias}"
    path = Path(spec)
    if path.is_file():
        return str(path.resolve())
    if path.is_dir():
        return str((path / "tokenizer.json").resolve())
    return f"hf:{spec.strip()}"


def load_tokenizer(spec: str | None) -> TextTokenizer | None:
    if not spec:
        return None
    bundled = _bundled_tokenizer_path(spec)
    path = Path(spec)
    local_file: Path | None = None
    if bundled is not None:
        local_file = bundled
    elif path.is_file():
        local_file = path
    elif path.is_dir():
        candidate = path / "tokenizer.json"
        if not candidate.is_file():
            raise ConfigError(
                f"tokenizer directory {spec!r} does not contain tokenizer.json"
            )
        local_file = candidate
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise ConfigError(
            "tokenizer support requires the optional dependencies: "
            "pip install 'llmbench[tokenizer]'"
        ) from exc
    try:
        if local_file is not None:
            tokenizer = Tokenizer.from_file(str(local_file))
        else:
            tokenizer = Tokenizer.from_pretrained(spec)
    except Exception as exc:
        raise ConfigError(
            f"failed to load tokenizer {spec!r}: {type(exc).__name__}: {exc}"
        ) from exc
    return LocalTokenizer(tokenizer)
