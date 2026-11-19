from __future__ import annotations

from pathlib import Path
from typing import Protocol

from llmbench.config import ConfigError

_BUNDLED_TOKENIZER_ROOT = Path(__file__).with_name("tokenizers")
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


def available_bundled_tokenizers() -> tuple[str, ...]:
    if not _BUNDLED_TOKENIZER_ROOT.is_dir():
        return ()
    return tuple(
        sorted(
            child.name
            for child in _BUNDLED_TOKENIZER_ROOT.iterdir()
            if child.is_dir() and (child / "tokenizer.json").is_file()
        )
    )


def _bundled_tokenizer_path(spec: str) -> Path | None:
    alias = spec
    explicit = False
    for prefix in _BUNDLED_PREFIXES:
        if spec.startswith(prefix):
            alias = spec[len(prefix) :]
            explicit = True
            break
    if not alias or Path(alias).name != alias:
        if explicit:
            raise ConfigError(f"invalid bundled tokenizer alias {alias!r}")
        return None
    path = _BUNDLED_TOKENIZER_ROOT / alias / "tokenizer.json"
    if path.is_file():
        return path
    if explicit:
        available = ", ".join(available_bundled_tokenizers()) or "(none)"
        raise ConfigError(
            f"unknown bundled tokenizer {alias!r}; available: {available}"
        )
    return None


def load_tokenizer(spec: str | None) -> TextTokenizer | None:
    if not spec:
        return None
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise ConfigError(
            "tokenizer support requires the optional dependencies: "
            "pip install 'llmbench[tokenizer]'"
        ) from exc
    try:
        path = Path(spec)
        if path.is_file():
            tokenizer = Tokenizer.from_file(str(path))
        elif path.is_dir():
            tokenizer_file = path / "tokenizer.json"
            if not tokenizer_file.is_file():
                raise ConfigError(
                    f"tokenizer directory {spec!r} does not contain tokenizer.json"
                )
            tokenizer = Tokenizer.from_file(str(tokenizer_file))
        elif (bundled := _bundled_tokenizer_path(spec)) is not None:
            tokenizer = Tokenizer.from_file(str(bundled))
        else:
            tokenizer = Tokenizer.from_pretrained(spec)
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(
            f"failed to load tokenizer {spec!r}: {type(exc).__name__}: {exc}"
        ) from exc
    return LocalTokenizer(tokenizer)
