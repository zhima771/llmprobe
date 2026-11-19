from __future__ import annotations

from typing import Protocol

from llmbench.config import ConfigError


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
    if "/" in spec and not spec.endswith(".json"):
        tokenizer = Tokenizer.from_pretrained(spec)
    else:
        tokenizer = Tokenizer.from_file(spec)
    return LocalTokenizer(tokenizer)
