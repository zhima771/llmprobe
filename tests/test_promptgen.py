import random
import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest

from llmprobe.config import CellSpec, ConfigError, WorkloadConfig
from llmprobe.promptgen import generate_prompt
from llmprobe.tokenization import load_tokenizer
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


class NonRoundTripTokenizer:
    @property
    def is_tokenizer(self) -> bool:
        return True

    @property
    def vocab_size(self) -> int:
        return 100

    def decode_ids(self, token_ids: list[int]) -> str:
        return " ".join(f"a{token_id}" for token_id in token_ids)

    def encode_ids(self, text: str) -> list[int]:
        ids = [int(word[1:]) for word in text.split() if word.startswith("a")]
        return ids + [0] if ids else []

    def count(self, text: str) -> int:
        return len(self.encode_ids(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])


def workload_config(
    *,
    input_length: int,
    output_length: int,
    concurrency: int = 1,
    system_prompt: str = "",
) -> WorkloadConfig:
    return WorkloadConfig(
        cells=(CellSpec(input_length, output_length, concurrency),),
        system_prompt=system_prompt,
    )


def install_fake_tokenizers(monkeypatch: pytest.MonkeyPatch, tokenizer_class: type) -> None:
    module = types.ModuleType("tokenizers")
    module.Tokenizer = tokenizer_class  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tokenizers", module)


def test_prompt_is_deterministic_and_varies_by_request() -> None:
    workload = workload_config(
        concurrency=2, input_length=64, output_length=8, system_prompt="be terse"
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
    workload = workload_config(
        input_length=40,
        output_length=8,
        system_prompt="system instructions",
    )
    cell = build_cells(workload)[0]
    tokenizer = WordTokenizer()
    prompt = generate_prompt(9, cell, 0, 3, workload, tokenizer)
    assert prompt.tokenizer_used is True
    assert tokenizer.count(prompt.system) + tokenizer.count(prompt.user) == 40
    assert prompt.estimated_tokens == 40
    assert prompt.user.startswith("t")

    user_ids = tokenizer.encode_ids(prompt.user)
    assert len(user_ids) == 38
    assert all(0 <= token_id < tokenizer.vocab_size for token_id in user_ids)


def test_prompt_set_changes_by_repeat() -> None:
    workload = workload_config(
        input_length=32, output_length=4, system_prompt="sys"
    )
    cell = build_cells(workload)[0]
    tokenizer = WordTokenizer()
    first = generate_prompt(11, cell, 0, 0, workload, tokenizer)
    second = generate_prompt(11, cell, 1, 0, workload, tokenizer)
    assert first.user != second.user


def test_prompt_is_invariant_to_global_random_state() -> None:
    workload = workload_config(
        input_length=48, output_length=4, system_prompt="sys"
    )
    cell = build_cells(workload)[0]
    tokenizer = WordTokenizer()
    plain_first = generate_prompt(21, cell, 0, 2, workload, None)
    tokenized_first = generate_prompt(21, cell, 0, 2, workload, tokenizer)

    original_state = random.getstate()
    try:
        random.seed(999)
        for _ in range(20):
            random.random()

        assert generate_prompt(21, cell, 0, 2, workload, None) == plain_first
        assert generate_prompt(21, cell, 0, 2, workload, tokenizer) == tokenized_first
    finally:
        random.setstate(original_state)


def test_non_round_trip_tokenizer_returns_best_effort_prompt() -> None:
    workload = workload_config(input_length=10, output_length=4)
    cell = build_cells(workload)[0]
    tokenizer = NonRoundTripTokenizer()

    first = generate_prompt(5, cell, 0, 1, workload, tokenizer)
    second = generate_prompt(5, cell, 0, 1, workload, tokenizer)

    actual_tokens = tokenizer.count(first.system) + len(tokenizer.encode_ids(first.user))
    assert first == second
    assert first.target_tokens == 10
    assert actual_tokens == 11
    assert first.estimated_tokens == actual_tokens
    assert first.tokenizer_used is True


def test_system_prompt_must_fit_in_content_token_budget() -> None:
    workload = workload_config(
        input_length=2, output_length=4, system_prompt="one two"
    )
    cell = build_cells(workload)[0]
    with pytest.raises(ConfigError, match="too long"):
        generate_prompt(1, cell, 0, 0, workload, WordTokenizer())


def test_completions_text_concatenates_system_and_user() -> None:
    workload = workload_config(
        input_length=8, output_length=2, system_prompt="sys"
    )
    cell = build_cells(workload)[0]
    prompt = generate_prompt(1, cell, 0, 0, workload, None)
    assert prompt.completions_text.startswith("sys\n\n")


def test_load_tokenizer_without_spec_returns_none() -> None:
    assert load_tokenizer(None) is None


def test_load_tokenizer_uses_from_file_for_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RecordingTokenizer:
        calls: ClassVar[list[tuple[str, str]]] = []

        @classmethod
        def from_file(cls, spec: str) -> object:
            cls.calls.append(("file", spec))
            return object()

        @classmethod
        def from_pretrained(cls, spec: str) -> object:
            cls.calls.append(("pretrained", spec))
            return object()

    install_fake_tokenizers(monkeypatch, RecordingTokenizer)
    spec = tmp_path / "tokenizer.model"
    spec.write_text("{}", encoding="utf-8")

    tokenizer = load_tokenizer(str(spec))

    assert tokenizer is not None
    assert tokenizer.is_tokenizer is True
    assert RecordingTokenizer.calls == [("file", str(spec))]


def test_load_tokenizer_uses_from_pretrained_for_non_file_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingTokenizer:
        calls: ClassVar[list[tuple[str, str]]] = []

        @classmethod
        def from_file(cls, spec: str) -> object:
            cls.calls.append(("file", spec))
            return object()

        @classmethod
        def from_pretrained(cls, spec: str) -> object:
            cls.calls.append(("pretrained", spec))
            return object()

    install_fake_tokenizers(monkeypatch, RecordingTokenizer)
    spec = "missing-tokenizer.json"

    tokenizer = load_tokenizer(spec)

    assert tokenizer is not None
    assert RecordingTokenizer.calls == [("pretrained", spec)]


def test_load_tokenizer_wraps_loading_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenTokenizer:
        @classmethod
        def from_pretrained(cls, spec: str) -> object:
            raise RuntimeError("bad tokenizer payload")

    install_fake_tokenizers(monkeypatch, BrokenTokenizer)
    spec = "broken/tokenizer"

    with pytest.raises(ConfigError) as exc_info:
        load_tokenizer(spec)

    message = str(exc_info.value)
    assert spec in message
    assert "RuntimeError" in message
    assert "bad tokenizer payload" in message
