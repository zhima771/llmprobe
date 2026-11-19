import pytest

from llmbench.config import ConfigError, WorkloadConfig
from llmbench.promptgen import generate_prompt
from llmbench.tokenization import load_tokenizer
from llmbench.workload import build_cells


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

    user_ids = tokenizer.encode_ids(prompt.user)
    assert len(user_ids) == 38
    assert all(0 <= token_id < tokenizer.vocab_size for token_id in user_ids)


def test_prompt_set_changes_by_repeat() -> None:
    workload = WorkloadConfig(
        concurrencies=(1,), input_lengths=(32,), output_lengths=(4,), system_prompt="sys"
    )
    cell = build_cells(workload)[0]
    tokenizer = WordTokenizer()
    first = generate_prompt(11, cell, 0, 0, workload, tokenizer)
    second = generate_prompt(11, cell, 1, 0, workload, tokenizer)
    assert first.user != second.user


def test_system_prompt_must_fit_in_content_token_budget() -> None:
    workload = WorkloadConfig(
        concurrencies=(1,), input_lengths=(2,), output_lengths=(4,), system_prompt="one two"
    )
    cell = build_cells(workload)[0]
    with pytest.raises(ConfigError, match="too long"):
        generate_prompt(1, cell, 0, 0, workload, WordTokenizer())


def test_completions_text_concatenates_system_and_user() -> None:
    workload = WorkloadConfig(
        concurrencies=(1,), input_lengths=(8,), output_lengths=(2,), system_prompt="sys"
    )
    cell = build_cells(workload)[0]
    prompt = generate_prompt(1, cell, 0, 0, workload, None)
    assert prompt.completions_text.startswith("sys\n\n")


def test_load_tokenizer_without_spec_returns_none() -> None:
    assert load_tokenizer(None) is None
