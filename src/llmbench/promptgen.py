from __future__ import annotations

import math
import random

from llmbench.config import ConfigError, WorkloadConfig
from llmbench.models import Cell, GeneratedPrompt
from llmbench.tokenization import TextTokenizer
from llmbench.workload import cell_seed

_WORDS = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "network",
    "kernel",
    "tensor",
    "token",
    "stream",
    "prompt",
    "decode",
    "prefill",
    "latency",
    "throughput",
    "request",
    "response",
    "cache",
    "queue",
    "server",
    "client",
    "model",
    "inference",
    "benchmark",
    "measurement",
    "interval",
    "concurrency",
    "workload",
    "synthetic",
    "data",
]


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
        for _ in range(10):
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
        raise ConfigError(
            f"system_prompt is too long for target input length {cell.input_length}"
        )
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
