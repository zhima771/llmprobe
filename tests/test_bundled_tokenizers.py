from pathlib import Path

import pytest

from llmbench.config import ConfigError
from llmbench.tokenization import available_bundled_tokenizers, load_tokenizer


def test_bundled_tokenizer_aliases_are_available() -> None:
    aliases = available_bundled_tokenizers()
    assert "qwen3.8-flash-next" in aliases
    assert "qwen3.6-35b" in aliases
    assert "deepseek-v4.1-flash" in aliases
    assert "glm-5.3" in aliases


def test_bundled_alias_loads_a_real_tokenizer() -> None:
    pytest.importorskip("tokenizers")
    tokenizer = load_tokenizer("qwen3.6-35b")
    assert tokenizer is not None
    assert tokenizer.vocab_size > 100_000
    assert tokenizer.count("hello world") >= 1


def test_local_model_directory_resolves_tokenizer_json(tmp_path: Path) -> None:
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    tokenizer = Tokenizer(
        models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(model_dir / "tokenizer.json"))
    loaded = load_tokenizer(str(model_dir))
    assert loaded is not None
    assert loaded.count("hello world") == 2


def test_unknown_bundled_alias_lists_available_aliases() -> None:
    with pytest.raises(ConfigError, match="qwen3.6-35b"):
        load_tokenizer("bundled:not-a-real-tokenizer")
