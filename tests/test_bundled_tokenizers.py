import hashlib
import json
import sys
from pathlib import Path

import pytest

from llmprobe.config import ConfigError
from llmprobe.tokenization import (
    available_bundled_tokenizers,
    canonical_tokenizer_key,
    load_tokenizer,
)


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


def test_bundled_alias_rejects_path_traversal() -> None:
    with pytest.raises(ConfigError, match="invalid bundled tokenizer alias"):
        load_tokenizer("bundled:..")
    with pytest.raises(ConfigError, match="invalid bundled tokenizer alias"):
        load_tokenizer("bundled:../tokenizer")


def test_unknown_explicit_alias_fails_before_optional_import(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "tokenizers", None)
    with pytest.raises(ConfigError, match="unknown bundled tokenizer"):
        load_tokenizer("bundled:does-not-exist")


def test_equivalent_alias_spellings_have_same_canonical_key() -> None:
    assert canonical_tokenizer_key("qwen3.6-35b") == canonical_tokenizer_key(
        "bundled:qwen3.6-35b"
    )


def test_manifest_metadata_and_license_files_are_consistent() -> None:
    root = Path(__file__).parents[1] / "src" / "llmprobe" / "tokenizers"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["tokenizers"]) == set(available_bundled_tokenizers())
    for entry in manifest["tokenizers"].values():
        data = (root / entry["file"]).read_bytes()
        assert len(data) == entry["size_bytes"]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        assert entry["source_revision"]
        assert entry["license_spdx"]
        assert entry["license_url"]
        assert (root / entry["license_file"]).is_file()
