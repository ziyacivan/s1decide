"""Text sanitisation at ingestion.

`json.dumps(..., ensure_ascii=False)` writes U+0085 and friends out raw. `str.splitlines()`,
most JavaScript, and several JSONL readers treat them as line terminators, so one of them inside
a state silently splits a row in two. Four reached `train.jsonl` from MMLU question text before
anyone looked, and the only reason anyone looked was that a careless edit swapped one reader for
the other and the JSON stopped parsing.
"""

from __future__ import annotations

import json

import pytest
from data.build.pipeline import clean_text

from s1decide.tasks import repo_root

BREAKERS = ("", " ", " ", "\v", "\f", "\x1c", "\x1d", "\x1e")

NEWLINE = "\n"


@pytest.mark.parametrize("char", BREAKERS)
def test_every_line_breaking_character_becomes_a_space(char: str) -> None:
    cleaned = clean_text(f"before{char}after")
    assert cleaned == "before after"
    assert len(cleaned.splitlines()) == 1
    assert len(cleaned.split(NEWLINE)) == 1


def test_ordinary_text_is_untouched_apart_from_whitespace() -> None:
    assert clean_text("  a  b  ") == "a b"
    assert clean_text("Which intent does this request have?") == (
        "Which intent does this request have?"
    )


def test_non_latin_text_survives_cleaning() -> None:
    """Japanese and Turkish states must not be mangled by a whitespace pass."""
    assert clean_text("今週は午前五時に起こして") == "今週は午前五時に起こして"
    assert clean_text("beni cuma günü sabah dokuzda uyandır") == (
        "beni cuma günü sabah dokuzda uyandır"
    )


def test_a_real_newline_is_also_collapsed() -> None:
    assert clean_text(f"a{NEWLINE}b") == "a b"


def test_cleaning_is_idempotent() -> None:
    once = clean_text("a b  c")
    assert clean_text(once) == once


@pytest.mark.parametrize("split", ["train", "val", "test", "eval"])
def test_the_written_corpus_parses_the_same_under_both_readers(split: str) -> None:
    """The regression itself: a reader that splits on Unicode boundaries sees the same rows."""
    path = repo_root() / "data" / "processed" / f"{split}.jsonl"
    if not path.is_file():
        pytest.skip("no built corpus; run `uv run task data`")
    text = path.read_text(encoding="utf-8")
    by_newline = [line for line in text.split(NEWLINE) if line.strip()]
    by_splitlines = [line for line in text.splitlines() if line.strip()]
    assert by_newline == by_splitlines, (
        f"{split}.jsonl has {len(by_splitlines) - len(by_newline)} rows that a splitlines-based "
        "reader would break apart"
    )
    json.loads(by_newline[0])
    json.loads(by_newline[-1])


@pytest.mark.parametrize("split", ["train", "val", "test", "eval"])
def test_no_written_row_contains_a_line_breaking_character(split: str) -> None:
    path = repo_root() / "data" / "processed" / f"{split}.jsonl"
    if not path.is_file():
        pytest.skip("no built corpus; run `uv run task data`")
    text = path.read_text(encoding="utf-8")
    for char in BREAKERS:
        assert char not in text, f"{split}.jsonl contains U+{ord(char):04X}"
