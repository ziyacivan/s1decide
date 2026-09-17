"""Tests for option-to-token labelling.

The tokenizer-backed tests are the load-bearing ones: ADR 0001 is only valid if every
answer label really is a single token for the base model. They skip without the
tokenizer in the local cache, so the CPU suite still runs on a fresh clone.
"""

from __future__ import annotations

import pytest

from s1decide.primitives import Choice, Noul, Question, Score
from s1decide.tokens import (
    LETTER_LABELS,
    MAX_SINGLE_TOKEN_OPTIONS,
    NOUL_LABELS,
    allowed_token_ids,
    check_single_token,
    labels_for_count,
    labels_for_question,
    token_id,
)

ALL_LABELS = (*LETTER_LABELS, *NOUL_LABELS)


# --- label allocation --------------------------------------------------------


def test_letter_labels_are_the_alphabet() -> None:
    assert tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ") == LETTER_LABELS
    assert MAX_SINGLE_TOKEN_OPTIONS == 26


def test_labels_for_count_returns_a_prefix_of_the_alphabet() -> None:
    assert labels_for_count(2) == ("A", "B")
    assert labels_for_count(26)[-1] == "Z"


@pytest.mark.parametrize("count", [0, 1, -3])
def test_labels_for_count_rejects_degenerate_counts(count: int) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        labels_for_count(count)


def test_labels_for_count_refuses_high_cardinality_rather_than_truncating() -> None:
    with pytest.raises(NotImplementedError, match="two-stage"):
        labels_for_count(27)


def test_labels_for_question_dispatches_on_type() -> None:
    choice = Question(name="c", spec=Choice(instructions="q", options=("x", "y", "z")))
    score = Question(name="s", spec=Score(instructions="q", levels=("lo", "hi")))
    noul = Question(name="n", spec=Noul(instructions="q"))
    assert labels_for_question(choice) == ("A", "B", "C")
    assert labels_for_question(score) == ("A", "B")
    assert labels_for_question(noul) == ("no", "yes")


# --- tokenizer-backed: the property ADR 0001 depends on ----------------------


@pytest.mark.tokenizer
def test_every_label_is_a_single_token_bare_and_with_a_leading_space(tokenizer) -> None:
    """A prompt ending in a newline and one ending in a space tokenize differently."""
    assert check_single_token(tokenizer, ALL_LABELS) == {}


@pytest.mark.tokenizer
def test_digits_are_not_usable_as_labels(tokenizer) -> None:
    """The measured reason letters are the alphabet: ' 0' encodes to two tokens.

    If this ever starts passing, the 26-option ceiling can be raised to 36 and ADR 0001
    should be revisited.
    """
    offenders = check_single_token(tokenizer, tuple("0123456789"))
    assert set(offenders) == set("0123456789")


@pytest.mark.tokenizer
def test_labels_map_to_distinct_token_ids(tokenizer) -> None:
    ids = allowed_token_ids(tokenizer, LETTER_LABELS)
    assert len(set(ids)) == len(LETTER_LABELS)


@pytest.mark.tokenizer
def test_bare_and_spaced_labels_are_different_tokens(tokenizer) -> None:
    """Sanity check that `leading_space` actually selects a different id."""
    assert token_id(tokenizer, "A") != token_id(tokenizer, "A", leading_space=True)


@pytest.mark.tokenizer
def test_token_id_rejects_a_multi_token_label(tokenizer) -> None:
    with pytest.raises(ValueError, match="not a single token"):
        token_id(tokenizer, "definitely not one token")


@pytest.mark.tokenizer
def test_noul_labels_are_single_tokens(tokenizer) -> None:
    assert check_single_token(tokenizer, NOUL_LABELS) == {}


# --- tokenizer-free behaviour ------------------------------------------------


class FakeTokenizer:
    """Encodes each character as its ordinal, so token counts are predictable."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]


class CollidingTokenizer:
    """Maps everything to the same id, to exercise the collision guard."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [7]


def test_token_id_reports_the_offending_encoding() -> None:
    with pytest.raises(ValueError, match=r"is not a single token, encodes to \["):
        token_id(FakeTokenizer(), "AB")


def test_allowed_token_ids_rejects_colliding_labels() -> None:
    with pytest.raises(ValueError, match="collide on token ids"):
        allowed_token_ids(CollidingTokenizer(), ("A", "B"))


def test_check_single_token_reports_both_encodings() -> None:
    offenders = check_single_token(FakeTokenizer(), ("A", "AB"))
    assert "A" in offenders  # bare is 1 token, but ' A' is 2 under this fake
    assert offenders["AB"]["bare"] == [ord("A"), ord("B")]
