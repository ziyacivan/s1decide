"""Validation and derived-value tests for the three primitives."""

from __future__ import annotations

import dataclasses

import pytest

from s1decide.primitives import (
    MAX_OPTIONS,
    MIN_OPTIONS,
    Choice,
    Noul,
    Question,
    Result,
    Score,
)

TONE = Choice(instructions="What is the customer's tone?", options=("calm", "frustrated", "angry"))
URGENCY = Score(
    instructions="How urgent is this ticket?", levels=("can wait", "this week", "today")
)
BILLING = Noul(instructions="This ticket is about billing.")


# --- Choice ------------------------------------------------------------------


def test_choice_keeps_option_order() -> None:
    assert TONE.options == ("calm", "frustrated", "angry")
    assert TONE.labels == TONE.options
    assert TONE.qtype == "choice"


def test_choice_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        TONE.options = ("a", "b")  # type: ignore[misc]


def test_choice_accepts_a_list_and_normalises_to_a_tuple() -> None:
    assert Choice(instructions="q", options=["a", "b"]).options == ("a", "b")


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ((), "at least 2"),
        (("only",), "at least 2"),
        (("a", "a"), "must be unique"),
        (("a", "A"), "ignoring case"),
        (("a", ""), "empty entries"),
        (("a", "   "), "empty entries"),
        (("a", " b"), "padded with whitespace"),
        (("a", "b "), "padded with whitespace"),
        (("a", "b\nc"), "single-line"),
    ],
)
def test_choice_rejects_bad_options(options: tuple[str, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Choice(instructions="q", options=options)


def test_choice_rejects_too_many_options() -> None:
    too_many = tuple(f"option-{i}" for i in range(MAX_OPTIONS + 1))
    with pytest.raises(ValueError, match="at most 255"):
        Choice(instructions="q", options=too_many)


def test_choice_accepts_exactly_the_bounds() -> None:
    assert (
        len(Choice(instructions="q", options=tuple(f"o{i}" for i in range(MIN_OPTIONS))).options)
        == 2
    )
    assert (
        len(Choice(instructions="q", options=tuple(f"o{i}" for i in range(MAX_OPTIONS))).options)
        == 255
    )


def test_choice_rejects_non_string_options() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        Choice(instructions="q", options=("a", 2))  # type: ignore[arg-type]


@pytest.mark.parametrize("instructions", ["", "   ", "\n"])
def test_empty_instructions_are_rejected(instructions: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        Choice(instructions=instructions, options=("a", "b"))


def test_non_string_instructions_are_rejected() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        Choice(instructions=None, options=("a", "b"))  # type: ignore[arg-type]


# --- Score -------------------------------------------------------------------


def test_score_preserves_level_order() -> None:
    assert URGENCY.levels == ("can wait", "this week", "today")
    assert URGENCY.labels == URGENCY.levels
    assert URGENCY.qtype == "score"


def test_score_applies_the_same_validation_as_choice() -> None:
    with pytest.raises(ValueError, match="levels must be unique"):
        Score(instructions="q", levels=("low", "low"))
    with pytest.raises(ValueError, match="levels must have at least 2"):
        Score(instructions="q", levels=("only",))


# --- Noul --------------------------------------------------------------------


def test_noul_labels_are_fixed_with_yes_at_index_one() -> None:
    assert BILLING.labels == ("no", "yes")
    assert BILLING.labels[1] == "yes"
    assert BILLING.qtype == "noul"


# --- Question ----------------------------------------------------------------


def test_question_delegates_to_its_spec() -> None:
    q = Question(name="tone", spec=TONE)
    assert q.qtype == "choice"
    assert q.labels == TONE.options
    assert q.instructions == TONE.instructions


@pytest.mark.parametrize("name", ["", "   ", "has\nnewline"])
def test_question_rejects_bad_names(name: str) -> None:
    with pytest.raises(ValueError):
        Question(name=name, spec=TONE)


def test_question_rejects_a_non_spec() -> None:
    with pytest.raises(TypeError, match="must be Choice, Score or Noul"):
        Question(name="q", spec="not a spec")  # type: ignore[arg-type]


# --- Result ------------------------------------------------------------------


def choice_result(probs: tuple[float, ...]) -> Result:
    return Result(name="tone", qtype="choice", options=TONE.options, probabilities=probs)


def test_result_exposes_distribution_and_confidence() -> None:
    r = choice_result((0.1, 0.7, 0.2))
    assert r.distribution == {"calm": 0.1, "frustrated": 0.7, "angry": 0.2}
    assert r.choice == "frustrated"
    assert r.confidence == pytest.approx(0.7)
    assert r.argmax_index == 1


def test_result_breaks_ties_towards_the_lowest_index() -> None:
    assert choice_result((0.5, 0.5, 0.0)).argmax_index == 0


@pytest.mark.parametrize(
    ("probs", "message"),
    [
        ((0.5, 0.5), "3 options but 2 probabilities"),
        ((0.3, 0.3, 0.3), "must sum to 1"),
        ((1.5, -0.5, 0.0), r"must lie in \[0, 1\]"),
        ((float("nan"), 0.5, 0.5), "must be finite"),
        ((float("inf"), 0.0, 0.0), "must be finite"),
    ],
)
def test_result_rejects_invalid_probabilities(probs: tuple[float, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        choice_result(probs)


def test_result_accepts_probabilities_within_float_tolerance() -> None:
    third = 1.0 / 3.0
    choice_result((third, third, third))


def test_score_result_reports_expected_level() -> None:
    r = Result(name="urgency", qtype="score", options=URGENCY.levels, probabilities=(0.2, 0.3, 0.5))
    assert r.score == "today"
    assert r.expected_level == pytest.approx(0.2 * 0 + 0.3 * 1 + 0.5 * 2)


def test_expected_level_exposes_a_bimodal_distribution_the_argmax_hides() -> None:
    """A 50/0/50 split has a confident-looking argmax but a mean that sits in the middle."""
    r = Result(name="urgency", qtype="score", options=URGENCY.levels, probabilities=(0.5, 0.0, 0.5))
    assert r.score == "can wait"
    assert r.expected_level == pytest.approx(1.0)


def test_noul_result_reports_probability_of_yes() -> None:
    r = Result(name="billing", qtype="noul", options=("no", "yes"), probabilities=(0.25, 0.75))
    assert r.noul == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("qtype", "attribute"),
    [
        ("choice", "score"),
        ("choice", "expected_level"),
        ("choice", "noul"),
        ("score", "choice"),
        ("score", "noul"),
        ("noul", "choice"),
        ("noul", "score"),
        ("noul", "expected_level"),
    ],
)
def test_result_refuses_accessors_from_the_wrong_primitive(qtype: str, attribute: str) -> None:
    r = Result(name="q", qtype=qtype, options=("no", "yes"), probabilities=(0.5, 0.5))
    with pytest.raises(TypeError, match="not a"):
        getattr(r, attribute)
