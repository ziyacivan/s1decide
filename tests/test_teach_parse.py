"""Parsing a teacher's committed level.

The first version of this parser required a line start before `FINAL:` and a word boundary after
the digit. Magistral emits `FINAL: 1FINAL: 1` — the marker twice, run together — which satisfies
neither, so a pilot where the model answered every single time scored 0 of 100 committed. That
is the most expensive shape of bug available here: it looks exactly like a model failure, and it
would have disqualified a working teacher on a number that was entirely our own fault.
"""

from __future__ import annotations

import pytest
from data.build.teach import LEVELS, RUBRIC, build_prompt, parse_level, prompt_hash, trace_stats

NL = chr(10)


@pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
def test_a_plain_answer_parses_to_a_zero_based_index(level: int) -> None:
    assert parse_level(f"reasoning...{NL}FINAL: {level}") == level - 1


def test_the_marker_repeated_without_a_separator_still_parses() -> None:
    """The exact string that broke the first parser."""
    assert parse_level(f"...therefore level 1.{NL}FINAL: 1FINAL: 1") == 0


def test_the_marker_at_the_very_start_parses() -> None:
    assert parse_level("FINAL: 4") == 3


def test_the_marker_mid_line_parses() -> None:
    assert parse_level("So my answer is FINAL: 3 and that is that") == 2


def test_the_last_marker_wins() -> None:
    """A model that reconsiders mid-trace is taken at its final word."""
    assert parse_level(f"FINAL: 2{NL}wait, no.{NL}FINAL: 5") == 4


def test_a_trace_that_never_commits_is_none() -> None:
    assert parse_level("I think it is probably around level 3 or so, hard to say") is None


def test_a_truncated_trace_is_none_rather_than_a_guess() -> None:
    """A dropped row is the honest outcome; inventing a label poisons the hardest cases."""
    assert parse_level("...comparing the options, the best fit is level 2 - slight: a pass") is None


def test_empty_output_is_none() -> None:
    assert parse_level("") is None
    assert parse_level(None) is None


def test_a_two_digit_number_is_not_read_as_its_first_digit() -> None:
    assert parse_level("FINAL: 12") is None


def test_an_out_of_range_level_is_not_accepted() -> None:
    assert parse_level("FINAL: 7") is None
    assert parse_level("FINAL: 0") is None


def test_the_marker_is_case_insensitive() -> None:
    assert parse_level("final: 2") == 1


@pytest.mark.parametrize("level", range(len(LEVELS)))
def test_every_parsed_level_indexes_the_level_names(level: int) -> None:
    assert LEVELS[level]


# --- the prompt ------------------------------------------------------------------


def test_the_prompt_carries_the_state_question_and_every_rubric_level() -> None:
    prompt = build_prompt("the customer is furious", "How strong is the frustration?")
    assert "the customer is furious" in prompt
    assert "How strong is the frustration?" in prompt
    for level in RUBRIC:
        assert level in prompt
    assert prompt.rstrip().endswith("FINAL: <number 1-5>")


def test_the_prompt_is_identical_for_both_teachers() -> None:
    """Two teachers scoring differently worded prompts would disagree about the prompt."""
    a = build_prompt("state", "question")
    b = build_prompt("state", "question")
    assert prompt_hash(a) == prompt_hash(b)


def test_different_states_hash_differently() -> None:
    assert prompt_hash(build_prompt("a", "q")) != prompt_hash(build_prompt("b", "q"))


# --- trace statistics ---------------------------------------------------------------


def test_trace_stats_reports_mean_p90_and_max() -> None:
    stats = trace_stats([10, 20, 30, 40, 100])
    assert stats["n"] == 5
    assert stats["mean"] == pytest.approx(40.0)
    assert stats["p90"] == 100.0
    assert stats["max"] == 100.0


def test_trace_stats_of_nothing_is_zeros_not_a_crash() -> None:
    assert trace_stats([])["n"] == 0
