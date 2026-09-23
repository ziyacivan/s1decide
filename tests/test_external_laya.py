"""Laya's answers must land on our options, in our order, or the comparison is not like for like."""

from __future__ import annotations

import pytest
from eval.external_laya import from_laya_answer, to_laya_question


def row(qtype: str, options: list[str], stage: int | None = None) -> dict:
    return {"id": "r", "qtype": qtype, "options": options, "instructions": "q?", "stage": stage}


def test_a_noul_row_is_asked_as_noul_and_yes_is_index_one() -> None:
    r = row("noul", ["no", "yes"])
    assert to_laya_question(r) == {"type": "noul", "instructions": "q?"}
    assert from_laya_answer(r, {"type": "noul", "noul": 0.8}) == pytest.approx([0.2, 0.8])


def test_a_stage1_row_keeps_our_instructions_and_is_asked_as_noul() -> None:
    r = {**row("noul", ["no", "yes"], stage=1), "instructions": "Which intent?\nCandidate: x"}
    assert to_laya_question(r)["instructions"] == "Which intent?\nCandidate: x"


def test_choice_probabilities_follow_our_option_order_not_laya_s() -> None:
    r = row("choice", ["b", "a", "c"])
    answer = {"type": "choice", "probabilities": {"a": 0.2, "b": 0.7, "c": 0.1}}
    assert from_laya_answer(r, answer) == pytest.approx([0.7, 0.2, 0.1])
    assert to_laya_question(r)["criteria"] == {"b": "b", "a": "a", "c": "c"}


def test_score_levels_map_by_index() -> None:
    r = row("score", ["none", "slight", "moderate"])
    answer = {"type": "score", "probabilities": {"0": 0.5, "1": 0.3, "2": 0.2}}
    assert from_laya_answer(r, answer) == pytest.approx([0.5, 0.3, 0.2])
    assert to_laya_question(r)["criteria"] == ["none", "slight", "moderate"]


def test_a_missing_option_is_an_error_not_a_zero() -> None:
    r = row("choice", ["a", "b"])
    with pytest.raises(ValueError, match="lacks options"):
        from_laya_answer(r, {"type": "choice", "probabilities": {"a": 1.0}})


def test_a_type_mismatch_is_an_error() -> None:
    with pytest.raises(ValueError, match="expected a noul"):
        from_laya_answer(row("noul", ["no", "yes"]), {"type": "choice", "probabilities": {}})
