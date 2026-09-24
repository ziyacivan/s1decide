"""The TypeSafe head-to-head must score every model, Jev's published answers included, alike."""

from __future__ import annotations

import pytest
from eval.external_http import to_systemone_question
from eval.external_laya import from_laya_answer, to_laya_question
from eval.typesafe_headtohead import case_means, score_row, to_slice_row


def record(qtype: str = "choice") -> dict:
    if qtype == "choice":
        question = {
            "type": "choice",
            "instructions": "What is line 3?",
            "criteria": {"work": "Bills work.", "tax": "Is tax."},
            "label": "work",
            "src": "x",
        }
        target = {"work": 0.8, "tax": 0.2}
    else:
        question = {"type": "noul", "instructions": "It is vague.", "label": True, "src": "x"}
        target = {"true": 0.3, "false": 0.7}
    return {
        "state": {"b": 1, "a": "é"},
        "questions": {"decision": question},
        "_meta": {"id": f"typesafe/{qtype}", "group_id": "case-1", "family": "f", "target": target},
    }


def test_a_choice_record_keeps_ids_descriptions_and_the_original_question() -> None:
    row = to_slice_row(record("choice"))
    assert row["options"] == ["work: Bills work.", "tax: Is tax."]
    assert row["option_keys"] == ["work", "tax"]
    assert row["target"] == pytest.approx([0.8, 0.2])
    assert row["answer_idx"] == 0
    assert row["state"] == '{"a": "é", "b": 1}'
    assert "label" not in row["systemone_question"]
    assert "src" not in row["systemone_question"]


def test_a_noul_record_maps_true_to_yes() -> None:
    row = to_slice_row(record("noul"))
    assert row["options"] == ["no", "yes"]
    assert row["target"] == pytest.approx([0.7, 0.3])
    assert row["answer_idx"] == 0


def test_external_models_are_asked_the_original_question() -> None:
    row = to_slice_row(record("choice"))
    assert to_laya_question(row) == row["systemone_question"]
    assert to_systemone_question(row)["criteria"] == {"work": "Bills work.", "tax": "Is tax."}


def test_external_answers_map_back_by_criterion_id() -> None:
    row = to_slice_row(record("choice"))
    answer = {"type": "choice", "probabilities": {"tax": 0.1, "work": 0.9}}
    assert from_laya_answer(row, answer) == pytest.approx([0.9, 0.1])


def test_score_row_is_argmax_agreement_and_half_l1() -> None:
    assert score_row([0.6, 0.4], [0.8, 0.2]) == pytest.approx((1.0, 0.2))
    assert score_row([0.4, 0.6], [0.8, 0.2]) == pytest.approx((0.0, 0.4))


def test_cases_weigh_equally_and_missing_rows_score_zero_and_one() -> None:
    rows = [
        {"id": "a", "case": "c1"},
        {"id": "b", "case": "c1"},
        {"id": "c", "case": "c2"},
    ]
    scores = {"a": (1.0, 0.0), "b": (1.0, 0.0)}  # "c" unanswered
    out = case_means(scores, rows)
    assert out["agreement"] == pytest.approx(0.5)  # c1 = 1, c2 = 0, equal weight
    assert out["tvd"] == pytest.approx(0.5)
