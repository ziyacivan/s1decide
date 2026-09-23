"""Teacher-labelled `Score` rows joining the corpus.

Two properties carry the weight here. An ordinal scale's option order is part of its meaning, so
it must survive the build; and a soft target is a distribution over option positions, so it must
stay attached to the positions it describes.
"""

from __future__ import annotations

import json
import random

import pytest
from data.build.pipeline import (
    ORDERED_QTYPES,
    TEACHER_SCORE_FAMILY,
    shuffle_options,
    teacher_score_rows,
)

RUBRIC = ["none", "slight", "moderate", "strong", "decisive"]


def fold_row(agreement: str, first: int, second: int, answer: int) -> dict:
    return {
        "id": "score-teach-00001-abcd1234",
        "state": "the invoice was charged twice",
        "instructions": "How strongly does this express frustration?",
        "options": list(RUBRIC),
        "answer_idx": answer,
        "source": "PolyAI/banking77",
        "license": "cc-by-4.0",
        "parent_id": "teach-00001-abcd1234",
        "teacher_levels": {"first": first, "second": second},
        "agreement": agreement,
    }


def write(tmp_path, rows) -> object:
    path = tmp_path / "score_teacher.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return path


# --- targets ----------------------------------------------------------------------


def test_exact_agreement_becomes_a_hard_target(tmp_path) -> None:
    row = teacher_score_rows(write(tmp_path, [fold_row("exact", 2, 2, 2)]))[0]
    assert row["target_type"] == "hard"
    assert row["target"] == [0.0, 0.0, 1.0, 0.0, 0.0]


def test_one_level_apart_becomes_a_soft_target_split_equally(tmp_path) -> None:
    """Where an ordinal rubric is genuinely ambiguous, not where it failed. The split is equal
    because nothing measured justifies a tilt: over the kept rows teacher 2 was the higher one on
    1,010 adjacent disagreements and the lower on 795, so there is no reliable side."""
    row = teacher_score_rows(write(tmp_path, [fold_row("within_one", 1, 2, 1)]))[0]
    assert row["target_type"] == "soft"
    assert row["target"] == [0.0, 0.5, 0.5, 0.0, 0.0]


def test_every_target_is_a_distribution(tmp_path) -> None:
    rows = teacher_score_rows(
        write(tmp_path, [fold_row("exact", 0, 0, 0), fold_row("within_one", 3, 4, 3)])
    )
    for row in rows:
        assert sum(row["target"]) == pytest.approx(1.0)
        assert len(row["target"]) == len(row["options"])
        assert row["target"][row["answer_idx"]] > 0, "the hard label must have mass in the target"


def test_the_row_carries_its_licence_and_both_teacher_levels(tmp_path) -> None:
    row = teacher_score_rows(write(tmp_path, [fold_row("within_one", 1, 2, 1)]))[0]
    assert row["license"] == "cc-by-4.0" and row["source"] == "PolyAI/banking77"
    assert row["teacher_levels"] == {"first": 1, "second": 2}
    assert row["family"] == TEACHER_SCORE_FAMILY
    assert row["role"] == "train"


def test_a_missing_fold_file_builds_a_smaller_corpus_rather_than_failing(tmp_path) -> None:
    assert teacher_score_rows(tmp_path / "nothing.jsonl") == []


# --- option order ------------------------------------------------------------------


def test_an_ordinal_scale_is_never_shuffled() -> None:
    """`Score` *is* an ordering. It was shuffled with everything else until 2026-09-23, which
    produced option lists like ["none", "strong", "slight", "decisive", "moderate"] and made the
    expected level — and ADR 0007's E[|k-y|] penalty — arithmetic over a scrambled axis."""
    assert "score" in ORDERED_QTYPES and "noul" in ORDERED_QTYPES
    row = {"qtype": "score", "options": tuple(RUBRIC), "answer_idx": 2}
    for seed in range(20):
        assert shuffle_options(dict(row), random.Random(seed))["options"] == tuple(RUBRIC)


def test_a_choice_is_still_shuffled() -> None:
    """The positional prior is a real problem everywhere the positions mean nothing."""
    row = {"qtype": "choice", "options": ("a", "b", "c", "d", "e"), "answer_idx": 0}
    orders = {shuffle_options(dict(row), random.Random(s))["options"] for s in range(20)}
    assert len(orders) > 1


def test_a_shuffle_moves_the_answer_with_the_options() -> None:
    row = {"qtype": "choice", "options": ("a", "b", "c"), "answer_idx": 0}
    out = shuffle_options(dict(row), random.Random(3))
    assert out["options"][out["answer_idx"]] == "a"


def test_a_shuffle_moves_a_soft_target_with_the_options() -> None:
    """A target left behind desynchronised from `answer_idx` on 289 of the first 400 teacher
    rows: the hard label and the soft target named different options."""
    row = {
        "qtype": "choice",
        "options": ("a", "b", "c", "d"),
        "answer_idx": 1,
        "target": [0.0, 0.5, 0.5, 0.0],
    }
    out = shuffle_options(dict(row), random.Random(7))
    assert out["target"][out["answer_idx"]] == 0.5
    carried = {out["options"][i] for i, v in enumerate(out["target"]) if v > 0}
    assert carried == {"b", "c"}
    assert sum(out["target"]) == pytest.approx(1.0)


# --- the evaluation batch (teacher run 2) ---------------------------------------------


def test_eval_batch_rows_are_marked_as_such(tmp_path) -> None:
    rows = teacher_score_rows(
        write(tmp_path, [fold_row("exact", 2, 2, 2)]),
        loaded_from=f"{TEACHER_SCORE_FAMILY}_eval",
    )
    assert rows[0]["loaded_from"] == f"{TEACHER_SCORE_FAMILY}_eval"
    assert rows[0]["family"] == TEACHER_SCORE_FAMILY


def test_folding_an_eval_batch_over_the_training_rows_is_refused() -> None:
    """The default --out is the training fold; an eval batch written there replaces it."""
    from data.build.agree import main

    assert main(["--first", "a", "--second", "b", "--split", "eval"]) == 2
