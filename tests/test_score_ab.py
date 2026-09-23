"""The before/after `Score` report: its metrics checked on numbers whose answer is known."""

from __future__ import annotations

import json
import math

import pytest
from eval.score_ab import (
    load_score_rows,
    quadratic_weighted_kappa,
    row_deltas,
    score_metrics,
)

LEVELS = ["none", "slight", "moderate", "strong", "decisive"]


def row(answer: int, soft: tuple[int, int] | None = None, id_: str = "r") -> dict:
    out = {"id": id_, "options": LEVELS, "answer_idx": answer, "target_type": "hard"}
    if soft:
        target = [0.0] * 5
        target[soft[0]] = target[soft[1]] = 0.5
        out.update(target=target, target_type="soft")
    return out


def peaked(level: int, height: float = 20.0) -> list[float]:
    return [height if i == level else 0.0 for i in range(5)]


def test_qwk_is_one_for_perfect_agreement_and_negative_when_reversed() -> None:
    assert quadratic_weighted_kappa([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], 5) == pytest.approx(1.0)
    assert quadratic_weighted_kappa([4, 3, 2, 1, 0], [0, 1, 2, 3, 4], 5) < 0


def test_qwk_charges_a_far_miss_more_than_a_near_one() -> None:
    truth = [0, 1, 2, 3, 4] * 4
    near = [min(t + 1, 4) for t in truth]
    far = [4 - t for t in truth]
    assert quadratic_weighted_kappa(near, truth, 5) > quadratic_weighted_kappa(far, truth, 5)


def test_a_confident_correct_model_scores_well_on_every_metric() -> None:
    rows = [row(i % 5, id_=str(i)) for i in range(20)]
    m = score_metrics(rows, [peaked(r["answer_idx"]) for r in rows], [0.2] * 5)
    assert m["accuracy"] == 1.0
    assert m["qwk"] == pytest.approx(1.0)
    assert m["kl"] == pytest.approx(0.0, abs=1e-6)
    assert m["bss"] > 0.99


def test_kl_on_a_soft_row_is_zero_when_the_prediction_is_the_target() -> None:
    soft = row(1, soft=(1, 2))
    logits = [-50.0, 0.0, 0.0, -50.0, -50.0]
    m = score_metrics([soft], [logits], [0.2] * 5)
    assert m["kl"] == pytest.approx(0.0, abs=1e-6)
    assert m["soft_rows_argmax_on_a_teacher_level"] == 1.0


def test_the_base_rate_control_itself_has_zero_skill() -> None:
    rows = [row(i % 5, id_=str(i)) for i in range(10)]
    rate = [0.2] * 5
    m = score_metrics(rows, [[math.log(p) for p in rate]] * len(rows), rate)
    assert m["bss"] == pytest.approx(0.0, abs=1e-9)


def test_row_deltas_count_flips_and_the_sign_of_kl_change() -> None:
    rows = [row(0, id_="a"), row(1, id_="b")]
    before = score_metrics(rows, [peaked(0), peaked(0)], [0.2] * 5)  # a right, b wrong
    after = score_metrics(rows, [peaked(1), peaked(1)], [0.2] * 5)  # a wrong, b right
    summary = row_deltas(rows, before, after)["summary"]["all"]
    assert summary["right_to_wrong"] == 1
    assert summary["wrong_to_right"] == 1
    assert summary["share_kl_improved"] == 0.5


def test_only_teacher_score_rows_are_loaded(tmp_path) -> None:
    path = tmp_path / "val.jsonl"
    lines = [
        {"family": "score_teacher", "qtype": "score", "id": "keep"},
        {"family": "ordinal_control", "qtype": "score", "id": "rule-labelled"},
        {"family": "go_emotions", "qtype": "noul", "id": "other"},
    ]
    path.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    assert [r["id"] for r in load_score_rows(path)] == ["keep"]
