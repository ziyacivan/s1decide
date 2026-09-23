"""S1's checkpoint evaluation: fixed slice, weighted metrics, and the prior-shift marginal."""

from __future__ import annotations

import pytest
from train.periodic_eval import base_rates, build_eval_rows, eval_group, eval_metrics


def val_row(i: int, stage: int | None = None, parent: str = "p", answer: int = 0, **extra) -> dict:
    return {
        "id": f"r{i}",
        "qtype": extra.pop("qtype", "noul"),
        "family": extra.pop("family", "go_emotions"),
        "options": ["yes", "no"],
        "answer_idx": answer,
        "stage": stage,
        "parent_id": parent,
        **extra,
    }


def test_the_slice_keeps_every_non_stage1_row_and_samples_stage1_questions() -> None:
    rows = [val_row(i) for i in range(5)]
    for q in range(10):
        rows.append(val_row(100 + q, stage=1, parent=f"q{q}", answer=1))
        rows += [val_row(1000 * (q + 1) + n, stage=1, parent=f"q{q}", answer=0) for n in range(40)]
    out, report = build_eval_rows(rows, stage1_questions=3, negatives=4, seed=1)
    assert report["non_stage1_rows"] == 5
    assert report["stage1_questions"] == 3
    assert len(out) == 5 + 3 * (1 + 4)
    negatives = [r for r in out if r.get("stage") == 1 and r["answer_idx"] == 0]
    assert all(r["eval_weight"] == pytest.approx(10.0) for r in negatives)  # 40 kept as 4
    assert out == build_eval_rows(rows, stage1_questions=3, negatives=4, seed=1)[0]


def test_teacher_score_rows_are_reported_apart_from_the_rule_labelled_ones() -> None:
    assert eval_group({"qtype": "score", "family": "score_teacher"}) == "score/teacher"
    assert eval_group({"qtype": "score", "family": "ordinal_control"}) == "score"
    assert eval_group({"qtype": "noul", "stage": 1}) == "noul/stage1"


def _example(target: list[float], answer: int, weight: float = 1.0) -> dict:
    return {
        "qtype": "score",
        "family": "score_teacher",
        "target": target,
        "answer_idx": answer,
        "eval_weight": weight,
    }


def test_a_shifted_prior_shows_in_the_marginal_even_when_kl_is_modest() -> None:
    """The smoke adapter's failure: predictions pushed up the scale against a low target."""
    examples = [_example([1, 0, 0, 0, 0], 0) for _ in range(8)] + [
        _example([0, 0, 0, 0, 1], 4) for _ in range(2)
    ]
    up = [[0.0, 0.0, 0.0, 0.0, 3.0]] * 10  # always predicts the top level
    rates = {"score/teacher/5": [0.8, 0.0, 0.0, 0.0, 0.2]}
    m = eval_metrics(examples, up, rates)["score/teacher"]
    assert m["mean_predicted_level"] == pytest.approx(4.0)
    assert m["mean_target_level"] == pytest.approx(0.8)
    assert m["predicted_marginal"][4] == pytest.approx(1.0)
    assert m["target_marginal"][0] == pytest.approx(0.8)
    assert m["accuracy"] == pytest.approx(0.2)
    assert m["bss"] < 0  # worse than the training prior


def test_weights_count_as_rows_they_stand_for() -> None:
    right = _example([1, 0, 0, 0, 0], 0, weight=9.0)
    wrong = _example([0, 1, 0, 0, 0], 1, weight=1.0)
    m = eval_metrics([right, wrong], [[5, 0, 0, 0, 0]] * 2, {"score/teacher/5": [0.2] * 5})
    assert m["score/teacher"]["accuracy"] == pytest.approx(0.9)


def test_a_missing_control_is_an_error_not_a_uniform_default() -> None:
    with pytest.raises(KeyError, match="score/teacher/5"):
        eval_metrics([_example([1, 0, 0, 0, 0], 0)], [[1, 0, 0, 0, 0]], {})


def test_base_rates_are_the_training_target_marginal_per_group_and_size() -> None:
    rows = [
        {"qtype": "noul", "family": "f", "options": ["yes", "no"], "answer_idx": 0},
        {"qtype": "noul", "family": "f", "options": ["yes", "no"], "answer_idx": 0},
        {"qtype": "noul", "family": "f", "options": ["yes", "no"], "answer_idx": 1},
    ]
    assert base_rates(rows)["noul/2"] == pytest.approx([2 / 3, 1 / 3])


def test_eval_asks_the_model_for_the_last_position_only() -> None:
    """Full-vocabulary logits at every position OOMed the first S1 smoke's row-0 eval."""
    torch = pytest.importorskip("torch")
    from train.periodic_eval import predict_logits

    calls: list[dict] = []

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, attention_mask, **kwargs):
            calls.append(kwargs)
            keep = kwargs.get("logits_to_keep", input_ids.shape[1])
            logits = torch.arange(10.0).repeat(input_ids.shape[0], keep, 1)
            return type("Out", (), {"logits": logits})()

    examples = [
        {"input_ids": [1, 2, 3], "label_token_ids": [4, 7], "target": [1.0, 0.0], "qtype": "noul"}
    ]
    model = Tiny().train()
    assert predict_logits(model, examples, pad_token_id=0) == [[4.0, 7.0]]
    assert calls[0]["logits_to_keep"] == 1
    assert model.training  # train mode restored


def test_a_long_evaluation_reports_progress() -> None:
    torch = pytest.importorskip("torch")
    from train.periodic_eval import predict_logits

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, attention_mask, **kwargs):
            return type("Out", (), {"logits": torch.zeros(input_ids.shape[0], 1, 10)})()

    example = {"input_ids": [1], "label_token_ids": [2, 3], "target": [1.0, 0.0], "qtype": "noul"}
    seen: list[tuple[int, int]] = []
    predict_logits(
        Tiny(),
        [example] * 10,
        0,
        batch_size=2,
        on_progress=lambda d, t: seen.append((d, t)),
        every=2,
    )
    assert seen == [(4, 10), (8, 10)]
