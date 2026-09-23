"""S1's row sampler: family weights between groups, uniform by row inside `Score`."""

from __future__ import annotations

import json

import pytest
from train.sampling import (
    row_weights,
    sample_indices,
    score_level_marginal,
)

from s1decide.tasks import repo_root


def row(qtype: str, family: str, answer: int = 0, stage: int | None = None, **extra) -> dict:
    return {
        "qtype": qtype,
        "family": family,
        "options": ["a", "b", "c", "d", "e"],
        "answer_idx": answer,
        "stage": stage,
        **extra,
    }


def test_score_mass_is_spread_evenly_over_rows_not_equalised_over_families() -> None:
    rows = [row("score", "ordinal_control")] + [row("score", "score_teacher")] * 9
    weights = {"score/ordinal_control": 9.0, "score/score_teacher": 1.0}
    out = row_weights(rows, weights)
    assert out == pytest.approx([1.8] * 10)  # total mass 18 over 10 rows
    assert sum(out) == pytest.approx(18.0)


def test_other_groups_keep_their_family_weights() -> None:
    rows = [row("noul", "go_emotions"), row("noul", "mmlu_noul")]
    assert row_weights(rows, {"noul/go_emotions": 1.5, "noul/mmlu_noul": 3.0}) == [1.5, 3.0]


def test_a_family_without_a_weight_is_an_error_not_a_default() -> None:
    with pytest.raises(KeyError, match="choice/new_source"):
        row_weights([row("choice", "new_source")], {})


def test_sampling_is_deterministic_and_proportional() -> None:
    a = sample_indices([1.0, 3.0], 20000, seed=7)
    assert a == sample_indices([1.0, 3.0], 20000, seed=7)
    assert a.count(1) / len(a) == pytest.approx(0.75, abs=0.02)


def _corpus() -> tuple[list[dict], dict]:
    root = repo_root()
    path = root / "data" / "processed" / "train.jsonl"
    weights_path = root / "data" / "processed" / "sampling_weights.json"
    if not path.is_file() or not weights_path.is_file():
        pytest.skip("no built corpus; run `uv run task data`")
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()
    ]
    return rows, json.loads(weights_path.read_text(encoding="utf-8"))["weights"]


def test_the_sampled_score_marginal_matches_the_training_marginal() -> None:
    """The owner's rule after the smoke regression: sampling must not move the Score prior.

    30,000 draws with the S1 seed. The tolerance is ~3 standard errors at the ~3,000 `Score` rows
    such a draw contains.
    """
    rows, weights = _corpus()
    drawn = [rows[i] for i in sample_indices(row_weights(rows, weights), 30000, seed=20260923)]
    expected = score_level_marginal(rows)
    got = score_level_marginal(drawn)
    assert set(got) == set(expected)
    # The 5-level scale holds the 4,804 teacher rows and is what the val check measures.
    assert max(abs(a - b) for a, b in zip(expected[5], got[5], strict=True)) < 0.03
    # ~130 draws each on the 3- and 4-level control scales: looser, same order of error.
    for levels in (3, 4):
        assert max(abs(a - b) for a, b in zip(expected[levels], got[levels], strict=True)) < 0.15


def test_the_sampled_soft_share_of_score_is_the_natural_one() -> None:
    rows, weights = _corpus()
    drawn = [rows[i] for i in sample_indices(row_weights(rows, weights), 30000, seed=20260923)]

    def soft_share(subset: list[dict]) -> float:
        score = [r for r in subset if r["qtype"] == "score"]
        return sum(r.get("target_type") == "soft" for r in score) / len(score)

    assert soft_share(drawn) == pytest.approx(soft_share(rows), abs=0.03)
