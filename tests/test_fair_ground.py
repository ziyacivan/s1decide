"""The fair-ground report: its controls must be the ones it says they are."""

from __future__ import annotations

import pytest
from eval.fair_ground import FAMILIES, family_report, own_marginal, training_prior


def noul(i: int, answer: int, weight: float = 1.0) -> dict:
    return {
        "id": f"n{i}",
        "family": "boolq",
        "qtype": "noul",
        "stage": None,
        "options": ["no", "yes"],
        "answer_idx": answer,
        "eval_weight": weight,
    }


def test_an_unseen_choice_count_gets_a_uniform_prior_and_is_listed() -> None:
    rows = [{"qtype": "choice", "stage": None, "family": "anli", "options": ["a", "b", "c"]}]
    rates, added = training_prior({"choice/4": [0.25] * 4}, rows)
    assert rates["choice/3"] == pytest.approx([1 / 3] * 3)
    assert added == ["choice/3"]


def test_a_missing_noul_prior_is_never_invented() -> None:
    with pytest.raises(KeyError, match="noul/2"):
        training_prior({}, [noul(0, 1)])


def test_own_marginal_is_weighted() -> None:
    assert own_marginal([1, 1, 0], [1.0, 1.0, 2.0], 2) == pytest.approx([0.5, 0.5])


def test_the_own_marginal_predictor_has_zero_skill_against_itself() -> None:
    rows = [noul(i, 1 if i < 30 else 0) for i in range(100)]
    marginal = own_marginal([r["answer_idx"] for r in rows], [1.0] * 100, 2)
    report = family_report(rows, [marginal] * 100, {"noul/2": [0.5, 0.5]})
    assert report["noul"]["bss_vs_own_marginal"] == pytest.approx(0.0, abs=1e-12)
    # Against a 50/50 training prior the same constant predictor does have skill: knowing
    # the base rate is worth something, which is why the own-marginal bar is the harder one.
    assert report["noul"]["model"]["bss"] > 0


def test_a_perfect_model_has_full_skill_on_both_controls() -> None:
    rows = [noul(i, i % 2) for i in range(20)]
    probs = [[0.0, 1.0] if r["answer_idx"] else [1.0, 0.0] for r in rows]
    report = family_report(rows, probs, {"noul/2": [0.7, 0.3]})
    assert report["noul"]["bss_vs_own_marginal"] == pytest.approx(1.0)
    assert report["noul"]["model"]["bss"] == pytest.approx(1.0)


def test_every_eval_family_is_described() -> None:
    from data.build.sources import SOURCES

    eval_families = {s.family for s in SOURCES if s.role in {"heldout", "ood"}}
    assert eval_families == set(FAMILIES)
