"""Metric correctness, checked against hand-computable cases.

These are the numbers the project publishes, so each metric is pinned to a case whose value
can be worked out on paper rather than read off the implementation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from eval.metrics import (
    DEFAULT_BINS,
    Prediction,
    accuracy,
    auroc,
    base_rate_probabilities,
    brier_multiclass,
    brier_top_label,
    build_report,
    ece,
    equal_mass_bins,
    mce,
    nll,
    option_count_bucket,
    summarize,
    summarize_by,
    uniform_probabilities,
)


def pred(logits, answer_idx, family="f", qtype="choice", id="x") -> Prediction:
    return Prediction(
        id=id, family=family, qtype=qtype, logits=tuple(logits), answer_idx=answer_idx
    )


# --- Prediction ---------------------------------------------------------------


def test_prediction_softmaxes_its_logits() -> None:
    p = pred([0.0, math.log(3.0)], 1)
    assert p.probabilities() == pytest.approx([0.25, 0.75])
    assert p.n_options == 2


def test_prediction_temperature_flattens() -> None:
    p = pred([0.0, 2.0], 1)
    assert max(p.probabilities(10.0)) < max(p.probabilities(1.0)) < max(p.probabilities(0.1))


def test_prediction_coerces_a_bool_index_to_an_int() -> None:
    """A bool would index numpy as a mask, corrupting the base-rate control silently."""
    p = pred([0.0, 1.0], True)
    assert p.answer_idx == 1
    assert not isinstance(p.answer_idx, bool)


def test_prediction_round_trips_through_json() -> None:
    p = pred([0.5, -0.5, 1.0], 2, family="banking77", qtype="choice", id="t-1")
    assert Prediction.from_json(p.to_json()) == p


@pytest.mark.parametrize(
    ("logits", "answer_idx"),
    [([1.0], 0), ([1.0, 2.0], 2), ([1.0, 2.0], -1)],
)
def test_prediction_validates(logits, answer_idx) -> None:
    with pytest.raises(ValueError):
        pred(logits, answer_idx)


def test_prediction_rejects_non_positive_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        pred([0.0, 1.0], 0).probabilities(0.0)


# --- buckets ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "bucket"),
    [
        (2, "2"),
        (3, "3-5"),
        (5, "3-5"),
        (6, "6-16"),
        (16, "6-16"),
        (17, "17-77"),
        (77, "17-77"),
        (78, "78+"),
        (255, "78+"),
    ],
)
def test_option_count_buckets_tile_the_range(n: int, bucket: str) -> None:
    assert option_count_bucket(n) == bucket


def test_option_count_bucket_rejects_degenerate() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        option_count_bucket(1)


# --- point metrics ------------------------------------------------------------


def test_accuracy_counts_argmax_hits() -> None:
    probs = [[0.6, 0.4], [0.4, 0.6], [0.7, 0.3]]
    assert accuracy(probs, [0, 0, 0]) == pytest.approx(2 / 3)


def test_accuracy_breaks_ties_towards_the_lowest_index() -> None:
    assert accuracy([[0.5, 0.5]], [0]) == 1.0
    assert accuracy([[0.5, 0.5]], [1]) == 0.0


def test_nll_matches_hand_computation() -> None:
    assert nll([[0.25, 0.75]], [1]) == pytest.approx(-math.log(0.75))
    assert nll([[0.5, 0.5], [0.25, 0.75]], [0, 1]) == pytest.approx(
        (-math.log(0.5) - math.log(0.75)) / 2
    )


def test_nll_is_finite_for_a_zero_probability_answer() -> None:
    assert nll([[1.0, 0.0]], [1]) > 25.0


def test_brier_multiclass_matches_hand_computation() -> None:
    # (0.25-0)^2 + (0.75-1)^2 = 0.125
    assert brier_multiclass([[0.25, 0.75]], [1]) == pytest.approx(0.125)


def test_brier_multiclass_is_zero_for_a_perfect_prediction() -> None:
    assert brier_multiclass([[0.0, 1.0]], [1]) == pytest.approx(0.0)


def test_brier_top_label_uses_confidence_against_correctness() -> None:
    # correct with confidence 0.75 -> (0.75-1)^2 ; wrong with 0.75 -> (0.75-0)^2
    assert brier_top_label([[0.25, 0.75]], [1]) == pytest.approx(0.0625)
    assert brier_top_label([[0.25, 0.75]], [0]) == pytest.approx(0.5625)


# --- calibration --------------------------------------------------------------


def test_perfect_calibration_has_zero_ece() -> None:
    """Half the predictions at confidence 1.0 correct, half at 0.5 right half the time."""
    probs = [[1.0, 0.0]] * 8 + [[0.5, 0.5]] * 8
    labels = [0] * 8 + [0, 1] * 4
    assert ece(probs, labels, n_bins=2) == pytest.approx(0.0, abs=1e-12)


def test_total_overconfidence_has_ece_equal_to_the_confidence() -> None:
    probs = [[0.9, 0.1]] * 10
    labels = [1] * 10  # always wrong, always 0.9 confident
    assert ece(probs, labels, n_bins=5) == pytest.approx(0.9)
    assert mce(probs, labels, n_bins=5) == pytest.approx(0.9)


def test_equal_mass_bins_split_the_count_evenly() -> None:
    probs = [[c, 1 - c] for c in np.linspace(0.51, 0.99, 30)]
    bins = equal_mass_bins(probs, [0] * 30, n_bins=15)
    assert len(bins) == 15
    assert {b.count for b in bins} == {2}
    assert sum(b.count for b in bins) == 30


def test_equal_mass_bins_handle_a_count_not_divisible_by_the_bin_count() -> None:
    probs = [[c, 1 - c] for c in np.linspace(0.51, 0.99, 17)]
    bins = equal_mass_bins(probs, [0] * 17, n_bins=5)
    assert sum(b.count for b in bins) == 17
    assert max(b.count for b in bins) - min(b.count for b in bins) <= 1


def test_equal_mass_bins_are_ordered_by_confidence() -> None:
    rng = np.random.default_rng(0)
    probs = [[c, 1 - c] for c in rng.uniform(0.5, 1.0, 100)]
    bins = equal_mass_bins(probs, [0] * 100)
    assert [b.mean_confidence for b in bins] == sorted(b.mean_confidence for b in bins)


def test_bins_never_exceed_the_number_of_predictions() -> None:
    bins = equal_mass_bins([[0.6, 0.4], [0.7, 0.3]], [0, 0], n_bins=DEFAULT_BINS)
    assert len(bins) == 2


def test_equal_mass_binning_differs_from_equal_width_on_a_skewed_model() -> None:
    """The reason the project specifies equal mass: 99 confident predictions and one that isn't."""
    probs = [[0.99, 0.01]] * 99 + [[0.5, 0.5]]
    labels = [0] * 100
    bins = equal_mass_bins(probs, labels, n_bins=10)
    assert len(bins) == 10
    assert max(b.count for b in bins) <= 11  # equal-width would put 99 in one bin


# --- discrimination -----------------------------------------------------------


def test_auroc_is_one_when_confidence_ranks_correctness_perfectly() -> None:
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)


def test_auroc_is_zero_when_ranking_is_exactly_inverted() -> None:
    assert auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == pytest.approx(0.0)


def test_auroc_is_half_for_all_ties() -> None:
    assert auroc([0.5] * 6, [1, 1, 1, 0, 0, 0]) == pytest.approx(0.5)


def test_auroc_handles_partial_ties_by_average_rank() -> None:
    # one positive tied with one negative at 0.5, plus a clear positive above
    assert auroc([0.5, 0.5, 0.9], [0, 1, 1]) == pytest.approx(0.75)


def test_auroc_is_none_when_a_class_is_missing() -> None:
    assert auroc([0.1, 0.9], [1, 1]) is None
    assert auroc([0.1, 0.9], [0, 0]) is None


def test_auroc_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="scores but"):
        auroc([0.1, 0.2], [1])


# --- summarize ----------------------------------------------------------------


def test_summarize_reports_every_headline_metric() -> None:
    probs = [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]
    s = summarize(probs, [0, 1, 1], n_bins=3)
    for key in (
        "n",
        "accuracy",
        "ece",
        "mce",
        "brier_multiclass",
        "brier_top_label",
        "nll",
        "auroc_confidence",
        "mean_confidence",
        "mean_options",
        "bins",
    ):
        assert key in s
    assert s["n"] == 3
    assert s["accuracy"] == pytest.approx(2 / 3)


def test_summarize_reports_auroc_none_when_undefined() -> None:
    assert summarize([[0.9, 0.1], [0.8, 0.2]], [0, 0])["auroc_confidence"] is None


def test_summarize_by_partitions_without_losing_anything() -> None:
    probs = [[0.9, 0.1]] * 4
    groups = ["a", "a", "b", "b"]
    out = summarize_by(probs, [0, 1, 0, 1], groups, n_bins=2)
    assert set(out) == {"a", "b"}
    assert sum(g["n"] for g in out.values()) == 4


def test_summarize_by_skips_groups_below_min_count() -> None:
    out = summarize_by([[0.9, 0.1]] * 3, [0, 0, 0], ["a", "a", "b"], n_bins=2, min_count=2)
    assert set(out) == {"a"}


def test_summarize_by_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        summarize_by([[0.5, 0.5]], [0], ["a", "b"])


# --- validation ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("probs", "labels", "message"),
    [
        ([[0.5, 0.4]], [0], "sum to"),
        ([[0.5, 0.5]], [2], "out of range"),
        ([[1.0]], [0], "at least 2"),
        ([[float("nan"), 1.0]], [0], "non-finite"),
        ([[0.5, 0.5], [0.5, 0.5]], [0], "2 probability rows but 1 labels"),
    ],
)
def test_bad_input_is_rejected(probs, labels, message) -> None:
    with pytest.raises(ValueError, match=message):
        summarize(probs, labels)


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="no predictions"):
        summarize([], [])


# --- negative controls --------------------------------------------------------


def test_uniform_control_has_chance_accuracy_and_matching_confidence() -> None:
    probs = uniform_probabilities([4] * 100)
    labels = [i % 4 for i in range(100)]
    s = summarize(probs, labels, n_bins=1)
    assert s["accuracy"] == pytest.approx(0.25)
    assert s["mean_confidence"] == pytest.approx(0.25)
    assert s["ece"] == pytest.approx(0.0, abs=1e-9)


def test_constant_confidence_leaves_a_small_residual_ece_from_binning() -> None:
    """Honest wart: with identical confidences, equal-mass bins split arbitrarily.

    Each bin's accuracy then wobbles around the true rate by sampling noise, so ECE is small
    but not zero. Worth pinning so the number is never mistaken for miscalibration.
    """
    probs = uniform_probabilities([4] * 100)
    labels = [i % 4 for i in range(100)]
    assert 0.0 < ece(probs, labels, n_bins=15) < 0.1


def test_base_rate_control_learns_the_label_marginal() -> None:
    """80% of the fit split answers index 1, so the control should predict close to that."""
    fit = [pred([0.0, 0.0], 1, id=f"f{i}") for i in range(80)]
    fit += [pred([0.0, 0.0], 0, id=f"g{i}") for i in range(20)]
    evaluate = [pred([0.0, 0.0], 1, id="e0")]
    row = base_rate_probabilities(fit, evaluate, alpha=1.0)[0]
    assert row[1] == pytest.approx(81 / 102, abs=0.01)


def test_base_rate_control_falls_back_to_uniform_for_unseen_groups() -> None:
    fit = [pred([0.0, 0.0], 0, family="seen")]
    evaluate = [pred([0.0, 0.0, 0.0], 0, family="unseen")]
    assert base_rate_probabilities(fit, evaluate)[0] == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_base_rate_control_separates_families_and_option_counts() -> None:
    fit = [pred([0.0, 0.0], 0, family="a", id=f"a{i}") for i in range(10)]
    fit += [pred([0.0, 0.0], 1, family="b", id=f"b{i}") for i in range(10)]
    rows = base_rate_probabilities(
        fit, [pred([0.0, 0.0], 0, family="a"), pred([0.0, 0.0], 0, family="b")]
    )
    assert rows[0][0] > rows[0][1]
    assert rows[1][1] > rows[1][0]


def test_base_rate_control_can_be_better_calibrated_than_a_real_model() -> None:
    """The reason this control exists at all (qwen-rlcd's warning), demonstrated.

    A constant predictor that matches the label marginal is calibrated by construction while
    knowing nothing about the state. Reporting ECE without it is therefore meaningless.
    """
    n = 400
    fit = [pred([0.0, 0.0], int(i % 4 == 0), id=f"f{i}") for i in range(n)]  # 25% positive
    evaluate = [pred([0.0, 0.0], int(i % 4 == 0), id=f"e{i}") for i in range(n)]
    labels = [p.answer_idx for p in evaluate]

    control = summarize(base_rate_probabilities(fit, evaluate), labels)
    overconfident = summarize(
        [[0.05, 0.95] if i % 4 == 0 else [0.95, 0.05] for i in range(n)], labels
    )

    assert control["ece"] < overconfident["ece"]  # better calibrated...
    assert control["accuracy"] < overconfident["accuracy"]  # ...and much worse at deciding
    assert control["auroc_confidence"] is None or control["auroc_confidence"] == pytest.approx(
        0.5, abs=0.05
    )


def test_base_rate_rejects_non_positive_smoothing() -> None:
    with pytest.raises(ValueError, match="alpha"):
        base_rate_probabilities([pred([0.0, 0.0], 0)], [pred([0.0, 0.0], 0)], alpha=0.0)


# --- build_report -------------------------------------------------------------


def make_predictions(n: int = 60) -> list[Prediction]:
    rng = np.random.default_rng(7)
    out = []
    for i in range(n):
        k = (2, 4, 8)[i % 3]
        out.append(
            Prediction(
                id=f"p{i}",
                family=("alpha", "beta")[i % 2],
                qtype=("choice", "noul", "score")[i % 3],
                logits=tuple(rng.normal(size=k)),
                answer_idx=int(rng.integers(k)),
            )
        )
    return out


def test_build_report_covers_every_breakdown() -> None:
    preds = make_predictions()
    report = build_report(preds, control_fit=preds, meta={"run_id": "t"})
    payload = report.to_json()
    assert payload["overall"]["n"] == len(preds)
    assert set(payload["by_option_count"]) == {"2", "3-5", "6-16"}
    assert set(payload["by_family"]) == {"alpha", "beta"}
    assert set(payload["by_qtype"]) == {"choice", "noul", "score"}
    assert set(payload["controls"]) == {"uniform", "base_rate"}
    assert payload["meta"]["run_id"] == "t"


def test_build_report_accepts_a_per_bucket_temperature_map() -> None:
    preds = make_predictions()
    hot = build_report(preds, temperature={"2": 5.0, "3-5": 5.0, "6-16": 5.0})
    cold = build_report(preds, temperature=1.0)
    assert hot.overall["mean_confidence"] < cold.overall["mean_confidence"]
    assert hot.overall["accuracy"] == cold.overall["accuracy"]  # temperature never moves argmax


def test_build_report_omits_the_base_rate_control_when_no_fit_split_is_given() -> None:
    assert set(build_report(make_predictions()).controls) == {"uniform"}


def test_build_report_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="no predictions"):
        build_report([])
