"""Brier skill score and the risk-coverage curve.

Both exist because ECE alone is gameable: on our own eval set a base-rate control scores a
better ECE than the calibrated model while being 33 accuracy points worse. These two metrics
are the ones that control cannot win.
"""

from __future__ import annotations

import math

import pytest
from eval.metrics import (
    RISK_COVERAGE_GRID,
    SELECTIVE_COVERAGES,
    Prediction,
    brier_multiclass,
    brier_skill_score,
    build_report,
    risk_coverage,
    summarize,
)

# --- Brier skill score --------------------------------------------------------


def test_skill_is_zero_against_itself_and_one_against_a_perfect_score() -> None:
    assert brier_skill_score(0.25, 0.25) == pytest.approx(0.0)
    assert brier_skill_score(0.0, 0.25) == pytest.approx(1.0)


def test_skill_is_negative_when_the_model_is_worse_than_the_control() -> None:
    assert brier_skill_score(0.40, 0.25) == pytest.approx(-0.6)


def test_skill_is_undefined_against_a_perfect_reference() -> None:
    assert brier_skill_score(0.1, 0.0) is None


def test_skill_ranks_a_confident_correct_model_above_a_hedging_one() -> None:
    """The property that matters: hedging cannot buy skill, though it can buy ECE."""
    labels = [0, 0, 1, 1]
    confident = [[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9]]
    hedging = [[0.5, 0.5]] * 4
    reference = brier_multiclass(hedging, labels)
    assert brier_skill_score(brier_multiclass(confident, labels), reference) > 0.9
    assert brier_skill_score(brier_multiclass(hedging, labels), reference) == pytest.approx(0.0)


# --- risk-coverage ------------------------------------------------------------


def test_full_coverage_is_plain_accuracy() -> None:
    probabilities = [[0.9, 0.1], [0.6, 0.4], [0.3, 0.7], [0.55, 0.45]]
    labels = [0, 1, 1, 0]
    curve = risk_coverage(probabilities, labels)
    assert curve["curve"][-1]["coverage"] == 1.0
    assert curve["curve"][-1]["accuracy"] == pytest.approx(0.75)


def test_a_model_that_knows_when_it_is_wrong_gains_accuracy_as_coverage_drops() -> None:
    """Confidence that ranks correctness is the whole point of the metric."""
    probabilities = [[0.99, 0.01], [0.95, 0.05], [0.55, 0.45], [0.51, 0.49]]
    labels = [0, 0, 1, 1]  # the two confident ones are right, the two unsure ones are wrong
    curve = risk_coverage(probabilities, labels, coverages=(0.5, 1.0))
    assert curve["curve"][0]["accuracy"] == pytest.approx(1.0)
    assert curve["curve"][1]["accuracy"] == pytest.approx(0.5)
    assert curve["aurc"] is not None


def test_confidence_uncorrelated_with_correctness_gives_a_flat_curve() -> None:
    """The base-rate control's shape: plenty of confidence, no information in its ordering."""
    # Confidences .95 .90 .85 .80; correctness alternates down that ranking, so abstaining on
    # the least confident half removes one right answer and one wrong one.
    probabilities = [[0.95, 0.05], [0.10, 0.90], [0.85, 0.15], [0.20, 0.80]]
    labels = [0, 0, 0, 0]
    curve = risk_coverage(probabilities, labels, coverages=(0.5, 1.0))
    assert curve["curve"][1]["accuracy"] == pytest.approx(0.5)
    assert curve["curve"][0]["accuracy"] == pytest.approx(curve["curve"][1]["accuracy"])


def test_selective_accuracy_is_reported_at_the_declared_coverages() -> None:
    probabilities = [[0.9, 0.1]] * 10
    labels = [0] * 10
    curve = risk_coverage(probabilities, labels)
    assert set(curve["selective_accuracy"]) == {f"{c:g}" for c in SELECTIVE_COVERAGES}


def test_coverage_outside_the_unit_interval_is_refused() -> None:
    with pytest.raises(ValueError, match="coverage"):
        risk_coverage([[0.6, 0.4]], [0], coverages=(0.0,))
    with pytest.raises(ValueError, match="coverage"):
        risk_coverage([[0.6, 0.4]], [0], coverages=(1.5,))


def test_a_single_global_temperature_leaves_the_curve_untouched() -> None:
    """Softmax with one temperature is monotone, so it cannot re-rank.

    This is the property the risk-coverage section claims, stated precisely: it holds for *one*
    temperature. Our fitted temperatures are per option-count bucket, and the next test shows
    that those genuinely do move the curve — which is why the summary says so rather than
    claiming the metric is calibration-invariant.
    """
    logits = [[2.0, 0.5], [1.0, 0.9], [0.2, 3.0], [1.5, 1.4]]
    predictions = [
        Prediction(id=str(i), family="f", qtype="choice", logits=row, answer_idx=i % 2)
        for i, row in enumerate(logits)
    ]
    cold = risk_coverage(
        [p.probabilities(1.0) for p in predictions], [p.answer_idx for p in predictions]
    )
    warm = risk_coverage(
        [p.probabilities(2.5) for p in predictions], [p.answer_idx for p in predictions]
    )
    for a, b in zip(cold["curve"], warm["curve"]):
        assert a["accuracy"] == pytest.approx(b["accuracy"])


def test_per_bucket_temperatures_can_re_rank_across_buckets() -> None:
    """Two buckets scaled differently swap which question looks most confident."""
    two_option = Prediction(id="a", family="f", qtype="noul", logits=[2.0, 0.0], answer_idx=0)
    four_option = Prediction(
        id="b", family="f", qtype="choice", logits=[2.4, 0.0, 0.0, 0.0], answer_idx=0
    )
    flat = [two_option.probabilities(1.0), four_option.probabilities(1.0)]
    scaled = [two_option.probabilities(3.0), four_option.probabilities(1.0)]
    assert max(flat[0]) > max(flat[1])
    assert max(scaled[0]) < max(scaled[1]), "the per-bucket temperature must reverse the order"


# --- wiring -------------------------------------------------------------------


def test_summarize_carries_selective_accuracy_and_aurc() -> None:
    summary = summarize([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7]] * 5, [0, 1, 1] * 5)
    assert set(summary["selective_accuracy"]) == {f"{c:g}" for c in SELECTIVE_COVERAGES}
    assert 0.0 <= summary["aurc"] <= 1.0


def test_report_carries_skill_against_every_control() -> None:
    predictions = [
        Prediction(
            id=str(i),
            family="f",
            qtype="choice",
            logits=[3.0, 0.0, 0.0] if i % 2 == 0 else [0.0, 3.0, 0.0],
            answer_idx=i % 2,
        )
        for i in range(40)
    ]
    report = build_report(predictions, control_fit=predictions)
    assert set(report.skill) == {"uniform", "base_rate"}
    for control, skill in report.skill.items():
        assert skill["brier_multiclass"] > 0, f"an accurate model must beat the {control} control"
        assert skill["accuracy_gain"] > 0
    assert len(report.risk_coverage["curve"]) == len(RISK_COVERAGE_GRID)
    payload = report.to_json()
    assert "skill" in payload and "risk_coverage" in payload


def test_aurc_is_lower_for_the_model_that_ranks_its_errors_better() -> None:
    # Both are exactly 50% accurate. They differ only in whether the confident half is the
    # right half, which is the only thing this metric is sensitive to.
    labels = [0, 0, 0, 0] * 5
    ranks_well = [[0.95, 0.05], [0.90, 0.10], [0.45, 0.55], [0.40, 0.60]] * 5
    ranks_badly = [[0.05, 0.95], [0.10, 0.90], [0.55, 0.45], [0.60, 0.40]] * 5
    assert summarize(ranks_well, labels)["accuracy"] == pytest.approx(0.5)
    assert math.isclose(
        summarize(ranks_well, labels)["accuracy"], summarize(ranks_badly, labels)["accuracy"]
    )
    assert summarize(ranks_well, labels)["aurc"] < summarize(ranks_badly, labels)["aurc"]


def test_the_threshold_is_the_confidence_that_produces_that_coverage() -> None:
    """acc@80% is what an operator gets; thr@80% is what they set to get it."""
    probabilities = [
        [c, 1 - c] for c in (0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55)
    ]
    labels = [0] * 10
    curve = risk_coverage(probabilities, labels)
    # 80% of 10 questions is the 8 most confident; the cut is the 8th confidence down.
    assert curve["selective_threshold"]["0.8"] == pytest.approx(0.65)
    assert curve["selective_threshold"]["0.9"] == pytest.approx(0.60)


def test_thresholds_fall_as_coverage_rises() -> None:
    """Answering more questions always means accepting less confident ones."""
    probabilities = [[0.5 + i / 100, 0.5 - i / 100] for i in range(40)]
    curve = risk_coverage(probabilities, [0] * 40)
    assert curve["selective_threshold"]["0.9"] < curve["selective_threshold"]["0.8"]


def test_every_selective_accuracy_has_a_matching_threshold() -> None:
    summary = summarize([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7]] * 5, [0, 1, 1] * 5)
    assert set(summary["selective_threshold"]) == set(summary["selective_accuracy"])


def test_the_threshold_actually_retains_the_promised_coverage() -> None:
    """The contract: keep everything at or above the threshold and you get that coverage."""
    probabilities = [[0.5 + i / 200, 0.5 - i / 200] for i in range(100)]
    labels = [0] * 100
    curve = risk_coverage(probabilities, labels)
    threshold = curve["selective_threshold"]["0.8"]
    kept = [row for row in probabilities if max(row) >= threshold]
    assert len(kept) == 80
