"""Case-control sampling must give the same answers as the full fan-out.

This is the gate the plan puts in front of the first 3.3-hour evaluation: if weighted metrics on
a sample do not reproduce unweighted metrics on the population, then the cheap mode is not a
cheaper way to get the same number, it is a different number — and nobody would know which runs
were affected.

Every test here builds a full stage-1 population, samples it, and compares.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
from eval.case_control import DEFAULT_EVAL_NEGATIVES, EVAL_MODES, sample_case_control
from eval.metrics import (
    Prediction,
    accuracy,
    auroc,
    brier_multiclass,
    brier_top_label,
    build_report,
    ece,
    nll,
    risk_coverage,
)

from s1decide.calibrate import fit_temperature

#: Weighted estimates are unbiased, not exact: a sample of 16 negatives from ~95 carries real
#: sampling error. The tolerances below are what that error looks like at this population size,
#: and they are deliberately not loose enough to hide a missing weight — an unweighted metric on
#: this population is wrong by 0.3 or more, an order of magnitude outside every bound here.
ACCURACY_TOLERANCE = 0.02
CALIBRATION_TOLERANCE = 0.05


def population(n_questions: int = 120, n_options: int = 96, seed: int = 0) -> list[dict]:
    """A full stage-1 fan-out: one positive and ``n_options - 1`` negatives per question.

    Logits are drawn so that the positive usually, but not always, outscores the negatives —
    a model that is good and not perfect, which is the case the metrics have to handle.
    """
    rng = random.Random(seed)
    rows = []
    for q in range(n_questions):
        for option in range(n_options):
            is_answer = option == 0
            yes = rng.gauss(2.0, 1.5) if is_answer else rng.gauss(-2.0, 1.5)
            rows.append(
                {
                    "id": f"q{q}-o{option}",
                    "family": "banking77",
                    "qtype": "noul",
                    "stage": 1,
                    "parent_id": f"q{q}",
                    "logits": (0.0, yes),
                    "answer_idx": 1 if is_answer else 0,
                }
            )
    return rows


def to_predictions(rows) -> list[Prediction]:
    return [
        Prediction(
            id=r["id"],
            family=r["family"],
            qtype=r["qtype"],
            logits=r["logits"],
            answer_idx=r["answer_idx"],
            weight=r.get("eval_weight", 1.0),
        )
        for r in rows
    ]


@pytest.fixture(scope="module")
def full() -> list[Prediction]:
    return to_predictions(population())


@pytest.fixture(scope="module")
def sampled() -> list[Prediction]:
    rows, _ = sample_case_control(population(), negatives=DEFAULT_EVAL_NEGATIVES)
    return to_predictions(rows)


def probs(predictions, temperature: float = 1.0):
    return [p.probabilities(temperature) for p in predictions]


def labels(predictions):
    return [p.answer_idx for p in predictions]


def weights(predictions):
    return [p.weight for p in predictions]


# --- the sampler itself ---------------------------------------------------------


def test_every_positive_is_kept() -> None:
    """Positives are the scarce class; sampling them would throw away the signal."""
    rows, _ = sample_case_control(population(n_questions=20), negatives=4)
    assert sum(r["answer_idx"] == 1 for r in rows) == 20


def test_the_weights_reconstruct_the_population_size() -> None:
    rows, report = sample_case_control(population(n_questions=20, n_options=96), negatives=8)
    assert sum(r["eval_weight"] for r in rows) == pytest.approx(20 * 96)
    assert report["rows_dropped"] == 20 * 95 - 20 * 8


def test_a_question_with_few_negatives_is_kept_whole_and_weighs_one() -> None:
    """Nothing was dropped, so nothing stands in for anything."""
    rows, _ = sample_case_control(population(n_questions=3, n_options=5), negatives=16)
    assert len(rows) == 15
    assert all(r["eval_weight"] == 1.0 for r in rows)


def test_non_stage_one_rows_pass_through_unweighted() -> None:
    rows = [{"id": "x", "stage": None, "answer_idx": 0, "parent_id": None}]
    out, report = sample_case_control(rows)
    assert out[0]["eval_weight"] == 1.0
    assert report["questions"] == 0


def test_sampling_is_seeded() -> None:
    a, _ = sample_case_control(population(n_questions=10), negatives=4, seed=3)
    b, _ = sample_case_control(population(n_questions=10), negatives=4, seed=3)
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_zero_negatives_is_refused() -> None:
    with pytest.raises(ValueError, match="negatives"):
        sample_case_control(population(n_questions=2), negatives=0)


def test_both_modes_are_named() -> None:
    assert set(EVAL_MODES) == {"case-control", "full"}


# --- the gate: weighted sample == unweighted population --------------------------


def test_weighted_accuracy_matches_the_full_fan_out(full, sampled) -> None:
    assert accuracy(probs(sampled), labels(sampled), weights(sampled)) == pytest.approx(
        accuracy(probs(full), labels(full)), abs=ACCURACY_TOLERANCE
    )


def test_weighted_brier_matches_the_full_fan_out(full, sampled) -> None:
    for metric in (brier_multiclass, brier_top_label):
        assert metric(probs(sampled), labels(sampled), weights(sampled)) == pytest.approx(
            metric(probs(full), labels(full)), abs=CALIBRATION_TOLERANCE
        )


def test_weighted_nll_matches_the_full_fan_out(full, sampled) -> None:
    assert nll(probs(sampled), labels(sampled), weights(sampled)) == pytest.approx(
        nll(probs(full), labels(full)), abs=CALIBRATION_TOLERANCE
    )


def test_weighted_ece_matches_the_full_fan_out(full, sampled) -> None:
    assert ece(probs(sampled), labels(sampled), weights=weights(sampled)) == pytest.approx(
        ece(probs(full), labels(full)), abs=CALIBRATION_TOLERANCE
    )


def test_weighted_auroc_matches_the_full_fan_out(full, sampled) -> None:
    """AUROC is a ranking statistic, so a missing weight shows up here loudly."""

    def compute(predictions):
        rows = [np.asarray(p.probabilities()) for p in predictions]
        confidence = np.array([float(r.max()) for r in rows])
        correct = np.array(
            [float(int(np.argmax(r)) == p.answer_idx) for r, p in zip(rows, predictions)]
        )
        return confidence, correct

    cf, yf = compute(full)
    cs, ys = compute(sampled)
    assert auroc(cs, ys, weights(sampled)) == pytest.approx(auroc(cf, yf), abs=0.05)


def test_weighted_risk_coverage_matches_the_full_fan_out(full, sampled) -> None:
    a = risk_coverage(probs(sampled), labels(sampled), weights=weights(sampled))
    b = risk_coverage(probs(full), labels(full))
    for key in a["selective_accuracy"]:
        assert a["selective_accuracy"][key] == pytest.approx(
            b["selective_accuracy"][key], abs=ACCURACY_TOLERANCE
        )


def test_weighted_temperature_matches_the_full_fan_out(full, sampled) -> None:
    """The one that matters most: a temperature fitted on the sample is deployed on everything."""
    fitted = fit_temperature([p.logits for p in sampled], labels(sampled), weights=weights(sampled))
    reference = fit_temperature([p.logits for p in full], labels(full))
    assert fitted == pytest.approx(reference, rel=0.10)


def test_ignoring_the_weights_breaks_the_base_rate_control(full, sampled) -> None:
    """The tolerances above are only meaningful if an unweighted sample would fail somewhere.

    It would, but not everywhere, and the distinction is worth stating rather than glossing.
    **Top-line accuracy is inherently robust to this sampling**: going from 1:95 to 1:16 moves
    the positives from 1.0% of the set to 5.9%, so accuracy can shift by at most ~5pp times the
    gap between per-class accuracies — on this population, under a point. Measured: 0.9084 full
    against 0.9093 unweighted. That is not the weights working, it is accuracy not caring.

    The quantity that does break is anything reading the **label marginal**, which is exactly
    what case-control sampling distorts by construction. The base-rate control learns that
    marginal, so it is the honest canary: weighted it recovers the population's, unweighted it
    learns the sample's and becomes a far easier baseline to beat.
    """
    stripped = [
        Prediction(
            id=p.id, family=p.family, qtype=p.qtype, logits=p.logits, answer_idx=p.answer_idx
        )
        for p in sampled
    ]
    reference = build_report(full, control_fit=full).controls["base_rate"]["accuracy"]
    weighted = build_report(sampled, control_fit=sampled).controls["base_rate"]["accuracy"]
    unweighted = build_report(stripped, control_fit=stripped).controls["base_rate"]["accuracy"]

    assert weighted == pytest.approx(reference, abs=ACCURACY_TOLERANCE)
    assert abs(unweighted - reference) > ACCURACY_TOLERANCE * 2, (
        "if an unweighted control matched the population's, these tests would prove nothing"
    )


def test_the_effective_count_reports_the_population_not_the_sample(full, sampled) -> None:
    """`n` is how many questions were scored; `effective_n` is how many they stand for."""
    weighted = build_report(sampled, control_fit=sampled).overall
    assert weighted["n"] < len(full)
    assert weighted["effective_n"] == pytest.approx(len(full), rel=0.01)


def test_a_weighted_report_matches_a_full_report(full, sampled) -> None:
    """End to end, through build_report, including the base-rate control."""
    weighted = build_report(sampled, control_fit=sampled)
    reference = build_report(full, control_fit=full)
    assert weighted.overall["accuracy"] == pytest.approx(
        reference.overall["accuracy"], abs=ACCURACY_TOLERANCE
    )
    assert weighted.overall["ece"] == pytest.approx(
        reference.overall["ece"], abs=CALIBRATION_TOLERANCE
    )
    assert weighted.controls["base_rate"]["accuracy"] == pytest.approx(
        reference.controls["base_rate"]["accuracy"], abs=ACCURACY_TOLERANCE
    ), "the control must learn the population base rate, not the sample's"
    assert weighted.overall["effective_n"] == pytest.approx(reference.overall["n"], rel=0.01)
