"""Vector scaling: temperature plus a per-position bias.

Temperature scaling cannot change which option wins. Vector scaling can, and that is both the
reason to want it and the reason to be careful with it: the option→token mapping is shuffled per
example, so a position that is still systematically favoured means the model prefers a *slot*
rather than an answer. Correcting that is a genuine gain; fitting 26 biases to 40 examples is a
confident-looking artefact of the sample.

Everything here runs on mock logits with a known planted bias, so the tests check recovery of
something whose true value is known rather than agreement with whatever the fitter happened to
produce.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from eval.metrics import Prediction

from s1decide.calibrate import (
    CALIBRATION_METHODS,
    MIN_VECTOR_SAMPLES_PER_PARAMETER,
    Calibration,
    fit_calibration,
    fit_temperature,
    fit_vector_scaling,
    nll_at_temperature,
    nll_with_vector,
)


def mock_logits(
    n: int,
    n_options: int = 3,
    *,
    sharpness: float = 2.0,
    slot_bias: float = 0.0,
    margin: float = 2.0,
    seed: int = 0,
) -> tuple[list[list[float]], list[int]]:
    """Logits from a model that is over-confident by ``sharpness`` and favours slot 0.

    Args:
        n: Rows.
        n_options: Options per row.
        sharpness: Multiplies the logits; > 1 is over-confident.
        slot_bias: Added to option 0 only — the planted positional preference.
        margin: How much the correct option is favoured before distortion.
        seed: Reproducibility.

    Returns:
        ``(logits, labels)``.
    """
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for _ in range(n):
        y = int(rng.integers(0, n_options))
        z = rng.normal(0.0, 1.0, n_options)
        z[y] += margin
        z *= sharpness
        z[0] += slot_bias
        rows.append([float(v) for v in z])
        labels.append(y)
    return rows, labels


# --- the fitter -----------------------------------------------------------------


def test_the_bias_is_centred_so_the_fit_is_identifiable() -> None:
    """Softmax is shift-invariant, so an uncentred bias would drift along a flat direction."""
    logits, labels = mock_logits(2000, slot_bias=1.0)
    _temperature, bias = fit_vector_scaling(logits, labels)
    assert sum(bias) == pytest.approx(0.0, abs=1e-9)


def test_a_planted_slot_preference_is_recovered_with_the_right_sign() -> None:
    """The model was made to over-favour slot 0; the correction must push slot 0 down."""
    logits, labels = mock_logits(3000, slot_bias=1.0, seed=3)
    _temperature, bias = fit_vector_scaling(logits, labels)
    assert bias[0] < 0
    assert bias[0] < min(bias[1:]), "slot 0 must be penalised more than any other"


def test_no_planted_bias_gives_a_small_correction() -> None:
    logits, labels = mock_logits(3000, slot_bias=0.0, seed=4)
    _temperature, bias = fit_vector_scaling(logits, labels)
    assert max(abs(b) for b in bias) < 0.25


def test_vector_scaling_never_scores_worse_than_temperature_alone() -> None:
    """Temperature is the b=0 special case, so the optimum cannot be worse on the fitting set."""
    logits, labels = mock_logits(3000, slot_bias=1.2, seed=5)
    temperature, bias = fit_vector_scaling(logits, labels)
    only_t = fit_temperature(logits, labels)
    assert (
        nll_with_vector(logits, labels, temperature, bias)
        <= nll_at_temperature(logits, labels, only_t) + 1e-9
    )


def test_it_beats_temperature_by_a_real_margin_when_a_bias_exists() -> None:
    logits, labels = mock_logits(4000, slot_bias=1.5, seed=6)
    temperature, bias = fit_vector_scaling(logits, labels)
    gain = nll_at_temperature(logits, labels, fit_temperature(logits, labels)) - nll_with_vector(
        logits, labels, temperature, bias
    )
    assert gain > 0.01, f"only {gain:.4f} nats recovered from a planted 1.5-logit slot bias"


def test_the_fit_is_deterministic() -> None:
    logits, labels = mock_logits(1500, slot_bias=0.8, seed=7)
    assert fit_vector_scaling(logits, labels) == fit_vector_scaling(logits, labels)


def test_weights_shift_the_fit_toward_the_weighted_population() -> None:
    """Case-control sampling relies on this exactly as temperature scaling does."""
    logits, labels = mock_logits(1200, slot_bias=1.0, seed=8)
    heavy = [5.0 if y == 0 else 1.0 for y in labels]
    assert fit_vector_scaling(logits, labels, weights=heavy) != fit_vector_scaling(logits, labels)


def test_ragged_rows_are_refused_rather_than_silently_padded() -> None:
    with pytest.raises(ValueError, match="equal length"):
        fit_vector_scaling([[0.0, 1.0], [0.0, 1.0, 2.0]], [0, 1])


def test_an_empty_fit_is_refused() -> None:
    with pytest.raises(ValueError, match="no data"):
        fit_vector_scaling([], [])


def test_mismatched_labels_are_refused() -> None:
    with pytest.raises(ValueError, match="labels"):
        fit_vector_scaling([[0.0, 1.0], [1.0, 0.0]], [0])


# --- Calibration carrying both methods --------------------------------------------


def predictions(n: int, n_options: int = 3, *, slot_bias: float = 1.0, seed: int = 9):
    logits, labels = mock_logits(n, n_options, slot_bias=slot_bias, seed=seed)
    return [
        Prediction(id=str(i), family="f", qtype="choice", logits=tuple(z), answer_idx=y)
        for i, (z, y) in enumerate(zip(logits, labels))
    ]


def test_fitting_produces_both_methods() -> None:
    cal = fit_calibration(predictions(3000), quantization="nf4-bf16")
    assert cal.temperatures, "temperature scaling must still be fitted"
    assert cal.vectors, "vector scaling must be fitted alongside it"
    assert cal.method == "temperature", "the deployed method stays temperature until chosen"


def test_the_deployed_method_is_named_in_the_file(tmp_path) -> None:
    cal = fit_calibration(predictions(3000), quantization="nf4-bf16")
    payload = json.loads(cal.save(tmp_path / "calibration.json").read_text(encoding="utf-8"))
    assert payload["method"] in CALIBRATION_METHODS
    assert "vectors" in payload and "temperatures" in payload


def test_both_methods_round_trip_through_the_file(tmp_path) -> None:
    cal = fit_calibration(predictions(3000), quantization="nf4-bf16")
    loaded = Calibration.load(cal.save(tmp_path / "c.json"))
    assert loaded.vectors == cal.vectors
    assert loaded.temperatures == cal.temperatures


def test_a_file_written_before_vector_scaling_still_loads(tmp_path) -> None:
    """Old runs are temperature-only, which is what they are — not a load failure."""
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "version": "0.1",
                "quantization": "nf4-bf16",
                "default": 1.0,
                "temperatures": {"2": 1.5},
                "meta": {},
            }
        ),
        encoding="utf-8",
    )
    loaded = Calibration.load(path)
    assert loaded.vectors == {}
    assert loaded.method == "temperature"


def test_applying_the_two_methods_gives_different_probabilities() -> None:
    cal = fit_calibration(predictions(3000), quantization="nf4-bf16")
    row = predictions(1, seed=99)[0].logits
    assert not np.allclose(cal.apply(row, "temperature"), cal.apply(row, "vector"))


def test_temperature_cannot_change_the_winner_and_vector_can() -> None:
    """The defining difference, and the reason vector scaling is reported rather than assumed."""
    cal = Calibration(
        temperatures={"3-5": 2.0},
        vectors={"3": {"temperature": 2.0, "bias": [-4.0, 2.0, 2.0], "n": 999}},
        quantization="q",
    )
    row = [1.0, 0.5, 0.0]
    assert int(np.argmax(cal.apply(row, "temperature"))) == int(np.argmax(row))
    assert int(np.argmax(cal.apply(row, "vector"))) != int(np.argmax(row))


def test_an_option_count_without_a_vector_fit_falls_back_to_temperature() -> None:
    """Falling back to an unfitted zero bias would be a different method under the same name."""
    cal = Calibration(temperatures={"3-5": 2.0}, vectors={}, quantization="q", method="vector")
    row = [1.0, 0.5, 0.0]
    assert np.allclose(cal.apply(row), cal.apply(row, "temperature"))


def test_with_method_carries_both_fits() -> None:
    cal = fit_calibration(predictions(3000), quantization="nf4-bf16")
    switched = cal.with_method("vector")
    assert switched.method == "vector"
    assert switched.vectors == cal.vectors
    assert switched.temperatures == cal.temperatures


def test_an_unknown_method_is_refused() -> None:
    cal = fit_calibration(predictions(3000), quantization="nf4-bf16")
    with pytest.raises(ValueError, match="unknown method"):
        cal.with_method("isotonic")
    with pytest.raises(ValueError, match="unknown method"):
        cal.apply([1.0, 0.0], "isotonic")


def test_a_bias_of_the_wrong_length_is_refused() -> None:
    """A length mismatch means the bias belongs to a different question shape."""
    with pytest.raises(ValueError, match="bias vector"):
        Calibration(temperatures={}, vectors={"3": {"temperature": 1.0, "bias": [0.0, 0.0]}})


# --- the data requirement ----------------------------------------------------------


def test_too_little_data_for_the_option_count_is_skipped_not_fitted() -> None:
    """26 biases from 40 examples is an artefact of the sample, not a calibration."""
    cal = fit_calibration(predictions(60, n_options=5), quantization="q", min_count=10)
    assert "5" not in cal.vectors
    assert "skipped" in cal.meta["vector_scaling"]["5"]


def test_the_requirement_scales_with_the_option_count() -> None:
    needed_for_3 = MIN_VECTOR_SAMPLES_PER_PARAMETER * 4
    needed_for_5 = MIN_VECTOR_SAMPLES_PER_PARAMETER * 6
    assert needed_for_5 > needed_for_3
    cal = fit_calibration(predictions(needed_for_3 + 10, n_options=3), quantization="q")
    assert "3" in cal.vectors


def test_the_fit_report_lets_the_two_methods_be_compared() -> None:
    cal = fit_calibration(predictions(3000), quantization="nf4-bf16")
    report = cal.meta["vector_scaling"]["3"]
    assert report["nll_vector"] <= report["nll_temperature_only"] + 1e-9
    assert "max_abs_bias" in report and report["n"] == 3000
