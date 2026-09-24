"""Temperature scaling: does it find the right temperature, and does it refuse the wrong one?"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from eval.metrics import Prediction, ece, summarize

from s1decide.calibrate import (
    CALIBRATION_VERSION,
    Calibration,
    CalibrationSet,
    default_method,
    fit_calibration,
    fit_calibration_cells,
    fit_temperature,
    nll_at_temperature,
    option_count_bucket,
    select_methods,
)


def synthetic(n: int, k: int, true_temperature: float, seed: int = 0) -> list[Prediction]:
    """Logits whose softmax at ``true_temperature`` generated the labels.

    So the recoverable optimum really is ``true_temperature``: sampling labels from the
    tempered distribution is exactly the model temperature scaling assumes.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        logits = rng.normal(scale=3.0, size=k)
        z = logits / true_temperature
        p = np.exp(z - z.max())
        p /= p.sum()
        out.append(
            Prediction(
                id=f"s{i}",
                family="synthetic",
                qtype="choice",
                logits=tuple(logits),
                answer_idx=int(rng.choice(k, p=p)),
            )
        )
    return out


# --- fit_temperature ----------------------------------------------------------


@pytest.mark.parametrize("true_temperature", [0.5, 1.0, 2.0, 4.0])
def test_fit_temperature_recovers_the_generating_temperature(true_temperature: float) -> None:
    preds = synthetic(4000, 5, true_temperature, seed=int(true_temperature * 10))
    fitted = fit_temperature([p.logits for p in preds], [p.answer_idx for p in preds])
    assert fitted == pytest.approx(true_temperature, rel=0.15)


def test_fit_temperature_finds_a_minimum_not_just_a_point() -> None:
    preds = synthetic(2000, 4, 2.5, seed=3)
    logits = [p.logits for p in preds]
    labels = [p.answer_idx for p in preds]
    best = fit_temperature(logits, labels)
    at_best = nll_at_temperature(logits, labels, best)
    for other in (best * 0.8, best * 1.25, 1.0):
        assert nll_at_temperature(logits, labels, other) >= at_best - 1e-9


def test_fit_temperature_reduces_ece_on_an_overconfident_model() -> None:
    preds = synthetic(3000, 4, 3.0, seed=11)
    labels = [p.answer_idx for p in preds]
    before = ece([p.probabilities(1.0) for p in preds], labels)
    t = fit_temperature([p.logits for p in preds], labels)
    after = ece([p.probabilities(t) for p in preds], labels)
    assert after < before / 2


def test_fit_temperature_never_changes_accuracy() -> None:
    preds = synthetic(500, 6, 3.0, seed=5)
    labels = [p.answer_idx for p in preds]
    t = fit_temperature([p.logits for p in preds], labels)
    assert summarize([p.probabilities(t) for p in preds], labels)["accuracy"] == pytest.approx(
        summarize([p.probabilities(1.0) for p in preds], labels)["accuracy"]
    )


def test_fit_temperature_is_deterministic() -> None:
    preds = synthetic(300, 3, 1.7, seed=9)
    args = ([p.logits for p in preds], [p.answer_idx for p in preds])
    assert fit_temperature(*args) == fit_temperature(*args)


@pytest.mark.parametrize(
    ("logits", "labels", "message"),
    [([], [], "no data"), ([[0.0, 1.0]], [], "1 logit rows but 0 labels")],
)
def test_fit_temperature_rejects_bad_input(logits, labels, message) -> None:
    with pytest.raises(ValueError, match=message):
        fit_temperature(logits, labels)


def test_fit_temperature_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="invalid bounds"):
        fit_temperature([[0.0, 1.0]], [0], bounds=(2.0, 1.0))


def test_nll_at_temperature_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="temperature"):
        nll_at_temperature([[0.0, 1.0]], [0], 0.0)


# --- Calibration --------------------------------------------------------------


def test_calibration_applies_the_bucket_temperature() -> None:
    cal = Calibration(temperatures={"2": 2.0, "3-5": 4.0}, quantization="nf4-bf16")
    assert cal.temperature_for(2) == 2.0
    assert cal.temperature_for(4) == 4.0
    assert cal.temperature_for(10) == 1.0  # unfitted bucket falls back to the default
    two = cal.apply([0.0, 4.0])
    assert two == pytest.approx(np.exp([0.0, 2.0]) / np.exp([0.0, 2.0]).sum())


def test_identity_calibration_is_a_plain_softmax() -> None:
    cal = Calibration.identity("bf16")
    assert cal.apply([0.0, math.log(3.0)]) == pytest.approx([0.25, 0.75])


def test_calibration_rejects_non_positive_temperatures() -> None:
    with pytest.raises(ValueError, match="positive"):
        Calibration(temperatures={"2": 0.0})
    with pytest.raises(ValueError, match="positive"):
        Calibration(temperatures={}, default=-1.0)


def test_calibration_round_trips_through_disk(tmp_path) -> None:
    cal = Calibration(
        temperatures={"2": 1.25, "3-5": 2.5},
        quantization="nf4-bf16",
        meta={"run_id": "r1"},
    )
    path = cal.save(tmp_path / "calibration.json")
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == CALIBRATION_VERSION
    assert Calibration.load(path) == cal


def test_loading_a_future_version_fails_loudly(tmp_path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"version": "9.9", "temperatures": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        Calibration.load(path)


def test_calibration_refuses_a_quantization_mismatch() -> None:
    """Fitting in BF16 and shipping Q4 is a named anti-pattern; it must fail, not warn."""
    cal = Calibration(temperatures={"2": 2.0}, quantization="bf16")
    cal.check_matches("bf16")
    with pytest.raises(ValueError, match="re-fit at the deployment quantization"):
        cal.check_matches("q4_k_m")


# --- fit_calibration ----------------------------------------------------------


def test_fit_calibration_fits_each_bucket_separately() -> None:
    """Two buckets generated at different temperatures must get different fits."""
    preds = [
        Prediction(id=f"a{i}", family="f", qtype="noul", logits=p.logits, answer_idx=p.answer_idx)
        for i, p in enumerate(synthetic(1500, 2, 1.0, seed=1))
    ] + [
        Prediction(id=f"b{i}", family="f", qtype="choice", logits=p.logits, answer_idx=p.answer_idx)
        for i, p in enumerate(synthetic(1500, 4, 4.0, seed=2))
    ]
    cal = fit_calibration(preds, quantization="nf4-bf16")
    assert set(cal.temperatures) == {"2", "3-5"}
    assert cal.temperatures["2"] == pytest.approx(1.0, rel=0.2)
    assert cal.temperatures["3-5"] == pytest.approx(4.0, rel=0.25)
    assert cal.temperatures["3-5"] > cal.temperatures["2"]


def test_fit_calibration_skips_buckets_that_are_too_small() -> None:
    preds = [
        *synthetic(40, 4, 2.0, seed=4),
        Prediction(id="lonely", family="f", qtype="noul", logits=(0.0, 1.0), answer_idx=0),
    ]
    cal = fit_calibration(preds, quantization="q", min_count=30)
    assert "3-5" in cal.temperatures
    assert "2" not in cal.temperatures
    assert "fewer than 30" in cal.meta["buckets"]["2"]["skipped"]
    assert cal.temperature_for(2) == 1.0


def test_fit_calibration_records_the_nll_it_improved() -> None:
    cal = fit_calibration(synthetic(600, 4, 3.0, seed=6), quantization="nf4-bf16")
    info = cal.meta["buckets"]["3-5"]
    assert info["nll_after"] < info["nll_before"]
    assert info["n"] == 600
    assert cal.meta["n_predictions"] == 600
    assert cal.quantization == "nf4-bf16"


def test_fit_calibration_carries_caller_metadata() -> None:
    cal = fit_calibration(
        synthetic(50, 4, 1.0), quantization="q", min_count=10, meta={"run_id": "z"}
    )
    assert cal.meta["run_id"] == "z"


def test_option_count_bucket_is_shared_with_the_eval() -> None:
    """The library owns the bucket definition so a calibration file and a report agree."""
    from eval.metrics import option_count_bucket as eval_bucket

    assert eval_bucket is option_count_bucket


@pytest.fixture
def binary_predictions() -> list[Prediction]:
    """Enough two-option rows to fit both a temperature and a bias vector."""
    return synthetic(400, 2, true_temperature=1.7)


# --- which method a quantization deploys -------------------------------------------


@pytest.mark.parametrize(
    "quantization,expected",
    [
        ("q4_k_m", "vector"),
        ("Q5_K_M", "vector"),
        ("q8_0", "vector"),
        ("qwen3.8-27b-gguf", "vector"),
        ("nf4-bf16", "temperature"),
        ("bf16", "temperature"),
        ("int8", "temperature"),
    ],
)
def test_gguf_artefacts_deploy_vector_scaling(quantization, expected) -> None:
    """Measured, not preferred: moving nf4 -> Q4_K_M shifted every one of 300 rows' log-odds by
    about -2.07 nats, and a temperature is monotone so it recovered exactly zero of the decision
    agreement. A per-option bias is the parameter that can absorb a shift."""
    assert default_method(quantization) == expected


def test_a_gguf_fit_deploys_vector_when_there_is_data_for_one(binary_predictions) -> None:
    fit = fit_calibration(binary_predictions, quantization="q4_k_m", primitive="noul")
    assert fit.primitive == "noul"
    assert fit.method == "vector"
    assert fit.temperatures, "the temperature is still fitted and still reported"


def test_a_gguf_fit_with_no_room_for_a_bias_says_so_instead_of_pretending(
    binary_predictions,
) -> None:
    """Deploying "vector" with no fitted vector would fall back to the temperature on every
    question while the file claimed otherwise."""
    fit = fit_calibration(binary_predictions[:5], quantization="q4_k_m", min_count=1)
    assert fit.method == "temperature"
    assert "deployed" in fit.meta["vector_scaling"]


def test_the_fit_records_that_its_target_was_labels(binary_predictions) -> None:
    """nf4 is a diagnostic reference for measuring drift, not a judge of the right answer -- on
    the Noul study it was the less accurate of the two runtimes."""
    fit = fit_calibration(binary_predictions, quantization="nf4-bf16")
    assert "never agreement with another runtime" in fit.meta["fit_target"]


# --- one file, many fits -----------------------------------------------------------


def test_a_set_keeps_one_fit_per_quantization_and_primitive(tmp_path, binary_predictions) -> None:
    noul = fit_calibration(binary_predictions, quantization="q4_k_m", primitive="noul")
    other = fit_calibration(binary_predictions, quantization="nf4-bf16", primitive="score")
    saved = CalibrationSet().add(noul).add(other)
    path = saved.save(tmp_path / "calibration.json")

    loaded = CalibrationSet.load(path)
    assert set(loaded.entries) == {"q4_k_m/noul", "nf4-bf16/score"}
    assert loaded.get("q4_k_m", "noul").method == "vector"
    assert loaded.get("nf4-bf16", "score").method == "temperature"


def test_a_missing_primitive_falls_back_within_its_quantization(binary_predictions) -> None:
    shared = fit_calibration(binary_predictions, quantization="q4_k_m", primitive="all")
    assert CalibrationSet().add(shared).get("q4_k_m", "noul") is shared


def test_a_missing_quantization_does_not_borrow_from_another(binary_predictions) -> None:
    """The specific mistake the whole quantization study exists to prevent. An identity fit is
    honestly uncalibrated; a borrowed one is quietly wrong by about two nats."""
    fit = fit_calibration(binary_predictions, quantization="q4_k_m", primitive="noul")
    got = CalibrationSet().add(fit).get("nf4-bf16", "noul")
    assert got.quantization == "nf4-bf16"
    assert got.temperatures == {} and got.default == 1.0


def test_an_older_single_fit_file_still_loads(tmp_path, binary_predictions) -> None:
    """Files written before the set schema hold one fit and no primitive; they are still valid."""
    single = fit_calibration(binary_predictions, quantization="nf4-bf16")
    path = single.save(tmp_path / "calibration.json")
    loaded = CalibrationSet.load(path)
    assert set(loaded.entries) == {"nf4-bf16/all"}


# --- ADR 0008: method chosen per (primitive, bucket), out of sample ----------------


def shifted(n: int, shift: float, seed: int = 0) -> list[Prediction]:
    """Two-option rows whose labels come from the logits with a constant bias on option 0.

    A temperature is monotone in the log-odds and cannot remove a constant shift; a bias can.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        logits = rng.normal(scale=2.0, size=2)
        z = logits + np.array([shift, 0.0])
        p = np.exp(z - z.max())
        p /= p.sum()
        out.append(
            Prediction(
                id=f"v{seed}-{i}",
                family="synthetic",
                qtype="noul",
                logits=tuple(logits),
                answer_idx=int(rng.choice(2, p=p)),
            )
        )
    return out


def test_selection_picks_vector_when_the_error_is_a_positional_shift() -> None:
    chosen = select_methods(shifted(600, 2.0), quantization="nf4-bf16", primitive="noul")
    assert chosen["2"]["method"] == "vector"
    assert chosen["2"]["rows"] == 600


def test_selection_keeps_temperature_when_there_is_no_shift() -> None:
    """Out of sample the extra parameters of vector scaling buy nothing, so it must not win."""
    chosen = select_methods(
        synthetic(600, 2, true_temperature=1.7), quantization="nf4-bf16", primitive="noul"
    )
    assert chosen["2"]["method"] == "temperature"


def test_cells_fit_each_primitive_separately_and_deploy_the_chosen_method() -> None:
    noul = shifted(600, 2.0)
    choice = [
        Prediction(
            id=f"c{p.id}", family="other", qtype="choice", logits=p.logits, answer_idx=p.answer_idx
        )
        for p in synthetic(600, 2, true_temperature=1.7, seed=3)
    ]
    cells = fit_calibration_cells(
        noul + choice, quantization="nf4-bf16", group_of=lambda p: p.qtype
    )
    assert set(cells.entries) == {"nf4-bf16/choice", "nf4-bf16/noul"}
    noul_fit = cells.get("nf4-bf16", "noul")
    assert noul_fit.method_by_bucket == {"2": "vector"}
    assert noul_fit.meta["rows"] == 600
    # `apply` without a method deploys the cell's choice, not the fit-wide default.
    logits = (0.3, 0.1)
    assert list(noul_fit.apply(logits)) == list(noul_fit.apply(logits, method="vector"))
    assert list(noul_fit.apply(logits)) != list(noul_fit.apply(logits, method="temperature"))


def test_excluded_families_are_left_out_of_the_fit() -> None:
    rows = shifted(600, 2.0)
    control = [
        Prediction(
            id=f"x{i}", family="ordinal_control", qtype="noul", logits=(5.0, -5.0), answer_idx=0
        )
        for i in range(200)
    ]
    cells = fit_calibration_cells(
        rows + control,
        quantization="nf4-bf16",
        group_of=lambda p: p.qtype,
        exclude_families=["ordinal_control"],
    )
    fit = cells.get("nf4-bf16", "noul")
    assert fit.meta["rows"] == 600
    assert fit.meta["excluded_families"] == ["ordinal_control"]


def test_method_by_bucket_round_trips_and_is_validated(tmp_path) -> None:
    fit = fit_calibration(shifted(400, 2.0), quantization="nf4-bf16", primitive="noul")
    import dataclasses

    chosen = dataclasses.replace(fit, method_by_bucket={"2": "vector"})
    loaded = Calibration.load(chosen.save(tmp_path / "c.json"))
    assert loaded.method_by_bucket == {"2": "vector"}
    with pytest.raises(ValueError, match="unknown method"):
        dataclasses.replace(fit, method_by_bucket={"2": "platt"})
