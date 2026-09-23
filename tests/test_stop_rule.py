"""The S1 stop rule, exactly as the owner amended it on 2026-09-24."""

from __future__ import annotations

from train.stop_rule import marginal_gap, stop_decision


def entry(rows: int, teacher: tuple[float, float] = (0.34, 0.59), **bss: float) -> dict:
    """A checkpoint with teacher-Score (bss, accuracy) and other groups' BSS."""
    metrics = {
        "score/teacher": {
            "bss": teacher[0],
            "accuracy": teacher[1],
            "predicted_marginal": [0.5, 0.5],
            "target_marginal": [0.5, 0.5],
        }
    }
    for group, value in bss.items():
        metrics[group] = {
            "bss": value,
            "accuracy": 0.9,
            "predicted_marginal": [0.5, 0.5],
            "target_marginal": [0.5, 0.5],
        }
    return {"rows": rows, "metrics": metrics}


def test_row_zero_alone_never_stops() -> None:
    assert stop_decision([entry(0)])["stop"] is False


def test_one_teacher_checkpoint_below_row_zero_is_a_warning_not_a_stop() -> None:
    decision = stop_decision([entry(0), entry(5000, teacher=(0.30, 0.59))])
    assert decision["stop"] is False
    assert any("first time" in w for w in decision["warnings"])


def test_two_consecutive_teacher_checkpoints_below_row_zero_stop() -> None:
    history = [entry(0), entry(5000, teacher=(0.30, 0.59)), entry(10000, teacher=(0.34, 0.55))]
    decision = stop_decision(history)
    assert decision["stop"] is True
    assert "5000 and 10000" in decision["reasons"][0]


def test_a_recovery_in_between_resets_the_count() -> None:
    history = [
        entry(0),
        entry(5000, teacher=(0.30, 0.59)),
        entry(10000, teacher=(0.40, 0.60)),
        entry(15000, teacher=(0.30, 0.59)),
    ]
    assert stop_decision(history)["stop"] is False


def test_bss_turning_negative_after_being_positive_stops_immediately() -> None:
    history = [entry(0, noul=-0.08), entry(5000, noul=0.26), entry(10000, noul=-0.01)]
    decision = stop_decision(history)
    assert decision["stop"] is True
    assert "noul BSS turned negative" in decision["reasons"][0]


def test_bss_that_was_never_positive_does_not_stop() -> None:
    """Row-0 `noul` BSS was -0.083 in the smoke; staying negative is not 'turning'."""
    assert stop_decision([entry(0, noul=-0.08), entry(5000, noul=-0.05)])["stop"] is False


def test_marginal_drift_is_a_warning_only() -> None:
    row0 = entry(0)
    drifted = entry(5000)
    drifted["metrics"]["score/teacher"]["predicted_marginal"] = [0.9, 0.1]
    decision = stop_decision([row0, drifted])
    assert decision["stop"] is False
    assert any("marginal drift" in w for w in decision["warnings"])


def test_marginal_gap_is_total_variation() -> None:
    assert marginal_gap({"predicted_marginal": [0.9, 0.1], "target_marginal": [0.5, 0.5]}) == 0.4
