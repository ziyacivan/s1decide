"""The reference-run registry is only worth having if it is enforced.

A baseline named in prose drifts silently. These tests fail the moment a named run stops being
committed, changes prompt format, or loses a metric a comparison depends on.
"""

from __future__ import annotations

import json

import pytest
from eval.baselines import (
    COMPARED_METRICS,
    REFERENCE_RUNS,
    compare_to_reference,
    load_reference,
)


@pytest.mark.parametrize("name", sorted(REFERENCE_RUNS))
def test_every_reference_run_is_committed(name: str) -> None:
    assert load_reference(name)["meta"]["run_id"] == REFERENCE_RUNS[name].run_id


@pytest.mark.parametrize("name", sorted(REFERENCE_RUNS))
def test_every_reference_run_matches_its_declared_format(name: str) -> None:
    """The registry's format is what a comparison is gated on, so it must not be a guess."""
    payload = load_reference(name)
    assert payload["meta"]["format_version"] == REFERENCE_RUNS[name].format_version


@pytest.mark.parametrize("name", sorted(REFERENCE_RUNS))
def test_every_reference_run_carries_the_metrics_a_comparison_needs(name: str) -> None:
    payload = load_reference(name)
    for block in ("overall", "overall_calibrated"):
        for metric, _lower_is_better in COMPARED_METRICS:
            assert metric in payload[block], f"{name}.{block} is missing {metric}"


def test_the_zero_shot_reference_is_the_current_prompt_format() -> None:
    """A stale reference would silently make every S1 comparison unusable."""
    from s1decide.prompt import FORMAT_VERSION

    assert REFERENCE_RUNS["zeroshot"].format_version == FORMAT_VERSION


def test_the_zero_shot_reference_carries_its_skill_scores() -> None:
    """Brier skill is the headline number S1 has to move, so it has to be in the baseline."""
    payload = load_reference("zeroshot")
    skill = payload["skill_calibrated"]["base_rate"]
    assert skill["brier_multiclass"] > 0
    assert skill["accuracy_gain"] > 0
    assert payload["risk_coverage_calibrated"]["curve"]


def test_an_unknown_reference_name_is_refused() -> None:
    with pytest.raises(KeyError, match="unknown reference run"):
        load_reference("s3-grpo")


def test_a_missing_run_directory_says_what_to_do(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="not committed"):
        load_reference("zeroshot", results_root=tmp_path)


def test_comparing_against_itself_is_all_zeroes() -> None:
    payload = load_reference("zeroshot")
    comparison = compare_to_reference(payload)
    assert comparison["reference"] == REFERENCE_RUNS["zeroshot"].run_id
    for metric, result in comparison["metrics"].items():
        assert result["delta"] == pytest.approx(0.0), metric


def test_an_improvement_is_reported_in_the_right_direction() -> None:
    payload = json.loads(json.dumps(load_reference("zeroshot")))
    payload["overall_calibrated"]["accuracy"] += 0.05
    payload["overall_calibrated"]["ece"] -= 0.01
    metrics = compare_to_reference(payload)["metrics"]
    assert metrics["accuracy"]["better"] is True
    assert metrics["ece"]["better"] is True


def test_a_regression_is_reported_as_a_regression() -> None:
    payload = json.loads(json.dumps(load_reference("zeroshot")))
    payload["overall_calibrated"]["accuracy"] -= 0.05
    payload["overall_calibrated"]["ece"] += 0.01
    metrics = compare_to_reference(payload)["metrics"]
    assert metrics["accuracy"]["better"] is False
    assert metrics["ece"]["better"] is False


def test_comparing_across_prompt_formats_is_refused() -> None:
    """The 0.1 run is still committed; nothing should ever diff it against a 0.2 run."""
    payload = json.loads(json.dumps(load_reference("zeroshot")))
    payload["meta"]["format_version"] = "0.1"
    with pytest.raises(ValueError, match="prompts differ"):
        compare_to_reference(payload)
