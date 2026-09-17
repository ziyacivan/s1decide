"""The mock engine and the engine contract helpers."""

from __future__ import annotations

import pytest

from s1decide.engine.base import Engine, EngineError, EngineOutput, validate_output
from s1decide.engine.mock import MockEngine


def test_mock_conforms_to_the_engine_protocol() -> None:
    assert isinstance(MockEngine(), Engine)
    assert MockEngine().name == "mock"


def test_mock_returns_one_row_per_suffix_and_one_logit_per_label() -> None:
    out = MockEngine().score("p", ["s1", "s2"], [("A", "B", "C"), ("no", "yes")])
    assert isinstance(out, EngineOutput)
    assert len(out) == 2
    assert len(out.logits[0]) == 3
    assert len(out.logits[1]) == 2
    assert out.meta["rows"] == 2


def test_mock_is_deterministic_and_seed_sensitive() -> None:
    a = MockEngine(seed=0).score("p", ["s"], [("A", "B")])
    b = MockEngine(seed=0).score("p", ["s"], [("A", "B")])
    c = MockEngine(seed=1).score("p", ["s"], [("A", "B")])
    assert a.logits == b.logits
    assert a.logits != c.logits


def test_mock_logits_stay_within_scale() -> None:
    out = MockEngine(scale=2.0).score("p", [f"s{i}" for i in range(50)], [("A", "B", "C")] * 50)
    assert all(-2.0 <= x <= 2.0 for row in out.logits for x in row)


def test_mock_differs_across_labels_suffixes_and_prefixes() -> None:
    e = MockEngine()
    row = e.score("p", ["s"], [("A", "B", "C")]).logits[0]
    assert len(set(row)) == 3
    assert e.score("p", ["s"], [("A",)]).logits != e.score("p", ["t"], [("A",)]).logits
    assert e.score("p", ["s"], [("A",)]).logits != e.score("q", ["s"], [("A",)]).logits


def test_mock_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="label sets"):
        MockEngine().score("p", ["s1", "s2"], [("A", "B")])


def test_mock_rejects_non_positive_scale() -> None:
    with pytest.raises(ValueError, match="scale"):
        MockEngine(scale=0.0)


# --- validate_output ---------------------------------------------------------


def test_validate_output_accepts_a_matching_answer() -> None:
    validate_output(EngineOutput(logits=((0.1, 0.2), (1.0,))), ["a", "b"], [("x", "y"), ("z",)])


def test_validate_output_rejects_wrong_row_count() -> None:
    with pytest.raises(EngineError, match="rows"):
        validate_output(EngineOutput(logits=((0.1, 0.2),)), ["a", "b"], [("x", "y"), ("z",)])


def test_validate_output_rejects_wrong_label_count() -> None:
    with pytest.raises(EngineError, match="logits for"):
        validate_output(EngineOutput(logits=((0.1,),)), ["a"], [("x", "y")])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), "0.5"])
def test_validate_output_rejects_non_finite_or_non_numeric(bad) -> None:
    with pytest.raises(EngineError, match="non-finite"):
        validate_output(EngineOutput(logits=((bad, 0.0),)), ["a"], [("x", "y")])
