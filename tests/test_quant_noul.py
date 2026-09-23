"""Comparing two runtimes on the same rows.

The sampling and the comparison are pure; the scoring needs a GPU and a server and is exercised
by the run itself. What is worth pinning here is that the comparison refuses to compare things
that are not comparable, and that its arithmetic says what the research note says.
"""

from __future__ import annotations

import pytest
from eval.quantization_noul import compare, sample_noul_rows


def rows(n: int, stage: str | None = None) -> list[dict]:
    return [
        {
            "id": f"r{i}",
            "qtype": "noul",
            "stage": stage,
            "state": f"s{i}",
            "instructions": "Is it?",
            "answer_idx": i % 2,
        }
        for i in range(n)
    ]


def pass_of(ids: list[str], p: list[float], truth: list[int], runtime: str = "x") -> dict:
    return {
        "runtime": runtime,
        "ids": ids,
        "p_yes": p,
        "answer_idx": truth,
        "stage": ["None"] * len(ids),
    }


def test_comparing_different_rows_is_refused() -> None:
    """Silently zipping two passes over different samples would produce a plausible number."""
    a = pass_of(["r0", "r1"], [0.5, 0.5], [0, 1])
    b = pass_of(["r0", "r2"], [0.5, 0.5], [0, 1])
    with pytest.raises(ValueError, match="different rows"):
        compare(a, b)


def test_a_one_sided_shift_is_counted_as_one_sided() -> None:
    """The shape the real run produced: every row moving the same way."""
    a = pass_of(["r0", "r1", "r2"], [0.6, 0.7, 0.8], [1, 1, 1])
    b = pass_of(["r0", "r1", "r2"], [0.5, 0.6, 0.7], [1, 1, 1])
    out = compare(a, b)
    assert out["shift_down"] == 3 and out["shift_up"] == 0
    assert out["mean_shift"] == pytest.approx(-0.1)
    assert out["sign_test_p"] == pytest.approx(0.25)


def test_agreement_is_about_the_decision_not_the_probability() -> None:
    """Two runtimes can differ by a lot and still decide the same thing, which is the finding:
    88.7% of decisions agreed while every single probability had moved."""
    a = pass_of(["r0"], [0.95], [1])
    b = pass_of(["r0"], [0.55], [1])
    out = compare(a, b)
    assert out["exact_agreement"] == 1.0
    assert out["mean_absolute_shift"] == pytest.approx(0.40)


def test_a_shift_across_the_boundary_is_a_disagreement() -> None:
    a = pass_of(["r0"], [0.55], [1])
    b = pass_of(["r0"], [0.45], [1])
    assert compare(a, b)["exact_agreement"] == 0.0


def test_accuracy_is_reported_for_both_sides() -> None:
    """Agreement with the other runtime is not correctness, and the note leans on the difference."""
    a = pass_of(["r0", "r1"], [0.9, 0.9], [1, 0])
    b = pass_of(["r0", "r1"], [0.9, 0.1], [1, 0])
    out = compare(a, b)
    assert out["accuracy_first"] == pytest.approx(0.5)
    assert out["accuracy_second"] == pytest.approx(1.0)


def test_the_sample_is_balanced_between_genuine_and_stage_one() -> None:
    """Drawn at random the sample would be almost all stage-1: the fan-out dwarfs everything."""
    pool = rows(20) + rows(2000, stage="1")
    picked = sample_noul_rows(pool, 100)
    assert sum(1 for r in picked if str(r.get("stage")) == "1") == 80
    assert len(picked) == 100


def test_the_sample_is_deterministic() -> None:
    pool = rows(50) + rows(500, stage="1")
    assert [r["id"] for r in sample_noul_rows(pool, 40)] == [
        r["id"] for r in sample_noul_rows(pool, 40)
    ]


def test_the_committed_logits_are_not_quantised_by_the_compute_dtype() -> None:
    """A resolution floor under every calibration number, guarded on the committed artefact.

    `HFEngine` used to run the output head in bf16 and cast afterwards, which widens the type
    without restoring the value: at the magnitude a logit sits at, bf16's spacing is 0.125, so
    the 300 `Noul` rows produced only 84 distinct probabilities and every log-odds landed on an
    exact multiple of an eighth. llama.cpp's fp32 produced 300. An equal-mass ECE bin narrower
    than that floor measures the dtype, not the model.
    """
    import json
    import math
    from pathlib import Path

    from s1decide.tasks import repo_root

    saved = Path(repo_root()) / "results/quant-noul/hf.json"
    if not saved.is_file():
        pytest.skip("no committed quant-noul run")
    p = json.loads(saved.read_text(encoding="utf-8"))["p_yes"]
    assert len(p) == 300
    assert len(set(p)) > 250, (
        f"only {len(set(p))} distinct probabilities in 300 rows; the answer-position logits are "
        "being rounded by the compute dtype again"
    )

    def log_odds(x: float) -> float:
        x = min(max(x, 1e-9), 1 - 1e-9)
        return math.log(x / (1 - x))

    on_eighths = sum(1 for v in map(log_odds, p) if abs(v * 8 - round(v * 8)) < 1e-3)
    assert on_eighths < 30, f"{on_eighths} of 300 log-odds sit on exact eighths — that is bf16"
