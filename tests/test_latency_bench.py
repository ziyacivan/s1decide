"""Benchmark plumbing, checked on CPU with the mock engine.

The timings themselves need a GPU and are the point of `uv run task bench`; what is tested
here is that the harness measures and reports the right things.
"""

from __future__ import annotations

import json
import statistics

import pytest
from eval.latency_bench import BenchConfig, Measurement, _fit_two_term, make_questions, make_state

from s1decide.prompt import render

# --- inputs -------------------------------------------------------------------


def test_make_state_reaches_the_target_length() -> None:
    encode = str.split  # one "token" per word, enough to exercise the loop
    state = make_state(encode, 200)
    assert len(encode(state)) >= 200
    assert state == state.strip()


def test_make_state_is_deterministic() -> None:
    assert make_state(str.split, 120) == make_state(str.split, 120)


@pytest.mark.parametrize("n", [1, 4, 16, 64])
def test_make_questions_returns_unique_valid_questions(n: int) -> None:
    questions = make_questions(n)
    assert len(questions) == n
    assert len({q.name for q in questions}) == n


def test_make_questions_mixes_all_three_primitives() -> None:
    assert {q.qtype for q in make_questions(6)} == {"noul", "choice", "score"}


def test_make_questions_render_into_one_prefix(ticket_state) -> None:
    rendered = render(ticket_state, make_questions(8))
    assert len(rendered.suffixes) == 8
    for i in range(8):
        assert rendered.full(i).startswith(rendered.prefix)


# --- Measurement --------------------------------------------------------------


def measurement(n: int, wall: list[float], suffix_tokens: int = 100) -> Measurement:
    return Measurement(
        n_questions=n,
        wall_ms=wall,
        prefill_ms=[w * 0.6 for w in wall],
        suffix_ms=[w * 0.4 for w in wall],
        passes=1,
        rows_per_pass=n,
        prefix_tokens=1500,
        suffix_tokens=suffix_tokens,
        peak_gib=21.0,
        separate_ms=[w * n for w in wall],
    )


def test_measurement_reports_median_and_quantiles() -> None:
    m = measurement(4, [100.0, 110.0, 120.0, 130.0, 140.0])
    payload = m.to_json()
    assert payload["median_ms"] == pytest.approx(120.0)
    assert payload["min_ms"] == 100.0
    assert payload["max_ms"] == 140.0
    assert payload["p10_ms"] <= payload["median_ms"] <= payload["p90_ms"]
    assert payload["samples"] == 5


def test_measurement_reports_the_separate_call_comparison() -> None:
    m = measurement(4, [100.0])
    assert m.median_separate_ms == pytest.approx(400.0)
    assert m.to_json()["median_separate_ms"] == pytest.approx(400.0)


def test_measurement_without_separate_samples_reports_none() -> None:
    m = measurement(1, [100.0])
    m.separate_ms = []
    assert m.median_separate_ms is None


def test_measurement_reports_cost_per_suffix_token() -> None:
    m = measurement(4, [100.0], suffix_tokens=200)
    # suffix is 40 ms of a 100 ms call, over 200 tokens
    assert m.to_json()["ms_per_suffix_token"] == pytest.approx(0.2)


def test_measurement_stdev_is_zero_for_a_single_sample() -> None:
    assert measurement(1, [100.0]).to_json()["stdev_ms"] == 0.0


def test_measurement_is_json_serialisable() -> None:
    json.dumps(measurement(4, [100.0, 101.0]).to_json())


# --- decomposition ------------------------------------------------------------


def test_fit_recovers_a_known_per_pass_and_per_token_cost() -> None:
    """Synthesise suffix times from a known model and check the fit finds it back."""
    per_pass, per_token = 25.0, 0.8
    points = []
    for n, passes, tokens in ((1, 1, 60), (4, 1, 240), (16, 3, 960), (64, 10, 3840)):
        total = per_pass * passes + per_token * tokens
        m = measurement(n, [total / 0.4], suffix_tokens=tokens)
        m.passes = passes
        m.suffix_ms = [total]
        points.append(m)
    fit = _fit_two_term(points)
    assert fit["per_pass_ms"] == pytest.approx(per_pass, rel=0.01)
    assert fit["per_token_ms"] == pytest.approx(per_token, rel=0.01)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-6)


def test_fit_attributes_a_token_bound_workload_to_tokens() -> None:
    """If cost tracks tokens and not passes, the fit must say so."""
    points = []
    for n, passes, tokens in ((1, 1, 60), (4, 1, 240), (16, 8, 960), (64, 32, 3840)):
        m = measurement(n, [1.0], suffix_tokens=tokens)
        m.passes = passes
        m.suffix_ms = [1.0 * tokens]
        points.append(m)
    fit = _fit_two_term(points)
    assert fit["per_token_ms"] > 0.5
    assert abs(fit["per_pass_ms"]) < 0.2 * fit["per_token_ms"] * 60


# --- config -------------------------------------------------------------------


def test_default_config_matches_the_brief() -> None:
    config = BenchConfig()
    assert config.question_counts == (1, 4, 16, 64)
    assert config.repeats == 20
    assert config.state_tokens == 1500


def test_config_is_json_serialisable() -> None:
    assert json.loads(json.dumps(BenchConfig().to_json()))["repeats"] == 20


# --- plotting from a stored file ----------------------------------------------


def test_plot_is_drawn_from_the_committed_json(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from eval.latency_bench import plot

    points = []
    for n, wall in ((1, 1500.0), (4, 1700.0), (16, 2500.0), (64, 5800.0)):
        m = measurement(n, [wall, wall * 1.01], suffix_tokens=60 * n)
        points.append(m.to_json())
    payload = {
        "meta": {
            "run_id": "unit",
            "model": "x/y",
            "quantization": "nf4-bf16",
            "gpu": "RTX 3090",
            "repeats": 2,
            "state_tokens_actual": 1500,
        },
        "target": {"ratio_64_over_1": 5800.0 / 1500.0, "met": False},
        "decomposition": {"per_pass_ms": 1.0, "per_token_ms": 1.0, "r_squared": 1.0},
        "points": points,
    }
    (tmp_path / "latency.json").write_text(json.dumps(payload), encoding="utf-8")
    path = plot(tmp_path)
    assert path.is_file()
    assert path.stat().st_size > 5000


def test_median_of_an_even_sample_count_is_the_mean_of_the_middle_two() -> None:
    """Pinned so the reported median is unambiguous with the default 20 repeats."""
    assert statistics.median([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)
