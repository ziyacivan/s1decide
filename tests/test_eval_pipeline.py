"""The eval harness end to end on the mock engine, plus dataset normalisation.

The pipeline test needs no GPU, no weights and no network: it runs synthetic items through
the mock engine, so CI can prove the harness itself works. The dataset tests skip unless
`pngwn/system-one-decisions` is already in the local cache.
"""

from __future__ import annotations

import json

import pytest
from eval.data import Coverage, EvalItem, bundle_stats, group_by_state
from eval.metrics import Prediction
from eval.run_eval import (
    RunConfig,
    default_run_id,
    read_jsonl,
    report_from_predictions,
    score_items,
    write_jsonl,
)

from s1decide.calibrate import Calibration, fit_calibration
from s1decide.engine.mock import MockEngine
from s1decide.prompt import FORMAT_VERSION


def item(i: int, qtype: str = "choice", n_options: int = 4, state: str | None = None) -> EvalItem:
    if qtype == "noul":
        options = ("no", "yes")
    elif qtype == "score":
        options = ("low", "medium", "high")
    else:
        options = tuple(f"option {j}" for j in range(n_options))
    return EvalItem(
        id=f"it-{i:04d}",
        family=("alpha", "beta")[i % 2],
        qtype=qtype,
        state=state if state is not None else f"state number {i // 3}",
        instructions=f"Question {i}?",
        options=options,
        answer_idx=i % len(options),
        split="test",
    )


# --- EvalItem -> Question -----------------------------------------------------


def test_choice_item_becomes_a_choice_question() -> None:
    q = item(0, "choice", 4).to_question()
    assert q.qtype == "choice"
    assert q.labels == tuple(f"option {j}" for j in range(4))


def test_score_item_keeps_level_order() -> None:
    q = item(0, "score").to_question()
    assert q.qtype == "score"
    assert q.labels == ("low", "medium", "high")


def test_noul_item_uses_our_label_order() -> None:
    """Index 1 must be `yes` so `Result.noul` means P(true)."""
    q = item(0, "noul").to_question()
    assert q.qtype == "noul"
    assert q.labels == ("no", "yes")


def test_to_question_rejects_options_that_do_not_match_the_primitive() -> None:
    bad = EvalItem(
        id="bad",
        family="f",
        qtype="noul",
        state="s",
        instructions="q?",
        options=("yes", "no"),
        answer_idx=0,
        split="test",
    )
    with pytest.raises(ValueError, match="do not match"):
        bad.to_question()


def test_question_name_defaults_to_the_item_id() -> None:
    assert item(3).to_question().name == "it-0003"


# --- grouping -----------------------------------------------------------------


def test_group_by_state_bundles_and_preserves_order() -> None:
    items = [item(i) for i in range(9)]  # three per state
    groups = group_by_state(items)
    assert len(groups) == 3
    assert [len(g) for _, g in groups] == [3, 3, 3]
    assert [i.id for i in groups[0][1]] == ["it-0000", "it-0001", "it-0002"]


def test_bundle_stats_describes_the_bundling() -> None:
    stats = bundle_stats(group_by_state([item(i) for i in range(9)]))
    assert stats == {
        "states": 3,
        "questions": 9,
        "mean_questions_per_state": 3.0,
        "max_questions_per_state": 3,
        "bundle_size_counts": {3: 3},
    }


def test_coverage_reports_what_it_dropped() -> None:
    c = Coverage(
        total=100,
        kept=90,
        dropped_high_cardinality=10,
        max_options_kept=26,
        dropped_families={"banking77": 10},
    )
    assert c.fraction_kept == 0.9
    assert c.to_json()["dropped_families"] == {"banking77": 10}


# --- scoring ------------------------------------------------------------------


def test_score_items_returns_one_prediction_per_item_in_order() -> None:
    items = [item(i) for i in range(12)]
    preds, timing = score_items(MockEngine(), items, progress_every=0)
    assert [p.id for p in preds] == [i.id for i in items]
    assert all(p.n_options == len(i.options) for p, i in zip(preds, items))
    assert all(p.answer_idx == i.answer_idx for p, i in zip(preds, items))
    assert timing["states"] == 4
    assert timing["questions"] == 12


def test_score_items_bundling_does_not_change_the_answer() -> None:
    """One state with three questions must score each exactly as it would alone."""
    shared = [item(i, state="one shared state") for i in range(3)]
    engine = MockEngine()
    bundled, _ = score_items(engine, shared, progress_every=0)
    for original, prediction in zip(shared, bundled):
        alone, _ = score_items(engine, [original], progress_every=0)
        assert alone[0].logits == prediction.logits


def test_score_items_carries_family_and_qtype_through() -> None:
    items = [item(0, "choice"), item(1, "noul"), item(2, "score")]
    preds, _ = score_items(MockEngine(), items, progress_every=0)
    assert [p.qtype for p in preds] == ["choice", "noul", "score"]
    assert [p.family for p in preds] == [i.family for i in items]


# --- report -------------------------------------------------------------------


def make_run(tmp_path, n: int = 120):
    val = [item(i) for i in range(n)]
    test = [item(i + n) for i in range(n)]
    engine = MockEngine()
    val_preds, _ = score_items(engine, val, progress_every=0)
    test_preds, _ = score_items(engine, test, progress_every=0)
    calibration = fit_calibration(val_preds, quantization="mock", min_count=10)
    config = RunConfig(model="mock", engine="mock", quantization="mock", run_id="unit-test")
    coverage = {
        "val": Coverage(len(val), len(val), 0, 26, {}),
        "test": Coverage(len(test), len(test), 0, 26, {}),
    }
    return report_from_predictions(
        config=config,
        test_predictions=test_preds,
        val_predictions=val_preds,
        calibration=calibration,
        coverage=coverage,
    )


def test_report_has_every_required_section(tmp_path) -> None:
    report = make_run(tmp_path)
    for key in (
        "meta",
        "coverage",
        "calibration",
        "overall",
        "overall_calibrated",
        "by_option_count",
        "by_family",
        "by_qtype",
        "controls",
        "validation",
    ):
        assert key in report, key


def test_report_always_carries_brier_and_both_controls(tmp_path) -> None:
    """AGENTS.md anti-pattern: never report ECE without Brier and the base-rate control."""
    report = make_run(tmp_path)
    assert "brier_top_label" in report["overall"]
    assert "brier_multiclass" in report["overall"]
    assert set(report["controls"]) == {"uniform", "base_rate"}


def test_report_records_provenance(tmp_path) -> None:
    meta = make_run(tmp_path)["meta"]
    assert meta["run_id"] == "unit-test"
    assert meta["format_version"] == FORMAT_VERSION
    assert meta["dataset_license"] == "CC-BY-NC-4.0"
    assert "commit" in meta["git"]
    assert meta["quantization"] == "mock"


def test_calibration_does_not_move_accuracy(tmp_path) -> None:
    report = make_run(tmp_path)
    assert report["overall"]["accuracy"] == pytest.approx(report["overall_calibrated"]["accuracy"])


def test_report_is_json_serialisable(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(make_run(tmp_path), indent=2), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["overall"]["n"] == 120


# --- storage ------------------------------------------------------------------


def test_predictions_round_trip_through_jsonl(tmp_path) -> None:
    preds, _ = score_items(MockEngine(), [item(i) for i in range(5)], progress_every=0)
    path = write_jsonl(tmp_path / "predictions.jsonl", [p.to_json() for p in preds])
    assert [Prediction.from_json(r) for r in read_jsonl(path)] == preds


def test_jsonl_is_utf8_with_lf(tmp_path) -> None:
    path = write_jsonl(tmp_path / "p.jsonl", [{"state": "ödeme alındı"}])
    assert b"\r\n" not in path.read_bytes()
    assert "ödeme alındı" in path.read_text(encoding="utf-8")


def test_calibration_file_round_trips_in_a_run(tmp_path) -> None:
    preds, _ = score_items(MockEngine(), [item(i) for i in range(60)], progress_every=0)
    cal = fit_calibration(preds, quantization="mock", min_count=10)
    assert Calibration.load(cal.save(tmp_path / "calibration.json")) == cal


def test_default_run_id_is_descriptive_and_sortable() -> None:
    rid = default_run_id(
        RunConfig(model="unsloth/Qwen3.8-27B-unsloth-bnb-4bit", quantization="nf4-bf16")
    )
    assert rid.startswith("20")
    assert "qwen38-27b-unsloth-bnb-4bit" in rid
    assert rid.endswith("nf4-bf16-test-zeroshot")


# --- the real dataset (skips without the local cache) -------------------------


@pytest.fixture(scope="module")
def dataset_available() -> bool:
    datasets = pytest.importorskip("datasets")
    try:
        datasets.load_dataset("pngwn/system-one-decisions", split="val")
    except Exception as exc:
        pytest.skip(f"dataset not cached: {type(exc).__name__}")
    return True


@pytest.mark.weights
def test_real_dataset_normalises_and_reports_coverage(dataset_available: bool) -> None:
    from eval.data import load_system_one_decisions

    items, coverage = load_system_one_decisions("val")
    assert coverage.total == 1452
    assert coverage.kept == len(items)
    # banking77 (77 options) and tickets_queue (52) exceed the 26 single-token labels.
    assert coverage.dropped_high_cardinality == 160
    assert set(coverage.dropped_families) == {"banking77", "tickets_queue"}
    assert all(i.n_options <= 26 for i in items)


@pytest.mark.weights
def test_real_noul_rows_are_remapped_to_our_label_order(dataset_available: bool) -> None:
    """The dataset says ('yes','no') with 0=yes; we store ('no','yes') with 1=yes."""
    from datasets import load_dataset
    from eval.data import load_system_one_decisions

    raw = load_dataset("pngwn/system-one-decisions", split="val")
    items, _ = load_system_one_decisions("val")
    by_id = {i.id: i for i in items}

    checked = 0
    for index, row in enumerate(raw):
        if row["question_type"] != "noul":
            continue
        from eval.data import _item_id

        key = _item_id("val", index, row["state"], row["question"])
        if key not in by_id:
            continue
        got = by_id[key]
        assert got.options == ("no", "yes")
        assert got.options[got.answer_idx] == row["options"][row["answer_index"]]
        checked += 1
    assert checked > 100


@pytest.mark.weights
def test_real_items_build_valid_questions(dataset_available: bool) -> None:
    from eval.data import load_system_one_decisions

    items, _ = load_system_one_decisions("val")
    for i in items:
        q = i.to_question()
        assert q.labels == i.options
        assert 0 <= i.answer_idx < len(q.labels)


@pytest.mark.weights
def test_real_splits_do_not_share_states(dataset_available: bool) -> None:
    """No leakage: a calibration fitted on val must not have seen test states."""
    from eval.data import load_system_one_decisions

    val, _ = load_system_one_decisions("val")
    test, _ = load_system_one_decisions("test")
    assert not {i.state for i in val} & {i.state for i in test}


# --- generated summary --------------------------------------------------------


def test_summary_is_generated_from_the_report(tmp_path) -> None:
    """The summary table must be produced from metrics.json, never hand-written."""
    from eval.summary import render_summary, write_summary

    report = make_run(tmp_path)
    (tmp_path / "metrics.json").write_text(json.dumps(report), encoding="utf-8")
    path = write_summary(tmp_path)
    text = path.read_text(encoding="utf-8")

    assert "GENERATED from metrics.json" in text
    assert report["meta"]["run_id"] in text
    # headline numbers appear with the precision the renderer chose
    assert f"{report['overall']['accuracy']:.4f}" in text
    assert f"{report['overall']['ece']:.4f}" in text
    # every section that carries a metric
    for heading in ("## Coverage", "## Headline", "## Calibration", "## By option-count bucket"):
        assert heading in text
    # both controls are always present, per the anti-pattern rule
    assert "control: uniform" in text
    assert "control: base_rate" in text
    assert render_summary(report) == text


def test_summary_marks_a_dirty_tree(tmp_path) -> None:
    from eval.summary import render_summary

    report = make_run(tmp_path)
    report["meta"]["git"] = {"commit": "a" * 40, "branch": "master", "dirty": True}
    assert "tree dirty at run time" in render_summary(report)
    report["meta"]["git"]["dirty"] = False
    assert "tree dirty at run time" not in render_summary(report)


def test_summary_writes_lf_and_utf8(tmp_path) -> None:
    from eval.summary import write_summary

    (tmp_path / "metrics.json").write_text(json.dumps(make_run(tmp_path)), encoding="utf-8")
    path = write_summary(tmp_path)
    assert b"\r\n" not in path.read_bytes()
    path.read_text(encoding="utf-8")
