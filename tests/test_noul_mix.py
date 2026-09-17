"""Genuine `Noul` supply, and the per-primitive mix that makes it visible.

A `qtype` count reported `Noul` at 79% of training while the primitive we actually publish was
9%, from a single source. The difference is stage-1 rows: expanding a 60- or 151-option
question emits one `noul`-shaped row per candidate ("is `card arrival` the intent of this
message?"), which is an option-membership question wearing a Noul's clothes. These tests keep
the two apart and keep the genuine supply above a floor.
"""

from __future__ import annotations

import json

import pytest
from data.build.pipeline import (
    MIN_GENUINE_NOUL_FAMILIES,
    MIN_GENUINE_NOUL_SHARE,
    primitive_mix,
)

from s1decide.tasks import repo_root


def row(qtype: str, family: str, split: str = "train", stage: int | None = None) -> dict:
    return {"qtype": qtype, "family": family, "split": split, "stage": stage}


# --- primitive_mix -------------------------------------------------------------


def test_stage_one_rows_are_counted_apart_from_genuine_ones() -> None:
    rows = [row("noul", "banking77", stage=1)] * 8 + [row("noul", "go_emotions")] * 2
    mix = primitive_mix(rows)["train"]["noul"]
    assert mix["total"] == 10
    assert mix["stage1"] == 8
    assert mix["genuine"] == 2
    assert mix["genuine_share"] == pytest.approx(0.2)


def test_the_share_is_of_the_whole_split_not_of_the_primitive() -> None:
    """The question is "how much of training is genuine Noul", not "how much of Noul"."""
    rows = [row("noul", "go_emotions")] * 2 + [row("choice", "mmlu")] * 8
    assert primitive_mix(rows)["train"]["noul"]["genuine_share"] == pytest.approx(0.2)


def test_the_family_breakdown_shows_a_primitive_carried_by_one_source() -> None:
    rows = [row("noul", "go_emotions")] * 5 + [row("noul", "banking77", stage=1)] * 5
    assert primitive_mix(rows)["train"]["noul"]["genuine_by_family"] == {"go_emotions": 5}


def test_splits_are_reported_separately() -> None:
    rows = [row("noul", "go_emotions"), row("noul", "go_emotions", split="val")]
    mix = primitive_mix(rows)
    assert set(mix) == {"train", "val"}
    assert mix["val"]["noul"]["genuine"] == 1


def test_an_empty_input_does_not_divide_by_zero() -> None:
    assert primitive_mix([]) == {}


# --- the floor itself ----------------------------------------------------------


def test_the_floor_is_a_floor_not_a_target() -> None:
    assert 0.10 <= MIN_GENUINE_NOUL_SHARE <= 0.25
    assert MIN_GENUINE_NOUL_FAMILIES >= 2


@pytest.fixture(scope="module")
def manifest() -> dict:
    path = repo_root() / "data" / "processed" / "manifest.json"
    if not path.is_file():
        pytest.skip("no built corpus; run `uv run task data`")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_built_corpus_meets_the_genuine_noul_floor(manifest: dict) -> None:
    noul = manifest["by_primitive"]["train"]["noul"]
    assert noul["genuine_share"] >= MIN_GENUINE_NOUL_SHARE, (
        f"genuine Noul is {noul['genuine_share']:.1%} of training, below the "
        f"{MIN_GENUINE_NOUL_SHARE:.0%} floor — {noul['stage1']} of {noul['total']} noul rows are "
        "stage-1 option-membership questions, which are a different task"
    )


def test_genuine_noul_comes_from_more_than_one_source(manifest: dict) -> None:
    families = manifest["by_primitive"]["train"]["noul"]["genuine_by_family"]
    assert len(families) >= MIN_GENUINE_NOUL_FAMILIES, (
        f"genuine Noul comes only from {sorted(families)}; one source carrying a whole "
        "primitive makes its quirks indistinguishable from the primitive's behaviour"
    )


def test_the_manifest_separates_stage_one_for_every_primitive(manifest: dict) -> None:
    for split, per_qtype in manifest["by_primitive"].items():
        for qtype, counts in per_qtype.items():
            assert counts["total"] == counts["stage1"] + counts["genuine"], f"{split}/{qtype}"


def test_only_noul_has_stage_one_rows(manifest: dict) -> None:
    """Stage 1 scores one candidate at a time, so it can only ever be a yes/no question."""
    for split, per_qtype in manifest["by_primitive"].items():
        for qtype, counts in per_qtype.items():
            if qtype != "noul":
                assert counts["stage1"] == 0, f"{split}/{qtype} has stage-1 rows"


# --- the MMLU-derived source ---------------------------------------------------


def mmlu_rows(*items: tuple[str, list[str], int]) -> list[dict]:
    return [{"question": q, "choices": c, "answer": a} for q, c, a in items]


def test_one_question_yields_one_true_statement_and_two_false_ones() -> None:
    from data.build.sources import _mmlu_noul

    rows = list(_mmlu_noul(mmlu_rows(("What is 2+2?", ["3", "4", "5", "6"], 1))))
    assert len(rows) == 3
    assert sum(r["answer_idx"] for r in rows) == 1, "exactly one statement may be true"
    assert {r["state"] for r in rows} == {"What is 2+2?"}
    true_row = next(r for r in rows if r["answer_idx"] == 1)
    assert '"4"' in true_row["instructions"]


def test_the_false_statements_name_distractors_not_the_answer() -> None:
    from data.build.sources import _mmlu_noul

    rows = list(_mmlu_noul(mmlu_rows(("q", ["a", "b", "c", "d"], 0))))
    for r in rows:
        if r["answer_idx"] == 0:
            assert '"a"' not in r["instructions"], "a false statement must not name the answer"


def test_every_row_is_a_two_option_yes_no() -> None:
    from data.build.sources import _mmlu_noul

    for r in _mmlu_noul(mmlu_rows(("q", ["a", "b", "c", "d"], 2))):
        assert r["options"] == ("no", "yes")
        assert r["answer_idx"] in (0, 1)


def test_a_question_with_duplicate_options_is_skipped() -> None:
    """A statement that is simultaneously true and false teaches the wrong thing."""
    from data.build.sources import _mmlu_noul

    assert list(_mmlu_noul(mmlu_rows(("q", ["a", "a", "b", "c"], 0)))) == []


def test_a_question_with_an_empty_option_is_skipped() -> None:
    from data.build.sources import _mmlu_noul

    assert list(_mmlu_noul(mmlu_rows(("q", ["a", "", "b", "c"], 0)))) == []


def test_an_out_of_range_answer_is_skipped_rather_than_crashing_the_build() -> None:
    from data.build.sources import _mmlu_noul

    assert list(_mmlu_noul(mmlu_rows(("q", ["a", "b"], 7)))) == []


def test_distractor_sampling_is_deterministic() -> None:
    """A rebuild must reproduce the corpus exactly (`uv run task data` is seeded)."""
    from data.build.sources import _mmlu_noul

    items = mmlu_rows(("q1", ["a", "b", "c", "d"], 0), ("q2", ["w", "x", "y", "z"], 3))
    first = [r["instructions"] for r in _mmlu_noul(items)]
    second = [r["instructions"] for r in _mmlu_noul(items)]
    assert first == second


def test_the_negative_count_is_configurable() -> None:
    from data.build.sources import _mmlu_noul

    assert len(list(_mmlu_noul(mmlu_rows(("q", ["a", "b", "c", "d"], 0)), negatives=1))) == 2
    assert len(list(_mmlu_noul(mmlu_rows(("q", ["a", "b", "c", "d"], 0)), negatives=3))) == 4


def test_the_mmlu_noul_source_is_registered_as_permissive_training_data() -> None:
    from data.build.licences import classify_licence
    from data.build.sources import SOURCES

    source = next(s for s in SOURCES if s.family == "mmlu_noul")
    assert source.role == "train"
    assert source.primitive == "noul"
    assert classify_licence(source.license) == "train", (
        f"{source.license} is not on the ADR 0004 allowlist, so this source cannot be trained on"
    )


def test_mmlu_noul_avoids_the_benchmark_split() -> None:
    """Training on MMLU's test split would invalidate any later MMLU evaluation of these weights."""
    from data.build.sources import SOURCES

    source = next(s for s in SOURCES if s.family == "mmlu_noul")
    assert source.split != "test"
    assert source.split != "auxiliary_train", "auxiliary_train aggregates share-alike sources"
