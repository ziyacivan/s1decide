"""Stage-1 class balance and family weighting (Phase 1 Step 2d).

Row counts are not what the model sees. These tests are mostly about that gap: the raw corpus is
75% stage-1 rows by design, and the weighted draw the trainer takes is not.
"""

from __future__ import annotations

import json
import random

import pytest
from data.build.balance import (
    DEFAULT_TARGET_MIX,
    MAX_OVERSAMPLE,
    STAGE1_TRAIN_NEGATIVES,
    effective_mix,
    family_weights,
    mix_table,
    rank_negatives,
    row_group,
    subsample_stage1,
)

from s1decide.tasks import repo_root


def make(
    family: str, qtype: str = "noul", stage: int | None = None, split: str = "train", **extra
) -> dict:
    return {"family": family, "qtype": qtype, "stage": stage, "split": split, **extra}


# --- grouping ------------------------------------------------------------------


def test_stage_one_is_its_own_group_not_a_noul() -> None:
    """They share a shape and are different tasks; collapsing them is what hid the problem."""
    assert row_group(make("banking77", stage=1)) == "stage1"
    assert row_group(make("go_emotions")) == "noul"


def test_a_stage_two_row_is_an_ordinary_choice() -> None:
    assert row_group(make("banking77", qtype="choice", stage=2)) == "choice"


def test_score_keeps_its_own_group() -> None:
    assert row_group(make("ordinal_control", qtype="score")) == "score"


# --- hard-negative ordering -----------------------------------------------------


def test_scores_order_the_negatives_hardest_first() -> None:
    order = rank_negatives([0, 1, 2], {0: 0.1, 1: 0.9, 2: 0.4}, random.Random(0))
    assert order == [1, 2, 0]


def test_without_scores_the_order_is_random_but_complete() -> None:
    order = rank_negatives([0, 1, 2, 3], None, random.Random(0))
    assert sorted(order) == [0, 1, 2, 3]


def test_unscored_candidates_sort_behind_scored_ones_rather_than_being_dropped() -> None:
    """A partial scoring pass must not silently shrink the candidate pool."""
    order = rank_negatives([0, 1, 2], {1: 0.5}, random.Random(0))
    assert order[0] == 1
    assert sorted(order) == [0, 1, 2]


def test_the_fallback_ordering_is_seeded() -> None:
    a = rank_negatives(list(range(20)), None, random.Random(7))
    b = rank_negatives(list(range(20)), None, random.Random(7))
    assert a == b


# --- subsampling ----------------------------------------------------------------


def stage1_rows(parent: str, negatives: int, split: str = "train") -> list[dict]:
    rows = [make("banking77", stage=1, split=split, parent_id=parent, answer_idx=1)]
    rows += [
        make("banking77", stage=1, split=split, parent_id=parent, answer_idx=0)
        for _ in range(negatives)
    ]
    return rows


def test_subsampling_keeps_the_positive_and_k_negatives() -> None:
    kept, report = subsample_stage1(stage1_rows("p", 50), negatives=6)
    assert len(kept) == 7
    assert sum(r["answer_idx"] for r in kept) == 1
    assert report["rows_dropped"] == 44


def test_subsampling_leaves_other_splits_alone() -> None:
    """val and test carry the deployed ratio; that is the whole point of item 4."""
    rows = stage1_rows("p", 50, split="val")
    kept, _ = subsample_stage1(rows, negatives=6, splits=("train",))
    assert len(kept) == 51


def test_subsampling_leaves_non_stage_one_rows_alone() -> None:
    rows = [make("go_emotions") for _ in range(20)]
    kept, report = subsample_stage1(rows, negatives=1)
    assert len(kept) == 20
    assert report["questions"] == 0


def test_a_question_with_fewer_negatives_than_k_is_kept_whole() -> None:
    kept, report = subsample_stage1(stage1_rows("p", 2), negatives=6)
    assert len(kept) == 3
    assert report["rows_dropped"] == 0


def test_the_report_says_whether_negatives_were_chosen_or_sampled() -> None:
    _, random_report = subsample_stage1(stage1_rows("p", 10), negatives=3)
    assert random_report["hard_negative_source"] == "random"
    _, scored = subsample_stage1(stage1_rows("p", 10), negatives=3, scores={"p": {0: 0.9, 1: 0.2}})
    assert scored["hard_negative_source"] == "zero_shot_scores"
    assert scored["questions_with_scores"] == 1


def test_a_negative_negative_count_is_refused() -> None:
    with pytest.raises(ValueError, match="negatives"):
        subsample_stage1([], negatives=-1)


# --- weights ---------------------------------------------------------------------


def corpus() -> list[dict]:
    rows = [make("banking77", stage=1) for _ in range(700)]
    rows += [make("clinc_oos", stage=1) for _ in range(300)]
    rows += [make("go_emotions") for _ in range(100)]
    rows += [make("mmlu", qtype="choice") for _ in range(100)]
    rows += [make("ordinal_control", qtype="score") for _ in range(10)]
    return rows


def test_weighting_moves_the_mix_toward_the_target() -> None:
    rows = corpus()
    weights, _ = family_weights(rows)
    mix = effective_mix(rows, weights)
    assert mix["stage1"]["raw_share"] > 0.8
    assert mix["stage1"]["effective_share"] < mix["stage1"]["raw_share"]
    assert mix["noul"]["effective_share"] > mix["noul"]["raw_share"]


def test_families_inside_a_group_are_equalised() -> None:
    """A group must not be carried by whichever source happens to be largest."""
    rows = corpus()
    weights, _ = family_weights(rows)
    mass = {}
    for row in rows:
        if row_group(row) != "stage1":
            continue
        key = row["family"]
        mass[key] = mass.get(key, 0.0) + weights[f"stage1/{key}"]
    assert mass["banking77"] == pytest.approx(mass["clinc_oos"], rel=1e-6)


def test_an_unreachable_target_is_clipped_and_reported_not_silently_met() -> None:
    rows = corpus()
    _, report = family_weights(rows)
    clipped = {c["group"] for c in report["clipped"]}
    assert "score" in clipped, "10% from 10 of 1,210 rows cannot be reached inside the cap"
    entry = next(c for c in report["clipped"] if c["group"] == "score")
    assert entry["needed_oversample"] > MAX_OVERSAMPLE
    assert entry["capped_at"] == MAX_OVERSAMPLE


def test_an_absent_group_does_not_take_a_share_of_the_mix() -> None:
    rows = [make("banking77", stage=1) for _ in range(10)]
    weights, report = family_weights(rows)
    assert set(report["groups"]) == {"stage1"}
    assert effective_mix(rows, weights)["stage1"]["effective_share"] == pytest.approx(1.0)


def test_weighting_an_empty_corpus_is_not_an_error() -> None:
    assert family_weights([]) == ({}, {"groups": {}, "clipped": []})


def test_missing_weights_fall_back_to_one_rather_than_dropping_rows() -> None:
    rows = corpus()
    mix = effective_mix(rows, {})
    assert mix["stage1"]["effective_share"] == pytest.approx(mix["stage1"]["raw_share"])


# --- the report table -------------------------------------------------------------


def test_the_table_separates_stage_one_from_genuine_within_noul() -> None:
    rows = corpus()
    weights, _ = family_weights(rows)
    table = mix_table(rows, weights)
    noul = {(r["stage"], r["family"]) for r in table if r["primitive"] == "noul"}
    assert ("stage-1", "banking77") in noul
    assert ("genuine", "go_emotions") in noul


def test_the_table_is_ordered_by_what_the_model_sees() -> None:
    rows = corpus()
    weights, _ = family_weights(rows)
    shares = [r["effective_share"] for r in mix_table(rows, weights)]
    assert shares == sorted(shares, reverse=True)


def test_the_table_carries_before_and_after_counts() -> None:
    rows = corpus()
    weights, _ = family_weights(rows)
    table = mix_table(rows, weights, raw_counts={"stage1/banking77": 70000})
    entry = next(r for r in table if r["family"] == "banking77" and r["stage"] == "stage-1")
    assert entry["rows_before"] == 70000
    assert entry["rows_after"] == 700


def test_shares_sum_to_one() -> None:
    rows = corpus()
    weights, _ = family_weights(rows)
    table = mix_table(rows, weights)
    assert sum(r["raw_share"] for r in table) == pytest.approx(1.0)
    assert sum(r["effective_share"] for r in table) == pytest.approx(1.0)


# --- the built corpus -------------------------------------------------------------


@pytest.fixture(scope="module")
def manifest() -> dict:
    path = repo_root() / "data" / "processed" / "manifest.json"
    if not path.is_file():
        pytest.skip("no built corpus; run `uv run task data`")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_default_k_lands_inside_the_ratio_ceiling() -> None:
    from data.build.pipeline import MAX_STAGE1_NO_YES_RATIO

    assert STAGE1_TRAIN_NEGATIVES <= MAX_STAGE1_NO_YES_RATIO


def test_the_target_mix_is_a_distribution() -> None:
    assert sum(DEFAULT_TARGET_MIX.values()) == pytest.approx(1.0)
    assert all(0 < v < 1 for v in DEFAULT_TARGET_MIX.values())


def test_the_manifest_records_which_negatives_were_used(manifest: dict) -> None:
    report = manifest["stage1_subsampling"]
    assert report["negatives_per_question"] == STAGE1_TRAIN_NEGATIVES
    assert report["splits"] == ["train"]
    assert report["hard_negative_source"] in {"random", "zero_shot_scores"}


def test_the_manifest_carries_weights_the_trainer_can_read(manifest: dict) -> None:
    weights = manifest["sampling_weights"]
    assert weights, "a training run must be able to reproduce the mix it was given"
    assert all("/" in key for key in weights)
    assert all(value > 0 for value in weights.values())


def test_the_weights_file_matches_the_manifest() -> None:
    directory = repo_root() / "data" / "processed"
    if not (directory / "sampling_weights.json").is_file():
        pytest.skip("no built corpus; run `uv run task data`")
    side_car = json.loads((directory / "sampling_weights.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert side_car["weights"] == manifest["sampling_weights"], (
        "a run and its corpus must not disagree about the mix that produced it"
    )
