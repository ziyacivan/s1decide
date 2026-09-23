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
    MAX_STAGE1_NO_YES_RATIO,
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
    """Asserted on the **effective** mix, not on row counts.

    Raw counts are stage-1-heavy on purpose — training keeps six negatives per question and the
    expansion is what it is. What the model sees is the weighted draw, so that is what the floor
    applies to. The raw share is reported beside it precisely so the gap stays visible instead
    of looking like the floor was met by accident.
    """
    effective = manifest["effective_mix"]["noul"]["effective_share"]
    raw = manifest["effective_mix"]["noul"]["raw_share"]
    assert effective >= MIN_GENUINE_NOUL_SHARE, (
        f"genuine Noul is {effective:.1%} of the effective training mix (raw {raw:.1%}), below "
        f"the {MIN_GENUINE_NOUL_SHARE:.0%} floor — raise the target share in DEFAULT_TARGET_MIX "
        "or add genuine Noul sources; do not lower the floor"
    )


def test_the_raw_noul_share_is_reported_even_when_it_is_below_the_floor(manifest: dict) -> None:
    """The gap between raw and effective is the point of reporting both."""
    noul = manifest["effective_mix"]["noul"]
    assert "raw_share" in noul and "effective_share" in noul
    assert noul["rows"] > 0


def test_stage_one_training_rows_are_within_the_no_yes_ceiling(manifest: dict) -> None:
    """The reason item 4 exists: uncapped, this ratio is ~96:1 and the model learns to say no."""
    import json as _json

    from s1decide.tasks import repo_root

    path = repo_root() / "data" / "processed" / "train.jsonl"
    if not path.is_file():
        pytest.skip("no built corpus; run `uv run task data`")
    stage1 = [
        row
        for row in (
            _json.loads(line)
            for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()
        )
        if row.get("stage") == 1
    ]
    yes = sum(1 for row in stage1 if row["answer_idx"] == 1)
    no = len(stage1) - yes
    assert yes, "no positive stage-1 rows at all — the expansion is broken, not merely unbalanced"
    assert no / yes <= MAX_STAGE1_NO_YES_RATIO, (
        f"train stage-1 is {no / yes:.1f}:1 no:yes, above the {MAX_STAGE1_NO_YES_RATIO}:1 ceiling"
    )


def test_evaluation_splits_keep_the_full_fan_out(manifest: dict) -> None:
    """val and test must carry the deployed ratio, not a comfortable one.

    A temperature fitted on a 6:1 sample and deployed at 96:1 is fitted to the wrong
    distribution, so the subsampling is deliberately train-only.
    """
    import json as _json

    from s1decide.tasks import repo_root

    for split in ("val", "test"):
        path = repo_root() / "data" / "processed" / f"{split}.jsonl"
        if not path.is_file():
            pytest.skip("no built corpus; run `uv run task data`")
        stage1 = [
            row
            for row in (
                _json.loads(line)
                for line in path.read_text(encoding="utf-8").split("\n")
                if line.strip()
            )
            if row.get("stage") == 1
        ]
        yes = sum(1 for row in stage1 if row["answer_idx"] == 1)
        ratio = (len(stage1) - yes) / max(1, yes)
        assert ratio > MAX_STAGE1_NO_YES_RATIO * 2, (
            f"{split} stage-1 is only {ratio:.1f}:1 — that looks subsampled, and evaluation "
            "must see the full fan-out"
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


# --- every primitive needs a real evaluation set --------------------------------

#: Families whose labels come from a rule rather than from a model or a human annotation.
#:
#: They are exact by construction, which makes them a control and not a measurement: a model that
#: scores well on `ordinal_control` has learned the rule, which is not the claim the model card
#: makes for `Score`.
RULE_BASED_FAMILIES: frozenset[str] = frozenset({"ordinal_control"})

#: Genuine (non-stage-1) rows a primitive needs per evaluation split, from a non-rule-based
#: family, before a number about it can be published.
MIN_EVAL_ROWS_PER_PRIMITIVE = 300


def _genuine_eval_rows(split: str) -> dict[str, int]:
    """Non-stage-1 rows per primitive in one split, excluding rule-labelled families."""
    path = repo_root() / "data" / "processed" / f"{split}.jsonl"
    counts: dict[str, int] = {"choice": 0, "score": 0, "noul": 0}
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("stage") == 1 or row["family"] in RULE_BASED_FAMILIES:
            continue
        if row["qtype"] in counts:
            counts[row["qtype"]] += 1
    return counts


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Score has no non-rule-based evaluation rows yet: all 4,804 teacher-labelled rows landed "
        "in train, so val and test hold only ordinal_control, whose labels come from a rule. "
        "Score accuracy on a teacher rubric is therefore unmeasurable and Score calibration would "
        "be fitted on the control set. Closed by the queued second teacher run over 600 val and "
        "600 test states (docs/phase-1-plan.md); remove this marker when it lands."
    ),
)
@pytest.mark.parametrize("split", ["val", "test"])
def test_every_primitive_has_a_real_evaluation_set(split: str) -> None:
    """A primitive with no genuine eval rows cannot appear in the model card with a number.

    Strict xfail on purpose: when the second teacher run lands this starts passing, the suite
    fails on the unexpected pass, and the marker has to be removed deliberately rather than the
    gap quietly closing unnoticed.
    """
    counts = _genuine_eval_rows(split)
    below = {k: v for k, v in counts.items() if v < MIN_EVAL_ROWS_PER_PRIMITIVE}
    assert not below, (
        f"{split}: {below} genuine non-rule-based rows, "
        f"need {MIN_EVAL_ROWS_PER_PRIMITIVE} per primitive"
    )
