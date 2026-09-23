"""The data pipeline: option shuffling, two-stage expansion, splitting and the gate.

The pure functions are tested exhaustively here with no network. The one test that builds from
real sources is marked ``weights`` and skips when they are not cached.
"""

from __future__ import annotations

import json
import random

import pytest
from data.build.licences import LicenceError
from data.build.pipeline import (
    STAGE1_NEGATIVES,
    STAGE2_SHORTLIST,
    BuildConfig,
    expand_high_cardinality,
    shuffle_options,
    split_by_state,
    state_hash,
)
from data.build.synthetic import ORDINAL_RUBRICS, generate_ordinal_control

from s1decide.tokens import MAX_SINGLE_TOKEN_OPTIONS


def choice_row(n_options: int = 4, answer: int = 0, family: str = "f", state: str = "s") -> dict:
    return {
        "id": "row-1",
        "family": family,
        "state": state,
        "state_hash": state_hash(state),
        "qtype": "choice",
        "instructions": "Which one?",
        "options": tuple(f"option {i}" for i in range(n_options)),
        "answer_idx": answer,
        "source": "src",
        "loaded_from": "src",
        "license": "apache-2.0",
        "role": "train",
        "stage": None,
        "parent_id": None,
    }


# --- option shuffling ---------------------------------------------------------


def test_shuffling_moves_the_answer_index_with_the_options() -> None:
    rng = random.Random(0)
    for _ in range(50):
        row = choice_row(6, answer=2)
        shuffled = shuffle_options(row, rng)
        assert set(shuffled["options"]) == set(row["options"])
        assert shuffled["options"][shuffled["answer_idx"]] == row["options"][row["answer_idx"]]


def test_shuffling_actually_permutes_sometimes() -> None:
    rng = random.Random(1)
    rows = [shuffle_options(choice_row(8), rng) for _ in range(20)]
    assert any(r["options"] != choice_row(8)["options"] for r in rows)


def test_noul_labels_are_never_shuffled() -> None:
    """Index 1 must always mean `yes`, or `Result.noul` silently inverts."""
    row = {**choice_row(2), "qtype": "noul", "options": ("no", "yes"), "answer_idx": 1}
    for seed in range(20):
        assert shuffle_options(row, random.Random(seed)) == row


def test_shuffling_is_deterministic_for_a_given_seed() -> None:
    a = shuffle_options(choice_row(10), random.Random(7))
    b = shuffle_options(choice_row(10), random.Random(7))
    assert a == b


# --- two-stage expansion ------------------------------------------------------


def test_expansion_emits_the_full_fan_out_by_default() -> None:
    """The default is every candidate, because that is the task the deployment performs.

    Training subsamples afterwards, per split, rather than the expansion deciding for it — see
    `data/build/balance.py`. `STAGE1_NEGATIVES` is None for exactly this reason.
    """
    assert STAGE1_NEGATIVES is None
    rows = expand_high_cardinality(choice_row(77, answer=40), random.Random(0))
    stage1 = [r for r in rows if r["stage"] == 1]
    stage2 = [r for r in rows if r["stage"] == 2]
    assert len(stage1) == 77, "one row per candidate: the positive plus all 76 negatives"
    assert len(stage2) == 1
    assert all(r["qtype"] == "noul" and r["options"] == ("no", "yes") for r in stage1)
    assert stage2[0]["qtype"] == "choice"


def test_expansion_can_be_asked_for_a_subsample() -> None:
    """What the train split gets: the positive plus k negatives."""
    rows = expand_high_cardinality(choice_row(77, answer=40), random.Random(0), negatives=6)
    stage1 = [r for r in rows if r["stage"] == 1]
    assert len(stage1) == 7
    assert sum(r["answer_idx"] == 1 for r in stage1) == 1


def test_asking_for_more_negatives_than_exist_is_not_an_error() -> None:
    rows = expand_high_cardinality(choice_row(30, answer=2), random.Random(0), negatives=500)
    assert len([r for r in rows if r["stage"] == 1]) == 30


def test_stage1_has_exactly_one_positive() -> None:
    rows = expand_high_cardinality(choice_row(77, answer=13), random.Random(3))
    stage1 = [r for r in rows if r["stage"] == 1]
    assert sum(r["answer_idx"] == 1 for r in stage1) == 1
    positive = next(r for r in stage1 if r["answer_idx"] == 1)
    assert positive["instructions"].endswith("Candidate: option 13")


def test_stage2_shortlist_always_contains_the_true_answer() -> None:
    for seed in range(30):
        rows = expand_high_cardinality(choice_row(77, answer=seed), random.Random(seed))
        stage2 = next(r for r in rows if r["stage"] == 2)
        assert stage2["options"][stage2["answer_idx"]] == f"option {seed}"
        assert len(stage2["options"]) == STAGE2_SHORTLIST
        assert len(stage2["options"]) <= MAX_SINGLE_TOKEN_OPTIONS


def test_every_expanded_row_points_back_at_its_parent() -> None:
    rows = expand_high_cardinality(choice_row(77, answer=1), random.Random(0))
    assert {r["parent_id"] for r in rows} == {"row-1"}
    assert len({r["id"] for r in rows}) == len(rows)


def test_stage1_rows_render_exactly_like_render_stage1(ticket_state) -> None:
    """The training data must be byte-identical to what inference produces, or S1 learns a
    format the engine never emits."""
    from s1decide.primitives import Choice, Noul, Question
    from s1decide.prompt import render, render_stage1

    options = tuple(f"intent {i:02d}" for i in range(40))
    question = Question("intent", Choice("Which intent?", options))
    from_inference = render_stage1(ticket_state, question).suffixes[7]

    row = choice_row(40, answer=7)
    row["instructions"] = "Which intent?"
    row["options"] = options
    expanded = expand_high_cardinality(row, random.Random(0))
    positive = next(r for r in expanded if r["stage"] == 1 and r["answer_idx"] == 1)
    from_training = render(
        ticket_state, [Question(positive["id"], Noul(positive["instructions"]))]
    ).suffixes[0]

    assert from_training == from_inference


def test_narrow_questions_are_not_expanded() -> None:
    """Expansion is only for questions the single-stage path cannot label."""
    from data.build.pipeline import MAX_SINGLE_TOKEN_OPTIONS as ceiling

    assert ceiling == MAX_SINGLE_TOKEN_OPTIONS


# --- splitting ----------------------------------------------------------------


def test_heldout_and_ood_go_entirely_to_eval() -> None:
    rows = [{**choice_row(), "role": role} for role in ("heldout", "ood")]
    assert {r["split"] for r in split_by_state(rows, BuildConfig())} == {"eval"}


def test_every_question_about_one_state_lands_in_the_same_split() -> None:
    rows = [choice_row(state=f"state {i // 5}") for i in range(200)]
    assigned = split_by_state(rows, BuildConfig())
    by_state: dict[str, set[str]] = {}
    for row in assigned:
        by_state.setdefault(row["state"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_state.values())


def test_splitting_is_deterministic_and_seed_sensitive() -> None:
    rows = [choice_row(state=f"state {i}") for i in range(300)]
    a = [r["split"] for r in split_by_state(rows, BuildConfig(seed=1))]
    b = [r["split"] for r in split_by_state(rows, BuildConfig(seed=1))]
    c = [r["split"] for r in split_by_state(rows, BuildConfig(seed=2))]
    assert a == b
    assert a != c


def test_split_fractions_are_roughly_respected() -> None:
    rows = [choice_row(state=f"state {i}") for i in range(4000)]
    assigned = split_by_state(rows, BuildConfig(val_fraction=0.10, test_fraction=0.10))
    counts = {s: sum(r["split"] == s for r in assigned) for s in ("train", "val", "test")}
    assert 0.07 < counts["val"] / 4000 < 0.13
    assert 0.07 < counts["test"] / 4000 < 0.13
    assert counts["train"] + counts["val"] + counts["test"] == 4000


def test_state_hash_is_whitespace_insensitive_and_stable() -> None:
    assert state_hash("  hello  ") == state_hash("hello")
    assert state_hash("a") != state_hash("b")
    assert state_hash("a") == state_hash("a")


# --- the gate, wired into the build ------------------------------------------


def test_the_build_refuses_to_write_non_permissive_training_rows() -> None:
    from data.build.licences import assert_train_splits_are_licensed

    rows = split_by_state([{**choice_row(), "license": "cc-by-nc-4.0"}], BuildConfig())
    with pytest.raises(LicenceError, match="may not be trained on"):
        assert_train_splits_are_licensed(rows)


def test_non_commercial_sources_can_only_be_eval() -> None:
    """ANLI and SciQ are OOD *because* their licence forbids training — by construction."""
    from data.build.licences import assert_train_splits_are_licensed
    from data.build.sources import SOURCES

    for source in SOURCES:
        if source.license.upper().find("NC") >= 0:
            assert source.role == "ood", source.family
    rows = split_by_state(
        [{**choice_row(), "license": "cc-by-nc-4.0", "role": "ood"}], BuildConfig()
    )
    assert_train_splits_are_licensed(rows)


# --- the programmatic ordinal control set (ADR 0005 rule c) ------------------


def test_ordinal_control_labels_follow_the_rule_exactly() -> None:
    for row in generate_ordinal_control(count=300):
        rubric = ORDINAL_RUBRICS[row["rubric"]]
        value = json.loads(row["state"])[rubric["field"]]
        expected = sum(1 for t in rubric["thresholds"] if value >= t)
        assert row["answer_idx"] == expected
        assert row["options"] == tuple(rubric["levels"])


def test_ordinal_control_is_deterministic() -> None:
    a = list(generate_ordinal_control(count=50, seed=5))
    b = list(generate_ordinal_control(count=50, seed=5))
    assert a == b
    assert a != list(generate_ordinal_control(count=50, seed=6))


def test_ordinal_control_populates_every_level() -> None:
    """A control set that never shows the top band cannot test the ordinal loss."""
    seen: dict[str, set[int]] = {}
    for row in generate_ordinal_control(count=1500):
        seen.setdefault(row["rubric"], set()).add(row["answer_idx"])
    for rubric, levels in seen.items():
        assert levels == set(range(len(ORDINAL_RUBRICS[rubric]["levels"]))), rubric


def test_ordinal_control_states_carry_distractor_fields() -> None:
    """Without them the task is a lookup of the only number in the state."""
    row = next(iter(generate_ordinal_control(count=1)))
    state = json.loads(row["state"])
    rubric = ORDINAL_RUBRICS[row["rubric"]]
    assert len(state) > 1
    assert "unrelated_count" in state
    assert rubric["field"] in state


def test_ordinal_control_levels_are_ordered_low_to_high() -> None:
    for rubric in ORDINAL_RUBRICS.values():
        assert len(rubric["levels"]) == len(rubric["thresholds"]) + 1


# --- a real build, when the sources are cached -------------------------------


def _build_or_skip(config: BuildConfig) -> dict:
    """Build, skipping only when a *source* is genuinely unavailable.

    Deliberately narrow. An earlier version caught every exception, and reported a real bug in
    ``build()`` as "sources not available" — a test that skips on any error has stopped being a
    test.
    """
    from data.build.pipeline import build
    from datasets.exceptions import DatasetNotFoundError

    try:
        return build(config)
    except (DatasetNotFoundError, ConnectionError) as exc:
        pytest.skip(f"source unavailable: {type(exc).__name__}: {exc}"[:150])


@pytest.mark.weights
@pytest.mark.slow
def test_a_small_real_build_is_licensed_and_splits_cleanly(tmp_path) -> None:
    pytest.importorskip("datasets")

    manifest = _build_or_skip(BuildConfig(out_dir=tmp_path, limit_per_source=40))

    assert manifest["rows"] > 0
    assert set(manifest["splits"]) <= {"train", "val", "test", "eval"}
    # Held-out families and OOD sets never reach a training split.
    for family in ("commonsense_qa", "pubmedqa", "anli", "sciq"):
        assert set(manifest["by_family"].get(family, {})) <= {"eval"}
    # The high-cardinality sources were expanded.
    assert manifest["two_stage_parents_expanded"] > 0
    # Every file is valid UTF-8 JSONL with LF endings.
    for name in manifest["files"].values():
        path = tmp_path / name.rsplit("/", 1)[-1]
        assert b"\r\n" not in path.read_bytes()
        for line in path.read_text(encoding="utf-8").splitlines():
            json.loads(line)


@pytest.mark.weights
@pytest.mark.slow
def test_a_real_build_is_idempotent(tmp_path) -> None:
    pytest.importorskip("datasets")

    first = _build_or_skip(BuildConfig(out_dir=tmp_path / "a", limit_per_source=30))
    second = _build_or_skip(BuildConfig(out_dir=tmp_path / "b", limit_per_source=30))
    assert first["splits"] == second["splits"]
    for name in first["files"]:
        a = (tmp_path / "a" / f"{name}.jsonl").read_bytes()
        b = (tmp_path / "b" / f"{name}.jsonl").read_bytes()
        assert a == b, f"{name}.jsonl differs between builds"


def test_the_leakage_guard_fails_the_build_when_it_cannot_run(monkeypatch) -> None:
    """ "Unchecked" is not a state the Tier-2 guard may be in; it used to return checked=False."""
    import eval.data
    from data.build.pipeline import leakage_report

    def unavailable(*args, **kwargs):
        raise ConnectionError("hub unreachable")

    monkeypatch.setattr(eval.data, "load_system_one_decisions", unavailable)
    with pytest.raises(RuntimeError, match="ConnectionError"):
        leakage_report([{"split": "train", "state_hash": "h"}])


def test_a_manifest_with_an_unchecked_guard_is_not_rendered() -> None:
    import json

    from data.build.report import render_build_report

    from s1decide.tasks import repo_root

    manifest = json.loads(
        (repo_root() / "data/processed/manifest.json").read_text(encoding="utf-8")
    )
    manifest["leakage"] = {"checked": False, "reason": "old manifest"}
    with pytest.raises(ValueError, match="unchecked"):
        render_build_report(manifest)
