"""MASSIVE's locale split, and the parallel-corpus trap underneath it.

MASSIVE translates the *same* utterance into all 51 locales and keeps its id. That makes two
mistakes very easy, and neither is visible in a row count:

1. Taking the first N rows of each training locale gives N meanings translated three times, not
   3N meanings. Verified before these tests existed: the first 3,000 train ids are identical
   across en-US, tr-TR and de-DE.
2. Holding out a locale is not holding out *content* unless the held-out rows also come from a
   different partition — otherwise the "unseen language" set is a translation of the training
   set, and the leakage guard cannot see it, because it hashes state text and two translations
   of one sentence are different strings.

The first is handled by disjoint `row_offset` slices, the second by taking the OOD locales from
the test partition. Both are asserted here rather than trusted.
"""

from __future__ import annotations

import itertools

import pytest
from data.build.sources import SOURCES, load_source

pytestmark = [pytest.mark.weights, pytest.mark.slow]

TRAIN_LOCALES = ("massive_en", "massive_tr", "massive_de")
OOD_LOCALES = ("massive_fr", "massive_ja")


def _sources() -> dict:
    return {s.family: s for s in SOURCES if s.family.startswith("massive")}


@pytest.fixture(scope="module")
def loaded() -> dict[str, list[dict]]:
    """Load every MASSIVE locale once; skip only when the dataset is genuinely unavailable."""
    pytest.importorskip("datasets")
    from datasets.exceptions import DatasetNotFoundError

    try:
        return {family: load_source(source) for family, source in _sources().items()}
    except (DatasetNotFoundError, ConnectionError) as exc:
        pytest.skip(f"source unavailable: {type(exc).__name__}: {exc}"[:150])


# --- registry shape (no download needed) ---------------------------------------


def test_five_locales_are_registered() -> None:
    assert set(_sources()) == {*TRAIN_LOCALES, *OOD_LOCALES}


def test_the_held_out_locales_are_never_trainable() -> None:
    """A locale that leaks into a train split stops being an unseen language."""
    sources = _sources()
    for family in OOD_LOCALES:
        assert sources[family].role == "ood", family
    for family in TRAIN_LOCALES:
        assert sources[family].role == "train", family


def test_the_held_out_locales_come_from_a_different_partition() -> None:
    """The load-bearing detail: same partition would make them translations of training rows."""
    sources = _sources()
    assert {sources[f].split for f in TRAIN_LOCALES} == {"train"}
    assert {sources[f].split for f in OOD_LOCALES} == {"test"}


def test_the_training_locales_take_disjoint_slices() -> None:
    offsets = [_sources()[f].row_offset for f in TRAIN_LOCALES]
    caps = [_sources()[f].max_rows for f in TRAIN_LOCALES]
    assert len(set(offsets)) == len(offsets), "identical offsets would give parallel translations"
    for (offset_a, cap_a), (offset_b, _cap_b) in itertools.combinations(zip(offsets, caps), 2):
        assert offset_a + cap_a <= offset_b or offset_b >= offset_a + cap_a, (
            f"slices [{offset_a}, {offset_a + cap_a}) and [{offset_b}, ...) overlap"
        )


def test_every_locale_is_the_same_first_party_licence() -> None:
    for family, source in _sources().items():
        assert source.license == "cc-by-4.0", family


def test_the_cap_keeps_massive_below_the_largest_single_source() -> None:
    """The comparison must be on locales *summed*, which is what the first cap got wrong.

    3,000 per locale looked reasonable against banking77's 4,000 source rows and turned out to
    be 48% of the training corpus, because each source row expands ~5x through the two-stage
    path and there are three locales. Compared correctly, all three together must stay below
    the largest single other source.
    """
    sources = _sources()
    trained = sum(sources[f].max_rows for f in TRAIN_LOCALES)
    largest_other = max(
        s.max_rows or 0 for s in SOURCES if not s.family.startswith("massive") and s.role == "train"
    )
    assert trained <= largest_other, (
        f"MASSIVE's three locales contribute {trained} source rows against {largest_other} "
        "from the largest other training source; compare summed locales, not one locale"
    )


def test_the_manifest_confirms_massive_does_not_dominate() -> None:
    """The source-row check above is a proxy; this is the quantity that actually matters."""
    import json

    from s1decide.tasks import repo_root

    path = repo_root() / "data" / "processed" / "manifest.json"
    if not path.is_file():
        pytest.skip("no built corpus; run `uv run task data`")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    by_family = manifest["by_family"]
    train_total = manifest["splits"].get("train", 0)
    massive = sum(v.get("train", 0) for k, v in by_family.items() if k.startswith("massive"))
    share = massive / train_total
    assert share <= 0.33, f"MASSIVE is {share:.1%} of training rows; the cap needs lowering"
    largest_other = max(
        v.get("train", 0) for k, v in by_family.items() if not k.startswith("massive")
    )
    assert massive <= largest_other * 1.1, (
        f"MASSIVE ({massive}) should sit at or below the largest other source ({largest_other})"
    )


# --- the actual data -----------------------------------------------------------


def test_each_locale_loads_the_expected_number_of_rows(loaded) -> None:
    for family, rows in loaded.items():
        assert len(rows) == _sources()[family].max_rows, family


def test_every_locale_shares_one_option_space(loaded) -> None:
    """60 English intent labels everywhere, so the two-stage negatives are comparable."""
    option_sets = {family: rows[0]["options"] for family, rows in loaded.items()}
    first = next(iter(option_sets.values()))
    assert len(first) == 60
    for family, options in option_sets.items():
        assert options == first, f"{family} has a different option space"


def test_the_question_is_asked_in_english_for_every_locale(loaded) -> None:
    """The task is intent classification, not translation; holding the prompt fixed means a
    locale's score measures reading that language, not handling a different prompt."""
    instructions = {rows[0]["instructions"] for rows in loaded.values()}
    assert len(instructions) == 1, instructions


def test_the_training_locales_share_no_utterance(loaded) -> None:
    """The parallel-corpus trap. Disjoint slices, asserted on the text itself."""
    states = {family: {row["state"] for row in loaded[family]} for family in TRAIN_LOCALES}
    for a, b in itertools.combinations(TRAIN_LOCALES, 2):
        assert not (states[a] & states[b]), f"{a} and {b} share utterances"


def test_held_out_locale_overlap_with_training_is_only_homographs(loaded) -> None:
    """Text-level overlap exists and is tiny: cross-lingual homographs, not leaked content.

    French "silence" is also an English word, so one of 1,000 French rows collides with a
    training utterance. That is a coincidence of spelling, not the same data — the partition
    split above is what guarantees the held-out locales are different *content*. This test
    pins the overlap at a handful of short strings so a genuine failure of the locale or
    partition filter, which would produce hundreds, still fails loudly.
    """
    trained = set().union(*({row["state"] for row in loaded[f]} for f in TRAIN_LOCALES))
    for family in OOD_LOCALES:
        overlap = trained & {row["state"] for row in loaded[family]}
        assert len(overlap) <= 5, (
            f"{family} shares {len(overlap)} utterances: {sorted(overlap)[:10]}"
        )
        for utterance in overlap:
            assert len(utterance.split()) <= 2, f"{family}: {utterance!r} is too long to be chance"


def test_the_leakage_guard_removes_those_homographs(loaded) -> None:
    """Whatever does collide must not survive into an evaluation set.

    The guard hashes state text, so a homograph is indistinguishable from real leakage to it —
    and dropping it is the right call either way. Asserted here so the OOD set is known clean
    at the point it is built, not assumed clean because the locales differ.
    """
    from data.build.licences import drop_leaked_rows

    train_rows = [{"state": row["state"]} for family in TRAIN_LOCALES for row in loaded[family]]
    for family in OOD_LOCALES:
        external = [{"state": row["state"]} for row in loaded[family]]
        kept, dropped = drop_leaked_rows(external, train_rows)
        assert dropped == len(external) - len(kept)
        remaining = {row["state"] for row in kept} & {row["state"] for row in train_rows}
        assert not remaining, f"{family} still shares {len(remaining)} states after the guard"


def test_the_two_held_out_locales_are_parallel_to_each_other(loaded) -> None:
    """Deliberate: same utterances in both, so near-vs-far is language and script alone.

    If French and Japanese covered different content, a gap between them could be explained by
    the content rather than by the language, and the whole point of the pair would be lost.
    """
    answers = {
        family: [row["options"][row["answer_idx"]] for row in loaded[family]]
        for family in OOD_LOCALES
    }
    near, far = (answers[f] for f in OOD_LOCALES)
    assert near == far, "the held-out locales must cover the same utterances in the same order"


def test_the_answer_label_is_an_intent_from_the_shared_space(loaded) -> None:
    for family, rows in loaded.items():
        for row in rows[:50]:
            assert row["options"][row["answer_idx"]] in row["options"], family
            assert 0 <= row["answer_idx"] < len(row["options"]), family


def test_states_are_stripped_and_non_empty(loaded) -> None:
    for family, rows in loaded.items():
        for row in rows[:200]:
            assert row["state"] == row["state"].strip(), family
            assert row["state"], family


def test_a_locale_that_matches_nothing_is_an_error_not_an_empty_source() -> None:
    """Silently loading zero rows would shrink the corpus without failing anything."""
    pytest.importorskip("datasets")
    from dataclasses import replace

    from datasets.exceptions import DatasetNotFoundError

    source = replace(_sources()["massive_en"], locale="xx-XX", max_rows=10)
    try:
        with pytest.raises(ValueError, match="matched no rows"):
            load_source(source)
    except (DatasetNotFoundError, ConnectionError) as exc:
        pytest.skip(f"source unavailable: {type(exc).__name__}"[:150])
