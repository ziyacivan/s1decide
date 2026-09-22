"""Folding two teachers into one corpus.

The questions here are: does a row survive on the right evidence, is the label it carries a
valid index into its own options, and is anything thrown away that we would need in order to
change our minds later.
"""

from __future__ import annotations

import json

import pytest
from data.build.agree import (
    FoldReport,
    agreement_stats,
    build_score_rows,
    fold_teachers,
    load_levels,
    rubric_options,
    source_index,
)

# --- what survives ---------------------------------------------------------------


def test_exact_agreement_is_kept() -> None:
    kept, report = fold_teachers({"a": 2}, {"a": 2})
    assert report.exact == 1 and report.kept == 1
    assert kept["a"]["agreement"] == "exact"


def test_one_level_apart_is_kept_and_marked() -> None:
    """Adjacent levels are where an ordinal rubric is genuinely ambiguous, not where it fails."""
    kept, report = fold_teachers({"a": 2}, {"a": 3})
    assert report.within_one == 1 and report.exact == 0
    assert kept["a"]["agreement"] == "within_one"


def test_two_levels_apart_is_dropped() -> None:
    kept, report = fold_teachers({"a": 1}, {"a": 3})
    assert kept == {} and report.far == 1


def test_a_teacher_that_did_not_commit_drops_the_row_without_counting_it_as_disagreement() -> None:
    """A truncated trace and a real disagreement are different failures with different fixes."""
    _, report = fold_teachers({"a": 3}, {"a": None})
    assert report.no_commit == 1
    assert report.far == 0 and report.compared == 0


def test_rows_only_one_teacher_saw_are_counted_as_missing() -> None:
    _, report = fold_teachers({"a": 1, "b": 2}, {"a": 1})
    assert report.missing == 1 and report.compared == 1


def test_a_one_level_disagreement_resolves_to_teacher_one() -> None:
    """Pinning the documented default, in both directions, so a change to it is deliberate.

    Measured over the 4,804 kept rows of the real run, resolving to the minimum instead would
    make 51.3% of the corpus "none" against 41.0% for this rule. Both levels stay on the row, so
    the choice is reversible without re-labelling.
    """
    assert fold_teachers({"a": 2}, {"a": 3})[0]["a"]["answer_idx"] == 2
    assert fold_teachers({"a": 3}, {"a": 2})[0]["a"]["answer_idx"] == 3


# --- the label ------------------------------------------------------------------


def test_the_answer_is_an_index_into_the_options_not_a_rubric_number() -> None:
    """`parse_level` returns a 0-based index. Treating it as a 1-based level put `answer_idx: -1`
    on every level-0 row — a valid Python index, the *last* option, and silent."""
    report = FoldReport()
    kept, _ = fold_teachers({"a": 0}, {"a": 0})
    rows = build_score_rows(
        [{"id": "a", "state": "s", "question": "q", "family": "f"}],
        kept,
        {"s": ("src", "mit")},
        report,
    )
    assert rows[0]["answer_idx"] == 0


@pytest.mark.parametrize("first,second", [(0, 0), (0, 1), (2, 3), (4, 4), (4, 3)])
def test_every_answer_indexes_a_real_option(first, second) -> None:
    report = FoldReport()
    kept, _ = fold_teachers({"a": first}, {"a": second})
    rows = build_score_rows(
        [{"id": "a", "state": "s", "question": "q", "family": "f"}],
        kept,
        {"s": ("src", "mit")},
        report,
    )
    row = rows[0]
    assert 0 <= row["answer_idx"] < len(row["options"])
    assert row["answer_idx"] in (first, second), "the label must be something a teacher said"


def test_both_teacher_levels_survive_onto_the_row() -> None:
    """The point of the whole design: the fusion rule can be changed without re-labelling."""
    report = FoldReport()
    kept, _ = fold_teachers({"a": 2}, {"a": 3})
    rows = build_score_rows(
        [{"id": "a", "state": "s", "question": "q", "family": "f"}],
        kept,
        {"s": ("src", "mit")},
        report,
    )
    assert rows[0]["teacher_levels"] == {"first": 2, "second": 3}


def test_the_rubric_becomes_five_ordered_option_names() -> None:
    options = rubric_options()
    assert options == ["none", "slight", "moderate", "strong", "decisive"]
    assert all(":" not in option and "-" not in option for option in options)


# --- provenance ------------------------------------------------------------------


def test_a_row_carries_the_licence_of_the_state_it_came_from() -> None:
    report = FoldReport()
    kept, _ = fold_teachers({"a": 1}, {"a": 1})
    rows = build_score_rows(
        [{"id": "a", "state": "s", "question": "q", "family": "f"}],
        kept,
        {"s": ("PolyAI/banking77", "cc-by-4.0")},
        report,
    )
    assert rows[0]["source"] == "PolyAI/banking77" and rows[0]["license"] == "cc-by-4.0"


def test_a_state_with_no_known_source_is_counted_rather_than_passed_off() -> None:
    """ADR 0004 gates training rows on licence; an unlicensed row must be visible, not silent."""
    report = FoldReport()
    kept, _ = fold_teachers({"a": 1}, {"a": 1})
    build_score_rows([{"id": "a", "state": "s", "question": "q", "family": "f"}], kept, {}, report)
    assert report.unlicensed == 1


def test_source_index_is_deterministic_when_a_state_appears_twice() -> None:
    index = source_index(
        [
            {"state": "same", "source": "first", "license": "mit"},
            {"state": "same", "source": "second", "license": "apache-2.0"},
        ]
    )
    assert index["same"] == ("first", "mit")


# --- reporting -------------------------------------------------------------------


def test_the_report_separates_the_two_ways_a_row_is_lost() -> None:
    _, report = fold_teachers({"a": 0, "b": 0, "c": 0}, {"a": 4, "b": None, "c": 0})
    payload = report.to_json()
    assert payload["far"] == 1 and payload["no_commit"] == 1
    assert payload["kept"] == 1
    assert payload["drop_rate"] == pytest.approx(
        1.0
    )  # 1 far of 1 compared; no_commit never compared


def test_rates_do_not_divide_by_zero_on_an_empty_comparison() -> None:
    assert FoldReport().to_json()["exact_rate"] == 0.0


def test_the_confusion_matrix_records_direction() -> None:
    """Which teacher said what, not just that they differed."""
    _, report = fold_teachers({"a": 1}, {"a": 2})
    assert report.to_json()["confusion"] == {"1->2": 1}


def test_load_levels_survives_a_half_written_final_line(tmp_path) -> None:
    (tmp_path / "rows.jsonl").write_text(
        json.dumps({"id": "a", "level": 3}) + '\n{"id": "b", "lev', encoding="utf-8"
    )
    assert load_levels(tmp_path) == {"a": 3}


def test_load_levels_of_a_directory_with_no_rows_is_empty(tmp_path) -> None:
    assert load_levels(tmp_path) == {}


# --- agreement statistics ---------------------------------------------------------


def test_perfect_agreement_is_kappa_one() -> None:
    stats = agreement_stats([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])
    assert stats["exact"] == 1.0
    assert stats["kappa"] == pytest.approx(1.0)
    assert stats["weighted_kappa"] == pytest.approx(1.0)


def test_independent_teachers_score_about_zero_not_their_raw_agreement() -> None:
    """The whole point of kappa: raw agreement on a skewed scale flatters two teachers who are
    telling you nothing. Both answer 0 four times in five, agree 68% of the time, know nothing."""
    pairs = [(x, y) for x in [0, 0, 0, 0, 1] for y in [0, 0, 0, 0, 1]]
    stats = agreement_stats(pairs)
    assert stats["exact"] > 0.6
    assert stats["kappa"] == pytest.approx(0.0, abs=1e-9)


def test_weighted_kappa_forgives_a_near_miss_more_than_a_far_one() -> None:
    """3-vs-4 is a near miss on an ordinal scale; 0-vs-4 is a different judgement."""
    base = [(0, 0), (1, 1), (2, 2), (3, 3)]
    near = agreement_stats([*base, (3, 4)])["weighted_kappa"]
    far = agreement_stats([*base, (0, 4)])["weighted_kappa"]
    assert near > far


def test_stats_of_nothing_do_not_divide_by_zero() -> None:
    assert agreement_stats([])["n"] == 0


def test_the_marginals_are_reported_so_a_threshold_shift_is_visible() -> None:
    """A teacher that sits half a level low is a systematic offset, not noise, and the means are
    what make that visible — it is why the resolution default is teacher 1's level."""
    stats = agreement_stats([(2, 1), (3, 2), (4, 3)])
    assert stats["first_mean"] - stats["second_mean"] == pytest.approx(1.0)
