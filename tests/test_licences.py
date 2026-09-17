"""The training-data licence gate (ADR 0004).

This is the test that makes the gate real. A comment saying "don't train on NC data" would not
have stopped anything; these assertions do. They run in the normal CPU suite — no GPU, no
network, no dataset download.
"""

from __future__ import annotations

import pytest
from data.build.licences import (
    TRAIN_LICENCE_ALLOWLIST,
    TRAIN_SPLITS,
    LicenceError,
    assert_train_splits_are_licensed,
    classify_licence,
    drop_leaked_rows,
    normalise_spdx,
    state_hashes,
    summarise_licences,
)


def row(license: str, split: str = "train", source: str = "src", state: str = "s") -> dict:
    return {"source": source, "license": license, "split": split, "state": state}


# --- normalisation ------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("apache-2.0", "Apache-2.0"),
        ("Apache 2.0", "Apache-2.0"),
        ("APACHE-2.0", "Apache-2.0"),
        ("mit", "MIT"),
        ("MIT", "MIT"),
        ("cc-by-4.0", "CC-BY-4.0"),
        ("cc-by-3.0", "CC-BY-3.0"),
        ("cc0-1.0", "CC0-1.0"),
        ("cc-by-sa-3.0", "CC-BY-SA-3.0"),
        ("cc-by-nc-4.0", "CC-BY-NC-4.0"),
        ("bsd-3-clause", "BSD-3-Clause"),
        (None, "unknown"),
        ("", "unknown"),
        ("unknown", "unknown"),
        ("other", "unknown"),
    ],
)
def test_normalise_spdx_canonicalises_common_spellings(raw, expected) -> None:
    assert normalise_spdx(raw) == expected


# --- classification -----------------------------------------------------------


@pytest.mark.parametrize("licence", sorted(TRAIN_LICENCE_ALLOWLIST))
def test_every_allowlisted_licence_classifies_as_train(licence: str) -> None:
    assert classify_licence(licence) == "train"


@pytest.mark.parametrize(
    "licence",
    ["cc-by-nc-4.0", "CC-BY-NC-SA-4.0", "cc-by-nc-nd-4.0", "cc-by-nd-4.0", "cc-by-nc-3.0"],
)
def test_non_commercial_and_no_derivatives_are_eval_only(licence: str) -> None:
    assert classify_licence(licence) == "eval-only"


@pytest.mark.parametrize(
    "licence",
    [
        "cc-by-sa-3.0",  # share-alike: not NC, but not on the allowlist either
        "cc-by-sa-4.0",
        "gpl-3.0",
        "unknown",
        "other",
        None,
        "",
        "some-bespoke-terms",
    ],
)
def test_anything_unrecognised_is_refused_not_assumed_permissive(licence) -> None:
    assert classify_licence(licence) == "refused"


def test_a_multi_licence_declaration_is_refused() -> None:
    """MultiNLI and FEVER declare several licences at once; that needs a human, not a guess."""
    assert classify_licence("cc-by-3.0, cc-by-sa-3.0, mit, other") == "refused"
    assert classify_licence("cc-by-sa-3.0, gpl-3.0") == "refused"


def test_share_alike_is_refused_rather_than_eval_only() -> None:
    """SA permits commercial use, so it is not eval-only; it is simply undecided (ADR 0005)."""
    assert classify_licence("cc-by-sa-4.0") == "refused"


# --- the gate -----------------------------------------------------------------


def test_clean_training_split_passes() -> None:
    assert_train_splits_are_licensed([row("apache-2.0"), row("cc-by-4.0"), row("mit", split="val")])


@pytest.mark.parametrize("split", sorted(TRAIN_SPLITS))
def test_gate_covers_every_training_split_including_validation(split: str) -> None:
    """A temperature fitted on data we may not use is as derived as a weight."""
    with pytest.raises(LicenceError):
        assert_train_splits_are_licensed([row("cc-by-nc-4.0", split=split)])


def test_non_commercial_data_is_allowed_in_an_eval_split() -> None:
    assert_train_splits_are_licensed([row("cc-by-nc-4.0", split="eval")])
    assert_train_splits_are_licensed([row("cc-by-nc-4.0", split="test")])


def test_the_external_eval_set_would_fail_if_anyone_put_it_in_train() -> None:
    """The exact mistake ADR 0004 exists to prevent."""
    with pytest.raises(LicenceError, match="system-one-decisions"):
        assert_train_splits_are_licensed(
            [row("cc-by-nc-4.0", split="train", source="pngwn/system-one-decisions")]
        )


def test_error_names_the_source_licence_and_row_count() -> None:
    rows = [row("cc-by-nc-4.0", source="facebook/anli") for _ in range(7)]
    with pytest.raises(LicenceError) as excinfo:
        assert_train_splits_are_licensed(rows)
    message = str(excinfo.value)
    assert "facebook/anli" in message
    assert "CC-BY-NC-4.0" in message
    assert "7 rows" in message
    assert "eval-only" in message


def test_unknown_licences_are_blocked_from_training() -> None:
    with pytest.raises(LicenceError, match="unknown"):
        assert_train_splits_are_licensed([row("unknown", source="Yelp/yelp_review_full")])


def test_rows_must_carry_the_schema_fields() -> None:
    for missing in ("split", "license", "source"):
        incomplete = row("mit")
        del incomplete[missing]
        with pytest.raises(LicenceError, match=missing):
            assert_train_splits_are_licensed([incomplete])


def test_an_empty_dataset_passes_vacuously() -> None:
    assert_train_splits_are_licensed([])


# --- audited sources, pinned --------------------------------------------------

#: Verdicts resolved from the Hugging Face Hub on 2026-09-17 and recorded in
#: docs/research/source-licences-2026-09-17.md. Pinned so that loosening the allowlist, or
#: quietly reclassifying a source, breaks a test rather than a release.
AUDITED = {
    "PolyAI/banking77": ("cc-by-4.0", "train"),
    "clinc/clinc_oos": ("cc-by-3.0", "train"),
    "AmazonScience/massive": ("cc-by-4.0", "train"),
    "google-research-datasets/go_emotions": ("apache-2.0", "train"),
    "cais/mmlu": ("mit", "train"),
    "tau/commonsense_qa": ("mit", "train"),
    "qiaojin/PubMedQA": ("mit", "train"),
    "SetFit/amazon_reviews_multi_en": ("apache-2.0", "train"),
    "jakartaresearch/google-play-review": ("cc-by-4.0", "train"),
    "facebook/anli": ("cc-by-nc-4.0", "eval-only"),
    "pngwn/system-one-decisions": ("cc-by-nc-4.0", "eval-only"),
    "allenai/sciq": ("cc-by-nc-3.0", "eval-only"),
    "nyu-mll/multi_nli": ("cc-by-3.0, cc-by-sa-3.0, mit, other", "refused"),
    "fever/fever": ("cc-by-sa-3.0, gpl-3.0", "refused"),
    "google/boolq": ("cc-by-sa-3.0", "refused"),
    "stanfordnlp/snli": ("cc-by-sa-4.0", "refused"),
    "allenai/ai2_arc": ("cc-by-sa-4.0", "refused"),
    "fancyzhx/ag_news": ("unknown", "refused"),
    "fancyzhx/dbpedia_14": ("cc-by-sa-3.0", "refused"),
    "Yelp/yelp_review_full": ("other", "refused"),
    "stanfordnlp/sst": ("unknown", "refused"),
}


@pytest.mark.parametrize(("source", "declared_and_verdict"), sorted(AUDITED.items()))
def test_audited_sources_keep_their_recorded_verdict(source, declared_and_verdict) -> None:
    declared, expected = declared_and_verdict
    assert classify_licence(declared) == expected, source


def test_the_external_eval_set_is_never_train_eligible() -> None:
    assert AUDITED["pngwn/system-one-decisions"][1] == "eval-only"


def test_at_least_one_train_eligible_source_exists_per_primitive() -> None:
    """If this fails, Phase 1 has no data for a primitive and that is a planning problem."""
    train_eligible = {s for s, (_, v) in AUDITED.items() if v == "train"}
    assert "PolyAI/banking77" in train_eligible  # Choice, high cardinality
    assert "google-research-datasets/go_emotions" in train_eligible  # Noul
    assert "SetFit/amazon_reviews_multi_en" in train_eligible  # Score, ordinal


# --- summaries ----------------------------------------------------------------


def test_summarise_counts_rows_per_source_and_split() -> None:
    summary = summarise_licences(
        [
            row("apache-2.0", split="train", source="a"),
            row("apache-2.0", split="train", source="a"),
            row("apache-2.0", split="test", source="a"),
            row("cc-by-nc-4.0", split="eval", source="b"),
        ]
    )
    assert summary["a"] == {
        "license": "Apache-2.0",
        "verdict": "train",
        "rows": 3,
        "splits": {"train": 2, "test": 1},
    }
    assert summary["b"]["verdict"] == "eval-only"


def test_summarise_rejects_a_source_with_two_licences() -> None:
    with pytest.raises(LicenceError, match="two different licences"):
        summarise_licences([row("mit", source="a"), row("apache-2.0", source="a")])


# --- Tier-2 leakage guard (Phase 1 plan, amendment B) -------------------------


def test_state_hashes_ignore_surrounding_whitespace() -> None:
    assert state_hashes([row("mit", state="  hello  ")]) == state_hashes(
        [row("mit", state="hello")]
    )


def test_leaked_external_rows_are_dropped_and_counted() -> None:
    """pngwn is derived from banking77, so its states can appear in our train splits."""
    train = [row("cc-by-4.0", state="shared ticket text"), row("cc-by-4.0", state="ours only")]
    external = [
        row("cc-by-nc-4.0", split="eval", state="shared ticket text"),
        row("cc-by-nc-4.0", split="eval", state="genuinely unseen"),
    ]
    kept, dropped = drop_leaked_rows(external, train)
    assert dropped == 1
    assert [r["state"] for r in kept] == ["genuinely unseen"]


def test_nothing_is_dropped_when_there_is_no_overlap() -> None:
    kept, dropped = drop_leaked_rows([row("mit", state="a")], [row("mit", state="b")])
    assert dropped == 0
    assert len(kept) == 1


def test_leakage_guard_handles_an_empty_train_set() -> None:
    kept, dropped = drop_leaked_rows([row("mit", state="a")], [])
    assert dropped == 0
    assert len(kept) == 1
