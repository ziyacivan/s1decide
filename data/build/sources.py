"""The source registry: what we may use, how to load it, and what it becomes.

Every entry carries its SPDX licence, and the licence gate (ADR 0004) is applied to the rows
this module produces — not to a comment about them. A source's ``role`` decides which splits
it may reach:

* ``train`` — split into train / val / test by state hash.
* ``heldout`` — a whole family unseen in training, evaluated as Tier 1.
* ``ood`` — fully out of distribution, evaluated as Tier 1. Both OOD sets are non-commercial,
  so the licence gate makes training on them impossible by construction rather than by care.

Loading note: ``datasets`` 4.x removed script-based datasets, which breaks
``PolyAI/banking77`` and ``AmazonScience/massive`` at their canonical ids.
``legacy-datasets/banking77`` is Hugging Face's own parquet mirror of the same CC-BY-4.0 data
and is used instead; the provenance is recorded here so the substitution is visible.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

__all__ = ["SOURCES", "Source", "load_source", "row_id"]


@dataclass(frozen=True)
class Source:
    """One dataset we are allowed to use, and what it becomes.

    Attributes:
        family: Family name recorded on every row; the unit held out by `heldout`.
        repo: Hugging Face dataset id actually loaded.
        canonical_repo: The id the licence was audited under, when it differs from ``repo``.
        license: SPDX identifier, checked by the gate.
        primitive: ``choice``, ``noul`` or ``score``.
        role: ``train``, ``heldout`` or ``ood``.
        config: Dataset config name, if any.
        split: Source split to read.
        revision: Dataset revision, for parquet-branch loads.
        locale: Keep only rows with this ``locale``. MASSIVE ships all 51 languages in one
            config, and each language is a separate family to us.
        row_offset: Skip this many rows after filtering, before applying ``max_rows``. Used to
            give the MASSIVE training locales *disjoint* slices of a parallel corpus.
        normaliser: Which normaliser to use, when several families share one. Defaults to
            ``family``.
        max_rows: Cap, to keep the build quick and the families balanced.
        note: Anything a reader should know about the substitution or the mapping.
    """

    family: str
    repo: str
    license: str
    primitive: str
    role: str
    config: str | None = None
    split: str = "train"
    revision: str | None = None
    locale: str | None = None
    row_offset: int = 0
    normaliser: str | None = None
    max_rows: int | None = None
    canonical_repo: str | None = None
    note: str = ""


SOURCES: tuple[Source, ...] = (
    Source(
        family="banking77",
        repo="legacy-datasets/banking77",
        canonical_repo="PolyAI/banking77",
        license="cc-by-4.0",
        primitive="choice",
        role="train",
        max_rows=4000,
        note="77 intents — exercises the two-stage path. PolyAI/banking77 is script-based and "
        "unloadable on datasets 4.x; this is HF's parquet mirror of the same CC-BY-4.0 data.",
    ),
    Source(
        family="clinc_oos",
        repo="clinc/clinc_oos",
        license="cc-by-3.0",
        primitive="choice",
        role="train",
        config="plus",
        max_rows=4000,
        note="151 intents including out-of-scope — the highest-cardinality source we have.",
    ),
    # MASSIVE: 60 intents in 51 languages, one `default` config, CC-BY-4.0 first-party.
    #
    # Capped at 1,200 source rows per locale. The cap is a balance decision, not a size one.
    #
    # It was 3,000, chosen by comparing source-row counts against banking77's 4,000. That was
    # wrong twice over, and the manifest said so: each source row expands to ~5 rows through the
    # two-stage path (one stage-2 row plus a positive and three hard negatives), and there are
    # *three* locales, so 3,000 per locale meant 36,180 training rows — **48% of the corpus**,
    # the single largest block by far. The comparison has to be made on expanded rows summed
    # across the locales, which is what the manifest reports.
    #
    #   source rows/locale   MASSIVE train rows   share of training
    #                 3000                36180               48.4%
    #                 1500                18090               32.0%
    #                 1200                14472               27.3%   <- chosen
    #                 1000                12060               23.8%
    #
    # 1,200 puts all three locales together at roughly banking77's single-source weight (15,810,
    # ~30%), which is the intended reading of "must not dominate": comparable to the largest
    # other source, not larger than all of them. Uncapped it would be ~104k source rows and the
    # model would learn "predict an intent" rather than "answer the question asked".
    #
    # 60 intents is above the 26-label ceiling, so every row goes through two-stage expansion.
    #
    # **MASSIVE is a parallel corpus**: the same utterance is translated into all 51 locales and
    # keeps its id, so taking the first 3,000 rows of each language would give three translations
    # of one set of 3,000 meanings rather than 9,000 distinct ones (verified — the first 3,000
    # train ids are identical across en/tr/de). The locales therefore take *disjoint* slices via
    # `row_offset`. Diversity is worth more here than alignment: the controlled
    # same-meaning-different-language comparison is what the fr/ja OOD pair is for, and it is
    # already perfectly controlled because those two *are* parallel to each other. Seeing one
    # meaning three times mostly invites memorising the utterance.
    #
    # Note for the leakage guard: it hashes state text, so it cannot see cross-locale
    # parallelism — two translations of one sentence are different strings. Disjoint slices are
    # what keeps that from mattering, not the guard.
    Source(
        family="massive_en",
        repo="AmazonScience/massive",
        license="cc-by-4.0",
        primitive="choice",
        role="train",
        revision="refs/convert/parquet",
        locale="en-US",
        row_offset=0,
        normaliser="massive",
        max_rows=1200,
        note="English. Utterances 0-1,199 of the locale. The control locale: the other four "
        "are read against it.",
    ),
    Source(
        family="massive_tr",
        repo="AmazonScience/massive",
        license="cc-by-4.0",
        primitive="choice",
        role="train",
        revision="refs/convert/parquet",
        locale="tr-TR",
        row_offset=1200,
        normaliser="massive",
        max_rows=1200,
        note="Turkish. Utterances 1,200-2,399, disjoint from the other locales. Agglutinative "
        "and Latin-script; the non-English language we most want to work, and the reason "
        "MASSIVE is in the corpus at all.",
    ),
    Source(
        family="massive_de",
        repo="AmazonScience/massive",
        license="cc-by-4.0",
        primitive="choice",
        role="train",
        revision="refs/convert/parquet",
        locale="de-DE",
        row_offset=2400,
        normaliser="massive",
        max_rows=1200,
        note="German. Utterances 2,400-3,599, disjoint from the other locales. A second "
        "Latin-script European language, so 'multilingual' is not one language plus English.",
    ),
    # The unseen-language OOD pair. Never in train or val. Two points chosen to separate two
    # different failures: fr-FR is close to the training languages (Latin script,
    # Indo-European, shares vocabulary with English and German), ja-JP is far from all of them
    # (non-Latin script, no shared script or family, and tokenises quite differently). If
    # accuracy holds on French but collapses on Japanese, the failure is script and
    # tokenisation rather than language transfer, and the two sets tell us which.
    Source(
        family="massive_fr",
        repo="AmazonScience/massive",
        license="cc-by-4.0",
        primitive="choice",
        role="ood",
        revision="refs/convert/parquet",
        locale="fr-FR",
        normaliser="massive",
        split="test",
        max_rows=1000,
        note="OOD, unseen language, NEAR: same script and family as the training locales. "
        "Parallel to massive_ja on purpose — same utterances, so the near/far difference is "
        "language and script alone.",
    ),
    Source(
        family="massive_ja",
        repo="AmazonScience/massive",
        license="cc-by-4.0",
        primitive="choice",
        role="ood",
        revision="refs/convert/parquet",
        locale="ja-JP",
        normaliser="massive",
        split="test",
        max_rows=1000,
        note="OOD, unseen language, FAR: different script, no shared family, different "
        "tokenisation behaviour. Parallel to massive_fr, as above.",
    ),
    Source(
        family="go_emotions",
        repo="google-research-datasets/go_emotions",
        license="apache-2.0",
        primitive="noul",
        role="train",
        config="simplified",
        max_rows=6000,
        note="Multi-label over 28 emotions, so one message yields several bundled Nouls.",
    ),
    Source(
        family="mmlu",
        repo="cais/mmlu",
        license="mit",
        primitive="choice",
        role="train",
        config="all",
        split="validation",
        max_rows=1500,
        note="4-option knowledge questions; the question is the state.",
    ),
    Source(
        family="commonsense_qa",
        repo="tau/commonsense_qa",
        license="mit",
        primitive="choice",
        role="heldout",
        max_rows=1500,
        note="Held-out family: 5-option commonsense, never trained on.",
    ),
    Source(
        family="pubmedqa",
        repo="qiaojin/PubMedQA",
        license="mit",
        primitive="noul",
        role="heldout",
        config="pqa_labeled",
        max_rows=1000,
        note="Held-out family: biomedical yes/no over an abstract. 'maybe' rows are dropped.",
    ),
    Source(
        family="anli",
        repo="facebook/anli",
        license="cc-by-nc-4.0",
        primitive="choice",
        role="ood",
        split="test_r1",
        max_rows=1000,
        note="OOD. Non-commercial, so the licence gate makes training on it impossible.",
    ),
    Source(
        family="sciq",
        repo="allenai/sciq",
        license="cc-by-nc-3.0",
        primitive="choice",
        role="ood",
        split="test",
        max_rows=1000,
        note="OOD. Non-commercial, as above.",
    ),
)


def row_id(family: str, index: int, state: str, instructions: str) -> str:
    """A stable id derived from the content, so a rebuild reproduces it exactly."""
    digest = hashlib.blake2b(f"{state}\x1f{instructions}".encode(), digest_size=6).hexdigest()
    return f"{family}-{index:06d}-{digest}"


def _humanise(label: str) -> str:
    """Turn an intent slug into readable option text (``card_arrival`` -> ``card arrival``)."""
    return re.sub(r"[_\-]+", " ", str(label).strip().strip("'\"")).strip()


def _class_names(dataset: Any, column: str) -> list[str] | None:
    feature = dataset.features.get(column)
    names = getattr(feature, "names", None)
    return [str(n) for n in names] if names else None


# --- per-source normalisers ---------------------------------------------------
# Each yields dicts with state / instructions / options / answer_idx; the pipeline adds
# family, licence, split and ids.


def _intent_rows(
    dataset: Any, text_column: str, label_column: str, question: str
) -> Iterator[dict[str, Any]]:
    names = _class_names(dataset, label_column)
    if names is None:
        raise ValueError(f"{label_column} has no class names; cannot build option labels")
    options = tuple(_humanise(n) for n in names)
    if len(set(options)) != len(options):
        raise ValueError("intent labels collide once humanised")
    for row in dataset:
        yield {
            "state": str(row[text_column]).strip().strip("'\""),
            "instructions": question,
            "options": options,
            "answer_idx": int(row[label_column]),
        }


def _banking77(dataset: Any) -> Iterator[dict[str, Any]]:
    return _intent_rows(dataset, "text", "label", "Which banking intent does this message have?")


def _clinc(dataset: Any) -> Iterator[dict[str, Any]]:
    return _intent_rows(dataset, "text", "intent", "Which intent does this request have?")


def _massive(dataset: Any) -> Iterator[dict[str, Any]]:
    """One Choice per utterance over MASSIVE's 60 intents.

    The question is asked in English for every locale on purpose. The task is intent
    classification, not translation, and holding the instruction fixed means a locale's score
    measures how well the model reads *that language's* utterance rather than how well it
    handles a differently-worded prompt. It also keeps the option labels identical across
    locales, so the two-stage expansion produces comparable negatives everywhere.
    """
    return _intent_rows(dataset, "utt", "intent", "Which intent does this request have?")


def _go_emotions(dataset: Any) -> Iterator[dict[str, Any]]:
    """One Noul per sampled emotion: the positives, plus a few negatives for balance."""
    names = _class_names(dataset, "labels")
    if names is None:
        feature = dataset.features["labels"]
        names = [str(n) for n in feature.feature.names]
    rng = random.Random(20260917)
    for row in dataset:
        text = str(row["text"]).strip()
        positives = {int(i) for i in row["labels"]}
        negatives = rng.sample([i for i in range(len(names)) if i not in positives], k=3)
        for index in sorted(positives) + negatives:
            emotion = _humanise(names[index])
            yield {
                "state": text,
                "instructions": f'This message expresses the emotion "{emotion}".',
                "options": ("no", "yes"),
                "answer_idx": 1 if index in positives else 0,
            }


def _mmlu(dataset: Any) -> Iterator[dict[str, Any]]:
    for row in dataset:
        options = tuple(str(c).strip() for c in row["choices"])
        yield {
            "state": str(row["question"]).strip(),
            "instructions": "Which option correctly answers the question in the state?",
            "options": options,
            "answer_idx": int(row["answer"]),
        }


def _commonsense_qa(dataset: Any) -> Iterator[dict[str, Any]]:
    for row in dataset:
        labels = list(row["choices"]["label"])
        options = tuple(str(t).strip() for t in row["choices"]["text"])
        key = str(row["answerKey"]).strip()
        if key not in labels:
            continue  # the test split ships without answers
        yield {
            "state": str(row["question"]).strip(),
            "instructions": "Which option best answers the question in the state?",
            "options": options,
            "answer_idx": labels.index(key),
        }


def _pubmedqa(dataset: Any) -> Iterator[dict[str, Any]]:
    for row in dataset:
        decision = str(row["final_decision"]).strip().lower()
        if decision not in {"yes", "no"}:
            continue  # 'maybe' has no place in a two-option Noul
        context = " ".join(str(c).strip() for c in row["context"]["contexts"])
        yield {
            "state": context,
            "instructions": f"The answer to this question is yes: {str(row['question']).strip()}",
            "options": ("no", "yes"),
            "answer_idx": 1 if decision == "yes" else 0,
        }


def _anli(dataset: Any) -> Iterator[dict[str, Any]]:
    names = _class_names(dataset, "label") or ["entailment", "neutral", "contradiction"]
    options = tuple(_humanise(n) for n in names)
    for row in dataset:
        yield {
            "state": f"Premise: {str(row['premise']).strip()}\nHypothesis: {str(row['hypothesis']).strip()}",
            "instructions": "What is the relationship of the hypothesis to the premise?",
            "options": options,
            "answer_idx": int(row["label"]),
        }


def _sciq(dataset: Any) -> Iterator[dict[str, Any]]:
    rng = random.Random(20260917)
    for row in dataset:
        correct = str(row["correct_answer"]).strip()
        distractors = [str(row[f"distractor{i}"]).strip() for i in (1, 2, 3)]
        options = [correct, *distractors]
        if len({o.casefold() for o in options}) != len(options):
            continue
        rng.shuffle(options)
        support = str(row.get("support", "")).strip()
        yield {
            "state": support or str(row["question"]).strip(),
            "instructions": str(row["question"]).strip()
            if support
            else "Which option answers the question in the state?",
            "options": tuple(options),
            "answer_idx": options.index(correct),
        }


_NORMALISERS: dict[str, Callable[[Any], Iterator[dict[str, Any]]]] = {
    "banking77": _banking77,
    "clinc_oos": _clinc,
    "massive": _massive,
    "go_emotions": _go_emotions,
    "mmlu": _mmlu,
    "commonsense_qa": _commonsense_qa,
    "pubmedqa": _pubmedqa,
    "anli": _anli,
    "sciq": _sciq,
}


def load_source(source: Source) -> list[dict[str, Any]]:
    """Load and normalise one source into question dicts.

    Args:
        source: The registry entry to load.

    Returns:
        Normalised rows, capped at ``source.max_rows`` *after* normalisation so that a
        multi-question source like GoEmotions is capped on questions, not on messages.
    """
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": source.split}
    if source.revision:
        kwargs["revision"] = source.revision
    dataset = (
        load_dataset(source.repo, source.config, **kwargs)
        if source.config
        else load_dataset(source.repo, **kwargs)
    )
    if source.locale is not None:
        # MASSIVE ships every language in one split, so the locale filter is part of loading
        # rather than of normalising — the normaliser stays a plain intent mapper.
        before = len(dataset)
        dataset = dataset.filter(lambda row: row["locale"] == source.locale)
        if len(dataset) == 0:
            raise ValueError(
                f"{source.family}: locale {source.locale!r} matched no rows of "
                f"{before} in {source.repo} ({source.split})"
            )

    if source.row_offset:
        if source.row_offset >= len(dataset):
            raise ValueError(
                f"{source.family}: row_offset {source.row_offset} is past the end of "
                f"{len(dataset)} rows — the disjoint slices would silently collapse to nothing"
            )
        dataset = dataset.select(range(source.row_offset, len(dataset)))

    rows: list[dict[str, Any]] = []
    for row in _NORMALISERS[source.normaliser or source.family](dataset):
        rows.append(row)
        if source.max_rows is not None and len(rows) >= source.max_rows:
            break
    return rows
