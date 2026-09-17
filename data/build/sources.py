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
    rows: list[dict[str, Any]] = []
    for row in _NORMALISERS[source.family](dataset):
        rows.append(row)
        if source.max_rows is not None and len(rows) >= source.max_rows:
            break
    return rows
