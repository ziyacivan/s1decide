"""Load and normalise evaluation data into the project's question schema.

Currently one source: ``pngwn/system-one-decisions``.

**Licence: CC-BY-NC-4.0 (non-commercial).** Fine for measuring a model. It is *not* fine as
training data for an Apache-2.0 release — see `docs/dataset-card.md` and the note in
`docs/adr/`. Nothing here trains anything; `family` is carried through so the eval can be read
per source.

Two normalisations happen at load time, both of which would be silent bugs otherwise:

1. **Noul answer indices are flipped.** The dataset lists options as ``('yes', 'no')`` — index
   0 is *yes*. :class:`s1decide.primitives.Noul` fixes labels as ``('no', 'yes')`` so that
   index 1 is always the true case and ``Result.noul`` means P(true). Loading remaps the index.
2. **Score levels are checked for ascending order**, which this dataset already satisfies
   (``'1 star'..'5 stars'``, ``'very low'..'critical'``), because
   :class:`s1decide.primitives.Score` treats position as the rubric.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from s1decide.primitives import Choice, Noul, Question, Score
from s1decide.tokens import MAX_SINGLE_TOKEN_OPTIONS

__all__ = [
    "DATASET_LICENSE",
    "SYSTEM_ONE_DECISIONS",
    "Coverage",
    "EvalItem",
    "group_by_state",
    "load_system_one_decisions",
]

SYSTEM_ONE_DECISIONS = "pngwn/system-one-decisions"
DATASET_LICENSE = "CC-BY-NC-4.0"

#: How the dataset encodes a Noul, and therefore what the remap below assumes.
_DATASET_NOUL_OPTIONS = ("yes", "no")


@dataclass(frozen=True)
class EvalItem:
    """One question, normalised to this project's conventions.

    ``options`` and ``answer_idx`` are in **our** label order, so ``answer_idx`` indexes
    directly into the logits the engine returns.

    Attributes:
        id: Stable id, derived from the source row so a run can be joined back to it.
        family: Task family (``banking77``, ``go_emotions``, ...).
        qtype: ``choice``, ``score`` or ``noul``.
        state: The shared state text.
        instructions: The question text.
        options: Option labels in our order.
        answer_idx: Index of the correct option in ``options``.
        split: Source split.
        source: Source dataset id.
        license: Source licence.
    """

    id: str
    family: str
    qtype: str
    state: str
    instructions: str
    options: tuple[str, ...]
    answer_idx: int
    split: str
    source: str = SYSTEM_ONE_DECISIONS
    license: str = DATASET_LICENSE

    @property
    def n_options(self) -> int:
        """Number of options."""
        return len(self.options)

    def to_question(self, name: str | None = None) -> Question:
        """Build the :class:`~s1decide.primitives.Question` this item asks.

        Args:
            name: Question name; defaults to the item id.

        Returns:
            A validated question whose ``labels`` are aligned with ``options``.
        """
        if self.qtype == "noul":
            spec: Choice | Score | Noul = Noul(instructions=self.instructions)
        elif self.qtype == "score":
            spec = Score(instructions=self.instructions, levels=self.options)
        else:
            spec = Choice(instructions=self.instructions, options=self.options)
        question = Question(name=name or self.id, spec=spec)
        if question.labels != self.options:
            raise ValueError(
                f"{self.id}: question labels {question.labels} do not match item options "
                f"{self.options}; answer_idx would point at the wrong thing"
            )
        return question


@dataclass(frozen=True)
class Coverage:
    """What a load kept and what it dropped, so a report can state its own limits.

    Attributes:
        total: Rows in the split.
        kept: Rows returned.
        dropped_high_cardinality: Rows dropped for having more options than there are
            single-token labels.
        max_options_kept: The cap applied.
        dropped_families: Per-family counts of what was dropped for high cardinality.
        dropped_invalid: Rows dropped because they could not be normalised into a valid
            question at all (e.g. options that collide once whitespace is stripped).
        invalid_examples: A few reasons, verbatim, so a silent drop can be investigated.
    """

    total: int
    kept: int
    dropped_high_cardinality: int
    max_options_kept: int
    dropped_families: dict[str, int]
    dropped_invalid: int = 0
    invalid_examples: tuple[str, ...] = ()

    @property
    def fraction_kept(self) -> float:
        """Fraction of the split actually evaluated."""
        return self.kept / self.total if self.total else 0.0

    def to_json(self) -> dict[str, Any]:
        """Serialise into ``metrics.json``."""
        return {
            "total": self.total,
            "kept": self.kept,
            "fraction_kept": self.fraction_kept,
            "dropped_high_cardinality": self.dropped_high_cardinality,
            "max_options_kept": self.max_options_kept,
            "dropped_families": dict(sorted(self.dropped_families.items())),
            "dropped_invalid": self.dropped_invalid,
            "invalid_examples": list(self.invalid_examples),
        }


def _item_id(split: str, index: int, state: str, question: str) -> str:
    digest = hashlib.blake2b(f"{state}\x1f{question}".encode(), digest_size=6).hexdigest()
    return f"{split}-{index:05d}-{digest}"


def _normalise(row: dict[str, Any], split: str, index: int) -> EvalItem:
    """Map one source row onto :class:`EvalItem`, remapping Noul indices.

    Raises:
        ValueError: If the row cannot be represented as a valid question.
    """
    qtype = str(row["question_type"])
    # Some sources (MMLU here) carry trailing spaces on option text. The primitives reject
    # padded labels because they would render inconsistently, so strip at the boundary — and
    # refuse if stripping makes two options collide, rather than quietly merging them.
    options = tuple(str(o).strip() for o in row["options"])
    answer_idx = int(row["answer_index"])
    if any(not o for o in options):
        raise ValueError(f"{split}[{index}]: an option is empty once stripped")
    folded = [o.casefold() for o in options]
    if len(set(folded)) != len(folded):
        raise ValueError(f"{split}[{index}]: options collide once stripped: {options}")

    if qtype == "noul":
        if options != _DATASET_NOUL_OPTIONS:
            raise ValueError(
                f"{split}[{index}]: expected noul options {_DATASET_NOUL_OPTIONS}, got {options}; "
                "the yes/no remap below would be wrong"
            )
        # ('yes', 'no') -> ('no', 'yes'): index 0 <-> 1.
        options = ("no", "yes")
        answer_idx = 1 - answer_idx
    elif qtype == "score" and not row.get("ordered", False):
        raise ValueError(f"{split}[{index}]: a score question must be marked ordered")

    return EvalItem(
        id=_item_id(split, index, str(row["state"]), str(row["question"])),
        family=str(row["task"]),
        qtype=qtype,
        state=str(row["state"]).strip(),
        instructions=str(row["question"]).strip(),
        options=options,
        answer_idx=answer_idx,
        split=split,
    )


def load_system_one_decisions(
    split: str,
    *,
    max_options: int = MAX_SINGLE_TOKEN_OPTIONS,
    limit: int | None = None,
) -> tuple[list[EvalItem], Coverage]:
    """Load one split of ``pngwn/system-one-decisions``, normalised.

    Args:
        split: ``train``, ``val`` or ``test``.
        max_options: Drop questions with more options than this. The default is the number of
            single-token labels available (26); ``banking77`` (77 options) and
            ``tickets_queue`` (52) exceed it and need the two-stage path from ADR 0001, which
            does not exist yet. Dropping is **reported**, never silent — see :class:`Coverage`.
        limit: Keep only the first N rows after filtering. For smoke runs.

    Returns:
        The items and a :class:`Coverage` record.
    """
    from datasets import load_dataset

    raw = load_dataset(SYSTEM_ONE_DECISIONS, split=split)
    items: list[EvalItem] = []
    dropped: dict[str, int] = {}
    invalid: list[str] = []
    for index, row in enumerate(raw):
        try:
            item = _normalise(row, split, index)
            question = item.to_question()
        except (ValueError, TypeError) as exc:
            invalid.append(str(exc))
            continue
        if item.n_options > max_options:
            dropped[item.family] = dropped.get(item.family, 0) + 1
            continue
        del question
        items.append(item)

    coverage = Coverage(
        total=len(raw),
        kept=len(items),
        dropped_high_cardinality=sum(dropped.values()),
        max_options_kept=max_options,
        dropped_families=dropped,
        dropped_invalid=len(invalid),
        invalid_examples=tuple(invalid[:5]),
    )
    if limit is not None:
        items = items[:limit]
    return items, coverage


def group_by_state(items: Iterable[EvalItem]) -> list[tuple[str, list[EvalItem]]]:
    """Bundle questions that share a state, so each state is prefilled once.

    This is the whole point of the engine: in this dataset roughly 40% of questions arrive in
    bundles of three or four about the same text.

    Args:
        items: Items to group.

    Returns:
        ``(state, items)`` pairs in first-appearance order, each item list in input order.
    """
    groups: dict[str, list[EvalItem]] = {}
    for item in items:
        groups.setdefault(item.state, []).append(item)
    return list(groups.items())


def bundle_stats(groups: Sequence[tuple[str, list[EvalItem]]]) -> dict[str, Any]:
    """Summarise how questions bundle, for the run's provenance record."""
    sizes = [len(g) for _, g in groups]
    counts: dict[int, int] = {}
    for s in sizes:
        counts[s] = counts.get(s, 0) + 1
    return {
        "states": len(groups),
        "questions": sum(sizes),
        "mean_questions_per_state": sum(sizes) / len(sizes) if sizes else 0.0,
        "max_questions_per_state": max(sizes) if sizes else 0,
        "bundle_size_counts": dict(sorted(counts.items())),
    }
