"""The three decision primitives, the question wrapper and the result type.

`Choice`, `Score` and `Noul` describe *what is being asked*; they carry no model
state and no probabilities. `Question` pairs one of them with a caller-supplied
name. `Result` carries what came back.

Everything here is frozen and validated at construction, so an invalid question
cannot reach the engine and an off-schema result cannot be built at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

__all__ = [
    "MAX_OPTIONS",
    "MIN_OPTIONS",
    "Choice",
    "Noul",
    "NoulLabel",
    "QType",
    "Question",
    "Result",
    "Score",
    "Spec",
]

#: Smallest number of options a question may offer. One option is not a decision.
MIN_OPTIONS = 2

#: Largest number of options a question may offer. Mirrors the documented limit of the
#: product this project is measured against; see docs/research/jev-landscape-2026-09-17.md
#: section 1, which also records that the limit is worth re-verifying.
#: Note this is the *schema* limit. Single-token labelling currently covers 26 options
#: (see s1decide.tokens); beyond that a two-stage scheme is required.
MAX_OPTIONS = 255

QType: TypeAlias = Literal["choice", "score", "noul"]

#: The two labels a Noul resolves over, in index order. `noul` is P(index 1) = P(yes).
NoulLabel: TypeAlias = Literal["no", "yes"]
NOUL_OPTIONS: tuple[str, str] = ("no", "yes")

#: Probability mass may deviate from 1 by at most this much before `Result` rejects it.
PROB_SUM_TOLERANCE = 1e-6


def _validate_instructions(instructions: str) -> str:
    """Validate and canonicalise the instruction text of a question.

    Args:
        instructions: The question as the caller wrote it.

    Returns:
        The instructions unchanged.

    Raises:
        TypeError: If ``instructions`` is not a string.
        ValueError: If it is empty or blank.
    """
    if not isinstance(instructions, str):
        raise TypeError(f"instructions must be a string, got {type(instructions).__name__}")
    if not instructions.strip():
        raise ValueError("instructions must not be empty")
    return instructions


def _validate_options(options: tuple[str, ...], *, kind: str) -> tuple[str, ...]:
    """Validate an option or level list.

    Enforces the count bounds, rejects labels that would break prompt rendering
    (empty, surrounded by whitespace, containing a newline), and rejects duplicates
    both exactly and case-insensitively — two options that differ only in case are a
    footgun for a model asked to pick exactly one.

    Args:
        options: The option labels, in caller order.
        kind: Field name used in error messages, e.g. ``"options"`` or ``"levels"``.

    Returns:
        The validated options as a tuple.

    Raises:
        TypeError: If any option is not a string.
        ValueError: If the count is out of bounds, a label is malformed, or two
            labels collide.
    """
    if len(options) < MIN_OPTIONS:
        raise ValueError(f"{kind} must have at least {MIN_OPTIONS} entries, got {len(options)}")
    if len(options) > MAX_OPTIONS:
        raise ValueError(f"{kind} must have at most {MAX_OPTIONS} entries, got {len(options)}")

    for option in options:
        if not isinstance(option, str):
            raise TypeError(f"every entry of {kind} must be a string, got {type(option).__name__}")
        if not option.strip():
            raise ValueError(f"{kind} must not contain empty entries")
        if option != option.strip():
            raise ValueError(f"{kind} entries must not be padded with whitespace: {option!r}")
        if "\n" in option or "\r" in option:
            raise ValueError(f"{kind} entries must be single-line: {option!r}")

    if len(set(options)) != len(options):
        duplicates = sorted({o for o in options if options.count(o) > 1})
        raise ValueError(f"{kind} must be unique, repeated: {duplicates}")

    folded = [o.casefold() for o in options]
    if len(set(folded)) != len(folded):
        duplicates = sorted({o for o in folded if folded.count(o) > 1})
        raise ValueError(f"{kind} must be unique ignoring case, collides: {duplicates}")

    return tuple(options)


@dataclass(frozen=True)
class Choice:
    """Pick exactly one option from a caller-supplied, unordered set.

    Attributes:
        instructions: The question, phrased as one atomic judgement.
        options: The allowed answers, 2 to :data:`MAX_OPTIONS` of them, unique.
    """

    instructions: str
    options: tuple[str, ...]

    qtype: QType = field(default="choice", init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "instructions", _validate_instructions(self.instructions))
        object.__setattr__(self, "options", _validate_options(tuple(self.options), kind="options"))

    @property
    def labels(self) -> tuple[str, ...]:
        """The answerable labels, in index order."""
        return self.options


@dataclass(frozen=True)
class Score:
    """Place the state on an ordered rubric.

    The order of ``levels`` *is* the rubric: index 0 is the lowest level and the last
    index the highest. Nothing can verify that the caller's order is semantically
    monotonic, so this is a documented contract, not a checked one — but the ordering
    is what the ordinal loss and :attr:`Result.expected_level` rely on.

    Attributes:
        instructions: The question, phrased as one one-dimensional judgement.
        levels: The rubric levels from lowest to highest, unique.
    """

    instructions: str
    levels: tuple[str, ...]

    qtype: QType = field(default="score", init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "instructions", _validate_instructions(self.instructions))
        object.__setattr__(self, "levels", _validate_options(tuple(self.levels), kind="levels"))

    @property
    def labels(self) -> tuple[str, ...]:
        """The answerable labels, lowest level first."""
        return self.levels


@dataclass(frozen=True)
class Noul:
    """The probability that a statement about the state is true.

    A Noul is a two-option :class:`Choice` over ``("no", "yes")`` — locked decision 3 in
    CLAUDE.md — so it shares the whole masked-logit path. Index 1 is ``yes``, and
    :attr:`Result.noul` reports its probability.

    Attributes:
        instructions: The statement to judge, phrased so that "yes" is unambiguous.
    """

    instructions: str

    qtype: QType = field(default="noul", init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "instructions", _validate_instructions(self.instructions))

    @property
    def labels(self) -> tuple[str, ...]:
        """``("no", "yes")`` — fixed, so that index 1 is always the true case."""
        return NOUL_OPTIONS


Spec: TypeAlias = Choice | Score | Noul


@dataclass(frozen=True)
class Question:
    """A named question about the shared state.

    Attributes:
        name: The caller's key for this question. Must be a non-empty, single-line
            string; it is echoed back on the :class:`Result` and used in error messages.
        spec: The primitive being asked.
    """

    name: str
    spec: Spec

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError(f"question name must be a string, got {type(self.name).__name__}")
        if not self.name.strip():
            raise ValueError("question name must not be empty")
        if "\n" in self.name or "\r" in self.name:
            raise ValueError(f"question name must be single-line: {self.name!r}")
        if not isinstance(self.spec, Choice | Score | Noul):
            raise TypeError(
                f"question spec must be Choice, Score or Noul, got {type(self.spec).__name__}"
            )

    @property
    def qtype(self) -> QType:
        """``"choice"``, ``"score"`` or ``"noul"``."""
        return self.spec.qtype

    @property
    def labels(self) -> tuple[str, ...]:
        """The answerable labels of the underlying spec, in index order."""
        return self.spec.labels

    @property
    def instructions(self) -> str:
        """The instruction text of the underlying spec."""
        return self.spec.instructions


@dataclass(frozen=True)
class Result:
    """One question's answer: a full distribution over its labels, plus derived views.

    Attributes:
        name: The question name this answers.
        qtype: Which primitive produced it.
        options: The labels, in the same index order the question defined.
        probabilities: Probability per label, aligned with ``options``, summing to 1.
    """

    name: str
    qtype: QType
    options: tuple[str, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", tuple(self.options))
        object.__setattr__(self, "probabilities", tuple(float(p) for p in self.probabilities))

        if len(self.options) != len(self.probabilities):
            raise ValueError(
                f"got {len(self.options)} options but {len(self.probabilities)} probabilities"
            )
        if not self.options:
            raise ValueError("a result must cover at least one option")
        for p in self.probabilities:
            if not math.isfinite(p):
                raise ValueError(f"probabilities must be finite, got {self.probabilities}")
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"probabilities must lie in [0, 1], got {p}")
        total = math.fsum(self.probabilities)
        if abs(total - 1.0) > PROB_SUM_TOLERANCE:
            raise ValueError(f"probabilities must sum to 1, got {total!r}")

    @property
    def distribution(self) -> dict[str, float]:
        """The distribution as a label to probability mapping, in index order."""
        return dict(zip(self.options, self.probabilities))

    @property
    def argmax_index(self) -> int:
        """Index of the most probable label.

        Ties resolve to the lowest index, which keeps results reproducible.
        """
        return max(range(len(self.probabilities)), key=self.probabilities.__getitem__)

    @property
    def confidence(self) -> float:
        """Probability of the most probable label.

        This is the *uncalibrated* confidence. Temperature scaling is applied
        separately, per option-count bucket and per deployment quantization; see
        :mod:`s1decide.calibrate`.
        """
        return self.probabilities[self.argmax_index]

    @property
    def choice(self) -> str:
        """The selected option. Only meaningful for a :class:`Choice`.

        Raises:
            TypeError: If this result did not come from a Choice.
        """
        if self.qtype != "choice":
            raise TypeError(f"result {self.name!r} is a {self.qtype}, not a choice")
        return self.options[self.argmax_index]

    @property
    def score(self) -> str:
        """The most probable rubric level. Only meaningful for a :class:`Score`.

        Raises:
            TypeError: If this result did not come from a Score.
        """
        if self.qtype != "score":
            raise TypeError(f"result {self.name!r} is a {self.qtype}, not a score")
        return self.options[self.argmax_index]

    @property
    def expected_level(self) -> float:
        """Probability-weighted mean level index. Only meaningful for a :class:`Score`.

        Reported alongside :attr:`score` because a bimodal distribution over an ordinal
        rubric — a known failure mode, see docs/research/jev-landscape-2026-09-17.md
        section 4 — has an argmax that hides the disagreement and a mean that shows it.

        Raises:
            TypeError: If this result did not come from a Score.
        """
        if self.qtype != "score":
            raise TypeError(f"result {self.name!r} is a {self.qtype}, not a score")
        return math.fsum(i * p for i, p in enumerate(self.probabilities))

    @property
    def noul(self) -> float:
        """Probability that the statement is true. Only meaningful for a :class:`Noul`.

        Raises:
            TypeError: If this result did not come from a Noul.
        """
        if self.qtype != "noul":
            raise TypeError(f"result {self.name!r} is a {self.qtype}, not a noul")
        return self.distribution["yes"]
