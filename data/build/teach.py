"""Teacher-labelled `Score` rows (ADR 0005 rule b).

Two open-weight teachers score a written rubric **with reasoning on**, and only rows where they
agree exactly or within one level are kept. The reasoning is the point: we are distilling
deliberate System-2 judgements into a System-1 student that answers in one forward pass. A
teacher that answers in a single token is doing the student's job, badly.

Everything about a run that a reader might want to check is recorded — both model ids and
revisions, both prompt hashes, the reasoning setting, the trace-length distribution, the
truncation rate, and the agreement rates. None of it is typed by hand.

**A truncated trace is a dropped row, not a guessed label.** If the model runs out of budget
before committing to a level, there is no judgement to record, and inventing one would put
exactly the hardest cases into the corpus with noise for labels.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "RUBRIC",
    "TeacherSetting",
    "agreement_report",
    "build_prompt",
    "parse_level",
    "prompt_hash",
    "trace_stats",
]

#: The rubric both teachers score. Five ordered levels, defined so that the boundaries are about
#: the text rather than about a feeling, because an ordinal scale whose levels are vibes cannot
#: be agreed on by two models or checked by a person.
RUBRIC: tuple[str, ...] = (
    "1 - none: the state contains nothing relevant to the question",
    "2 - slight: a passing or indirect mention, not enough to act on",
    "3 - moderate: relevant and explicit, but partial or hedged",
    "4 - strong: clearly and directly stated, enough to act on",
    "5 - decisive: stated unambiguously and central to the state",
)

#: Levels, as the student will see them.
LEVELS: tuple[str, ...] = ("none", "slight", "moderate", "strong", "decisive")

#: Matches the teacher's committed answer.
#:
#: Not anchored to a line start, and no word-boundary escape after the digit. Both were in
#: the first version and both were wrong: Magistral emits ``FINAL: 1FINAL: 1`` — the marker
#: twice, run together — so there is no line break before the second and no word boundary
#: after the first digit. That parser scored 0 of 100 committed on a pilot where the model
#: had in fact answered every time, which is the most expensive kind of bug: it looks like
#: a model failure and would have disqualified a perfectly good teacher.
#:
#: The negative lookahead keeps `12` from reading as 1; the last match wins, so a model that
#: reconsiders mid-trace is taken at its final word.
_ANSWER = re.compile(r"FINAL:\s*([1-5])(?![0-9])", re.IGNORECASE)


@dataclass(frozen=True)
class TeacherSetting:
    """One teacher run at one reasoning setting.

    Attributes:
        model: Hugging Face model id.
        label: Short name used in filenames and reports.
        effort: Native reasoning effort where the model has one; ``None`` where it does not.
        max_new_tokens: Generation cap. For a model without an effort knob this **is** the only
            control, and the report says so rather than calling it an effort level.
        revision: Model revision, recorded for the dataset card.
        quantization: How it was loaded.
    """

    model: str
    label: str
    effort: str | None = None
    max_new_tokens: int = 384
    revision: str | None = None
    quantization: str = "nf4-bf16"

    @property
    def effort_description(self) -> str:
        """What the reasoning setting actually was — never overstated."""
        if self.effort:
            return f"native effort={self.effort}, max_new_tokens={self.max_new_tokens}"
        return (
            f"no native effort control; bounded by max_new_tokens={self.max_new_tokens} "
            "(a cap, not an effort level)"
        )

    def to_json(self) -> dict[str, Any]:
        """For the run's meta and the dataset card."""
        return {
            "model": self.model,
            "label": self.label,
            "effort": self.effort,
            "max_new_tokens": self.max_new_tokens,
            "revision": self.revision,
            "quantization": self.quantization,
            "effort_description": self.effort_description,
        }


def build_prompt(state: str, question: str, rubric: Sequence[str] = RUBRIC) -> str:
    """The judgement prompt, identical for both teachers.

    Identical by construction: two teachers scoring differently worded prompts would disagree
    about the prompt as much as about the answer, and the agreement rate would mean nothing.

    Args:
        state: The text being judged.
        question: What is being looked for in it.
        rubric: Ordered level definitions.

    Returns:
        The prompt text.
    """
    levels = "\n".join(rubric)
    return (
        "You are grading a piece of text against an ordered rubric.\n\n"
        f"TEXT:\n{state.strip()}\n\n"
        f"QUESTION: {question.strip()}\n\n"
        f"RUBRIC:\n{levels}\n\n"
        "Think briefly about which level fits, then commit. Your last line must be exactly:\n"
        "FINAL: <number 1-5>"
    )


def prompt_hash(prompt: str) -> str:
    """Stable hash of a prompt, recorded so a later run can prove it asked the same thing."""
    return hashlib.blake2b(prompt.encode("utf-8"), digest_size=8).hexdigest()


def parse_level(text: str) -> int | None:
    """Extract the committed level from a teacher's output.

    Args:
        text: Everything the model generated, reasoning trace included.

    Returns:
        The level as a 0-based index, or ``None`` when the model never committed — which is
        what a truncated trace looks like, and is a dropped row rather than a guess.
    """
    matches = _ANSWER.findall(text or "")
    if not matches:
        return None
    # Last wins: a model that reconsiders mid-trace should be taken at its final word.
    return int(matches[-1]) - 1


def trace_stats(lengths: Sequence[int]) -> dict[str, float]:
    """Mean and p90 trace length, which is what the overnight estimate is built from.

    Args:
        lengths: Generated token counts.

    Returns:
        ``{"n", "mean", "p90", "max"}``; zeros when there is nothing to summarise.
    """
    if not lengths:
        return {"n": 0, "mean": 0.0, "p90": 0.0, "max": 0.0}
    ordered = sorted(lengths)
    index = min(len(ordered) - 1, round(0.9 * (len(ordered) - 1)))
    return {
        "n": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p90": float(ordered[index]),
        "max": float(ordered[-1]),
    }


@dataclass
class AgreementReport:
    """How two teachers compared on the same rows.

    Attributes:
        compared: Rows where both teachers committed to a level.
        exact: Rows where they gave the same level.
        within_one: Rows within one level, the exact ones included.
        dropped: Rows kept by neither rule, or where a teacher did not commit.
        confusion: ``{(a, b): count}`` for a look at *how* they disagree.
    """

    compared: int = 0
    exact: int = 0
    within_one: int = 0
    dropped: int = 0
    confusion: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Rates as well as counts, so nobody has to divide by hand."""
        return {
            "compared": self.compared,
            "exact": self.exact,
            "within_one": self.within_one,
            "dropped": self.dropped,
            "exact_rate": self.exact / self.compared if self.compared else 0.0,
            "within_one_rate": self.within_one / self.compared if self.compared else 0.0,
            "drop_rate": self.dropped / (self.compared + self.dropped)
            if (self.compared + self.dropped)
            else 0.0,
            "confusion": dict(sorted(self.confusion.items())),
        }


def agreement_report(
    first: Mapping[str, int | None], second: Mapping[str, int | None]
) -> AgreementReport:
    """Compare two teachers' labels over the rows they share.

    Args:
        first: ``{row_id: level}`` from teacher 1; ``None`` where it did not commit.
        second: The same from teacher 2.

    Returns:
        An :class:`AgreementReport`.
    """
    report = AgreementReport()
    for row_id in sorted(set(first) & set(second)):
        a, b = first[row_id], second[row_id]
        if a is None or b is None:
            report.dropped += 1
            continue
        report.compared += 1
        report.confusion[f"{a}->{b}"] = report.confusion.get(f"{a}->{b}", 0) + 1
        if a == b:
            report.exact += 1
            report.within_one += 1
        elif abs(a - b) == 1:
            report.within_one += 1
        else:
            report.dropped += 1
    return report


def write_summary(directory: Any, payload: Mapping[str, Any]) -> Any:
    """Write ``summary.json`` for a teacher run.

    Args:
        directory: The run directory.
        payload: Everything measured.

    Returns:
        The path written.
    """
    from pathlib import Path

    path = Path(directory) / "summary.json"
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


#: Questions paired with a state to make an ordinal judgement. Deliberately generic: the rubric
#: carries the ordinal structure, and the question only says what to look for. A question that
#: smuggled in its own scale would make the rubric decorative.
TEACH_QUESTIONS: tuple[str, ...] = (
    "How strongly does this text express frustration or dissatisfaction?",
    "How urgent is the need described in this text?",
    "How specific and actionable is the request in this text?",
    "How confident does the writer sound about what they are saying?",
    "How much domain or technical detail does this text contain?",
)


def build_teach_items(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    seed: int = 20260917,
    min_chars: int = 40,
    prefix: str = "teach",
    exclude_families: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Pair train-eligible states with rubric questions, deterministically.

    ADR 0005 rule (a): the *state* text may come only from train-eligible sources or from our
    own constructed states. Passing the training split in satisfies that by construction — the
    licence gate has already refused anything else — and it is also the only place the states
    are already deduplicated and cleaned.

    Args:
        rows: Training rows, each with ``state``, ``state_hash`` and ``family``.
        limit: How many items to emit.
        seed: Seeds the question assignment and the shuffle.
        min_chars: Skip states too short to grade; a four-word utterance has no ordinal
            structure to find and would only measure the rubric's tie-breaking.
        prefix: Id prefix. A val or test batch must not collide with the training batch's ids,
            because both end up in one corpus and resume matches by id.
        exclude_families: Families to skip. Used to keep the rule-labelled control set out of a
            teacher batch: its labels are exact by construction and a teacher would only add
            noise to them.

    Returns:
        Items with ``id``, ``state``, ``question`` and ``family``, one per distinct state.
    """
    import random

    rng = random.Random(seed)
    seen: set[str] = set()
    pool: list[dict[str, Any]] = []
    skip = set(exclude_families)
    for row in rows:
        state = str(row["state"]).strip()
        digest = str(row.get("state_hash") or state)
        if digest in seen or len(state) < min_chars or row.get("family") in skip:
            continue
        seen.add(digest)
        pool.append({"state": state, "family": row.get("family", "?"), "hash": digest})

    rng.shuffle(pool)
    items = []
    for index, entry in enumerate(pool[:limit]):
        question = TEACH_QUESTIONS[index % len(TEACH_QUESTIONS)]
        items.append(
            {
                "id": f"{prefix}-{index:05d}-{entry['hash'][:8]}",
                "state": entry["state"],
                "question": question,
                "family": entry["family"],
            }
        )
    return items
