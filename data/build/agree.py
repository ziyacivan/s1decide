"""Fold two teachers' labels into `Score` rows, keeping only what they agree on.

Two teachers label the same states independently. Where they agree exactly the label is as good
as this pipeline can make it; where they differ by one level on a five-level ordinal scale the
row is still informative, because adjacent levels are where an ordinal rubric is genuinely
ambiguous; where they differ by two or more, one of them is wrong and we cannot tell which, so
the row is dropped rather than guessed.

**Nothing is discarded from a kept row.** Both teachers' levels travel with it. The rule for
turning a one-level disagreement into a single ``answer_idx`` is a real design choice — collapse
to the lower level, to teacher 1's, or train against a soft target over both — and it is not
settled here. Writing both levels means that choice can be made, and remade, without spending
another day of GPU time re-labelling. ``answer_idx`` is filled in with a documented default so
the corpus is usable meanwhile.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data.build.teach import RUBRIC

__all__ = [
    "FoldReport",
    "agreement_stats",
    "build_score_rows",
    "fold_teachers",
    "load_levels",
    "main",
    "rubric_options",
    "source_index",
]

#: Levels on the rubric, and therefore the number of options a `Score` row offers.
LEVELS = 5


def agreement_stats(pairs: Sequence[tuple[int, int]], levels: int = LEVELS) -> dict[str, Any]:
    """How much two teachers agree, corrected for the agreement chance would produce anyway.

    Raw agreement flatters a skewed scale: two teachers who both answer "none" half the time
    agree a quarter of the time while knowing nothing. Cohen's κ subtracts that. The quadratic
    weighting is the one that belongs to an ordinal scale, where 3-vs-4 is a near miss and 0-vs-4
    is not — both are returned, because reporting only the weighted figure would be the
    flattering choice.

    Args:
        pairs: ``(first, second)`` levels for rows where both teachers committed.
        levels: Size of the scale.

    Returns:
        Observed and chance agreement, both κ, and each teacher's marginal distribution and mean.
        Zeroed when there is nothing to compare, rather than dividing by zero.
    """
    n = len(pairs)
    if n == 0:
        return {"n": 0, "exact": 0.0, "chance": 0.0, "kappa": 0.0, "weighted_kappa": 0.0}
    observed = sum(1 for x, y in pairs if x == y) / n
    first = [sum(1 for x, _ in pairs if x == k) / n for k in range(levels)]
    second = [sum(1 for _, y in pairs if y == k) / n for k in range(levels)]
    chance = sum(first[k] * second[k] for k in range(levels))
    disagreement = sum((x - y) ** 2 for x, y in pairs) / n
    expected = sum(
        first[i] * second[j] * (i - j) ** 2 for i in range(levels) for j in range(levels)
    )
    return {
        "n": n,
        "exact": observed,
        "chance": chance,
        "kappa": (observed - chance) / (1 - chance) if chance < 1 else 0.0,
        "weighted_kappa": 1 - disagreement / expected if expected else 0.0,
        "first_marginal": first,
        "second_marginal": second,
        "first_mean": sum(x for x, _ in pairs) / n,
        "second_mean": sum(y for _, y in pairs) / n,
    }


def rubric_options() -> list[str]:
    """The five ordinal levels as option strings, in order.

    The rubric lines read ``3 - moderate: relevant and explicit, but partial or hedged``; the
    option is the name alone. The numeric prefix is the teacher's answer token, not a label the
    student should learn to emit.

    Returns:
        The five level names, lowest first.
    """
    options = []
    for line in RUBRIC:
        _, _, rest = line.partition(" - ")
        options.append(rest.split(":", 1)[0].strip())
    return options


def load_levels(directory: Path) -> dict[str, int | None]:
    """Read one teacher run's committed levels.

    Args:
        directory: A run directory holding ``rows.jsonl``.

    Returns:
        Row id to level, with ``None`` where the teacher produced no parseable answer.
    """
    path = Path(directory) / "rows.jsonl"
    if not path.is_file():
        return {}
    levels: dict[str, int | None] = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # A half-written final line; the same tolerance resume has.
        levels[str(row["id"])] = row.get("level")
    return levels


def source_index(train_rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[str, str]]:
    """Map state text to the source and licence it came in under.

    The teacher items carry the state but not its provenance, and a `Score` row without a licence
    cannot pass the ADR 0004 gate. States were deduplicated upstream, so a state maps to one
    source; where it somehow maps to several, the first in file order wins, which is
    deterministic rather than arbitrary.

    Args:
        train_rows: Rows of the training split.

    Returns:
        State text to ``(source, licence)``.
    """
    index: dict[str, tuple[str, str]] = {}
    for row in train_rows:
        state = str(row.get("state", "")).strip()
        if state and state not in index:
            index[state] = (str(row.get("source", "?")), str(row.get("license", "?")))
    return index


@dataclass
class FoldReport:
    """What happened to every row the two teachers were both given.

    The counts separate the two ways a row can be lost, because they are different failures with
    different fixes. ``no_commit`` is a teacher that produced no parseable level — a truncated
    trace, a generation cap set too low. ``far`` is two teachers that answered confidently and
    disagreed by two levels or more, which is a rubric or a model problem, not a plumbing one.
    The existing `AgreementReport` counts both as one number, which is why this one does not.

    Attributes:
        compared: Rows where both teachers committed.
        exact: Kept, both levels identical.
        within_one: Kept, levels one apart.
        far: Dropped, levels two or more apart.
        no_commit: Dropped, at least one teacher did not commit.
        missing: Ids that only one of the two teachers ever saw.
        confusion: Level pair to count, over compared rows.
        unlicensed: Kept rows whose state had no source in the training split.
    """

    compared: int = 0
    exact: int = 0
    within_one: int = 0
    far: int = 0
    no_commit: int = 0
    missing: int = 0
    confusion: dict[str, int] = field(default_factory=dict)
    unlicensed: int = 0

    @property
    def kept(self) -> int:
        """Rows that survive into the corpus."""
        return self.exact + self.within_one

    def to_json(self) -> dict[str, Any]:
        """Counts and the rates derived from them, so nobody divides by hand."""
        compared = self.compared or 1
        return {
            "compared": self.compared,
            "kept": self.kept,
            "exact": self.exact,
            "within_one": self.within_one,
            "far": self.far,
            "no_commit": self.no_commit,
            "missing": self.missing,
            "unlicensed": self.unlicensed,
            "exact_rate": self.exact / compared,
            "kept_rate": self.kept / compared,
            "drop_rate": (self.far + self.no_commit) / compared,
            "confusion": dict(sorted(self.confusion.items())),
        }


def fold_teachers(
    first: Mapping[str, int | None], second: Mapping[str, int | None]
) -> tuple[dict[str, dict[str, Any]], FoldReport]:
    """Decide which rows survive, and on what evidence.

    Args:
        first: Row id to level from teacher 1.
        second: The same from teacher 2.

    Returns:
        A mapping of kept row id to the verdict for that row, and the report. Only kept rows
        appear in the mapping.
    """
    report = FoldReport()
    kept: dict[str, dict[str, Any]] = {}
    both = set(first) & set(second)
    report.missing = len(set(first) ^ set(second))
    for row_id in sorted(both):
        a, b = first[row_id], second[row_id]
        if a is None or b is None:
            report.no_commit += 1
            continue
        report.compared += 1
        report.confusion[f"{a}->{b}"] = report.confusion.get(f"{a}->{b}", 0) + 1
        distance = abs(a - b)
        if distance >= 2:
            report.far += 1
            continue
        if distance == 0:
            report.exact += 1
            agreement = "exact"
        else:
            report.within_one += 1
            agreement = "within_one"
        # A one-level disagreement is resolved to teacher 1's level. This is a default, not a
        # finding, and it is reversible because both levels are written onto the row — but it is
        # not arbitrary either. Measured over all 4,804 kept rows, the four candidate rules give
        # "none" shares of: teacher 1 41.0%, minimum 51.3%, maximum 38.3%, teacher 2 48.6%. The
        # minimum produces a corpus over half "none"; the maximum has the flattest bottom but
        # imports teacher 2's polarisation, hollowing level 3 to reach it. Teacher 1's level
        # keeps the shape closest to the better-spread teacher, which is what the rubric asks
        # for. See docs/research/teacher-agreement-2026-09-22.md.
        #
        # An earlier version of this comment justified the rule by teacher 2 sitting 0.24 levels
        # low. That was a 100-row artifact: at 6,000 rows the gap is 0.08, and on the adjacent
        # disagreements this branch actually decides, teacher 2 is the *higher* one more often
        # (1,010 against 795). The rule survives; that reason for it did not.
        #
        # `parse_level` already returns a 0-based index, so this is an `answer_idx` and needs no
        # arithmetic. Subtracting one here produced `answer_idx: -1` on every level-0 row, and
        # nothing downstream would have rejected it — the last option, silently.
        kept[row_id] = {
            "first": a,
            "second": b,
            "agreement": agreement,
            "answer_idx": a,
        }
    return kept, report


def build_score_rows(
    items: Sequence[Mapping[str, Any]],
    kept: Mapping[str, Mapping[str, Any]],
    sources: Mapping[str, tuple[str, str]],
    report: FoldReport,
    split: str = "train",
) -> list[dict[str, Any]]:
    """Turn surviving labels into dataset rows in the project's schema.

    Args:
        items: The teacher items, carrying id, state, question and family.
        kept: Output of :func:`fold_teachers`.
        sources: Output of :func:`source_index`.
        report: Updated in place with the count of rows that had no licence.
        split: Which split these rows belong to.

    Returns:
        Rows ready for ``data/processed/``, in item order.
    """
    options = rubric_options()
    rows = []
    for item in items:
        row_id = str(item["id"])
        verdict = kept.get(row_id)
        if verdict is None:
            continue
        state = str(item["state"]).strip()
        source, licence = sources.get(state, ("?", "?"))
        if source == "?":
            report.unlicensed += 1
        rows.append(
            {
                "id": f"score-{row_id}",
                "family": item.get("family", "?"),
                "state": state,
                "qtype": "score",
                "instructions": item["question"],
                "options": list(options),
                "answer_idx": int(verdict["answer_idx"]),
                "split": split,
                "source": source,
                "license": licence,
                "stage": 0,
                "parent_id": row_id,
                "teacher_levels": {
                    "first": verdict["first"],
                    "second": verdict["second"],
                },
                "agreement": verdict["agreement"],
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Command line for ``uv run task teach-fold``.

    Args:
        argv: Arguments; ``None`` reads ``sys.argv``.

    Returns:
        Process exit code. Non-zero when a run directory is missing or the two runs share no
        rows, which is what passing the wrong pair of run ids looks like.
    """
    import argparse

    from s1decide.tasks import repo_root

    parser = argparse.ArgumentParser(prog="task teach-fold")
    parser.add_argument("--first", required=True, help="teacher 1 run id")
    parser.add_argument("--second", required=True, help="teacher 2 run id")
    parser.add_argument("--items", default="data/processed/teach_items.jsonl")
    parser.add_argument("--out", default="data/processed/score_teacher.jsonl")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--dry-run", action="store_true", help="report only; write neither rows nor report"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = repo_root()
    first_dir, second_dir = root / "results" / args.first, root / "results" / args.second
    for directory in (first_dir, second_dir):
        if not (directory / "rows.jsonl").is_file():
            print(f"no rows.jsonl in {directory}", file=sys.stderr)
            return 1

    first, second = load_levels(first_dir), load_levels(second_dir)
    kept, report = fold_teachers(first, second)
    if report.compared == 0:
        print(
            f"{args.first} and {args.second} share no rows where both teachers committed",
            file=sys.stderr,
        )
        return 1

    items = [
        json.loads(line)
        for line in (root / args.items).read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    train = [
        json.loads(line)
        for line in (root / "data/processed/train.jsonl").read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    rows = build_score_rows(items, kept, source_index(train), report, split=args.split)

    pairs = [
        (first[i], second[i])
        for i in sorted(set(first) & set(second))
        if first[i] is not None and second[i] is not None
    ]
    payload = {
        "first": args.first,
        "second": args.second,
        "fold": report.to_json(),
        "agreement": agreement_stats(pairs),
        "rows_written": len(rows),
        "label_distribution": {
            str(level): sum(1 for row in rows if row["answer_idx"] == level)
            for level in range(LEVELS)
        },
    }
    print(json.dumps(payload, indent=2))

    if args.dry_run:
        return 0

    out = root / args.out
    out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    report_dir = root / "results" / f"{args.first}-fold"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "fold.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"\nwrote {len(rows):,} rows to {args.out} and the report to {report_dir.name}/fold.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
