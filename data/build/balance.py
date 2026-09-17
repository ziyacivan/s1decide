"""Class balance and family weighting for the training mix (Phase 1 Step 2d).

Two problems, one visible and one not.

**The visible one.** Expanding a 60- or 151-option question into stage-1 rows emits one positive
and one negative per remaining candidate. At full fan-out that is a 96:1 no:yes ratio, and a
model trained on it learns to answer "no". Training therefore keeps the positive plus ``k``
negatives per question; evaluation does not, because the 96:1 ratio *is* the deployed task and
an evaluation that hides it would report a calibration the deployment does not have.

**The invisible one.** Row counts are not what the model sees. A corpus that is 64% stage-1 rows
trains a stage-1 classifier with a `Noul` hobby, whatever the source counts say. So the mix is
steered by per-family sampling weights and reported as an *effective* mix — the composition the
trainer actually draws — separately from the raw counts. The two differ by design, and the
floors in `pipeline.py` apply to the effective one.

Weights are capped: a primitive with 727 rows cannot be weighted to 10% of a 60k-row mix
without showing each of those rows ten times an epoch, which trains memorisation rather than
the primitive. When a target cannot be met inside the cap, the achieved share is reported
alongside the target rather than the target being quietly restated as a result.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

__all__ = [
    "DEFAULT_TARGET_MIX",
    "MAX_OVERSAMPLE",
    "STAGE1_TRAIN_NEGATIVES",
    "effective_mix",
    "family_weights",
    "mix_table",
    "rank_negatives",
    "row_group",
    "subsample_stage1",
]

#: Negatives kept per stage-1 question in **train only**. 6 gives a 6:1 no:yes ratio, inside the
#: 8:1 ceiling the plan sets, while still showing the model more wrong candidates than right
#: ones — which is the truth about the task, just not 96 times over.
STAGE1_TRAIN_NEGATIVES = 6

#: The four things a training row can be. `stage1` is split out from `noul` because they are
#: different tasks that share a shape: one asks whether a candidate is the answer, the other
#: whether a statement about the state is true.
GROUPS: tuple[str, ...] = ("stage1", "noul", "choice", "score")

#: Target effective mix. Not row counts — the composition the trainer draws.
#: `stage1` stays the largest single group because the two-stage path is how every
#: high-cardinality question is answered, but it no longer has a majority.
DEFAULT_TARGET_MIX: Mapping[str, float] = {
    "stage1": 0.35,
    "noul": 0.25,
    "choice": 0.30,
    "score": 0.10,
}

#: How far a group may be oversampled beyond its natural share. `Score` has 727 rows against a
#: 10% target of a ~65k mix, which would need ~9x; past about 3x a small set is being memorised
#: rather than learned, and the honest response is to report the shortfall and go get more data
#: (Step 2e's teacher run) rather than to weight harder.
MAX_OVERSAMPLE = 3.0


def row_group(row: Mapping[str, Any]) -> str:
    """Which of :data:`GROUPS` a row belongs to.

    Stage-2 rows are ``choice``: a shortlist Choice is an ordinary Choice, just a narrower one.
    """
    if row.get("stage") == 1:
        return "stage1"
    qtype = row["qtype"]
    return qtype if qtype in GROUPS else "choice"


def rank_negatives(
    candidates: Sequence[int],
    scores: Mapping[int, float] | None,
    rng: random.Random,
) -> list[int]:
    """Order candidate negatives hardest-first.

    "Hard" means the model was most inclined to say yes. With zero-shot ``P(yes)`` per candidate
    the order is by descending score; without it the order is random, which is a weaker training
    signal but an unbiased one.

    Args:
        candidates: Option indices that are *not* the answer.
        scores: Optional ``{option_index: P(yes)}`` from a scoring pass.
        rng: Seeded source of randomness, used when ``scores`` is absent or partial.

    Returns:
        The candidates, hardest first. Candidates missing from ``scores`` sort last, in random
        order, rather than being dropped or treated as score 0.
    """
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    if not scores:
        return shuffled
    # Stable sort over the already-shuffled list: scored candidates lead in score order,
    # unscored ones keep their random order behind them.
    return sorted(shuffled, key=lambda index: -scores.get(index, float("-inf")))


def subsample_stage1(
    rows: Sequence[Mapping[str, Any]],
    *,
    negatives: int = STAGE1_TRAIN_NEGATIVES,
    splits: Iterable[str] = ("train",),
    scores: Mapping[str, Mapping[int, float]] | None = None,
    seed: int = 20260917,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep the positive plus ``negatives`` hard negatives per stage-1 question, in ``splits``.

    Rows in other splits pass through untouched: `val` and `test` keep the full fan-out, because
    the ratio they carry is the ratio the deployed two-stage path faces, and fitting a
    temperature on a 6:1 sample to deploy it at 96:1 would be fitting the wrong distribution.

    Args:
        rows: Every assigned row, already carrying ``split``, ``stage`` and ``parent_id``.
        negatives: Negatives kept per question.
        splits: Which splits to subsample.
        scores: Optional ``{parent_id: {option_index: P(yes)}}`` for hard-negative selection.
        seed: Seeds the fallback ordering.

    Returns:
        ``(rows, report)``. The report records how many rows were dropped and whether the
        negatives were chosen by score or at random — a training run should be able to say which.
    """
    if negatives < 0:
        raise ValueError(f"negatives must be >= 0, got {negatives}")
    targets = set(splits)
    rng = random.Random(seed)

    by_parent: dict[str, list[Mapping[str, Any]]] = {}
    passthrough: list[Mapping[str, Any]] = []
    for row in rows:
        if row.get("stage") == 1 and row["split"] in targets:
            by_parent.setdefault(str(row["parent_id"]), []).append(row)
        else:
            passthrough.append(row)

    kept: list[dict[str, Any]] = []
    dropped = 0
    scored_parents = 0
    for parent, group in sorted(by_parent.items()):
        positives = [r for r in group if r["answer_idx"] == 1]
        negative_rows = [r for r in group if r["answer_idx"] == 0]
        parent_scores = (scores or {}).get(parent)
        if parent_scores:
            scored_parents += 1
        order = rank_negatives(list(range(len(negative_rows))), parent_scores, rng)
        chosen = [negative_rows[i] for i in order[:negatives]]
        dropped += len(negative_rows) - len(chosen)
        kept.extend(dict(r) for r in positives)
        kept.extend(dict(r) for r in chosen)

    report = {
        "negatives_per_question": negatives,
        "splits": sorted(targets),
        "questions": len(by_parent),
        "rows_dropped": dropped,
        "hard_negative_source": "zero_shot_scores" if scored_parents else "random",
        "questions_with_scores": scored_parents,
    }
    return [dict(r) for r in passthrough] + kept, report


def family_weights(
    rows: Sequence[Mapping[str, Any]],
    *,
    targets: Mapping[str, float] = DEFAULT_TARGET_MIX,
    max_oversample: float = MAX_OVERSAMPLE,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Per-``(group, family)`` sampling weights that steer the mix toward ``targets``.

    Within a group every family is weighted to contribute equally, so a group is not carried by
    whichever source happens to be largest. Between groups, weights aim at ``targets`` but are
    clipped at ``max_oversample`` times a group's natural share.

    Args:
        rows: Training rows only.
        targets: Desired share per group. Need not sum to 1; it is normalised.
        max_oversample: Cap on a group's weight relative to its natural share.

    Returns:
        ``(weights, report)``. ``weights`` is keyed ``"group/family"``. The report names any
        group whose target was clipped, with the share actually achieved.
    """
    if not rows:
        return {}, {"groups": {}, "clipped": []}

    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row_group(row), row["family"])
        counts[key] = counts.get(key, 0) + 1
    group_totals: dict[str, int] = {}
    for (group, _family), n in counts.items():
        group_totals[group] = group_totals.get(group, 0) + n

    total = sum(group_totals.values())
    present = {g: t for g, t in targets.items() if group_totals.get(g)}
    scale = sum(present.values()) or 1.0

    weights: dict[str, float] = {}
    clipped: list[dict[str, Any]] = []
    group_report: dict[str, Any] = {}
    for group, target in present.items():
        natural = group_totals[group] / total
        wanted = target / scale
        factor = wanted / natural
        if factor > max_oversample:
            clipped.append(
                {
                    "group": group,
                    "target_share": wanted,
                    "natural_share": natural,
                    "needed_oversample": factor,
                    "capped_at": max_oversample,
                }
            )
            factor = max_oversample
        group_report[group] = {
            "rows": group_totals[group],
            "natural_share": natural,
            "target_share": wanted,
            "oversample": factor,
        }
        families = [f for (g, f) in counts if g == group]
        for family in families:
            # Equalise families inside the group: a family with half the rows gets twice the
            # per-row weight, so neither carries the group on its own.
            per_family_rows = counts[(group, family)]
            weights[f"{group}/{family}"] = factor * (
                group_totals[group] / (len(families) * per_family_rows)
            )

    return weights, {"groups": group_report, "clipped": clipped, "max_oversample": max_oversample}


def effective_mix(
    rows: Sequence[Mapping[str, Any]], weights: Mapping[str, float]
) -> dict[str, dict[str, float]]:
    """The composition a weighted sampler actually draws.

    Args:
        rows: Training rows.
        weights: Output of :func:`family_weights`. A missing key weights 1.0, so an unweighted
            run reports its raw mix rather than nothing.

    Returns:
        ``{group: {"rows", "raw_share", "effective_share"}}``.
    """
    mass: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        group = row_group(row)
        key = f"{group}/{row['family']}"
        mass[group] = mass.get(group, 0.0) + weights.get(key, 1.0)
        counts[group] = counts.get(group, 0) + 1
    total_mass = sum(mass.values()) or 1.0
    total_rows = sum(counts.values()) or 1
    return {
        group: {
            "rows": counts[group],
            "raw_share": counts[group] / total_rows,
            "effective_share": mass[group] / total_mass,
        }
        for group in sorted(counts)
    }


def mix_table(
    rows: Sequence[Mapping[str, Any]],
    weights: Mapping[str, float],
    *,
    raw_counts: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """One row per ``(primitive, stage, family)`` with counts before and after each step.

    Args:
        rows: Training rows after subsampling.
        weights: Output of :func:`family_weights`.
        raw_counts: Optional ``{"group/family": rows}`` from before subsampling, so the table
            can show what was dropped rather than only what survived.

    Returns:
        Rows sorted by descending effective share, each carrying ``primitive``, ``stage``,
        ``family``, ``rows_before``, ``rows_after``, ``raw_share`` and ``effective_share``.
    """
    counts: dict[tuple[str, str, str], int] = {}
    for row in rows:
        group = row_group(row)
        primitive = "noul" if group == "stage1" else group
        stage = "stage-1" if group == "stage1" else "genuine"
        counts[(primitive, stage, row["family"])] = (
            counts.get((primitive, stage, row["family"]), 0) + 1
        )

    total_rows = sum(counts.values()) or 1
    mass = {
        key: n * weights.get(f"{'stage1' if key[1] == 'stage-1' else key[0]}/{key[2]}", 1.0)
        for key, n in counts.items()
    }
    total_mass = sum(mass.values()) or 1.0

    table = [
        {
            "primitive": primitive,
            "stage": stage,
            "family": family,
            "rows_before": (raw_counts or {}).get(
                f"{'stage1' if stage == 'stage-1' else primitive}/{family}", n
            ),
            "rows_after": n,
            "raw_share": n / total_rows,
            "effective_share": mass[(primitive, stage, family)] / total_mass,
        }
        for (primitive, stage, family), n in counts.items()
    ]
    return sorted(table, key=lambda r: -r["effective_share"])
