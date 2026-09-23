"""Which training rows an S1 run draws, and how often: a row budget under family weights.

S1 is defined by a sample budget, not epochs. Rows are drawn **with replacement** in proportion
to per-row weights built from `data/processed/sampling_weights.json` (`data/build/balance.py`),
so the run sees the effective mix the manifest reports.

One exception, learned from the smoke adapter's `Score` regression (2026-09-23): inside
**uniform groups** — `score` by default — every row weighs the same. The group keeps the total
mass the family weights give it, but it is spread evenly across its rows instead of equalised
across families. Family equalisation inside `score` would give the 726 rule-labelled
`ordinal_control` rows as much mass as the 4,804 teacher rows together, and their near-flat level
distribution is exactly the prior that pulled the smoke adapter upward. Uniform-by-row keeps the
soft/hard split and the level marginal at their natural shares, and a test holds it to that.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "UNIFORM_GROUPS",
    "row_group",
    "row_target",
    "row_weights",
    "sample_indices",
    "score_level_marginal",
]

#: Groups whose rows are weighted uniformly rather than equalised across families.
UNIFORM_GROUPS: tuple[str, ...] = ("score",)


def row_group(row: Mapping[str, Any]) -> str:
    """``stage1``, ``noul``, ``score`` or ``choice`` — the groups `sampling_weights.json` keys."""
    if row.get("stage") == 1:
        return "stage1"
    qtype = row["qtype"]
    return qtype if qtype in {"noul", "score"} else "choice"


def row_weights(
    rows: Sequence[Mapping[str, Any]],
    weights: Mapping[str, float],
    uniform_groups: Sequence[str] = UNIFORM_GROUPS,
) -> list[float]:
    """Per-row sampling weight.

    Args:
        rows: Training rows.
        weights: ``"group/family"`` to per-row weight, from ``sampling_weights.json``.
        uniform_groups: Groups whose mass is spread evenly over their rows.

    Returns:
        One weight per row.

    Raises:
        KeyError: If a row's ``group/family`` has no weight. A missing key used to mean 1.0,
            which would silently mis-weight a family added after the weights were built.
    """
    raw = []
    for row in rows:
        key = f"{row_group(row)}/{row['family']}"
        if key not in weights:
            raise KeyError(f"no sampling weight for {key!r}; rebuild with `uv run task data`")
        raw.append(float(weights[key]))
    out = list(raw)
    for group in uniform_groups:
        index = [i for i, row in enumerate(rows) if row_group(row) == group]
        if index:
            mass = sum(raw[i] for i in index)
            for i in index:
                out[i] = mass / len(index)
    return out


def sample_indices(weights: Sequence[float], budget: int, seed: int) -> list[int]:
    """Draw ``budget`` row indices with replacement, proportionally to ``weights``.

    Deterministic in ``seed``; the draw order is the training order.
    """
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    return random.Random(seed).choices(range(len(weights)), weights=weights, k=budget)


def row_target(row: Mapping[str, Any]) -> list[float]:
    """A row's target distribution: its soft target if it has one, else one-hot on the answer."""
    target = row.get("target")
    if isinstance(target, list) and len(target) == len(row["options"]):
        return [float(v) for v in target]
    onehot = [0.0] * len(row["options"])
    onehot[int(row["answer_idx"])] = 1.0
    return onehot


def score_level_marginal(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[float]]:
    """Mean target distribution over the `Score` rows in ``rows``, per scale size.

    Keyed by the number of levels: `ordinal_control` mixes 3-, 4- and 5-level scales, and a
    marginal is only meaningful within one scale.
    """
    totals: dict[int, list[float]] = {}
    counts: dict[int, int] = {}
    for row in rows:
        if row_group(row) != "score":
            continue
        target = row_target(row)
        levels = len(target)
        acc = totals.setdefault(levels, [0.0] * levels)
        for k, value in enumerate(target):
            acc[k] += value
        counts[levels] = counts.get(levels, 0) + 1
    return {n: [v / counts[n] for v in acc] for n, acc in sorted(totals.items())}
