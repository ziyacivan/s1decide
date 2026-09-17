"""Case-control sampling of stage-1 evaluation rows, with importance weights.

Evaluating the two-stage path honestly means evaluating it at the ratio it faces: one candidate
in ~96 is the answer. Enumerating that costs ~3.3 hours per run on the 3090, which is the right
price for a number that goes in the model card and the wrong one for iteration.

Case-control sampling pays a thirtieth of it for the same answer. Keep **every positive** — they
are rare and carry most of the information — and a fixed number of negatives per question, then
weight each retained negative by how many it stands for. Every weighted statistic is then an
unbiased estimate of the full-fan-out statistic: accuracy, ECE, Brier, AUROC, the risk-coverage
curve, the base-rate control, and the fitted temperature.

This is standard case-control design, and its one real hazard is forgetting the weights
somewhere. `tests/test_case_control.py` checks the weighted numbers against the full fan-out on
the same population, which is the only way to know nothing was forgotten.

Non-stage-1 rows are never sampled: they are one question each and already cheap.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["DEFAULT_EVAL_NEGATIVES", "EVAL_MODES", "sample_case_control"]

#: The two ways a run may evaluate stage-1 rows.
#:
#: ``full`` enumerates every candidate. Mandatory for anything that reaches the model card:
#: baseline tags, the S1 final run, the quantization table, the fair baselines and the
#: head-to-head. ``case-control`` samples and weights, and is the default for smoke runs and
#: iteration. A run records which it used, because the two are only interchangeable if the
#: weights are right.
EVAL_MODES: tuple[str, ...] = ("case-control", "full")

#: Negatives retained per stage-1 question under case-control sampling. 16 keeps the sampling
#: error on the negative class small while cutting a 3.3-hour evaluation to roughly half an
#: hour; the positives are all kept regardless, and they are the scarce class.
DEFAULT_EVAL_NEGATIVES = 16


def sample_case_control(
    rows: Sequence[Mapping[str, Any]],
    *,
    negatives: int = DEFAULT_EVAL_NEGATIVES,
    seed: int = 20260917,
    parent_key: str = "parent_id",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep every positive and ``negatives`` negatives per stage-1 question, with weights.

    Args:
        rows: Evaluation rows. Stage-1 rows are recognised by ``stage == 1`` and grouped by
            ``parent_key``; everything else passes through with weight 1.
        negatives: Negatives to retain per question. A question with fewer is kept whole, and
            its retained negatives then weigh 1 — nothing was dropped, so nothing stands in.
        seed: Seeds the choice of which negatives to keep.
        parent_key: Field identifying the question a stage-1 row belongs to.

    Returns:
        ``(rows, report)``. Each returned row carries ``eval_weight``. The report records the
        rows kept and dropped and the largest weight, which is the one to look at when a
        weighted estimate seems noisy.

    Raises:
        ValueError: If ``negatives`` is below 1. Zero negatives is not a cheaper evaluation, it
            is a different one: there would be nothing to weight back.
    """
    if negatives < 1:
        raise ValueError(f"negatives must be >= 1, got {negatives}")

    rng = random.Random(seed)
    grouped: dict[Any, list[Mapping[str, Any]]] = {}
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("stage") == 1:
            grouped.setdefault(row.get(parent_key), []).append(row)
        else:
            out.append({**row, "eval_weight": 1.0})

    dropped = 0
    max_weight = 1.0
    for _parent, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        positives = [r for r in group if r["answer_idx"] == 1]
        negative_rows = [r for r in group if r["answer_idx"] == 0]
        keep = min(negatives, len(negative_rows))
        chosen = rng.sample(negative_rows, k=keep) if keep else []
        # Each retained negative stands for `len(negative_rows) / keep` of them. Positives are
        # all kept, so they stand only for themselves.
        weight = (len(negative_rows) / keep) if keep else 1.0
        max_weight = max(max_weight, weight)
        dropped += len(negative_rows) - keep
        out.extend({**r, "eval_weight": 1.0} for r in positives)
        out.extend({**r, "eval_weight": weight} for r in chosen)

    report = {
        "mode": "case-control",
        "negatives_per_question": negatives,
        "questions": len(grouped),
        "rows_kept": len(out),
        "rows_dropped": dropped,
        "max_weight": max_weight,
        "seed": seed,
    }
    return out, report
