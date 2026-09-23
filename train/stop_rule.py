"""When an S1 run stops itself, decided from its checkpoint evaluations (owner, 2026-09-24).

- **Stop** if teacher-`Score` BSS *or* accuracy is below its row-0 value at **two consecutive**
  checkpoints. One bad checkpoint is noise often enough that it should not end a night's run.
- **Stop immediately** if any primitive's BSS turns negative after having been positive at any
  earlier evaluation, row 0 included: worse than the training prior after having beaten it.
- **Warn, never stop**, on predicted-vs-target marginal drift — reported in every checkpoint entry
  so it is read, not so it acts.

Pure function of the evaluation history, so the rule is tested without a model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["TEACHER_GROUP", "marginal_gap", "stop_decision"]

#: The genuine, teacher-labelled `Score` rows — the primitive the smoke adapter regressed on.
TEACHER_GROUP = "score/teacher"


def marginal_gap(metrics: Mapping[str, Any]) -> float:
    """Total-variation distance between a group's predicted and target marginals."""
    predicted, target = metrics["predicted_marginal"], metrics["target_marginal"]
    return 0.5 * sum(abs(p - t) for p, t in zip(predicted, target, strict=True))


def _teacher_below_row0(entry: Mapping[str, Any], row0: Mapping[str, Any]) -> list[str]:
    now, base = entry["metrics"].get(TEACHER_GROUP), row0["metrics"].get(TEACHER_GROUP)
    if not now or not base:
        return []
    below = []
    for key in ("bss", "accuracy"):
        if now[key] is not None and base[key] is not None and now[key] < base[key]:
            below.append(f"{key} {now[key]:.4f} < row-0 {base[key]:.4f}")
    return below


def stop_decision(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the stop rule to the evaluations so far, row 0 first.

    Args:
        history: Checkpoint entries ``{"rows", "metrics": {group: {...}}}``, in order.

    Returns:
        ``{"stop": bool, "reasons": [...], "warnings": [...]}``.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    if len(history) < 2:
        return {"stop": False, "reasons": reasons, "warnings": warnings}
    row0, current, previous = history[0], history[-1], history[-2]

    now_below = _teacher_below_row0(current, row0)
    if now_below and len(history) >= 3 and _teacher_below_row0(previous, row0):
        reasons.append(
            f"{TEACHER_GROUP} below row 0 at two consecutive checkpoints "
            f"({previous['rows']} and {current['rows']} rows): {'; '.join(now_below)}"
        )
    elif now_below:
        warnings.append(
            f"{TEACHER_GROUP} below row 0 at {current['rows']} rows (first time): "
            f"{'; '.join(now_below)}"
        )

    for group, metrics in current["metrics"].items():
        bss = metrics.get("bss")
        was_positive = any(
            (past["metrics"].get(group) or {}).get("bss") is not None
            and past["metrics"][group]["bss"] > 0
            for past in history[:-1]
        )
        if bss is not None and bss < 0 and was_positive:
            reasons.append(f"{group} BSS turned negative ({bss:.4f}) after having been positive")

    for group, metrics in current["metrics"].items():
        gap, gap0 = marginal_gap(metrics), None
        if group in row0["metrics"]:
            gap0 = marginal_gap(row0["metrics"][group])
        if gap0 is not None and gap > gap0:
            warnings.append(
                f"{group} marginal drift: predicted-vs-target gap {gap:.3f} (row 0: {gap0:.3f})"
            )
    return {"stop": bool(reasons), "reasons": reasons, "warnings": warnings}
