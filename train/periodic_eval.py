"""The evaluation an S1 run performs at every checkpoint, on a fixed slice of `val`.

Owner's spec (2026-09-23): every 5,000 sampled rows, a case-control `val` evaluation with KL,
accuracy, ECE and BSS **per primitive**, and — because a shifted prior is what broke the smoke
adapter — the **predicted-vs-target marginal** for each: level distribution for `Score`, yes-rate
for `Noul` and stage 1, answer-position distribution for `Choice`.

The slice is fixed for the whole run so checkpoints are comparable: every non-stage-1 `val` row,
plus a seeded set of stage-1 questions under case-control sampling (every positive, a fixed
number of negatives, each weighted by how many it stands for — `eval/case_control.py`). All
statistics are weighted. The base-rate control that BSS is measured against is fitted on the
*training* rows, never on the rows being scored.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = [
    "base_rates",
    "build_eval_rows",
    "eval_group",
    "eval_metrics",
    "predict_logits",
]


def eval_group(row: Mapping[str, Any]) -> str:
    """``choice``, ``noul``, ``noul/stage1``, ``score`` or ``score/teacher`` for reporting."""
    if row.get("stage") == 1 or row.get("stage1"):
        return "noul/stage1"
    if row["qtype"] == "score":
        return "score/teacher" if row.get("family") == "score_teacher" else "score"
    return str(row["qtype"])


def build_eval_rows(
    val_rows: Sequence[Mapping[str, Any]],
    *,
    stage1_questions: int,
    negatives: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The fixed evaluation slice: all non-stage-1 rows, and a case-control stage-1 sample.

    Args:
        val_rows: The `val` split.
        stage1_questions: How many stage-1 questions (parents) to keep, chosen with ``seed``.
        negatives: Negatives per stage-1 question; each carries its importance weight.
        seed: Seeds both the question choice and the negative sampling.

    Returns:
        ``(rows, report)``; every row carries ``eval_weight``.
    """
    from eval.case_control import sample_case_control

    others = [dict(r) for r in val_rows if r.get("stage") != 1]
    stage1 = [r for r in val_rows if r.get("stage") == 1]
    parents = sorted({str(r["parent_id"]) for r in stage1})
    chosen = set(random.Random(seed).sample(parents, k=min(stage1_questions, len(parents))))
    sampled, cc_report = sample_case_control(
        [r for r in stage1 if str(r["parent_id"]) in chosen], negatives=negatives, seed=seed
    )
    rows = [{**r, "eval_weight": 1.0} for r in others] + sampled
    report = {
        "non_stage1_rows": len(others),
        "stage1_questions": len(chosen),
        "stage1_questions_available": len(parents),
        "case_control": cc_report,
        "rows": len(rows),
    }
    return rows, report


def base_rates(train_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    """The base-rate control per ``group/options``: the training target marginal.

    `Choice` options are shuffled per example, so its control is close to uniform by
    construction; `Score` and `Noul` carry real priors.
    """
    from train.sampling import row_target

    totals: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for row in train_rows:
        target = np.asarray(row_target(row), dtype=np.float64)
        key = f"{eval_group(row)}/{len(target)}"
        totals[key] = totals.get(key, np.zeros_like(target)) + target
        counts[key] = counts.get(key, 0) + 1
    return {key: (totals[key] / counts[key]).tolist() for key in sorted(totals)}


def predict_logits(
    model: Any,
    examples: Sequence[dict[str, Any]],
    pad_token_id: int,
    batch_size: int = 8,
) -> list[list[float]]:
    """Masked option logits at the answer position, fp32, in example order.

    Same read as training and inference: left-padded batch, last position, the row's option
    tokens only. Runs under ``no_grad`` with the model in eval mode, and restores train mode.
    """
    import torch

    from train.sft_lora import collate

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    out: list[list[float]] = []
    try:
        with torch.no_grad():
            for start in range(0, len(examples), batch_size):
                chunk = examples[start : start + batch_size]
                batch = collate(chunk, pad_token_id)
                logits = (
                    model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                    )
                    .logits[:, -1, :]
                    .float()
                )
                for i, ids in enumerate(batch["label_token_ids"]):
                    out.append(logits[i, torch.tensor(ids, device=device)].tolist())
    finally:
        if was_training:
            model.train()
    return out


def _softmax(values: Sequence[float]) -> np.ndarray:
    z = np.asarray(values, dtype=np.float64)
    z = np.exp(z - z.max())
    return z / z.sum()


def _weighted_marginal(vectors: list[np.ndarray], weights: np.ndarray) -> list[float]:
    width = max(len(v) for v in vectors)
    padded = np.zeros((len(vectors), width))
    for i, v in enumerate(vectors):
        padded[i, : len(v)] = v
    return (np.average(padded, axis=0, weights=weights)).tolist()


def eval_metrics(
    examples: Sequence[Mapping[str, Any]],
    logits: Sequence[Sequence[float]],
    rates: Mapping[str, Sequence[float]],
    n_bins: int = 15,
) -> dict[str, Any]:
    """Weighted KL, accuracy, ECE, Brier/BSS and marginals, per reporting group.

    Args:
        examples: Rendered eval examples carrying ``target``, ``answer_idx`` and ``eval_weight``.
        logits: Output of :func:`predict_logits`, aligned with ``examples``.
        rates: Output of :func:`base_rates`, keyed ``group/options``.
        n_bins: Equal-mass bins for ECE.

    Returns:
        ``{group: {...}}``. Marginals: for every group the weighted mean of the predicted
        one-hot argmax next to the weighted mean target (positions for `Choice`, levels for
        `Score`, yes/no for `Noul`); `Score` also carries the mean predicted and target level.

    Raises:
        KeyError: If a row has no base-rate control for its ``group/options`` — the control is
            part of the report, and a missing one is not silently replaced by uniform.
    """
    from eval.metrics import brier_multiclass, brier_skill_score, ece

    groups: dict[str, list[int]] = {}
    for i, ex in enumerate(examples):
        groups.setdefault(eval_group(ex), []).append(i)

    report: dict[str, Any] = {}
    for group, index in sorted(groups.items()):
        w = np.asarray([float(examples[i].get("eval_weight", 1.0)) for i in index])
        probs = [_softmax(logits[i]) for i in index]
        targets = [np.asarray(examples[i]["target"], dtype=np.float64) for i in index]
        labels = [int(examples[i]["answer_idx"]) for i in index]
        argmax = [int(np.argmax(p)) for p in probs]
        kl = [
            float(np.sum(t[t > 0] * (np.log(t[t > 0]) - np.log(np.maximum(p[t > 0], 1e-300)))))
            for t, p in zip(targets, probs, strict=True)
        ]
        controls = []
        for i in index:
            key = f"{eval_group(examples[i])}/{len(examples[i]['target'])}"
            if key not in rates:
                raise KeyError(f"no base-rate control for {key!r}")
            controls.append(list(rates[key]))
        brier = brier_multiclass([p.tolist() for p in probs], labels, w)
        reference = brier_multiclass(controls, labels, w)
        onehots = [np.eye(len(p))[a] for p, a in zip(probs, argmax, strict=True)]
        entry: dict[str, Any] = {
            "rows": len(index),
            "weight": float(w.sum()),
            "accuracy": float(
                np.average([a == y for a, y in zip(argmax, labels, strict=True)], weights=w)
            ),
            "kl": float(np.average(kl, weights=w)),
            "ece": ece([p.tolist() for p in probs], labels, n_bins, w),
            "brier": brier,
            "bss": brier_skill_score(brier, reference),
            "predicted_marginal": _weighted_marginal(onehots, w),
            "target_marginal": _weighted_marginal(targets, w),
        }
        if group.startswith("score"):
            levels = [np.arange(len(t)) for t in targets]
            entry["mean_predicted_level"] = float(np.average(argmax, weights=w))
            entry["mean_target_level"] = float(
                np.average(
                    [float(np.dot(k, t)) for k, t in zip(levels, targets, strict=True)], weights=w
                )
            )
        report[group] = entry
    return report
