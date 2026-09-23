"""Before/after on genuine `Score` rows: the base model and one adapter, on the same rows.

The smoke runs showed `Score` KL rising across a one-pass run, but from first-10 and last-10
windows over *different* rows, which cannot separate "worse at `Score`" from "harder late rows".
This scores one fixed set — the teacher-labelled `Score` rows of `val` (teacher run 2) — twice,
in one process: once on the base model, then with the adapter injected into the same weights.
Same rows, same prompts (`s1decide.prompt.render`, as the trainer uses), same masked-logit read
as `decide()`.

Reported side by side, raw (no temperature: this compares the two models, not a deployment):

- accuracy against ``answer_idx`` (teacher 1's level, the documented default), and for soft rows
  whether the argmax lands on either teacher's level;
- quadratic weighted kappa and MAE of the argmax, and MAE of the expected level;
- KL(target ‖ pred) with the soft targets — zero at the optimum on every row;
- ECE (15 equal-mass bins, top label) and multiclass Brier with its skill against a base-rate
  control fitted on the *training* `Score` rows, never on the rows being scored.

Per-row deltas (adapter minus base) are written in full and summarised.

    uv run python -m eval.score_ab --adapter results/smoke-27b-attempt3/adapter \\
        --out results/score-ab-<date>
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from eval.metrics import DEFAULT_BINS, brier_multiclass, brier_skill_score, ece

__all__ = [
    "SCORE_FAMILY",
    "load_score_rows",
    "main",
    "quadratic_weighted_kappa",
    "row_deltas",
    "score_metrics",
]

#: The teacher-labelled `Score` family; its val rows are the genuine evaluation set.
SCORE_FAMILY = "score_teacher"


def load_score_rows(path: Path) -> list[dict[str, Any]]:
    """Teacher-labelled `Score` rows of one split, in file order.

    Args:
        path: A processed split, e.g. ``data/processed/val.jsonl``.

    Returns:
        Rows of :data:`SCORE_FAMILY` with ``qtype == "score"``.
    """
    rows = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("family") == SCORE_FAMILY and row.get("qtype") == "score":
            rows.append(row)
    return rows


def target_of(row: dict[str, Any]) -> list[float]:
    """The row's target distribution: its soft target if it has one, else one-hot on the answer."""
    target = row.get("target")
    if isinstance(target, list) and len(target) == len(row["options"]):
        return [float(v) for v in target]
    onehot = [0.0] * len(row["options"])
    onehot[int(row["answer_idx"])] = 1.0
    return onehot


def softmax(logits: Sequence[float]) -> np.ndarray:
    """Numerically stable softmax in float64."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def quadratic_weighted_kappa(predicted: Sequence[int], actual: Sequence[int], levels: int) -> float:
    """Cohen's kappa with quadratic weights over an ordinal scale.

    Args:
        predicted: Predicted levels.
        actual: Reference levels.
        levels: Size of the scale.

    Returns:
        1 for perfect agreement, 0 for chance, negative for worse than chance. ``nan`` when the
        expected disagreement is zero (a single level on both sides).
    """
    observed = np.zeros((levels, levels))
    for p, a in zip(predicted, actual, strict=True):
        observed[a, p] += 1
    n = observed.sum()
    weights = np.array([[(i - j) ** 2 for j in range(levels)] for i in range(levels)], float)
    weights /= (levels - 1) ** 2
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
    denominator = float((weights * expected).sum())
    if denominator == 0:
        return float("nan")
    return float(1.0 - (weights * observed).sum() / denominator)


def _kl(target: Sequence[float], probabilities: np.ndarray) -> float:
    return float(
        sum(
            t * (math.log(t) - math.log(max(p, 1e-300)))
            for t, p in zip(target, probabilities)
            if t > 0
        )
    )


def score_metrics(
    rows: Sequence[dict[str, Any]],
    logits: Sequence[Sequence[float]],
    base_rate: Sequence[float],
    n_bins: int = DEFAULT_BINS,
) -> dict[str, Any]:
    """Every metric the before/after table reports, for one condition.

    Args:
        rows: The scored rows.
        logits: Masked option logits per row, in level order.
        base_rate: The control's level distribution, fitted on training rows.
        n_bins: Equal-mass bins for ECE.

    Returns:
        A flat dict of metrics plus ``per_row`` (argmax, expected level, KL, correct).
    """
    probabilities = [softmax(row_logits) for row_logits in logits]
    labels = [int(row["answer_idx"]) for row in rows]
    levels = len(rows[0]["options"])
    argmax = [int(np.argmax(p)) for p in probabilities]
    expected = [float(np.dot(np.arange(levels), p)) for p in probabilities]
    targets = [target_of(row) for row in rows]
    target_means = [float(np.dot(np.arange(levels), t)) for t in targets]
    kls = [_kl(t, p) for t, p in zip(targets, probabilities, strict=True)]
    soft = [row.get("target_type") == "soft" for row in rows]

    soft_hits = [
        a in {i for i, t in enumerate(target) if t > 0}
        for a, target, is_soft in zip(argmax, targets, soft, strict=True)
        if is_soft
    ]
    brier = brier_multiclass([p.tolist() for p in probabilities], labels)
    reference = brier_multiclass([list(base_rate)] * len(rows), labels)
    return {
        "rows": len(rows),
        "soft_rows": sum(soft),
        "accuracy": float(np.mean([a == y for a, y in zip(argmax, labels, strict=True)])),
        "accuracy_hard_rows": float(
            np.mean([a == y for a, y, s in zip(argmax, labels, soft, strict=True) if not s])
        ),
        "soft_rows_argmax_on_a_teacher_level": float(np.mean(soft_hits)) if soft_hits else None,
        "qwk": quadratic_weighted_kappa(argmax, labels, levels),
        "mae_argmax": float(np.mean([abs(a - y) for a, y in zip(argmax, labels, strict=True)])),
        "mae_expected_level": float(
            np.mean([abs(e - m) for e, m in zip(expected, target_means, strict=True)])
        ),
        "kl": float(np.mean(kls)),
        "kl_median": float(np.median(kls)),
        "ece": ece([p.tolist() for p in probabilities], labels, n_bins),
        "brier": brier,
        "brier_base_rate": reference,
        "bss": brier_skill_score(brier, reference),
        "mean_confidence": float(np.mean([p.max() for p in probabilities])),
        "predicted_level_distribution": [
            float(np.mean([a == k for a in argmax])) for k in range(levels)
        ],
        "per_row": [
            {"argmax": a, "expected_level": e, "kl": k, "correct": a == y}
            for a, e, k, y in zip(argmax, expected, kls, labels, strict=True)
        ],
    }


def row_deltas(
    rows: Sequence[dict[str, Any]], before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    """Per-row change from ``before`` to ``after``, and its summary.

    Returns:
        ``rows`` (id, target type, KL before/after/delta, correct before/after) and ``summary``:
        mean/median/p10/p90 KL delta, share of rows whose KL fell, and correctness flips.
    """
    per_row = []
    for row, b, a in zip(rows, before["per_row"], after["per_row"], strict=True):
        per_row.append(
            {
                "id": row["id"],
                "target_type": row.get("target_type", "hard"),
                "answer_idx": int(row["answer_idx"]),
                "kl_before": b["kl"],
                "kl_after": a["kl"],
                "kl_delta": a["kl"] - b["kl"],
                "argmax_before": b["argmax"],
                "argmax_after": a["argmax"],
                "correct_before": b["correct"],
                "correct_after": a["correct"],
            }
        )
    deltas = np.array([r["kl_delta"] for r in per_row])

    def summary_for(subset: list[dict[str, Any]]) -> dict[str, Any]:
        if not subset:
            return {"rows": 0}
        d = np.array([r["kl_delta"] for r in subset])
        return {
            "rows": len(subset),
            "kl_delta_mean": float(d.mean()),
            "kl_delta_median": float(np.median(d)),
            "kl_delta_p10": float(np.percentile(d, 10)),
            "kl_delta_p90": float(np.percentile(d, 90)),
            "share_kl_improved": float(np.mean(d < 0)),
            "right_to_wrong": sum(
                1 for r in subset if r["correct_before"] and not r["correct_after"]
            ),
            "wrong_to_right": sum(
                1 for r in subset if not r["correct_before"] and r["correct_after"]
            ),
        }

    by_answer = {
        str(level): summary_for([r for r in per_row if r["answer_idx"] == level])
        for level in sorted({r["answer_idx"] for r in per_row})
    }
    return {
        "summary": {
            "all": summary_for(per_row),
            "hard": summary_for([r for r in per_row if r["target_type"] != "soft"]),
            "soft": summary_for([r for r in per_row if r["target_type"] == "soft"]),
            "by_answer_level": by_answer,
        },
        "kl_delta_mean": float(deltas.mean()),
        "rows": per_row,
    }


def _training_base_rate(root: Path, levels: int) -> list[float]:
    counts = np.zeros(levels)
    for row in load_score_rows(root / "data/processed/train.jsonl"):
        counts += np.asarray(target_of(row))
    return (counts / counts.sum()).tolist()


def _score_all(engine: Any, rows: Sequence[dict[str, Any]]) -> list[list[float]]:
    from s1decide.primitives import Question, Score
    from s1decide.prompt import render

    out = []
    for row in rows:
        spec = Score(instructions=row["instructions"], levels=tuple(row["options"]))
        rendered = render(row["state"], [Question(name="q", spec=spec)])
        result = engine.score(rendered.prefix, list(rendered.suffixes), list(rendered.labels))
        out.append([float(v) for v in result.logits[0]])
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Score the val `Score` rows on the base model, then with ``--adapter``; write the report."""
    import torch

    from s1decide.engine.hf import HFEngine
    from s1decide.kernels import kernel_report
    from s1decide.tasks import repo_root

    parser = argparse.ArgumentParser(prog="eval.score_ab")
    parser.add_argument("--model", default="unsloth/Qwen3.8-27B-unsloth-bnb-4bit")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = repo_root()
    rows = load_score_rows(root / f"data/processed/{args.split}.jsonl")
    if not rows:
        raise SystemExit(f"no {SCORE_FAMILY} rows in {args.split}")
    levels = len(rows[0]["options"])
    base_rate = _training_base_rate(root, levels)

    engine = HFEngine.from_pretrained(args.model, local_files_only=True)
    before_logits = _score_all(engine, rows)

    from peft import PeftModel
    from train.sft_lora import adapted_module_names, check_lora_layout

    # Injected in place: the engine keeps the same causal-LM object it reads `.model` from.
    PeftModel.from_pretrained(engine.model, str(root / args.adapter), is_trainable=False)
    engine.model.eval()
    layout = check_lora_layout(adapted_module_names(engine.model), engine.model.config)
    after_logits = _score_all(engine, rows)

    before = score_metrics(rows, before_logits, base_rate)
    after = score_metrics(rows, after_logits, base_rate)
    deltas = row_deltas(rows, before, after)

    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    headline_keys = [k for k in before if k not in {"per_row"}]
    payload = {
        "model": args.model,
        "adapter": args.adapter,
        "split": args.split,
        "rows": len(rows),
        "calibration": "none — raw softmax of the masked option logits, both conditions",
        "base_rate_control": {"fitted_on": "train score_teacher rows", "distribution": base_rate},
        "adapter_layout": layout,
        "kernels": kernel_report(),
        "logit_precision": "fp32",
        "zero_shot": {k: before[k] for k in headline_keys},
        "adapter_applied": {k: after[k] for k in headline_keys},
        "deltas": deltas["summary"],
        "vram_peak_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        if torch.cuda.is_available()
        else None,
    }
    (out / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    with (out / "rows.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for b, a, d in zip(before_logits, after_logits, deltas["rows"], strict=True):
            handle.write(json.dumps({**d, "logits_before": b, "logits_after": a}) + "\n")
    print(json.dumps({k: payload[k] for k in ("zero_shot", "adapter_applied", "deltas")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
