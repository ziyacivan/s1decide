"""Laya (Convai Innovations, ModernBERT, 421M) as an external row in the evaluation table.

Laya is non-autoregressive with its own decision head, so it cannot run through the token-logit
engine every other row uses. It is evaluated on **the same slices and the same metrics**
instead, in three steps that keep its dependencies (torch 2.14, transformers 5.17) out of this
project's locked environment:

1. ``export`` (project env) — writes the slice: exactly S1's periodic `val` slice (every
   non-stage-1 row plus 300 stage-1 questions case-control, seed 20260923), and the same
   construction on `test`.
2. ``predict`` (Laya's own env, CPU) — asks Laya each row's question and writes its per-option
   probabilities in our option order. This half imports only the standard library and `laya`.
3. ``report`` (project env) — weighted KL, accuracy, ECE and BSS per primitive against the
   training-fitted base rate (`train.periodic_eval`), risk-coverage, and the base-rate control
   as its own row. Written to ``metrics.json`` with Laya's version, checkpoint revision, device
   and any warnings it raised.

Questions are passed with **the text our model sees**: stage-1 rows keep their
"…?\\nCandidate: X" instructions rather than being rephrased as statements, because a different
question would be a different comparison. Laya's probabilities are used as shipped — its card
applies post-hoc temperature per question type and option count — and the meta says so.

    uv run --no-sync python -m eval.external_laya export --split val --out results/<run>
    <laya-env>/python -m eval.external_laya predict --slice results/<run>/slice-val.jsonl \\
        --checkpoint typed-decisions --out results/<run>/predictions-val.jsonl
    uv run --no-sync python -m eval.external_laya report --dir results/<run> --splits val test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["from_laya_answer", "main", "to_laya_question"]

#: The S1 periodic-eval slice parameters; a different slice would be a different comparison.
SLICE = {"stage1_questions": 300, "negatives": 16, "seed": 20260923}

#: Laya checkpoints, as its model card names them.
CHECKPOINTS = {"laya": None, "typed-decisions": "typed-decisions", "multilingual": "multilingual"}


def to_laya_question(row: Mapping[str, Any]) -> dict[str, Any]:
    """Our row as a Laya question: same instructions, same options, same order.

    Stage-1 rows are `Noul`-shaped (``["no", "yes"]``) and are asked as `noul`.
    """
    options = list(row["options"])
    if row["qtype"] == "noul" or row.get("stage") == 1:
        return {"type": "noul", "instructions": row["instructions"]}
    if row["qtype"] == "score":
        return {"type": "score", "instructions": row["instructions"], "criteria": options}
    return {
        "type": "choice",
        "instructions": row["instructions"],
        "criteria": {option: option for option in options},
    }


def from_laya_answer(row: Mapping[str, Any], answer: Mapping[str, Any]) -> list[float]:
    """Laya's answer as probabilities over our options, in our order.

    Raises:
        ValueError: If the answer is missing an option or its type does not match the row.
    """
    options = list(row["options"])
    kind = answer.get("type")
    if row["qtype"] == "noul" or row.get("stage") == 1:
        if kind != "noul":
            raise ValueError(f"{row['id']}: expected a noul answer, got {kind!r}")
        yes = float(answer["noul"])
        if options != ["no", "yes"]:
            raise ValueError(f"{row['id']}: unexpected noul options {options}")
        return [1.0 - yes, yes]
    probs = answer.get("probabilities") or {}
    if row["qtype"] == "score":
        values = [probs.get(str(i)) for i in range(len(options))]
    else:
        values = [probs.get(option) for option in options]
    if any(v is None for v in values):
        raise ValueError(f"{row['id']}: answer lacks options: {sorted(probs)} vs {options}")
    total = sum(values)
    return [float(v) / total for v in values]


# --- 1. export (project env) -----------------------------------------------------------------


def _export(split: str, out: Path) -> Path:
    from train.periodic_eval import build_eval_rows

    from s1decide.tasks import repo_root

    root = repo_root()
    rows = [
        json.loads(line)
        for line in (root / f"data/processed/{split}.jsonl").read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    slice_rows, report = build_eval_rows(rows, **SLICE)
    keep = (
        "id",
        "state",
        "qtype",
        "family",
        "stage",
        "instructions",
        "options",
        "answer_idx",
        "target",
        "target_type",
        "eval_weight",
    )
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"slice-{split}.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in slice_rows:
            handle.write(json.dumps({k: row.get(k) for k in keep}, ensure_ascii=False) + "\n")
    (out / f"slice-{split}.json").write_text(
        json.dumps({**report, "slice": SLICE, "split": split}, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {len(slice_rows)} rows -> {path}")
    return path


# --- 2. predict (Laya env) -------------------------------------------------------------------


def _predict(slice_path: Path, checkpoint: str, out: Path, threads: int) -> Path:
    import time
    import warnings

    import laya
    import torch

    torch.set_num_threads(threads)
    caught: list[str] = []
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        agent = laya.load("convaiinnovations/laya", subfolder=CHECKPOINTS[checkpoint])
        caught += [f"{w.category.__name__}: {w.message}" for w in seen]
    rows = [json.loads(x) for x in slice_path.read_text(encoding="utf-8").split("\n") if x.strip()]
    started = time.perf_counter()
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for i, row in enumerate(rows):
            result = agent.predict(row["state"], {"q": to_laya_question(row)})
            probs = from_laya_answer(row, result["answers"]["q"])
            handle.write(json.dumps({"id": row["id"], "probabilities": probs}) + "\n")
            if (i + 1) % 500 == 0:
                rate = (i + 1) / (time.perf_counter() - started)
                print(f"  {i + 1}/{len(rows)} ({rate:.2f} rows/s)", flush=True)
    revision = None
    try:
        from huggingface_hub import HfApi

        revision = HfApi().model_info("convaiinnovations/laya").sha
    except Exception as exc:  # recorded in the meta rather than silently left blank
        caught.append(f"revision lookup failed: {type(exc).__name__}: {exc}")
    meta = {
        "checkpoint": checkpoint,
        "repo": "convaiinnovations/laya",
        "revision": revision,
        "subfolder": CHECKPOINTS[checkpoint],
        "laya_version": getattr(laya, "__version__", None),
        "torch": torch.__version__,
        "device": "cpu",
        "threads": threads,
        "rows": len(rows),
        "seconds": round(time.perf_counter() - started, 1),
        "load_warnings": caught,
        "probabilities": "as shipped: Laya's post-hoc temperature per question type and option count",
        "state_passed_as": "the raw state string, as our model receives it",
    }
    out.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(meta, indent=2))
    return out


# --- 3. report (project env) -----------------------------------------------------------------


def _report(directory: Path, splits: Sequence[str]) -> Path:
    import numpy as np
    from train.periodic_eval import base_rates, eval_group, eval_metrics
    from train.sampling import row_target

    from eval.metrics import risk_coverage
    from s1decide.tasks import repo_root

    root = repo_root()
    train_rows = [
        json.loads(x)
        for x in (root / "data/processed/train.jsonl").read_text(encoding="utf-8").split("\n")
        if x.strip()
    ]
    rates = base_rates(train_rows)
    payload: dict[str, Any] = {"model": {}, "splits": {}}
    for split in splits:
        rows = [
            json.loads(x)
            for x in (directory / f"slice-{split}.jsonl").read_text(encoding="utf-8").split("\n")
            if x.strip()
        ]
        predictions = {
            p["id"]: p["probabilities"]
            for p in (
                json.loads(x)
                for x in (directory / f"predictions-{split}.jsonl")
                .read_text(encoding="utf-8")
                .split("\n")
                if x.strip()
            )
        }
        missing = [r["id"] for r in rows if r["id"] not in predictions]
        if missing:
            raise ValueError(f"{split}: {len(missing)} rows have no prediction, e.g. {missing[:3]}")
        examples = [{**r, "target": row_target(r)} for r in rows]
        logits = [[math.log(max(p, 1e-12)) for p in predictions[r["id"]]] for r in rows]
        control = [
            [math.log(max(p, 1e-12)) for p in rates[f"{eval_group(r)}/{len(r['options'])}"]]
            for r in rows
        ]
        weights = [float(r.get("eval_weight", 1.0)) for r in rows]
        labels = [int(r["answer_idx"]) for r in rows]
        payload["splits"][split] = {
            "rows": len(rows),
            "model": eval_metrics(examples, logits, rates),
            "base_rate_control": eval_metrics(examples, control, rates),
            "risk_coverage": risk_coverage(
                [list(np.exp(np.asarray(lg))) for lg in logits], labels, weights=weights
            ),
        }
        meta_path = directory / f"predictions-{split}.meta.json"
        payload["model"][split] = json.loads(meta_path.read_text(encoding="utf-8"))
    out = directory / "metrics.json"
    out.write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {out}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """``export`` / ``predict`` / ``report``; see the module docstring."""
    parser = argparse.ArgumentParser(prog="eval.external_laya")
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--split", required=True)
    e.add_argument("--out", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--slice", required=True)
    p.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=6)
    r = sub.add_parser("report")
    r.add_argument("--dir", required=True)
    r.add_argument("--splits", nargs="+", default=["val", "test"])
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.cmd == "export":
        _export(args.split, Path(args.out))
    elif args.cmd == "predict":
        _predict(Path(args.slice), args.checkpoint, Path(args.out), args.threads)
    else:
        _report(Path(args.dir), args.splits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
