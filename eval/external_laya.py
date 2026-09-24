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

That native form is our model's text, not a statement, and external models read it as "is this
plausible?". ``--stage1-phrasing plain`` asks stage-1 rows instead as a labelled plain
proposition (`plain_proposition`); ``--only-stage1`` sends just those rows, and ``merge`` builds
a run directory from a native run with its stage-1 predictions replaced. Both forms are
reported; the meta names the one used.

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

__all__ = ["from_laya_answer", "main", "plain_proposition", "select_rows", "to_laya_question"]

#: How the pipeline appends a stage-1 candidate to the parent question; must match
#: `s1decide.prompt.STAGE1_CANDIDATE_PREFIX` (a test holds them together). Not imported, because
#: ``predict`` runs in the external model's environment where `s1decide` is not installed.
STAGE1_SEPARATOR = "\nCandidate: "

#: The stage-1 phrasings an external row can be asked in.
STAGE1_PHRASINGS = ("native", "plain")

#: The S1 periodic-eval slice parameters; a different slice would be a different comparison.
SLICE = {"stage1_questions": 300, "negatives": 16, "seed": 20260923}

#: Laya checkpoints, as its model card names them.
CHECKPOINTS = {"laya": None, "typed-decisions": "typed-decisions", "multilingual": "multilingual"}


def plain_proposition(instructions: str) -> str:
    """A stage-1 row's question as a statement that is true or false.

    ``"Which intent does this request have?\\nCandidate: play music"`` becomes
    ``'The answer to "Which intent does this request have?" is "play music".'``

    Raises:
        ValueError: If the instructions are not in the stage-1 form.
    """
    question, sep, candidate = instructions.rpartition(STAGE1_SEPARATOR)
    if not sep or not question.strip() or not candidate.strip():
        raise ValueError(f"not a stage-1 question: {instructions!r}")
    return f'The answer to "{question.strip()}" is "{candidate.strip()}".'


def select_rows(rows: Sequence[dict[str, Any]], only_stage1: bool) -> list[dict[str, Any]]:
    """The rows a predict call asks: all of them, or just the stage-1 ones."""
    return [r for r in rows if r.get("stage") == 1] if only_stage1 else list(rows)


def to_laya_question(row: Mapping[str, Any], stage1_phrasing: str = "native") -> dict[str, Any]:
    """Our row as a Laya question: same instructions, same options, same order.

    Stage-1 rows are `Noul`-shaped (``["no", "yes"]``) and are asked as `noul`, in our own text
    (``native``) or as a plain proposition (``plain``).
    """
    if stage1_phrasing not in STAGE1_PHRASINGS:
        raise ValueError(f"unknown stage-1 phrasing {stage1_phrasing!r}; known: {STAGE1_PHRASINGS}")
    if row.get("systemone_question") is not None:
        # A row that came from a `/v1/systemone` suite (the TypeSafe head-to-head) is asked in
        # its own original form, criteria descriptions included.
        return dict(row["systemone_question"])
    options = list(row["options"])
    if row.get("stage") == 1 and stage1_phrasing == "plain":
        return {"type": "noul", "instructions": plain_proposition(row["instructions"])}
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
        # `option_keys`: the criteria ids of a `/v1/systemone` suite row, whose option text
        # carries the description as well.
        values = [probs.get(key) for key in row.get("option_keys") or options]
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


def _predict(
    slice_path: Path,
    checkpoint: str,
    out: Path,
    threads: int,
    stage1_phrasing: str = "native",
    only_stage1: bool = False,
) -> Path:
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
    rows = select_rows(
        [json.loads(x) for x in slice_path.read_text(encoding="utf-8").split("\n") if x.strip()],
        only_stage1,
    )
    started = time.perf_counter()
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for i, row in enumerate(rows):
            result = agent.predict(row["state"], {"q": to_laya_question(row, stage1_phrasing)})
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
        "stage1_phrasing": stage1_phrasing,
        "only_stage1": only_stage1,
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


def _read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    return {
        p["id"]: p
        for p in (json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip())
    }


def _merge(native: Path, stage1: Path, out: Path, splits: Sequence[str]) -> Path:
    """A run directory from ``native`` with every stage-1 prediction taken from ``stage1``.

    ``stage1`` holds ``predictions-<split>.jsonl`` written with ``--only-stage1``; it must cover
    every stage-1 row of the slice and nothing else.
    """
    import shutil

    out.mkdir(parents=True, exist_ok=True)
    for split in splits:
        for suffix in (".jsonl", ".json"):
            shutil.copy(native / f"slice-{split}{suffix}", out / f"slice-{split}{suffix}")
        rows = [
            json.loads(x)
            for x in (native / f"slice-{split}.jsonl").read_text(encoding="utf-8").split("\n")
            if x.strip()
        ]
        wanted = {r["id"] for r in rows if r.get("stage") == 1}
        base = _read_predictions(native / f"predictions-{split}.jsonl")
        replaced = _read_predictions(stage1 / f"predictions-{split}.jsonl")
        if set(replaced) != wanted:
            raise ValueError(
                f"{split}: stage-1 predictions cover {len(set(replaced) & wanted)} of "
                f"{len(wanted)} stage-1 rows, plus {len(set(replaced) - wanted)} other rows"
            )
        with (out / f"predictions-{split}.jsonl").open("w", encoding="utf-8", newline="\n") as h:
            for r in rows:
                h.write(json.dumps(replaced.get(r["id"], base[r["id"]])) + "\n")
        meta = {
            **json.loads((native / f"predictions-{split}.meta.json").read_text(encoding="utf-8")),
            "stage1_phrasing": "plain",
            "stage1_predictions_from": str(stage1),
            "stage1_meta": json.loads(
                (stage1 / f"predictions-{split}.meta.json").read_text(encoding="utf-8")
            ),
            "non_stage1_predictions_from": str(native),
        }
        (out / f"predictions-{split}.meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    print(f"wrote {out}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """``export`` / ``predict`` / ``merge`` / ``report``; see the module docstring."""
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
    p.add_argument("--stage1-phrasing", choices=STAGE1_PHRASINGS, default="native")
    p.add_argument("--only-stage1", action="store_true")
    m = sub.add_parser("merge")
    m.add_argument("--native", required=True)
    m.add_argument("--stage1", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--splits", nargs="+", default=["val", "test"])
    r = sub.add_parser("report")
    r.add_argument("--dir", required=True)
    r.add_argument("--splits", nargs="+", default=["val", "test"])
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.cmd == "export":
        _export(args.split, Path(args.out))
    elif args.cmd == "predict":
        _predict(
            Path(args.slice),
            args.checkpoint,
            Path(args.out),
            args.threads,
            args.stage1_phrasing,
            args.only_stage1,
        )
    elif args.cmd == "merge":
        _merge(Path(args.native), Path(args.stage1), Path(args.out), args.splits)
    else:
        _report(Path(args.dir), args.splits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
