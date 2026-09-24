"""Fair-ground evaluation: out-of-distribution sets and unseen families and languages.

The comparison slices (`eval/external_laya.py`) are our home ground: held-out states from the
families S1 trained on. This module scores every model on the `eval` split instead — families
and languages S1 never saw — with the same code for every row of the table:

* held-out families: CommonsenseQA (`Choice`), PubMedQA (`Noul`);
* OOD sets: ANLI r1 and SciQ (`Choice`, non-commercial), BoolQ and SNLI (`Noul`, share-alike;
  eval-only under ADR 0005);
* unseen languages: MASSIVE fr-FR (near) and ja-JP (far), through the two-stage path —
  stage-1 per-candidate `Noul` rows, case-control sampled, and the stage-2 8-way `Choice`.

``export`` writes ``slice-eval.jsonl``; each model is scored on it by its usual harness
(`eval/adapter_slice.py score --slices … --splits eval`, `eval/external_laya.py predict`,
`eval/external_http.py`), and ``report`` scores the predictions per family.

**Two controls, both reported.** The *training prior* is the base-rate control every other table
uses (the training marginal per primitive and option count); an OOD set's labels need not follow
it, so each family also carries skill against its *own* label marginal — a constant predictor
that knew the set's base rate in advance, which is the harder bar. `Choice` option counts the
training set never had (ANLI's 3, CommonsenseQA's 5) get a uniform training prior, which is what
the prior is for any `Choice` (options are shuffled per example); the report lists them.

**Contamination.** ``--contamination`` names a JSON file of per-model, per-family training
overlap (``{"model": {"family": {"status": "clean|trained|unknown", "evidence": "…"}}}``); the
status is copied onto each family's row so a table cannot quote a contaminated row unmarked.

    uv run --no-sync python -m eval.fair_ground export --out results/<run>
    uv run --no-sync python -m eval.fair_ground report --dir results/<run> \\
        --contamination eval/contamination.json --model kev-9b
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["FAMILIES", "SLICE", "family_report", "main", "own_marginal", "training_prior"]

#: The fair-ground slice. More stage-1 questions than the comparison slice, because they are
#: split across two languages here.
SLICE = {"stage1_questions": 600, "negatives": 16, "seed": 20260924}

#: What each `eval` family is, for the table.
FAMILIES: dict[str, str] = {
    "commonsense_qa": "held-out family",
    "pubmedqa": "held-out family",
    "anli": "OOD (non-commercial)",
    "sciq": "OOD (non-commercial)",
    "boolq": "OOD (share-alike)",
    "snli": "OOD (share-alike)",
    "massive_fr": "unseen language, near",
    "massive_ja": "unseen language, far",
}

_KEEP = (
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


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]


def training_prior(
    rates: Mapping[str, Sequence[float]], rows: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, list[float]], list[str]]:
    """The training base rates, with a uniform prior added for unseen `Choice` option counts.

    Args:
        rates: Output of `train.periodic_eval.base_rates`.
        rows: The slice rows.

    Returns:
        ``(rates, added)`` — the completed rates and the keys that were filled uniformly.

    Raises:
        KeyError: If a non-`Choice` key is missing: a `Noul` or `Score` prior is a real prior and
            is never invented.
    """
    from train.periodic_eval import eval_group

    out = {k: list(v) for k, v in rates.items()}
    added: list[str] = []
    for row in rows:
        key = f"{eval_group(row)}/{len(row['options'])}"
        if key in out:
            continue
        if eval_group(row) != "choice":
            raise KeyError(f"no training prior for {key!r}")
        out[key] = [1.0 / len(row["options"])] * len(row["options"])
        added.append(key)
    return out, sorted(set(added))


def own_marginal(labels: Sequence[int], weights: Sequence[float], n_options: int) -> list[float]:
    """A set's own weighted label marginal: the constant predictor that knew the base rate."""
    totals = [0.0] * n_options
    for label, w in zip(labels, weights, strict=True):
        totals[int(label)] += float(w)
    total = sum(totals)
    return [t / total for t in totals]


def family_report(
    rows: Sequence[Mapping[str, Any]],
    probabilities: Sequence[Sequence[float]],
    rates: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """Per reporting group of one family: the model, the training-prior control, and skill
    against the family's own label marginal.

    Args:
        rows: One family's slice rows.
        probabilities: The model's probabilities, aligned with ``rows``.
        rates: Completed training priors (see :func:`training_prior`).
    """
    from train.periodic_eval import eval_group, eval_metrics
    from train.sampling import row_target

    from eval.metrics import brier_multiclass, brier_skill_score

    examples = [{**r, "target": row_target(r)} for r in rows]
    logits = [[math.log(max(p, 1e-12)) for p in probs] for probs in probabilities]
    control = [
        [math.log(max(p, 1e-12)) for p in rates[f"{eval_group(r)}/{len(r['options'])}"]]
        for r in rows
    ]
    model = eval_metrics(examples, logits, rates)
    prior = eval_metrics(examples, control, rates)
    out: dict[str, Any] = {}
    for group in model:
        index = [i for i, r in enumerate(rows) if eval_group(r) == group]
        labels = [int(rows[i]["answer_idx"]) for i in index]
        weights = [float(rows[i].get("eval_weight", 1.0)) for i in index]
        counts = {len(rows[i]["options"]) for i in index}
        entry: dict[str, Any] = {"model": model[group], "training_prior_control": prior[group]}
        if len(counts) == 1:
            k = counts.pop()
            marginal = own_marginal(labels, weights, k)
            reference = brier_multiclass([marginal] * len(index), labels, weights)
            entry["own_marginal"] = marginal
            entry["bss_vs_own_marginal"] = brier_skill_score(model[group]["brier"], reference)
        out[group] = entry
    return out


def _export(out: Path) -> Path:
    from train.periodic_eval import build_eval_rows

    from s1decide.tasks import repo_root

    rows = _read(repo_root() / "data/processed/eval.jsonl")
    unknown = sorted({str(r["family"]) for r in rows} - set(FAMILIES))
    if unknown:
        raise ValueError(f"eval families with no description in FAMILIES: {unknown}")
    slice_rows, report = build_eval_rows(rows, **SLICE)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "slice-eval.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in slice_rows:
            handle.write(json.dumps({k: row.get(k) for k in _KEEP}, ensure_ascii=False) + "\n")
    (out / "slice-eval.json").write_text(
        json.dumps({**report, "slice": SLICE, "split": "eval", "families": FAMILIES}, indent=2)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {len(slice_rows)} rows -> {path}")
    return path


def _report(directory: Path, contamination: Path | None, model_key: str | None) -> Path:
    from train.periodic_eval import base_rates

    from s1decide.tasks import repo_root

    rows = _read(directory / "slice-eval.jsonl")
    predictions = {p["id"]: p["probabilities"] for p in _read(directory / "predictions-eval.jsonl")}
    missing = [r["id"] for r in rows if r["id"] not in predictions]
    if missing:
        raise ValueError(f"{len(missing)} rows have no prediction, e.g. {missing[:3]}")
    rates, uniform = training_prior(
        base_rates(_read(repo_root() / "data/processed/train.jsonl")), rows
    )
    marks: dict[str, Any] = {}
    if contamination is not None:
        table = json.loads(contamination.read_text(encoding="utf-8"))
        if model_key not in table:
            raise KeyError(f"{contamination} has no entry for model {model_key!r}")
        marks = table[model_key]
    families: dict[str, Any] = {}
    for family in sorted({str(r["family"]) for r in rows}):
        sub = [r for r in rows if r["family"] == family]
        families[family] = {
            "kind": FAMILIES[family],
            "contamination": marks.get(family, {"status": "not assessed"}),
            "groups": family_report(sub, [predictions[r["id"]] for r in sub], rates),
        }
    payload = {
        "model": json.loads((directory / "predictions-eval.meta.json").read_text(encoding="utf-8")),
        "slice": json.loads((directory / "slice-eval.json").read_text(encoding="utf-8")),
        "controls": {
            "training_prior": "training marginal per primitive and option count",
            "uniform_training_prior_for": uniform,
            "own_marginal": "each family's own weighted label marginal, per group",
        },
        "contamination_source": str(contamination) if contamination else None,
        "families": families,
    }
    out = directory / "fair_ground.json"
    out.write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {out}")
    return out


def _write_jsonl(path: Path, items: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")


def _calibrate(raw: Path, calibration: Path, out: Path) -> Path:
    """Apply a committed S2 calibration set to a model's raw fair-ground logits.

    The calibration is the one fitted on the comparison slices' `val` (ADR 0008) — never on the
    fair-ground rows themselves, which would be fitting on the test.
    """
    import shutil

    from eval.adapter_slice import calibration_group
    from s1decide.calibrate import CalibrationSet

    cells = CalibrationSet.load(calibration)
    rows = _read(raw / "slice-eval.jsonl")
    logits = {p["id"]: p["logits"] for p in _read(raw / "predictions-eval.jsonl")}
    out.mkdir(parents=True, exist_ok=True)
    for suffix in (".jsonl", ".json"):
        shutil.copy(raw / f"slice-eval{suffix}", out / f"slice-eval{suffix}")
    _write_jsonl(
        out / "predictions-eval.jsonl",
        [
            {
                "id": row["id"],
                "probabilities": [
                    float(x)
                    for x in cells.get("nf4-bf16", calibration_group(row)).apply(logits[row["id"]])
                ],
            }
            for row in rows
        ],
    )
    meta = json.loads((raw / "predictions-eval.meta.json").read_text(encoding="utf-8"))
    meta["calibration"] = (
        f"S2 per cell (ADR 0008) from {calibration}, fitted on the comparison val slice"
    )
    (out / "predictions-eval.meta.json").write_text(
        json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {out}")
    return out


def _table(root: Path, models: Sequence[str]) -> Path:
    """One JSON across models: per family and reporting group, the headline numbers."""
    table: dict[str, Any] = {"models": list(models), "families": {}}
    for name in models:
        report = json.loads((root / name / "fair_ground.json").read_text(encoding="utf-8"))
        for family, entry in report["families"].items():
            fam = table["families"].setdefault(family, {"kind": entry["kind"], "groups": {}})
            for group, g in entry["groups"].items():
                fam["groups"].setdefault(group, {})[name] = {
                    "rows": g["model"]["rows"],
                    "accuracy": g["model"]["accuracy"],
                    "bss_vs_training_prior": g["model"]["bss"],
                    "bss_vs_own_marginal": g.get("bss_vs_own_marginal"),
                    "ece": g["model"]["ece"],
                    "kl": g["model"]["kl"],
                    "contamination": entry["contamination"].get("status"),
                }
    out = root / "table.json"
    out.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """``export`` / ``report`` / ``calibrate`` / ``table``; see the module docstring."""
    parser = argparse.ArgumentParser(prog="eval.fair_ground")
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--dir", required=True)
    r.add_argument("--contamination", default=None)
    r.add_argument("--model", default=None, help="the model's key in the contamination file")
    c = sub.add_parser("calibrate")
    c.add_argument("--raw", required=True)
    c.add_argument("--calibration", required=True)
    c.add_argument("--out", required=True)
    t = sub.add_parser("table")
    t.add_argument("--root", required=True)
    t.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.cmd == "export":
        _export(Path(args.out))
    elif args.cmd == "calibrate":
        _calibrate(Path(args.raw), Path(args.calibration), Path(args.out))
    elif args.cmd == "table":
        _table(Path(args.root), args.models)
    else:
        if (args.contamination is None) != (args.model is None):
            parser.error("--contamination and --model go together")
        _report(
            Path(args.dir),
            Path(args.contamination) if args.contamination else None,
            args.model,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
