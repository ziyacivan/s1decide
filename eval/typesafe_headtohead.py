"""Head-to-head on TypeSafe's public workflow evals, strict common subset.

**Reference-label caveat, verbatim from https://evals.typesafe.ai (read 2026-09-24):**

    Instead of debating the correctness of the harness and labels, we assume that the code is
    correct, and measure against the current smartest large models. For this eval, the
    reference labels are generated via an average of the responses of GPT-6 Astra and Claude
    Fable 5.1, both at high thinking, answering every question in the harness. All other
    models are evaluated using the provider's default reasoning settings.

The references are therefore the consensus of two closed models, not ground truth, and every
number here is agreement with that consensus. We use them only as evaluation labels; nothing
here is trained on.

**The subset.** TypeSafe publishes 5 cases per workflow (20 of 711). The live files have changed
since SemIf froze them, so the strict common subset is the frozen copy every published
comparison used: Kev's ``evals/external/typesafe-v1/development.jsonl`` (102 questions over the
20 cases, 66 `Noul` + 36 `Choice`), pinned by the SHA-256 in Kev's manifest and checked here.
**The file stays outside this repository** — no licence or terms are stated for TypeSafe's data
— and only derived scores are committed.

**Protocol** — SemIf's, as Kev implements it (`scripts/compare_typesafe.py`): per row,
agreement = argmax(p) equals the reference argmax, TVD = half the L1 distance to the reference; rows are averaged
within each case, then cases with equal weight. ``evaluated`` covers the rows a model answered
(over-context rows are not answered); ``all_rows`` scores each missing row as agreement 0 and
TVD 1. Jev's row is **TypeSafe's own published answers carried in the file**
(``_meta.published.typesafe``), scored by the same code — quoted, not queried.

    uv run --no-sync python -m eval.typesafe_headtohead export --source <kev>/evals/external/typesafe-v1/development.jsonl --out results/<run>
    (score each model on slice-typesafe.jsonl with its usual harness)
    uv run --no-sync python -m eval.typesafe_headtohead report --source <…>/development.jsonl \\
        --run s1decide-s1=results/<run> --run kev-9b=results/<kev-run> --out results/<run>/headtohead.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["case_means", "main", "score_row", "to_slice_row"]

#: The frozen copy's SHA-256, from Kev's `evals/external/typesafe-v1/manifest.json` @ c9c1f85.
FROZEN_SHA256 = "877ed7cd473e3973d29f0bb0ff2a597192955e477f2127794df358b6ba795879"

#: The verbatim caveat travels into every output.
CAVEAT = (
    "Instead of debating the correctness of the harness and labels, we assume that the code is "
    "correct, and measure against the current smartest large models. For this eval, the "
    "reference labels are generated via an average of the responses of GPT-6 Astra and Claude "
    "Fable 5.1, both at high thinking, answering every question in the harness. All other "
    "models are evaluated using the provider's default reasoning settings."
)

#: `Noul` keys in the suite, in our option order (``no``, ``yes``).
NOUL_KEYS = ("false", "true")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(source: Path) -> list[dict[str, Any]]:
    digest = _sha256(source)
    if digest != FROZEN_SHA256:
        raise ValueError(
            f"{source} is not the frozen subset (sha256 {digest}, expected {FROZEN_SHA256})"
        )
    return [json.loads(x) for x in source.read_text(encoding="utf-8").split("\n") if x.strip()]


def to_slice_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """One suite record as a slice row every harness here can score.

    The state is the record's JSON, keys sorted — what Kev's own loader does with a non-string
    state. `Choice` option text is ``"<id>: <description>"`` for our model; ``option_keys`` keeps
    the ids, and ``systemone_question`` the original question, which external models are asked.
    """
    (question,) = record["questions"].values()
    meta = record["_meta"]
    asked = {k: v for k, v in question.items() if k not in {"label", "src"}}
    target_map = meta["target"]
    if question["type"] == "noul":
        keys, options = list(NOUL_KEYS), ["no", "yes"]
    elif question["type"] == "choice":
        keys = list(question["criteria"])
        options = [f"{k}: {d}" if d else k for k, d in ((k, question["criteria"][k]) for k in keys)]
    else:
        raise ValueError(f"{meta['id']}: unsupported question type {question['type']!r}")
    target = [float(target_map.get(k, 0.0)) for k in keys]
    total = sum(target)
    target = [t / total for t in target]
    state = record["state"]
    return {
        "id": meta["id"],
        "case": meta["group_id"],
        "family": meta["family"],
        "qtype": question["type"],
        "stage": None,
        "state": state
        if isinstance(state, str)
        else json.dumps(state, sort_keys=True, ensure_ascii=False),
        "instructions": question["instructions"],
        "options": options,
        "option_keys": keys,
        "answer_idx": max(range(len(target)), key=target.__getitem__),
        "target": target,
        "target_type": "soft",
        "eval_weight": 1.0,
        "systemone_question": asked,
    }


def score_row(p: Sequence[float], reference: Sequence[float]) -> tuple[float, float]:
    """``(agreement, tvd)`` for one row; both vectors in the same option order."""
    modal = max(range(len(p)), key=lambda i: p[i])
    ref = max(range(len(reference)), key=lambda i: reference[i])
    return float(modal == ref), sum(abs(a - b) for a, b in zip(p, reference, strict=True)) / 2


def case_means(
    scores: Mapping[str, tuple[float, float]], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Equal-case means over ``rows``; a row without a score counts as agreement 0, TVD 1."""
    cases: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        cases.setdefault(str(row["case"]), []).append(scores.get(row["id"], (0.0, 1.0)))

    def mean(xs: Sequence[float]) -> float:
        return sum(xs) / len(xs)

    return {
        "rows": len(rows),
        "cases": len(cases),
        "agreement": mean([mean([a for a, _ in v]) for v in cases.values()]),
        "tvd": mean([mean([t for _, t in v]) for v in cases.values()]),
    }


def _summary(scores: Mapping[str, tuple[float, float]], rows: Sequence[dict[str, Any]]) -> dict:
    answered = [r for r in rows if r["id"] in scores]
    out: dict[str, Any] = {
        "answered": len(answered),
        "evaluated": case_means(scores, answered) if answered else None,
        "all_rows": case_means(scores, rows),
        "by_primitive": {},
    }
    for qtype in sorted({r["qtype"] for r in rows}):
        sub = [r for r in answered if r["qtype"] == qtype]
        out["by_primitive"][qtype] = case_means(scores, sub) if sub else None
    return out


def _export(source: Path, out: Path) -> Path:
    rows = [to_slice_row(r) for r in _records(source)]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "slice-typesafe.jsonl"  # gitignored: the data is not ours to redistribute
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["qtype"]] = counts.get(row["qtype"], 0) + 1
    (out / "slice-typesafe.json").write_text(
        json.dumps(
            {
                "source": "jaredpalmer/kev @ c9c1f85, evals/external/typesafe-v1/development.jsonl",
                "source_sha256": FROZEN_SHA256,
                "rows": len(rows),
                "cases": len({r["case"] for r in rows}),
                "by_primitive": counts,
                "state_rendering": "json.dumps(state, sort_keys=True, ensure_ascii=False)",
                "reference_label_caveat": CAVEAT,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {len(rows)} rows -> {path}")
    return path


def _report(source: Path, runs: Sequence[str], out: Path) -> Path:
    records = _records(source)
    rows = [to_slice_row(r) for r in records]
    by_id = {r["id"]: r for r in rows}
    table: dict[str, Any] = {}
    for spec in runs:
        name, _, directory = spec.partition("=")
        if not directory:
            raise ValueError(f"--run takes name=directory, got {spec!r}")
        path = Path(directory) / "predictions-typesafe.jsonl"
        predictions = [
            json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()
        ]
        scores = {
            p["id"]: score_row(p["probabilities"], by_id[p["id"]]["target"]) for p in predictions
        }
        meta_path = Path(directory) / "predictions-typesafe.meta.json"
        table[name] = {
            "source": str(path),
            "meta": json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.exists()
            else None,
            **_summary(scores, rows),
        }
    published: dict[str, tuple[float, float]] = {}
    for record, row in zip(records, rows, strict=True):
        answer = record["_meta"]["published"].get("typesafe")
        if answer is not None:
            p = [float(answer["p"].get(k, 0.0)) for k in row["option_keys"]]
            published[row["id"]] = score_row(p, row["target"])
    table["jev (TypeSafe's published answers, quoted from the file)"] = {
        "source": "_meta.published.typesafe (model typesafe:v13_snowy_elephant)",
        **_summary(published, rows),
    }
    payload = {
        "subset": {
            "source_sha256": FROZEN_SHA256,
            "rows": len(rows),
            "cases": len({r["case"] for r in rows}),
        },
        "protocol": "SemIf/Kev: agreement = argmax match, TVD = half L1; rows averaged within "
        "case, cases with equal weight",
        "reference_label_caveat": CAVEAT,
        "models": table,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """``export`` / ``report``; see the module docstring."""
    parser = argparse.ArgumentParser(prog="eval.typesafe_headtohead")
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--source", required=True)
    e.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--source", required=True)
    r.add_argument("--run", action="append", default=[], help="name=directory, repeatable")
    r.add_argument("--out", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.cmd == "export":
        _export(Path(args.source), Path(args.out))
    else:
        _report(Path(args.source), args.run, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
