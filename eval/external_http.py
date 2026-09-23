"""Any model served behind TypeSafe's ``/v1/systemone`` request format, as an external row.

Kev (jaredpalmer/kev) ships such a server; so can other System One projects. This is the
``predict`` half of `eval/external_laya.py` over HTTP instead of an in-process package: the same
exported slices, the same question mapping, the same answer mapping onto our options, and the
same ``report`` step (`python -m eval.external_laya report`), so every external row is scored by
identical code. Standard library only.

    uv run --no-sync python -m eval.external_http --url http://127.0.0.1:8009 \\
        --slice results/<run>/slice-val.jsonl --out results/<run>/predictions-val.jsonl \\
        --model-id jaredpalmer/kev-9b --source-commit c9c1f85
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eval.external_laya import from_laya_answer, to_laya_question

__all__ = ["main", "to_systemone_question"]


def to_systemone_question(row: Mapping[str, Any], describe_options: bool = False) -> dict[str, Any]:
    """Our row as a ``/v1/systemone`` question.

    Args:
        row: A slice row.
        describe_options: Repeat each `Choice` option as its own description (what Laya was
            given) instead of ``null`` (Kev's documented form for an undescribed option).
    """
    question = to_laya_question(row)
    if question["type"] == "choice" and not describe_options:
        question["criteria"] = {option: None for option in row["options"]}
    return question


def _post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/systemone",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    """Ask the server every row's question and write ``predictions-<split>.jsonl`` + meta."""
    parser = argparse.ArgumentParser(prog="eval.external_http")
    parser.add_argument("--url", required=True)
    parser.add_argument("--slice", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--model-id", required=True, help="what the server is serving, for the meta"
    )
    parser.add_argument("--source-commit", default=None, help="the server's source revision")
    parser.add_argument("--describe-options", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows = [
        json.loads(x) for x in Path(args.slice).read_text(encoding="utf-8").split("\n") if x.strip()
    ]
    out = Path(args.out)
    started = time.perf_counter()
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for i, row in enumerate(rows):
            answer = _post(
                args.url,
                {
                    "state": row["state"],
                    "questions": {"q": to_systemone_question(row, args.describe_options)},
                },
                args.timeout,
            )["answers"]["q"]
            probs = from_laya_answer(row, answer)
            handle.write(json.dumps({"id": row["id"], "probabilities": probs}) + "\n")
            if (i + 1) % 500 == 0:
                rate = (i + 1) / (time.perf_counter() - started)
                print(f"  {i + 1}/{len(rows)} ({rate:.2f} rows/s)", flush=True)
    meta = {
        "model_id": args.model_id,
        "source_commit": args.source_commit,
        "server": args.url,
        "protocol": "/v1/systemone (TypeSafe request format)",
        "choice_options_described": args.describe_options,
        "rows": len(rows),
        "seconds": round(time.perf_counter() - started, 1),
        "probabilities": "as served: the server's own calibration (Kev: one fitted temperature)",
        "state_passed_as": "the raw state string, as our model receives it",
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
