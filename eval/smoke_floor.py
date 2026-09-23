"""Re-read a finished smoke run's per-row losses against each row's own floor.

Written for the first two smoke runs (2026-09-23), whose reports showed `Score` loss rising while
the aggregate fell. A soft target cannot be fitted to zero loss: for a 0.5/0.5 split,
cross-entropy bottoms out at ``ln 2`` and ``E|k - y|`` at 0.5, so the mean loss over `Score` rows
moves with how many soft rows a window happens to contain.

Those runs logged only each row's total loss. The target of every row is recoverable, because the
selection is deterministic, so the floor of every row is too:

    floor  = H(target) + lambda * min_k sum_j target_j |k - j|     (lambda = 0 off `Score`)
    excess = logged loss - floor = KL(target || pred) + lambda * (E|k - y| - its minimum)

Off `Score` the excess is exactly KL. On `Score` it is KL plus the distance term's excess; the two
cannot be separated after the fact, which is why the trainer now logs them apart
(`train.ordinal_loss.loss_parts`).

    uv run python -m eval.smoke_floor smoke-small smoke-27b --out results/smoke-floor-2026-09-23
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["main", "reread", "row_floor"]


def row_floor(target: Sequence[float], qtype: str, lambda_distance: float) -> float:
    """The lowest loss any prediction can reach on this row.

    Args:
        target: The row's target distribution.
        qtype: ``score`` rows carry the distance term; others are plain cross-entropy.
        lambda_distance: The run's distance weight.

    Returns:
        ``H(target)``, plus ``lambda`` times the distance term's minimum on `Score` rows.
    """
    entropy = -sum(p * math.log(p) for p in target if p > 0)
    if qtype != "score":
        return entropy
    floor = min(sum(t * abs(k - j) for j, t in enumerate(target)) for k in range(len(target)))
    return entropy + lambda_distance * floor


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def reread(run_dir: Path, root: Path) -> dict[str, Any]:
    """Rebuild a run's seen rows and report raw loss, floor and excess per primitive.

    Args:
        run_dir: ``results/<run_id>`` of a run with ``loss_rows`` in its summary.
        root: Repository root (for ``data/processed/train.jsonl``).

    Returns:
        Per qtype: rows, soft rows in each 10-row window, and the first/last-10 means of the raw
        loss, the floor and the excess, plus the smallest excess seen (a negative one would mean
        the floor is wrong).
    """
    from train.sft_lora import TrainConfig, build_examples, plan_steps, select_rows
    from transformers import AutoTokenizer

    summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
    known = TrainConfig.__dataclass_fields__
    config = TrainConfig(**{k: v for k, v in summary["config"].items() if k in known})
    rows = [
        json.loads(line)
        for line in (root / "data/processed/train.jsonl").read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    tokenizer = AutoTokenizer.from_pretrained(config.model, local_files_only=True)
    examples = build_examples(select_rows(rows, config), tokenizer, config)[: config.limit]
    total_rows = plan_steps(len(examples), config) * config.batch_size * config.grad_accum
    seen = [examples[i % len(examples)] for i in range(total_rows)]

    logged = summary["loss_rows"]
    losses = [row[1] if isinstance(row, list) else row["total"] for row in logged]
    if len(losses) != len(seen):
        raise ValueError(f"{len(losses)} logged rows but {len(seen)} rebuilt")
    for example, row in zip(seen, logged, strict=True):
        qtype = row[0] if isinstance(row, list) else row["qtype"]
        expected = "noul/stage1" if example.get("stage1") else example["qtype"]
        if qtype != expected:
            raise ValueError(f"rebuilt row is {expected} but the run logged {qtype}")

    report: dict[str, Any] = {"run_id": summary["run_id"], "lambda": config.lambda_distance}
    for qtype in ("choice", "noul", "score"):
        index = [i for i, e in enumerate(seen) if e["qtype"] == qtype]
        raw = [losses[i] for i in index]
        floor = [row_floor(seen[i]["target"], qtype, config.lambda_distance) for i in index]
        excess = [r - f for r, f in zip(raw, floor, strict=True)]
        soft = [bool(seen[i].get("soft")) for i in index]
        report[qtype] = {
            "rows": len(index),
            "soft_rows_first_10": sum(soft[:10]),
            "soft_rows_last_10": sum(soft[-10:]),
            **{
                f"{name}_{window}": _mean(values[:10] if window == "first_10" else values[-10:])
                for name, values in (("raw", raw), ("floor", floor), ("excess", excess))
                for window in ("first_10", "last_10")
            },
            "min_excess": min(excess) if excess else None,
        }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: one or more run ids, one JSON out."""
    from s1decide.tasks import repo_root

    parser = argparse.ArgumentParser(prog="eval.smoke_floor")
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", required=True, help="output directory under the repo")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = repo_root()
    payload = {run: reread(root / "results" / run, root) for run in args.runs}
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "floor.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
