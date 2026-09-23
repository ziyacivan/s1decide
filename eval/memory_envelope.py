"""Collect the training memory envelope for one machine from the runs that measured it.

Each point is one training run's own `train_summary.json`: its sequence cap, rank, batch, the
adapter it actually trained, peak VRAM and throughput. Nothing is typed in: the hardware profile
points at the JSON this writes (`HardwareProfile.training_envelope`), so the number a config is
checked against is the number a run produced.

    uv run python -m eval.memory_envelope smoke-27b-attempt3 smoke-27b-seq2048 smoke-27b-rank16 \\
        --out results/memory-envelope-2026-09-23
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

__all__ = ["envelope_point", "main"]


def envelope_point(summary: dict[str, Any]) -> dict[str, Any]:
    """One run's contribution to the envelope.

    Args:
        summary: A run's ``train_summary.json``.

    Returns:
        The knobs that set memory, what was trained, and what it cost.

    Raises:
        ValueError: If the run did not train the full adapter its layout check expects — a peak
            measured on a smaller adapter is not a point on this envelope.
    """
    layout = summary.get("adapter_layout") or {}
    if not layout or layout.get("adapted") != layout.get("expected"):
        raise ValueError(f"{summary.get('run_id')}: adapter layout incomplete ({layout})")
    config = summary["config"]
    return {
        "run_id": summary["run_id"],
        "model": config["model"],
        "max_seq_len": config["max_seq_len"],
        "rank": config["rank"],
        "batch_size": config["batch_size"],
        "grad_accum": config["grad_accum"],
        "rows": summary["rows_seen"],
        "adapter_modules": layout["adapted"],
        "loader": summary["loader"]["path"],
        "vram_peak_gib": summary["vram_peak_gib"],
        "tokens_per_second": summary["tokens_per_second"],
        "seconds_per_row": summary["elapsed_seconds"] / summary["rows_seen"],
        "kernels": summary["kernels"]["gated_deltanet"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Aggregate named runs into ``envelope.json``."""
    from s1decide.hardware import detect_profile
    from s1decide.tasks import repo_root

    parser = argparse.ArgumentParser(prog="eval.memory_envelope")
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = repo_root()
    points = [
        envelope_point(
            json.loads((root / "results" / run / "train_summary.json").read_text(encoding="utf-8"))
        )
        for run in args.runs
    ]
    payload = {"hardware": detect_profile().name, "points": points}
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "envelope.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
