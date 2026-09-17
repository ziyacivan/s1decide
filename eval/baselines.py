"""The committed runs that later results are measured against.

A baseline named only in prose drifts: someone re-runs the eval, the numbers move, and nothing
says which run the model card meant. This module names them, and `tests/test_baselines.py`
fails if a named run is missing, has been re-scored under a different prompt format, or no
longer carries the metrics a comparison needs.

Comparisons are only meaningful within one `FORMAT_VERSION`: a format change alters every
prompt the model sees, so a 0.1 run and a 0.2 run are different experiments, not two points on
one curve. :func:`compare_to_reference` refuses to compare across formats rather than quietly
producing a delta that looks like progress.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["REFERENCE_RUNS", "ReferenceRun", "compare_to_reference", "load_reference"]

#: Metrics a comparison reports, and whether lower is better.
COMPARED_METRICS: tuple[tuple[str, bool], ...] = (
    ("accuracy", False),
    ("ece", True),
    ("brier_top_label", True),
    ("brier_multiclass", True),
    ("nll", True),
    ("aurc", True),
)


@dataclass(frozen=True)
class ReferenceRun:
    """One committed run that later work is compared against.

    Attributes:
        run_id: Directory name under ``results/``.
        purpose: What this run is the reference *for*.
        format_version: The prompt format it was produced under. A comparison against a run in
            a different format is refused.
        note: Anything a reader needs in order to read the numbers correctly.
    """

    run_id: str
    purpose: str
    format_version: str
    note: str = ""


REFERENCE_RUNS: dict[str, ReferenceRun] = {
    "zeroshot": ReferenceRun(
        run_id="20260917-format02-zeroshot",
        purpose="Zero-shot Qwen3.8-27B at nf4-bf16 — what S1 has to beat",
        format_version="0.2",
        note=(
            "Accepted 2026-09-17 as the reference for S1 comparisons. Accuracy is unchanged "
            "from the format-0.1 run within noise (0.7671 -> 0.7645, n=1520), uncalibrated ECE "
            "improves (0.1274 -> 0.1026) and calibrated ECE is a wash (0.0375 -> 0.0449). The "
            "0.1 run stays committed and stays labelled not comparable. Read the Brier skill "
            "score before the ECE: the base-rate control beats this model's calibrated ECE "
            "while scoring 33 accuracy points lower."
        ),
    ),
}


def _results_root() -> Path:
    """Repository ``results/`` directory."""
    from s1decide.tasks import repo_root

    return repo_root() / "results"


def load_reference(name: str, results_root: Path | None = None) -> dict[str, Any]:
    """Load a named reference run's ``metrics.json``.

    Args:
        name: A key of :data:`REFERENCE_RUNS`.
        results_root: Override for ``results/``, for tests.

    Returns:
        The parsed metrics payload.

    Raises:
        KeyError: If the name is not a registered reference.
        FileNotFoundError: If the run is registered but not committed.
    """
    if name not in REFERENCE_RUNS:
        raise KeyError(f"unknown reference run {name!r}; known: {sorted(REFERENCE_RUNS)}")
    reference = REFERENCE_RUNS[name]
    path = (results_root or _results_root()) / reference.run_id / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"reference run {name!r} points at {path}, which is not committed. "
            "Either commit the run or update eval/baselines.py."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def compare_to_reference(
    report: dict[str, Any],
    name: str = "zeroshot",
    *,
    key: str = "overall_calibrated",
    results_root: Path | None = None,
) -> dict[str, Any]:
    """Compare a report against a named reference, metric by metric.

    Args:
        report: A parsed ``metrics.json`` for the new run.
        name: Which reference to compare against.
        key: Which block to compare — ``overall`` or ``overall_calibrated``.
        results_root: Override for ``results/``, for tests.

    Returns:
        ``{"reference", "metrics": {name: {"reference", "measured", "delta", "better"}}}``.

    Raises:
        ValueError: If the two runs were produced under different prompt formats, which makes
            the comparison meaningless rather than merely noisy.
    """
    reference_payload = load_reference(name, results_root)
    reference = REFERENCE_RUNS[name]
    ours = report.get("meta", {}).get("format_version")
    theirs = reference_payload.get("meta", {}).get("format_version")
    if ours != theirs:
        raise ValueError(
            f"cannot compare a format-{ours} run against format-{theirs} reference "
            f"{reference.run_id!r}: the prompts differ, so the model saw different inputs"
        )

    metrics: dict[str, Any] = {}
    for metric, lower_is_better in COMPARED_METRICS:
        before = reference_payload.get(key, {}).get(metric)
        after = report.get(key, {}).get(metric)
        if before is None or after is None:
            continue
        delta = after - before
        metrics[metric] = {
            "reference": before,
            "measured": after,
            "delta": delta,
            "better": (delta < 0) if lower_is_better else (delta > 0),
        }
    return {"reference": reference.run_id, "block": key, "metrics": metrics}
