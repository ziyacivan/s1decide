"""Fill the README's metric blocks from committed runs.

`CLAUDE.md` forbids typing a metric by hand, and the README is where that rule is easiest to
break and hardest to notice. The blocks between ``<!--metrics:name-->`` markers are therefore
generated from `results/`, and `tests/test_readme.py` fails if they drift.

It is not a hypothetical risk: the first hand-written version of the latency block said
5,872 ms where the run said 5,871.50, because `format` rounds halves to even and I did not.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

__all__ = ["fill_slots", "render_latency_slot", "render_zeroshot_slot", "update_readme"]


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def render_zeroshot_slot(metrics: dict[str, Any]) -> str:
    """The calibration headline, from a run's ``metrics.json``.

    Skill comes before ECE deliberately: a base-rate control beats this model's ECE while being
    33 accuracy points worse, so ECE alone would be the most flattering possible framing.
    """
    calibrated = metrics["overall_calibrated"]
    bss = metrics["skill_calibrated"]["base_rate"]["brier_multiclass"]
    selective = metrics["risk_coverage_calibrated"]["selective_accuracy"]["0.8"]
    return "\n".join(
        [
            "",
            "| | value |",
            "|---|---|",
            f"| Accuracy | {calibrated['accuracy']:.4f} |",
            f"| Brier skill vs base rate | +{bss:.4f} |",
            f"| ECE (15 equal-mass bins) | {calibrated['ece']:.4f} |",
            f"| Accuracy at 80% coverage | {selective:.4f} |",
            "",
        ]
    )


def render_latency_slot(payload: dict[str, Any]) -> str:
    """The latency table, from a run's ``latency.json``."""
    lines = [
        "",
        "| questions in one call | median | vs one call each | speedup |",
        "|---|---|---|---|",
    ]
    for point in payload["points"]:
        separate = point.get("median_separate_ms")
        if separate:
            lines.append(
                f"| {point['n_questions']} | {point['median_ms']:,.0f} ms | "
                f"{separate:,.0f} ms | {separate / point['median_ms']:.2f}x |"
            )
        else:
            lines.append(
                f"| {point['n_questions']} | {point['median_ms']:,.0f} ms | not measured | — |"
            )
    lines.append("")
    return "\n".join(lines)


def fill_slots(readme: str, blocks: dict[str, str]) -> str:
    """Replace each named block's contents, leaving the markers in place.

    Args:
        readme: The README text.
        blocks: ``{name: contents}``.

    Returns:
        The updated text.

    Raises:
        ValueError: If a named block has no markers — silently adding one at the end would put
            a metrics table somewhere nobody looks.
    """
    for name, contents in blocks.items():
        pattern = re.compile(rf"(<!--metrics:{name}-->)(.*?)(<!--/metrics:{name}-->)", re.S)
        if not pattern.search(readme):
            raise ValueError(f"README has no <!--metrics:{name}--> block to fill")
        readme = pattern.sub(lambda m: m.group(1) + contents + m.group(3), readme)
    return readme


def latest_latency_run(results: Path) -> Path:
    """The newest committed bench run.

    Raises:
        FileNotFoundError: If there is none, rather than quietly leaving the README stale.
    """
    runs = sorted(p for p in results.glob("*-latency") if (p / "latency.json").is_file())
    runs = [r for r in runs if "invariance" not in r.name]
    if not runs:
        raise FileNotFoundError("no committed bench run; run `uv run task bench` first")
    return runs[-1]


def update_readme(root: Path | None = None) -> Path:
    """Regenerate the README's metric blocks in place.

    Args:
        root: Repository root; detected when omitted.

    Returns:
        The README path.
    """
    from eval.baselines import REFERENCE_RUNS
    from s1decide.tasks import repo_root

    root = root or repo_root()
    results = root / "results"
    zeroshot = _read(results / REFERENCE_RUNS["zeroshot"].run_id / "metrics.json")
    latency = _read(latest_latency_run(results) / "latency.json")

    readme = root / "README.md"
    readme.write_text(
        fill_slots(
            readme.read_text(encoding="utf-8"),
            {
                "zeroshot": render_zeroshot_slot(zeroshot),
                "latency": render_latency_slot(latency),
            },
        ),
        encoding="utf-8",
        newline="\n",
    )
    return readme
