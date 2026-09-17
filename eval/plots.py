"""Reliability diagrams and calibration plots.

Every plot is drawn from a committed ``metrics.json``, never from live model state, so the
figure in the repository and the numbers beside it cannot disagree.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["plot_bucket_grid", "plot_reliability", "plot_run"]

_OK = "#2a6f97"
_GAP = "#c1666b"
_GREY = "#8d99ae"


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display on the training box or CI
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "matplotlib is needed for plots: `uv sync` installs it in the dev group, "
            "or `uv sync --extra plots`"
        ) from exc
    return plt


def plot_reliability(
    summary: Mapping[str, Any],
    path: str | Path,
    *,
    title: str,
    subtitle: str = "",
) -> Path:
    """Draw one reliability diagram from a :func:`eval.metrics.summarize` result.

    Bins are equal-mass, so bar *positions* carry the confidence and bar widths are not
    meaningful; each bin is drawn at its mean confidence with its count annotated, and the
    gap to the diagonal is shaded.

    Args:
        summary: A summary mapping containing ``bins`` and the headline metrics.
        path: Where to write the PNG.
        title: Plot title.
        subtitle: Optional second line.

    Returns:
        The path written.
    """
    plt = _require_matplotlib()
    bins = summary["bins"]
    conf = [b["mean_confidence"] for b in bins]
    acc = [b["accuracy"] for b in bins]
    counts = [b["count"] for b in bins]

    fig, (ax, ax_hist) = plt.subplots(
        2, 1, figsize=(6.2, 6.6), height_ratios=[3, 1], sharex=True, constrained_layout=True
    )

    ax.plot([0, 1], [0, 1], "--", color=_GREY, lw=1, label="perfect calibration", zorder=1)
    for c, a in zip(conf, acc):
        ax.plot([c, c], [a, c], color=_GAP, lw=1.2, alpha=0.7, zorder=2)
    ax.plot(conf, acc, "o-", color=_OK, lw=1.6, ms=5, label="observed", zorder=3)
    ax.set_ylabel("accuracy in bin")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.15)
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    head = (
        f"n={summary['n']}  acc={summary['accuracy']:.3f}  ECE={summary['ece']:.3f}  "
        f"Brier(top)={summary['brier_top_label']:.3f}  NLL={summary['nll']:.3f}"
    )
    auroc = summary.get("auroc_confidence")
    head += f"  AUROC={auroc:.3f}" if auroc is not None else "  AUROC=n/a"
    ax.set_title(f"{title}\n{subtitle}\n{head}" if subtitle else f"{title}\n{head}", fontsize=10)

    ax_hist.bar(conf, counts, width=0.012, color=_OK, alpha=0.75)
    ax_hist.set_xlabel("confidence (equal-mass bins)")
    ax_hist.set_ylabel("count")
    ax_hist.grid(alpha=0.15)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


def plot_bucket_grid(
    by_bucket: Mapping[str, Mapping[str, Any]],
    path: str | Path,
    *,
    title: str,
) -> Path:
    """Draw one reliability panel per option-count bucket, side by side.

    Args:
        by_bucket: Bucket label to summary mapping.
        path: Where to write the PNG.
        title: Figure title.

    Returns:
        The path written.
    """
    plt = _require_matplotlib()
    buckets = list(by_bucket)
    if not buckets:
        raise ValueError("no buckets to plot")
    fig, axes = plt.subplots(
        1, len(buckets), figsize=(3.1 * len(buckets), 3.4), sharey=True, constrained_layout=True
    )
    axes = [axes] if len(buckets) == 1 else list(axes)
    for ax, bucket in zip(axes, buckets):
        s = by_bucket[bucket]
        conf = [b["mean_confidence"] for b in s["bins"]]
        acc = [b["accuracy"] for b in s["bins"]]
        ax.plot([0, 1], [0, 1], "--", color=_GREY, lw=1)
        ax.plot(conf, acc, "o-", color=_OK, lw=1.4, ms=4)
        ax.set_title(
            f"{bucket} options\nn={s['n']}  acc={s['accuracy']:.2f}  ECE={s['ece']:.3f}",
            fontsize=9,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("confidence")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("accuracy")
    fig.suptitle(title, fontsize=11)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


def plot_run(run_dir: str | Path) -> list[Path]:
    """Draw every figure for a finished run, from its committed ``metrics.json``.

    Args:
        run_dir: A ``results/<run_id>`` directory.

    Returns:
        The paths written.
    """
    directory = Path(run_dir)
    report = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    meta = report.get("meta", {})
    label = f"{meta.get('model', '?')} · {meta.get('quantization', '?')} · {meta.get('split', '?')}"
    written: list[Path] = []

    for key, name in (("overall", "uncalibrated"), ("overall_calibrated", "calibrated")):
        summary = report.get(key)
        if summary:
            written.append(
                plot_reliability(
                    summary,
                    directory / f"reliability-{name}.png",
                    title=f"Reliability — {name}",
                    subtitle=label,
                )
            )
    for key, name in (
        ("by_option_count", "uncalibrated"),
        ("by_option_count_calibrated", "calibrated"),
    ):
        buckets = report.get(key)
        if buckets:
            written.append(
                plot_bucket_grid(
                    buckets,
                    directory / f"reliability-by-options-{name}.png",
                    title=f"Reliability by option count — {name} — {label}",
                )
            )
    for name in ("uniform", "base_rate"):
        control = report.get("controls", {}).get(name)
        if control:
            written.append(
                plot_reliability(
                    control,
                    directory / f"reliability-control-{name}.png",
                    title=f"Negative control — {name}",
                    subtitle=label,
                )
            )
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Redraw the figures for a run directory."""
    import argparse

    parser = argparse.ArgumentParser(description="Draw plots for a finished eval run")
    parser.add_argument("run_dir", help="results/<run_id>")
    args = parser.parse_args(list(argv) if argv is not None else None)
    for path in plot_run(args.run_dir):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
