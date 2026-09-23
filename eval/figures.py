"""README figures, generated from committed `results/` JSON.

Every figure in the repository is built by `uv run task figures` from a committed run. None is
drawn by hand, none is copied in, and `tests/test_figures.py` fails if a PNG is older than the
JSON it was built from — the same rule the numbers follow, applied to the pictures.

One style for all of them: one figure size, one palette, 2x DPI so they stay readable at README
width, and a caption line naming the run id and the hardware so a figure lifted out of the page
still says where it came from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["FIGURES", "FigureSpec", "build_all", "figure_captions"]

#: One size for every figure, chosen to be legible at the ~900px GitHub renders a README at.
FIGSIZE = (7.2, 4.2)

#: 2x so the PNGs survive a high-DPI screen.
DPI = 192

#: One palette, shared with the reliability diagrams in `eval/plots.py`.
MODEL = "#2a6f97"
CONTROL = "#c1666b"
SEPARATE = "#8d99ae"
GRID = "#9aa5b1"
ACCENT = "#6a994e"


@dataclass(frozen=True)
class FigureSpec:
    """One figure and the committed artefact it is built from.

    Attributes:
        name: File stem under ``docs/figures/``.
        source: Path, relative to the repository root, of the JSON it is generated from.
        alt: Alt text. Required — a figure nobody can read is not a figure.
        caption: Rendered under the image in the README, with run id and hardware filled in.
    """

    name: str
    source: str
    alt: str
    caption: str


FIGURES: tuple[FigureSpec, ...] = (
    FigureSpec(
        name="latency",
        source="results/{latency_run}/latency.json",
        alt=(
            "Latency against questions per call, log scale. Answering all questions in one "
            "bundled call stays far below making one call each, and the gap widens with the "
            "number of questions."
        ),
        caption="Bundled vs one call each, log scale. Speed-up annotated at each point.",
    ),
    FigureSpec(
        name="reliability",
        source="results/{zeroshot_run}/metrics.json",
        alt=(
            "Reliability diagram. The calibrated model tracks the diagonal closely; the "
            "base-rate control also tracks it while being far less accurate, which is the "
            "point being made."
        ),
        caption="Zero-shot, calibrated, with the base-rate control on the same axes.",
    ),
    FigureSpec(
        name="risk-coverage",
        source="results/{zeroshot_run}/metrics.json",
        alt=(
            "Risk-coverage curve. Model accuracy rises as the least confident answers are "
            "declined; the base-rate control stays almost flat, because its confidence "
            "carries little ranking information."
        ),
        caption="Accuracy on the answered set as the least confident are declined.",
    ),
    FigureSpec(
        name="training-mix",
        source="data/processed/manifest.json",
        alt=(
            "Horizontal stacked bar of the training mix, raw rows against the weighted draw. "
            "Stage-1 rows dominate the raw corpus and are cut to roughly a third of what the "
            "model actually sees."
        ),
        caption="Raw rows vs the weighted draw the trainer takes, by primitive and stage.",
    ),
)


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - matplotlib is a declared dependency
        raise RuntimeError("matplotlib is needed to draw figures") from exc
    return plt


def _style(ax) -> None:
    """The shared look: light grid, no top/right spines, readable at README width."""
    ax.grid(alpha=0.18, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _read(root: Path, relative: str) -> dict[str, Any]:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def latest_latency_run(root: Path) -> str:
    """The newest committed bench run id."""
    runs = sorted(
        p.name
        for p in (root / "results").glob("*-latency")
        if (p / "latency.json").is_file() and "invariance" not in p.name
    )
    if not runs:
        raise FileNotFoundError("no committed bench run; run `uv run task bench`")
    return runs[-1]


def resolve_sources(root: Path) -> dict[str, str]:
    """Which run each figure is built from, so captions and staleness checks agree."""
    from eval.baselines import REFERENCE_RUNS

    return {
        "latency_run": latest_latency_run(root),
        "zeroshot_run": REFERENCE_RUNS["zeroshot"].run_id,
    }


def draw_latency(payload: dict[str, Any], path: Path) -> Path:
    """Bundled call vs one call each, log y, speed-up annotated."""
    plt = _require_matplotlib()
    points = payload["points"]
    n = [p["n_questions"] for p in points]
    bundled = [p["median_ms"] for p in points]
    separate = [p.get("median_separate_ms") for p in points]

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    ax.plot(n, bundled, "o-", color=MODEL, lw=2.0, ms=7, label="one bundled call", zorder=3)
    if all(s for s in separate):
        ax.plot(
            n,
            separate,
            "s--",
            color=SEPARATE,
            lw=1.6,
            ms=6,
            label="one call per question",
            zorder=2,
        )
        for x, b, s in zip(n, bundled, separate):
            ax.annotate(
                f"{s / b:.1f}x",
                (x, b),
                textcoords="offset points",
                xytext=(0, -18),
                ha="center",
                fontsize=9,
                color=MODEL,
                fontweight="bold",
            )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([str(x) for x in n])
    ax.set_xlabel("questions in the call")
    ax.set_ylabel("median wall time (ms, log)")
    # Not "one forward pass for every question" — at 64 questions this run needed 11 passes,
    # because VRAM caps the broadcast at 7 rows. The state is prefilled once; the questions
    # are not free.
    ax.set_title("One prefill, shared across every question in the call", fontsize=11)
    ax.margins(y=0.12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def draw_reliability(payload: dict[str, Any], path: Path) -> Path:
    """Calibrated model and base-rate control on one set of axes."""
    plt = _require_matplotlib()
    model = payload["overall_calibrated"]
    control = payload["controls"]["base_rate"]

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    ax.plot([0, 1], [0, 1], "--", color=GRID, lw=1.2, label="perfect calibration", zorder=1)
    for summary, colour, label in (
        (model, MODEL, f"s1decide zero-shot (acc {model['accuracy']:.3f})"),
        (control, CONTROL, f"base-rate control (acc {control['accuracy']:.3f})"),
    ):
        conf = [b["mean_confidence"] for b in summary["bins"]]
        acc = [b["accuracy"] for b in summary["bins"]]
        ax.plot(
            conf,
            acc,
            "o-",
            color=colour,
            lw=1.8,
            ms=5,
            label=f"{label}  ECE {summary['ece']:.3f}",
            zorder=3,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence (15 equal-mass bins)")
    ax.set_ylabel("accuracy in bin")
    ax.set_title("Both track the diagonal — only one of them knows anything", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def draw_risk_coverage(payload: dict[str, Any], path: Path) -> Path:
    """Model vs control, with the 80% and 90% coverage points marked."""
    plt = _require_matplotlib()
    curve = payload["risk_coverage_calibrated"]
    control = payload["controls"]["base_rate"]

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    xs = [p["coverage"] for p in curve["curve"]]
    ys = [p["accuracy"] for p in curve["curve"]]
    ax.plot(xs, ys, "-", color=MODEL, lw=2.0, label="s1decide zero-shot", zorder=3)

    # The control's full curve is not stored — only its selective accuracies at 80% and 90%
    # plus full coverage. Drawn as three markers rather than a line, because a line through
    # three points starting at 0.8 reads as a curve that was cut off.
    control_selective = control.get("selective_accuracy", {})
    cx = [*sorted(float(k) for k in control_selective), 1.0]
    cy = [
        *[control_selective[f"{k:g}"] for k in sorted(float(k) for k in control_selective)],
        control["accuracy"],
    ]
    ax.plot(
        cx,
        cy,
        "s",
        color=CONTROL,
        ms=8,
        label="base-rate control (3 measured points)",
        zorder=2,
    )
    ax.plot(cx, cy, ":", color=CONTROL, lw=1.2, alpha=0.6, zorder=1)

    # Annotations pushed apart: the default placement put the 80% and 90% labels on top of
    # each other, which is worse than no annotation at all.
    offsets = {"0.8": (-14, 26), "0.9": (16, -40)}
    alignment = {"0.8": "right", "0.9": "left"}
    for coverage in ("0.8", "0.9"):
        accuracy = curve["selective_accuracy"][coverage]
        threshold = curve["selective_threshold"][coverage]
        ax.plot([float(coverage)], [accuracy], "o", color=ACCENT, ms=10, zorder=4)
        ax.annotate(
            f"{float(coverage):.0%} answered\n{accuracy:.3f} correct\nconf ≥ {threshold:.2f}",
            (float(coverage), accuracy),
            textcoords="offset points",
            xytext=offsets[coverage],
            ha=alignment[coverage],
            fontsize=8.5,
            color=ACCENT,
            arrowprops={"arrowstyle": "-", "color": ACCENT, "lw": 0.8, "alpha": 0.7},
        )
    ax.set_xlim(0, 1.05)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("coverage — fraction answered, most confident first")
    ax.set_ylabel("accuracy on the answered set")
    ax.set_title("Declining the least confident answers buys accuracy", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _style(ax)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def draw_training_mix(manifest: dict[str, Any], path: Path) -> Path:
    """One horizontal stacked bar per view: raw rows, and the weighted draw."""
    plt = _require_matplotlib()
    mix = manifest["effective_mix"]
    order = ["stage1", "noul", "choice", "score"]
    labels = {
        "stage1": "Noul · stage-1 (option membership)",
        "noul": "Noul · genuine",
        "choice": "Choice",
        "score": "Score",
    }
    colours = {"stage1": SEPARATE, "noul": MODEL, "choice": ACCENT, "score": CONTROL}

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    rows = [("raw rows", "raw_share"), ("weighted draw", "effective_share")]
    for y, (row_label, key) in enumerate(rows):
        left = 0.0
        for group in order:
            if group not in mix:
                continue
            width = mix[group][key]
            ax.barh(y, width, left=left, color=colours[group], edgecolor="white", linewidth=1.2)
            if width > 0.055:
                ax.text(
                    left + width / 2,
                    y,
                    f"{width:.0%}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    fontweight="bold",
                )
            left += width
        ax.text(-0.012, y, row_label, ha="right", va="center", fontsize=10)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[g]) for g in order if g in mix]
    ax.legend(
        handles,
        [labels[g] for g in order if g in mix],
        frameon=False,
        fontsize=9,
        ncol=2,
        loc="upper center",
        # Below the axis label, not on top of it: the default placement collided with it.
        bbox_to_anchor=(0.5, -0.22),
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of the training mix")
    ax.set_title("Row counts are not what the model sees", fontsize=11)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.18, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def figure_captions(root: Path) -> dict[str, str]:
    """Caption text per figure, with run id and hardware filled in from the source JSON."""
    sources = resolve_sources(root)
    latency = _read(root, f"results/{sources['latency_run']}/latency.json")
    zeroshot = _read(root, f"results/{sources['zeroshot_run']}/metrics.json")
    gpu = latency["meta"].get("gpu", "unknown GPU")
    quantization = latency["meta"].get("quantization", "?")

    provenance = {
        "latency": f"`{sources['latency_run']}` · {gpu} · {quantization}",
        "reliability": f"`{sources['zeroshot_run']}` · {gpu} · {zeroshot['meta']['quantization']}",
        "risk-coverage": f"`{sources['zeroshot_run']}` · {gpu} · {zeroshot['meta']['quantization']}",
        "training-mix": "`data/processed/manifest.json` · CPU",
    }
    return {
        spec.name: f"{spec.caption} Generated from {provenance[spec.name]}." for spec in FIGURES
    }


def build_all(root: Path | None = None) -> list[Path]:
    """Draw every figure into ``docs/figures/``.

    Args:
        root: Repository root; detected when omitted.

    Returns:
        The paths written.
    """
    from s1decide.tasks import repo_root

    root = root or repo_root()
    out = root / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    sources = resolve_sources(root)

    latency = _read(root, f"results/{sources['latency_run']}/latency.json")
    zeroshot = _read(root, f"results/{sources['zeroshot_run']}/metrics.json")
    manifest = _read(root, "data/processed/manifest.json")

    written = [
        draw_latency(latency, out / "latency.png"),
        draw_reliability(zeroshot, out / "reliability.png"),
        draw_risk_coverage(zeroshot, out / "risk-coverage.png"),
        draw_training_mix(manifest, out / "training-mix.png"),
    ]
    (out / "sources.json").write_text(
        json.dumps({spec.name: spec.source.format(**sources) for spec in FIGURES}, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    # What each figure was drawn from, by content. Freshness is judged against this rather than
    # file times: a regenerated figure whose input did not change is byte-identical, keeps its
    # old commit time, and would otherwise look stale forever next to a newer source.
    (out / DIGESTS_FILE).write_text(
        json.dumps(
            {spec.name: source_digest(root / spec.source.format(**sources)) for spec in FIGURES},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return written


#: Per-figure digest of the source it was drawn from, next to the figures.
DIGESTS_FILE = "digests.json"


def source_digest(path: Path) -> str:
    """SHA-256 of a figure's source file, as the figure-freshness check compares it."""
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
