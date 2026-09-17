"""Render a run's ``metrics.json`` as a readable ``SUMMARY.md``.

Generated, never written by hand: `CLAUDE.md` forbids typing a metric into a document, and a
summary table is exactly the place that rule gets broken. Regenerate any time with
``uv run task eval --summary results/<run_id>``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["render_latency", "render_summary", "write_latency_summary", "write_summary"]

_COLUMNS = (
    ("n", "n", "{:d}"),
    ("accuracy", "acc", "{:.4f}"),
    ("ece", "ECE", "{:.4f}"),
    ("mce", "MCE", "{:.3f}"),
    ("brier_top_label", "Brier(top)", "{:.4f}"),
    ("brier_multiclass", "Brier(mc)", "{:.4f}"),
    ("nll", "NLL", "{:.4f}"),
    ("auroc_confidence", "AUROC", "{:.4f}"),
    ("mean_confidence", "conf", "{:.3f}"),
)


def _row(label: str, summary: Mapping[str, Any]) -> str:
    cells = []
    for key, _title, fmt in _COLUMNS:
        value = summary.get(key)
        cells.append("n/a" if value is None else fmt.format(value))
    return f"| {label} | " + " | ".join(cells) + " |"


def _table(rows: Sequence[tuple[str, Mapping[str, Any]]], first: str = "") -> list[str]:
    head = f"| {first} | " + " | ".join(t for _k, t, _f in _COLUMNS) + " |"
    rule = "|---" * (len(_COLUMNS) + 1) + "|"
    return [head, rule, *[_row(label, s) for label, s in rows]]


def render_summary(report: Mapping[str, Any]) -> str:
    """Render the Markdown body for one report.

    Args:
        report: A parsed ``metrics.json``.

    Returns:
        Markdown source.
    """
    meta = report["meta"]
    git = meta.get("git", {})
    out: list[str] = [
        f"# {meta['run_id']}",
        "",
        "<!-- GENERATED from metrics.json by eval/summary.py — do not edit by hand. -->",
        "",
        f"- **Model**: `{meta['model']}` · quantization `{meta['quantization']}` · engine `{meta['engine']}`",
        f"- **Data**: `{meta['dataset']}` ({meta['dataset_license']}), report split `{meta['split']}`, "
        f"calibration fitted on `{meta['val_split']}`",
        f"- **Prompt format**: `{meta['format_version']}` · template `{meta['template']}`",
        f"- **Commit**: `{(git.get('commit') or '?')[:12]}`"
        + (" **(tree dirty at run time)**" if git.get("dirty") else ""),
        f"- **Created**: {meta.get('created', '?')} · {meta.get('platform', '?')}",
        "",
        "## Coverage",
        "",
        "| split | kept | total | fraction | dropped >26 options | dropped invalid |",
        "|---|---|---|---|---|---|",
    ]
    for split, c in report.get("coverage", {}).items():
        dropped = ", ".join(f"{k} {v}" for k, v in c["dropped_families"].items()) or "—"
        out.append(
            f"| {split} | {c['kept']} | {c['total']} | {c['fraction_kept']:.1%} | "
            f"{c['dropped_high_cardinality']} ({dropped}) | {c['dropped_invalid']} |"
        )

    out += [
        "",
        "## Headline",
        "",
        *_table(
            [
                ("model, uncalibrated", report["overall"]),
                ("model, calibrated", report["overall_calibrated"]),
                *[(f"control: {k}", v) for k, v in report.get("controls", {}).items()],
            ],
            first="row",
        ),
        "",
        "Temperature scaling cannot change which option wins, so accuracy is identical in the "
        "first two rows by construction. Read the controls before the model: a predictor that "
        "ignores the state can post a competitive ECE, which is why ECE never appears here "
        "without accuracy, Brier and AUROC beside it.",
        "",
        "## Calibration",
        "",
        "| bucket | n | temperature | NLL before | NLL after |",
        "|---|---|---|---|---|",
    ]
    for bucket, info in report.get("calibration", {}).get("meta", {}).get("buckets", {}).items():
        if "temperature" in info:
            out.append(
                f"| {bucket} | {info['n']} | {info['temperature']:.4f} | "
                f"{info['nll_before']:.4f} | {info['nll_after']:.4f} |"
            )
        else:
            out.append(f"| {bucket} | {info['n']} | — ({info.get('skipped', 'skipped')}) | — | — |")

    for key, title in (
        ("by_option_count", "By option-count bucket"),
        ("by_qtype", "By primitive"),
        ("by_family", "By family"),
    ):
        groups = report.get(key, {})
        if not groups:
            continue
        calibrated = report.get(f"{key}_calibrated", {})
        rows: list[tuple[str, Mapping[str, Any]]] = []
        for name, summary in groups.items():
            rows.append((name, summary))
            if name in calibrated:
                rows.append((f"{name} (calibrated)", calibrated[name]))
        out += ["", f"## {title}", "", *_table(rows, first=title.split()[-1])]

    timing = meta.get("timing", {})
    if timing:
        out += [
            "",
            "## Timing",
            "",
            "| split | questions | states | wall (s) | ms/question |",
            "|---|---|---|---|---|",
        ]
        for split, t in timing.items():
            out.append(
                f"| {split} | {t['questions']} | {t['states']} | {t['wall_seconds']:.0f} | "
                f"{t['seconds_per_question'] * 1000:.0f} |"
            )

    out += [
        "",
        "## Files",
        "",
        "- `metrics.json` — the source of truth for every number above.",
        "- `predictions-*.jsonl` — raw masked logits per question; re-score with "
        "`uv run task eval --rescore results/<run_id>` without a GPU.",
        "- `calibration.json` — fitted temperatures, tied to the quantization above.",
        "- `reliability-*.png` — drawn from `metrics.json`.",
        "",
    ]
    return "\n".join(out)


def render_latency(payload: Mapping[str, Any]) -> str:
    """Render the Markdown body for a ``latency.json``.

    Args:
        payload: A parsed ``latency.json``.

    Returns:
        Markdown source.
    """
    meta = payload["meta"]
    target = payload["target"]
    fit = payload["decomposition"]
    points = payload["points"]
    baseline = points[0]["median_ms"]

    out = [
        f"# {meta['run_id']}",
        "",
        "<!-- GENERATED from latency.json by eval/summary.py — do not edit by hand. -->",
        "",
        f"- **Model**: `{meta['model']}` · `{meta['quantization']}` · {meta['gpu']}",
        f"- **State**: {meta['state_tokens_actual']} tokens "
        f"({meta.get('prefix_tokens', meta['state_tokens_actual'])}-token prefill including the "
        f"system prompt) · median of {meta['repeats']} timed runs after {meta['warmup']} warm-ups",
        f"- **Created**: {meta.get('created', '?')} · torch {meta.get('torch', '?')}",
        "",
        f"## Target: {target['rule']}",
        "",
        f"**{target['ratio_64_over_1']:.2f}x — {'MET' if target['met'] else 'MISSED'}.**",
        "",
        "## Measurements",
        "",
        "| questions | median (ms) | p10-p90 | prefill | suffix | passes x rows | suffix tok | "
        "ms/suffix tok | peak GiB | vs 1-question | one call each (ms) | bundling speedup |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p in points:
        separate = p["median_separate_ms"]
        out.append(
            f"| {p['n_questions']} | {p['median_ms']:.0f} | "
            f"{p['p10_ms']:.0f}-{p['p90_ms']:.0f} | {p['median_prefill_ms']:.0f} | "
            f"{p['median_suffix_ms']:.0f} | {p['passes']} x {p['rows_per_pass']} | "
            f"{p['suffix_tokens']} | {p['ms_per_suffix_token']:.3f} | {p['peak_gib']:.2f} | "
            f"{p['median_ms'] / baseline:.2f}x | "
            + (f"{separate:.0f} | {separate / p['median_ms']:.2f}x |" if separate else "— | — |")
        )

    out += [
        "",
        "## Where the time goes",
        "",
        "Least-squares fit of the suffix phase, `suffix_ms ~ a * passes + b * suffix_tokens`:",
        "",
        f"- **{fit['per_pass_ms']:.1f} ms per forward pass** (weight streaming and launch overhead)",
        f"- **{fit['per_token_ms']:.3f} ms per suffix token** (marginal cost of question text)",
        f"- R² = {fit['r_squared']:.4f}",
        "",
        f"A prefix token costs "
        f"{points[0]['median_prefill_ms'] / meta.get('prefix_tokens', meta['state_tokens_actual']):.3f} ms, so a suffix "
        "token costs essentially the same as a prefix token: batching the rows buys no "
        "per-token discount at this batch size, because the pass is already compute-bound.",
        "",
        "## Files",
        "",
        "- `latency.json` — the source of truth for every number above.",
        "- `latency.png` — drawn from it by `uv run task bench --plot results/<run_id>`.",
        "",
    ]
    return "\n".join(out)


def write_latency_summary(run_dir: str | Path) -> Path:
    """Write ``LATENCY.md`` next to a run's ``latency.json``.

    Args:
        run_dir: A ``results/<run_id>`` directory.

    Returns:
        The path written.
    """
    directory = Path(run_dir)
    payload = json.loads((directory / "latency.json").read_text(encoding="utf-8"))
    target = directory / "LATENCY.md"
    target.write_text(render_latency(payload), encoding="utf-8", newline="\n")
    return target


def write_summary(run_dir: str | Path) -> Path:
    """Write ``SUMMARY.md`` next to a run's ``metrics.json``.

    Args:
        run_dir: A ``results/<run_id>`` directory.

    Returns:
        The path written.
    """
    directory = Path(run_dir)
    report = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    target = directory / "SUMMARY.md"
    target.write_text(render_summary(report), encoding="utf-8", newline="\n")
    return target
