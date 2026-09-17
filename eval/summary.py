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

#: Selective-accuracy columns, pulled out of the nested `selective_accuracy` mapping.
_SELECTIVE_COLUMNS = (("0.8", "acc@80%"), ("0.9", "acc@90%"))


def _stale_format_banner(run_format: str | None) -> list[str]:
    """Warn, in the document itself, when a run predates the current prompt format.

    A metric produced under one prompt format is not comparable with one produced under
    another, and a reader of a committed ``SUMMARY.md`` has no other way to know.

    Args:
        run_format: The ``format_version`` recorded in the run.

    Returns:
        Markdown lines, empty when the run is current.
    """
    from s1decide.prompt import FORMAT_VERSION

    if run_format is None or run_format == FORMAT_VERSION:
        return []
    return [
        f"> **Format {run_format} — not comparable to {FORMAT_VERSION}.** This run was produced "
        f"under an older prompt format. Every number below is valid for format {run_format} and "
        f"must not be compared with a format-{FORMAT_VERSION} run; the prompts differ, so the "
        "model saw different inputs. Kept deliberately: the change between formats is itself a "
        "result.",
        "",
    ]


def _row(label: str, summary: Mapping[str, Any]) -> str:
    cells = []
    for key, _title, fmt in _COLUMNS:
        value = summary.get(key)
        cells.append("n/a" if value is None else fmt.format(value))
    selective = summary.get("selective_accuracy") or {}
    for key, _title in _SELECTIVE_COLUMNS:
        value = selective.get(key)
        cells.append("n/a" if value is None else f"{value:.4f}")
    return f"| {label} | " + " | ".join(cells) + " |"


def _table(rows: Sequence[tuple[str, Mapping[str, Any]]], first: str = "") -> list[str]:
    titles = [t for _k, t, _f in _COLUMNS] + [t for _k, t in _SELECTIVE_COLUMNS]
    head = f"| {first} | " + " | ".join(titles) + " |"
    rule = "|---" * (len(titles) + 1) + "|"
    return [head, rule, *[_row(label, s) for label, s in rows]]


def _skill_section(report: Mapping[str, Any]) -> list[str]:
    """The headline calibration number: Brier skill against the negative controls."""
    pairs = [
        ("uncalibrated", report.get("skill", {})),
        ("calibrated", report.get("skill_calibrated", {})),
    ]
    if not any(block for _label, block in pairs):
        return []
    out = [
        "",
        "## Skill against the controls",
        "",
        "`BSS = 1 - Brier / Brier_control`. 1.0 is perfect, 0.0 is no better than the control, "
        "negative is worse. **This is the headline calibration number**, and ECE sits beside it "
        "rather than above it, because ECE can be won by a predictor that never commits: the "
        "base-rate control does exactly that on this set. A proper scoring rule cannot be won "
        "that way, so stating the result as skill *against that control* puts the comparison in "
        "the open.",
        "",
        "| model | control | BSS (multiclass) | BSS (top label) | accuracy gain |",
        "|---|---|---|---|---|",
    ]
    for label, block in pairs:
        for control, skill in block.items():

            def fmt(value: float | None) -> str:
                return "n/a" if value is None else f"{value:+.4f}"

            out.append(
                f"| {label} | {control} | {fmt(skill.get('brier_multiclass'))} | "
                f"{fmt(skill.get('brier_top_label'))} | {fmt(skill.get('accuracy_gain'))} |"
            )
    return out


def _risk_coverage_section(report: Mapping[str, Any]) -> list[str]:
    """Accuracy as a function of how much of the set is answered."""
    curves = [
        ("uncalibrated", report.get("risk_coverage", {})),
        ("calibrated", report.get("risk_coverage_calibrated", {})),
    ]
    curves = [(label, curve) for label, curve in curves if curve.get("curve")]
    if not curves:
        return []
    out = [
        "",
        "## Risk-coverage",
        "",
        "Questions are answered most-confident-first; at coverage *c* the least confident "
        "`1 - c` are abstained on. This is the deployment question — *if the least confident "
        "20% go to a human, how good is what is left?* — and it depends only on the **order** "
        "of the confidences, not on their values, which makes it a second opinion rather than a "
        "restatement of the reliability diagram. Calibration still moves it, because our "
        "temperatures are fitted per option-count bucket and therefore re-rank questions across "
        "buckets; a single global temperature would leave these rows identical. "
        "`AURC` is the area under the risk curve; lower is better.",
        "",
        "| model | " + " | ".join(f"acc@{c:.0%}" for c in (0.2, 0.4, 0.6, 0.8, 1.0)) + " | AURC |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, curve in curves:
        by_coverage = {round(point["coverage"], 2): point["accuracy"] for point in curve["curve"]}
        cells = [
            f"{by_coverage[c]:.4f}" if c in by_coverage else "n/a"
            for c in (0.2, 0.4, 0.6, 0.8, 1.0)
        ]
        aurc = curve.get("aurc")
        out.append(
            f"| {label} | " + " | ".join(cells) + f" | {'n/a' if aurc is None else f'{aurc:.4f}'} |"
        )
    out += ["", "Drawn in `risk-coverage.png`, with the negative controls on the same axes."]
    return out


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
        *_stale_format_banner(meta.get("format_version")),
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
        "without accuracy, Brier and AUROC beside it — and why the section below reports skill "
        "against those controls rather than ECE alone.",
        *_skill_section(report),
        *_risk_coverage_section(report),
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
        "- `reliability-*.png`, `risk-coverage.png` — drawn from `metrics.json`.",
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
    targets = payload["targets"]
    retired = payload["retired_target"]
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
        "## Targets",
        "",
        "| target | measured | threshold | met |",
        "|---|---|---|---|",
    ]
    for result in targets.values():
        if not isinstance(result, dict) or "threshold" not in result:
            continue
        value = result["measured"]
        out.append(
            f"| {result['description']} | {value:.3f} | {result['direction']} "
            f"{result['threshold']} | {'yes' if result['met'] else '**no**'} |"
        )
    amortised = targets.get("amortised_ms_64", {}).get("measured")
    out += [
        "",
        f"**All targets met: {'yes' if targets.get('all_met') else 'no'}.**"
        + (f" Amortised cost at 64 questions: {amortised:.0f} ms/question." if amortised else ""),
        "",
        "The two format-overhead rows differ by the base model's chat tail "
        "(`<|im_end|>...<think></think>`), which its template imposes and no change to our "
        "format removes. The target applies to the controllable figure (ADR 0003).",
        "",
        f"The retired rule (`{retired['rule']}`) would read "
        f"**{retired['ratio_64_over_1']:.2f}x**. ADR 0003 explains why it was replaced: it is met "
        "only when `prefix_tokens >~ 61 x tokens_per_question`, which measures the benchmark's "
        "state size more than the engine.",
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
        "## Format overhead",
        "",
        "Boilerplate is what the renderer adds; content is the caller's instruction and option "
        "labels, which no format change can remove. Boilerplate splits into the chat tail, which "
        "the base model's template imposes, and **ours** — the headers and answer block, which is "
        "the only part ADR 0003 option B can shrink.",
        "",
        "| primitive | suffix tok | content | ours | chat tail | boilerplate | controllable |",
        "|---|---|---|---|---|---|---|",
    ]
    for qtype, b in payload.get("format_overhead", {}).get("by_qtype", {}).items():
        out.append(
            f"| {qtype} | {b['suffix_tokens']} | {b['content_tokens']} | "
            f"{b['our_overhead_tokens']} | {b['tail_tokens']} | "
            f"{b['boilerplate_fraction']:.0%} | {b['controllable_fraction']:.0%} |"
        )

    out += [
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
