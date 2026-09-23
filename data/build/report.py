"""Render ``data/processed/manifest.json`` as ``docs/dataset-build.md``.

Generated, never hand-written: every count here comes from the build. The hand-written licence
reasoning stays in ``docs/dataset-card.md``, which links here for the numbers — so prose and
figures cannot drift apart.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = ["render_build_report", "write_build_report"]

_SPLIT_ORDER = ("train", "val", "test", "eval")


def render_build_report(manifest: Mapping[str, Any]) -> str:
    """Render the Markdown body for one build manifest."""
    splits = manifest["splits"]
    out = [
        "# Built dataset",
        "",
        "<!-- GENERATED from data/processed/manifest.json by data/build/report.py.",
        "     Rebuild with: uv run task data -->",
        "",
        f"**{manifest['rows']:,} rows**, seed `{manifest['seed']}`. Licence reasoning and the "
        "eval-only rules are in [dataset-card.md](dataset-card.md); this file is the counts.",
        "",
        "## Splits",
        "",
        "| split | rows | purpose |",
        "|---|---|---|",
    ]
    purpose = {
        "train": "S1 training",
        "val": "calibration fitting and model selection",
        "test": "Tier-1 in-distribution report",
        "eval": "Tier-1 held-out families and OOD sets",
    }
    for split in _SPLIT_ORDER:
        if split in splits:
            out.append(f"| `{split}` | {splits[split]:,} | {purpose.get(split, '')} |")

    out += [
        "",
        "## Families",
        "",
        "A family in `eval` only is either a **held-out family** (in-distribution, never "
        "trained on) or an **OOD set**. Both are Tier 1.",
        "",
        "| family | licence | role | primitive | "
        + " | ".join(f"`{s}`" for s in _SPLIT_ORDER)
        + " |",
        "|---|---|---|---|" + "---|" * len(_SPLIT_ORDER),
    ]
    for family, counts in manifest["by_family"].items():
        source = manifest["sources"].get(family, {})
        cells = " | ".join(f"{counts.get(s, 0):,}" if counts.get(s) else "—" for s in _SPLIT_ORDER)
        out.append(
            f"| {family} | `{source.get('license', '?')}` | {source.get('role', '?')} | "
            f"{source.get('primitive', '?')} | {cells} |"
        )

    out += [
        "",
        "## Sources",
        "",
        "| family | loaded from | audited as | rows loaded | rows kept | note |",
        "|---|---|---|---|---|---|",
    ]
    for family, source in manifest["sources"].items():
        loaded, canonical = source["repo"], source["canonical_repo"]
        audited = f"`{canonical}`" if canonical != loaded else "same"
        out.append(
            f"| {family} | `{loaded}` | {audited} | {source['rows_loaded']:,} | "
            f"{source['rows_kept']:,} | {source['note']} |"
        )

    leakage = manifest.get("leakage", {})
    out += [
        "",
        "## Two-stage expansion",
        "",
        f"**{manifest['two_stage_parents_expanded']:,} questions** had more options than the 26 "
        "single-token labels and were expanded into the ADR 0001 two-stage form: one yes/no row "
        "per sampled candidate, plus one Choice over a shortlist containing the answer. Stage-1 "
        "rows are rendered byte-identically to `render_stage1`, so training and inference see "
        "the same prompt.",
        "",
        "## Tier-2 leakage guard",
        "",
    ]
    if not leakage.get("checked"):
        # The pipeline raises before a manifest like this can exist; reaching here means an old
        # manifest, and it must not be rendered as though the guard had run.
        raise ValueError("manifest records an unchecked Tier-2 leakage guard; rebuild the data")
    out += [
        f"`pngwn/system-one-decisions` is eval-only (ADR 0004) but is *derived from* sources "
        f"we now train on, so overlap is possible even though the datasets differ. Comparing "
        f"**state hashes** against our {leakage['train_states']:,} training states finds "
        f"**{leakage['leaked_rows']:,} of {leakage['external_rows']:,}** external rows "
        f"({leakage['leaked_rows'] / max(1, leakage['external_rows']):.1%}) that must be "
        f"dropped before any Tier-2 number is reported.",
        "",
        f"Affected families: {', '.join(f'`{f}`' for f in leakage['leaked_families']) or 'none'}.",
    ]

    out += _effective_mix_section(manifest)

    out += [
        "",
        "## Licences present",
        "",
        "| source | licence | verdict | rows |",
        "|---|---|---|---|",
    ]
    for source, info in manifest["licences"].items():
        out.append(f"| `{source}` | `{info['license']}` | {info['verdict']} | {info['rows']:,} |")
    out += [
        "",
        "Every row in `train` and `val` carries a licence from the ADR 0004 allowlist; the build "
        "fails otherwise, before anything is written.",
        "",
    ]
    return "\n".join(out)


def _effective_mix_section(manifest: Mapping[str, Any]) -> list[str]:
    """The training mix as rows and as the model actually sees it.

    These are different numbers and the difference is the point. Expansion makes the corpus
    three-quarters stage-1 rows; the sampler is weighted so the model does not spend three
    quarters of its time on one task. Reporting only the first would misdescribe the training
    run, and reporting only the second would hide what the corpus contains.
    """
    table = manifest.get("mix_table")
    mix = manifest.get("effective_mix")
    if not table or not mix:
        return []

    subsample = manifest.get("stage1_subsampling", {})
    weighting = manifest.get("weighting", {})

    out = [
        "",
        "## Effective training mix",
        "",
        f"Stage-1 questions keep the positive plus **{subsample.get('negatives_per_question', '?')} "
        f"negatives** in `train` only, chosen "
        f"**{'by zero-shot P(yes)' if subsample.get('hard_negative_source') == 'zero_shot_scores' else 'at random'}**. "
        f"`val` and `test` keep the **full fan-out**, because the ratio they carry "
        "(~96:1 no:yes) is the one the deployed two-stage path faces, and a temperature fitted "
        "on a subsample would be fitted to a distribution we never serve.",
        "",
        "| group | rows | raw share | effective share | target | oversample |",
        "|---|---|---|---|---|---|",
    ]
    groups = weighting.get("groups", {})
    for group in sorted(mix):
        info = mix[group]
        g = groups.get(group, {})
        target = f"{g['target_share']:.0%}" if "target_share" in g else "—"
        factor = f"{g['oversample']:.2f}x" if "oversample" in g else "—"
        out.append(
            f"| `{group}` | {info['rows']:,} | {info['raw_share']:.1%} | "
            f"**{info['effective_share']:.1%}** | {target} | {factor} |"
        )

    for clip in weighting.get("clipped", []):
        out += [
            "",
            f"> **`{clip['group']}` could not reach its {clip['target_share']:.0%} target.** It "
            f"would need {clip['needed_oversample']:.1f}x oversampling against a natural share of "
            f"{clip['natural_share']:.1%}, and the cap is {clip['capped_at']:.0f}x. Past that a "
            "small set is being memorised rather than learned. The fix is more data, not a "
            "bigger weight.",
        ]

    out += [
        "",
        "### Per primitive, stage and family",
        "",
        "`rows before` is the full fan-out the expansion would have produced; `rows after` is "
        "what training keeps. `effective` is the share of the weighted draw.",
        "",
        "| primitive | stage | family | rows before | rows after | raw | effective |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in table:
        out.append(
            f"| `{entry['primitive']}` | {entry['stage']} | `{entry['family']}` | "
            f"{entry['rows_before']:,} | {entry['rows_after']:,} | "
            f"{entry['raw_share']:.1%} | **{entry['effective_share']:.1%}** |"
        )
    out.append("")
    return out


def write_build_report(manifest_path: str | Path, out_path: str | Path) -> Path:
    """Write ``docs/dataset-build.md`` from a manifest."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_build_report(manifest), encoding="utf-8", newline="\n")
    return target
