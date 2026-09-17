"""Run an evaluation and write ``results/<run_id>/``.

The run is split in two halves that can be executed independently:

1. **Score.** Bundle questions by state, prefill each state once, and store the raw masked
   logits in ``predictions-<split>.jsonl``. This is the only part that needs a GPU.
2. **Report.** Fit temperature scaling on the validation predictions, apply it to test, and
   write ``metrics.json``, ``calibration.json`` and the figures.

Because step 2 reads only stored logits, ``--rescore`` redoes all of it — different bins,
different controls, a fixed metric — without touching the model. Every number that ends up in
a document comes from ``metrics.json``; none is ever typed by hand.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.data import Coverage, EvalItem, bundle_stats, group_by_state, load_system_one_decisions
from eval.metrics import DEFAULT_BINS, Prediction, build_report
from s1decide.calibrate import Calibration, fit_calibration
from s1decide.prompt import DEFAULT_TEMPLATE, FORMAT_VERSION, ChatTemplate, render

__all__ = ["RunConfig", "main", "report_from_predictions", "run", "score_items"]

DEFAULT_MODEL = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit"


@dataclass(frozen=True)
class RunConfig:
    """Everything that defines a run, recorded verbatim in ``metrics.json``.

    Attributes:
        model: Model id, or ``"mock"``.
        engine: ``hf`` or ``mock``.
        quantization: Label for the deployment quantization, e.g. ``nf4-bf16``. Calibration
            is tied to it.
        split: Split to report on.
        val_split: Split the temperature and the base-rate control are fitted on.
        max_options: Questions above this are dropped and the drop reported.
        limit: Optional cap on items per split, for smoke runs.
        n_bins: Equal-mass bins.
        reuse_buffer: Whether the engine keeps one pre-expanded broadcast cache across passes
            (ADR 0003 option C). Recorded because it is an engine change, and the condition on
            option C is that it must not move the metrics.
        run_id: Output directory name under ``results/``.
    """

    model: str = DEFAULT_MODEL
    engine: str = "hf"
    quantization: str = "nf4-bf16"
    split: str = "test"
    val_split: str = "val"
    max_options: int = 26
    limit: int | None = None
    n_bins: int = DEFAULT_BINS
    reuse_buffer: bool = True
    run_id: str = ""

    def to_json(self) -> dict[str, Any]:
        """Serialise for the report's ``meta``."""
        return {
            "model": self.model,
            "engine": self.engine,
            "quantization": self.quantization,
            "split": self.split,
            "val_split": self.val_split,
            "max_options": self.max_options,
            "limit": self.limit,
            "n_bins": self.n_bins,
            "reuse_buffer": self.reuse_buffer,
        }


def git_provenance() -> dict[str, Any]:
    """Current commit and whether the tree is dirty — so a result can be reproduced."""

    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=30, check=True
            ).stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return None

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def default_run_id(config: RunConfig) -> str:
    """A run id that sorts by time and names what was run."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    model = config.model.rsplit("/", 1)[-1].lower().replace(".", "").replace("_", "-")
    return f"{stamp}-{model}-{config.quantization}-{config.split}-zeroshot"


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    """Write UTF-8 JSONL with LF endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read UTF-8 JSONL."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_engine(config: RunConfig) -> Any:
    """Construct the engine named by ``config``.

    Returns:
        An :class:`~s1decide.engine.base.Engine`.
    """
    if config.engine == "mock":
        from s1decide.engine.mock import MockEngine

        return MockEngine(seed=0)
    if config.engine == "hf":
        import torch

        from s1decide.engine.hf import HFEngine

        return HFEngine.from_pretrained(
            config.model,
            dtype=torch.bfloat16,
            device_map="cuda",
            local_files_only=True,
            reuse_buffer=config.reuse_buffer,
        )
    raise ValueError(f"unknown engine {config.engine!r}; expected 'hf' or 'mock'")


def score_items(
    engine: Any,
    items: Sequence[EvalItem],
    *,
    template: ChatTemplate = DEFAULT_TEMPLATE,
    progress_every: int = 50,
    label: str = "",
) -> tuple[list[Prediction], dict[str, Any]]:
    """Score every item, bundling questions that share a state into one engine call.

    Args:
        engine: Any engine.
        items: Normalised items.
        template: Chat template to render with.
        progress_every: Print a progress line every N states; 0 disables.
        label: Prefix for progress lines.

    Returns:
        ``(predictions, timing)`` with predictions in the order the items were given.
    """
    groups = group_by_state(items)
    by_id: dict[str, Prediction] = {}
    started = time.perf_counter()
    engine_seconds = 0.0

    for index, (state, group) in enumerate(groups, start=1):
        questions = [item.to_question(name=item.id) for item in group]
        rendered = render(state, questions, template=template)
        call_started = time.perf_counter()
        output = engine.score(rendered.prefix, rendered.suffixes, rendered.labels)
        engine_seconds += time.perf_counter() - call_started
        for item, logits in zip(group, output.logits):
            by_id[item.id] = Prediction(
                id=item.id,
                family=item.family,
                qtype=item.qtype,
                logits=tuple(float(x) for x in logits),
                answer_idx=item.answer_idx,
            )
        if progress_every and (index % progress_every == 0 or index == len(groups)):
            elapsed = time.perf_counter() - started
            rate = index / elapsed
            print(
                f"  {label}{index}/{len(groups)} states  {len(by_id)}/{len(items)} questions  "
                f"{elapsed:6.1f}s  {rate:5.2f} states/s  eta {(len(groups) - index) / rate:6.1f}s",
                flush=True,
            )

    predictions = [by_id[item.id] for item in items]
    wall = time.perf_counter() - started
    return predictions, {
        "states": len(groups),
        "questions": len(predictions),
        "wall_seconds": wall,
        "engine_seconds": engine_seconds,
        "seconds_per_question": wall / len(predictions) if predictions else 0.0,
        "bundling": bundle_stats(groups),
    }


def report_from_predictions(
    *,
    config: RunConfig,
    test_predictions: Sequence[Prediction],
    val_predictions: Sequence[Prediction],
    calibration: Calibration,
    coverage: dict[str, Coverage],
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the full ``metrics.json`` payload from stored predictions.

    Produces the uncalibrated report, the calibrated report, per-bucket and per-family
    breakdowns, and both negative controls. Pure function of its inputs — no model, no GPU.
    """
    temperatures = dict(calibration.temperatures)
    uncalibrated = build_report(
        test_predictions,
        temperature=1.0,
        control_fit=val_predictions,
        n_bins=config.n_bins,
    )
    calibrated = build_report(
        test_predictions,
        temperature=temperatures,
        control_fit=None,
        n_bins=config.n_bins,
    )
    val_uncalibrated = build_report(val_predictions, temperature=1.0, n_bins=config.n_bins)
    val_calibrated = build_report(val_predictions, temperature=temperatures, n_bins=config.n_bins)

    meta = {
        "run_id": config.run_id,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        **config.to_json(),
        "format_version": FORMAT_VERSION,
        "template": DEFAULT_TEMPLATE.name,
        "dataset": "pngwn/system-one-decisions",
        "dataset_license": "CC-BY-NC-4.0",
        "git": git_provenance(),
        "platform": f"{platform.system()} {platform.release()} · Python {platform.python_version()}",
        **(extra_meta or {}),
    }

    return {
        "meta": meta,
        "coverage": {name: c.to_json() for name, c in coverage.items()},
        "calibration": calibration.to_json(),
        "overall": uncalibrated.overall,
        "overall_calibrated": calibrated.overall,
        "by_option_count": uncalibrated.by_option_count,
        "by_option_count_calibrated": calibrated.by_option_count,
        "by_family": uncalibrated.by_family,
        "by_family_calibrated": calibrated.by_family,
        "by_qtype": uncalibrated.by_qtype,
        "by_qtype_calibrated": calibrated.by_qtype,
        "controls": uncalibrated.controls,
        "validation": {
            "overall": val_uncalibrated.overall,
            "overall_calibrated": val_calibrated.overall,
        },
    }


def _headline(report: dict[str, Any]) -> str:
    """One-line summary for the console. The file is the source of truth, not this."""

    def row(name: str, s: dict[str, Any]) -> str:
        auroc = s.get("auroc_confidence")
        return (
            f"  {name:22} n={s['n']:5d}  acc={s['accuracy']:.4f}  ECE={s['ece']:.4f}  "
            f"Brier={s['brier_top_label']:.4f}  NLL={s['nll']:.4f}  "
            f"AUROC={'n/a' if auroc is None else f'{auroc:.4f}'}"
        )

    lines = [
        row("test uncalibrated", report["overall"]),
        row("test calibrated", report["overall_calibrated"]),
    ]
    for name, s in report.get("controls", {}).items():
        lines.append(row(f"control: {name}", s))
    return "\n".join(lines)


def run(config: RunConfig, out_root: Path | None = None) -> Path:
    """Score both splits, fit calibration, write everything.

    Args:
        config: What to run.
        out_root: Root for ``results/``; defaults to the repository's.

    Returns:
        The run directory.
    """
    from s1decide.tasks import repo_root

    resolved = (
        config
        if config.run_id
        else RunConfig(**{**config.__dict__, "run_id": default_run_id(config)})
    )
    root = out_root or (repo_root() / "results")
    directory = root / resolved.run_id
    directory.mkdir(parents=True, exist_ok=True)
    print(f"run {resolved.run_id}\n  -> {directory}")

    val_items, val_coverage = load_system_one_decisions(
        resolved.val_split, max_options=resolved.max_options, limit=resolved.limit
    )
    test_items, test_coverage = load_system_one_decisions(
        resolved.split, max_options=resolved.max_options, limit=resolved.limit
    )
    print(
        f"  {resolved.val_split}: {val_coverage.kept}/{val_coverage.total} kept "
        f"({val_coverage.fraction_kept:.1%}), {resolved.split}: {test_coverage.kept}/"
        f"{test_coverage.total} kept ({test_coverage.fraction_kept:.1%}); "
        f"dropped >{resolved.max_options} options: {test_coverage.dropped_families}"
    )

    engine = build_engine(resolved)
    print(f"  engine: {engine.name} ({resolved.model})")

    val_predictions, val_timing = score_items(engine, val_items, label=f"{resolved.val_split} ")
    write_jsonl(
        directory / f"predictions-{resolved.val_split}.jsonl",
        [p.to_json() for p in val_predictions],
    )
    test_predictions, test_timing = score_items(engine, test_items, label=f"{resolved.split} ")
    write_jsonl(
        directory / f"predictions-{resolved.split}.jsonl",
        [p.to_json() for p in test_predictions],
    )

    calibration = fit_calibration(
        val_predictions,
        quantization=resolved.quantization,
        meta={"run_id": resolved.run_id, "fit_split": resolved.val_split, "model": resolved.model},
    )
    calibration.save(directory / "calibration.json")
    print(f"  temperatures: {calibration.temperatures}")

    report = report_from_predictions(
        config=resolved,
        test_predictions=test_predictions,
        val_predictions=val_predictions,
        calibration=calibration,
        coverage={resolved.val_split: val_coverage, resolved.split: test_coverage},
        extra_meta={"timing": {resolved.val_split: val_timing, resolved.split: test_timing}},
    )
    (directory / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(_headline(report))

    from eval.summary import write_summary

    try:
        from eval.plots import plot_run

        for path in plot_run(directory):
            print(f"  wrote {path.name}")
    except ImportError as exc:
        print(f"  plots skipped: {exc}")
    print(f"  wrote {write_summary(directory).name}")
    return directory


def rescore(run_dir: Path, n_bins: int = DEFAULT_BINS) -> Path:
    """Recompute ``metrics.json`` and the figures from stored predictions. No model needed.

    Args:
        run_dir: An existing ``results/<run_id>`` directory.
        n_bins: Equal-mass bins to use this time.

    Returns:
        The run directory.
    """
    directory = Path(run_dir)
    previous = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    meta = previous["meta"]
    config = RunConfig(
        model=meta["model"],
        engine=meta["engine"],
        quantization=meta["quantization"],
        split=meta["split"],
        val_split=meta["val_split"],
        max_options=meta["max_options"],
        limit=meta.get("limit"),
        n_bins=n_bins,
        reuse_buffer=meta.get("reuse_buffer", True),
        run_id=meta["run_id"],
    )
    test_predictions = [
        Prediction.from_json(r) for r in read_jsonl(directory / f"predictions-{config.split}.jsonl")
    ]
    val_predictions = [
        Prediction.from_json(r)
        for r in read_jsonl(directory / f"predictions-{config.val_split}.jsonl")
    ]
    calibration = Calibration.load(directory / "calibration.json")
    coverage = {
        name: Coverage(
            total=c["total"],
            kept=c["kept"],
            dropped_high_cardinality=c["dropped_high_cardinality"],
            max_options_kept=c["max_options_kept"],
            dropped_families=c["dropped_families"],
        )
        for name, c in previous["coverage"].items()
    }
    report = report_from_predictions(
        config=config,
        test_predictions=test_predictions,
        val_predictions=val_predictions,
        calibration=calibration,
        coverage=coverage,
        extra_meta={"timing": meta.get("timing", {}), "rescored": True},
    )
    (directory / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(_headline(report))
    from eval.plots import plot_run
    from eval.summary import write_summary

    for path in plot_run(directory):
        print(f"  wrote {path.name}")
    print(f"  wrote {write_summary(directory).name}")
    return directory


def main(argv: Sequence[str] | None = None) -> int:
    """Command line for ``uv run task eval``."""
    parser = argparse.ArgumentParser(description="Run a decision evaluation")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--engine", default="hf", choices=("hf", "mock"))
    parser.add_argument("--quantization", default="nf4-bf16")
    parser.add_argument("--split", default="test")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--max-options", type=int, default=26)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--no-buffer-reuse",
        action="store_true",
        help="deep-copy the prefix cache each pass instead of reusing one expanded buffer",
    )
    parser.add_argument("--rescore", default=None, help="recompute metrics for an existing run dir")
    parser.add_argument("--summary", default=None, help="regenerate SUMMARY.md for a run dir")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.summary:
        from eval.summary import write_summary

        print(f"wrote {write_summary(Path(args.summary))}")
        return 0
    if args.rescore:
        rescore(Path(args.rescore), n_bins=args.bins)
        return 0

    run(
        RunConfig(
            model=args.model,
            engine=args.engine,
            quantization=args.quantization,
            split=args.split,
            val_split=args.val_split,
            max_options=args.max_options,
            limit=args.limit,
            n_bins=args.bins,
            reuse_buffer=not args.no_buffer_reuse,
            run_id=args.run_id,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
