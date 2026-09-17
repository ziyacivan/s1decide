"""Latency versus number of questions, on a fixed state.

The claim this project has to defend is that asking more questions about the same state costs
almost nothing extra: prefill once, broadcast, answer everything in one pass. ``CLAUDE.md``
turns that into a target — **a 64-question call under 2x the 1-question call**.

So the benchmark measures three things, not one:

1. **The curve**: median wall time at 1 / 4 / 16 / 64 questions, over ``repeats`` runs.
2. **What bundling actually saves**: the same questions asked one call at a time. This is the
   honest comparison for a user, and it is where the design wins even if the 2x bar is missed.
3. **A decomposition**, so a miss is diagnosed rather than guessed at. The suffix phase is
   modelled as ``a * passes + b * suffix_tokens`` and fitted by least squares: ``a`` is what a
   forward pass costs before it processes anything (weight streaming, launch overhead), ``b``
   is the marginal cost of a suffix token. Which term dominates decides what to fix.

Everything is written to ``results/<run_id>/latency.json``; the plot and any prose are drawn
from that file.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from s1decide.primitives import Choice, Noul, Question, Score
from s1decide.prompt import FORMAT_VERSION, render

__all__ = ["BenchConfig", "Measurement", "main", "make_questions", "make_state", "run"]

GIB = 2**30

#: A fixed, deterministic paragraph repeated to reach the target state length. Chosen to look
#: like the support tickets the eval uses, so token counts are representative.
_PARAGRAPH = (
    "Ticket {i}: the customer reports that their subscription was charged twice on the same "
    "day. They attached two bank statements and ask for a refund of the duplicate charge. "
    "Account age three years, plan Pro monthly, auto-renew enabled, no prior contact in the "
    "last twelve months. Agent note: verified both charges in the billing system. "
)


@dataclass(frozen=True)
class BenchConfig:
    """What to benchmark.

    Attributes:
        model: Model id to load.
        quantization: Label recorded in the output.
        state_tokens: Target length of the shared state, in tokens.
        question_counts: Question counts to measure.
        repeats: Timed runs per point (median is reported).
        warmup: Untimed runs per point, to let Triton compile and the allocator settle.
        separate_repeats: Timed runs for the one-call-per-question comparison, which is much
            slower and needs fewer samples to make its point.
        run_id: Output directory name under ``results/``.
    """

    model: str = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit"
    quantization: str = "nf4-bf16"
    state_tokens: int = 1500
    question_counts: tuple[int, ...] = (1, 4, 16, 64)
    repeats: int = 20
    warmup: int = 2
    separate_repeats: int = 2
    run_id: str = ""

    def to_json(self) -> dict[str, Any]:
        """Serialise for the output file."""
        return asdict(self)


@dataclass
class Measurement:
    """Timings for one question count.

    Attributes:
        n_questions: How many questions were asked in the call.
        wall_ms: Every timed sample, in milliseconds.
        prefill_ms: Prefill portion of each sample.
        suffix_ms: Suffix portion of each sample.
        passes: Forward passes the engine needed for the suffixes.
        rows_per_pass: Rows the engine judged it could afford per pass.
        prefix_tokens: Tokens in the shared prefix.
        suffix_tokens: Total tokens across all suffixes.
        peak_gib: Peak allocated VRAM during a timed sample.
        separate_ms: Samples for the same questions asked one call at a time, if measured.
    """

    n_questions: int
    wall_ms: list[float] = field(default_factory=list)
    prefill_ms: list[float] = field(default_factory=list)
    suffix_ms: list[float] = field(default_factory=list)
    passes: int = 0
    rows_per_pass: int = 0
    prefix_tokens: int = 0
    suffix_tokens: int = 0
    peak_gib: float = 0.0
    separate_ms: list[float] = field(default_factory=list)

    @property
    def median_ms(self) -> float:
        """Median wall time of the bundled call."""
        return statistics.median(self.wall_ms)

    @property
    def median_separate_ms(self) -> float | None:
        """Median wall time of the same questions asked one at a time."""
        return statistics.median(self.separate_ms) if self.separate_ms else None

    def to_json(self) -> dict[str, Any]:
        """Serialise, with the summary statistics precomputed."""
        quantiles = sorted(self.wall_ms)
        return {
            "n_questions": self.n_questions,
            "median_ms": self.median_ms,
            "min_ms": min(self.wall_ms),
            "max_ms": max(self.wall_ms),
            "p10_ms": quantiles[max(0, int(0.10 * (len(quantiles) - 1)))],
            "p90_ms": quantiles[min(len(quantiles) - 1, int(0.90 * (len(quantiles) - 1)))],
            "stdev_ms": statistics.stdev(self.wall_ms) if len(self.wall_ms) > 1 else 0.0,
            "median_prefill_ms": statistics.median(self.prefill_ms),
            "median_suffix_ms": statistics.median(self.suffix_ms),
            "median_separate_ms": self.median_separate_ms,
            "passes": self.passes,
            "rows_per_pass": self.rows_per_pass,
            "prefix_tokens": self.prefix_tokens,
            "suffix_tokens": self.suffix_tokens,
            "ms_per_suffix_token": statistics.median(self.suffix_ms) / max(1, self.suffix_tokens),
            "peak_gib": self.peak_gib,
            "samples": len(self.wall_ms),
            "wall_ms": self.wall_ms,
        }


def make_state(encode, target_tokens: int) -> str:
    """Build a deterministic state of about ``target_tokens`` tokens.

    Args:
        encode: Callable turning text into token ids.
        target_tokens: Desired length.

    Returns:
        The state text.
    """
    text, i = "", 0
    while len(encode(text)) < target_tokens:
        text += _PARAGRAPH.format(i=i)
        i += 1
    return text.strip()


def make_questions(n: int) -> list[Question]:
    """Build ``n`` questions mixing the three primitives, as a real call would.

    The cycle is Noul, Choice, Score so that suffix lengths vary the way they do in practice
    rather than all being the cheapest shape.

    Args:
        n: How many questions.

    Returns:
        Validated questions with unique names.
    """
    out: list[Question] = []
    for i in range(n):
        kind = i % 3
        if kind == 0:
            spec: Choice | Score | Noul = Noul(
                instructions=f"Statement {i}: the customer is asking for a refund."
            )
        elif kind == 1:
            spec = Choice(
                instructions=f"Question {i}: which team should handle this ticket?",
                options=("billing", "technical support", "account management", "fraud"),
            )
        else:
            spec = Score(
                instructions=f"Question {i}: how urgent is this ticket?",
                levels=("can wait", "this week", "today", "immediately"),
            )
        out.append(Question(name=f"q{i:03d}", spec=spec))
    return out


def _time_call(engine: Any, state: str, questions: Sequence[Question]) -> tuple[float, Any]:
    """Run one bundled call and return ``(wall_ms, engine_meta)``."""
    rendered = render(state, list(questions))
    started = time.perf_counter()
    output = engine.score(rendered.prefix, rendered.suffixes, rendered.labels)
    return (time.perf_counter() - started) * 1000.0, output.meta


def _time_separate(engine: Any, state: str, questions: Sequence[Question]) -> float:
    """Run each question as its own call and return the total wall time in milliseconds."""
    started = time.perf_counter()
    for question in questions:
        rendered = render(state, [question])
        engine.score(rendered.prefix, rendered.suffixes, rendered.labels)
    return (time.perf_counter() - started) * 1000.0


def _fit_two_term(measurements: Sequence[Measurement]) -> dict[str, float]:
    """Least-squares fit of ``suffix_ms ~ a * passes + b * suffix_tokens``.

    Returns:
        ``{"per_pass_ms", "per_token_ms", "r_squared"}``. With two free parameters and four
        points this is a sanity check on where the time goes, not a precision instrument.
    """
    import numpy as np

    design = np.array([[m.passes, m.suffix_tokens] for m in measurements], dtype=float)
    observed = np.array([statistics.median(m.suffix_ms) for m in measurements], dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, observed, rcond=None)
    predicted = design @ coefficients
    residual = float(((observed - predicted) ** 2).sum())
    total = float(((observed - observed.mean()) ** 2).sum())
    return {
        "per_pass_ms": float(coefficients[0]),
        "per_token_ms": float(coefficients[1]),
        "r_squared": 1.0 - residual / total if total > 0 else float("nan"),
    }


def run(config: BenchConfig, out_root: Path | None = None) -> Path:
    """Run the benchmark and write ``latency.json`` plus the plot.

    Args:
        config: What to measure.
        out_root: Root for ``results/``; defaults to the repository's.

    Returns:
        The run directory.
    """
    import torch

    from s1decide.engine.hf import HFEngine
    from s1decide.tasks import repo_root

    run_id = config.run_id or (
        datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        + f"-{config.model.rsplit('/', 1)[-1].lower().replace('.', '')}-{config.quantization}-latency"
    )
    directory = (out_root or (repo_root() / "results")) / run_id
    directory.mkdir(parents=True, exist_ok=True)
    print(f"bench {run_id}\n  -> {directory}")

    engine = HFEngine.from_pretrained(
        config.model, dtype=torch.bfloat16, device_map="cuda", local_files_only=True
    )
    state = make_state(engine.encode, config.state_tokens)
    print(f"  engine {engine.name} | state {len(engine.encode(state))} tokens")

    measurements: list[Measurement] = []
    for n in config.question_counts:
        questions = make_questions(n)
        for _ in range(config.warmup):
            _time_call(engine, state, questions)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        measurement = Measurement(n_questions=n)
        for _ in range(config.repeats):
            wall, meta = _time_call(engine, state, questions)
            measurement.wall_ms.append(wall)
            measurement.prefill_ms.append(meta["prefill_seconds"] * 1000.0)
            measurement.suffix_ms.append(meta["suffix_seconds"] * 1000.0)
            measurement.passes = meta["passes"]
            measurement.rows_per_pass = meta["rows_per_pass"]
            measurement.prefix_tokens = meta["prefix_tokens"]
            measurement.suffix_tokens = sum(meta["suffix_tokens"])
        measurement.peak_gib = torch.cuda.max_memory_allocated() / GIB

        for _ in range(config.separate_repeats):
            measurement.separate_ms.append(_time_separate(engine, state, questions))

        separate = measurement.median_separate_ms
        print(
            f"  n={n:3d}  median {measurement.median_ms:8.1f} ms  "
            f"(prefill {statistics.median(measurement.prefill_ms):7.1f} + suffix "
            f"{statistics.median(measurement.suffix_ms):7.1f})  "
            f"{measurement.passes} pass(es) x {measurement.rows_per_pass} rows  "
            f"{measurement.suffix_tokens:5d} suffix tok  peak {measurement.peak_gib:.2f} GiB  "
            f"separate {separate:8.1f} ms  speedup {separate / measurement.median_ms:5.2f}x",
            flush=True,
        )
        measurements.append(measurement)

    baseline = measurements[0].median_ms
    largest = measurements[-1]
    payload = {
        "meta": {
            # config first: its `run_id` field is the *requested* one and may be empty.
            **config.to_json(),
            "run_id": run_id,
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "format_version": FORMAT_VERSION,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "platform": f"{platform.system()} {platform.version()}",
            # The prefix is the system prompt plus the state block, so it is longer than
            # `state_tokens` asked for; report what the model actually prefilled.
            "prefix_tokens": measurements[0].prefix_tokens,
            "state_tokens_actual": len(engine.encode(state)),
        },
        "target": {
            "rule": "64-question call < 2x the 1-question call",
            "ratio_64_over_1": largest.median_ms / baseline,
            "met": largest.median_ms < 2.0 * baseline,
        },
        "decomposition": _fit_two_term(measurements),
        "points": [m.to_json() for m in measurements],
    }
    (directory / "latency.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    ratio = payload["target"]["ratio_64_over_1"]
    print(
        f"  target: {largest.n_questions}-question / 1-question = {ratio:.2f}x "
        f"-> {'MET' if payload['target']['met'] else 'MISSED'} (bar is 2.00x)"
    )
    fit = payload["decomposition"]
    print(
        f"  decomposition: {fit['per_pass_ms']:.1f} ms per forward pass + "
        f"{fit['per_token_ms']:.3f} ms per suffix token (R^2={fit['r_squared']:.3f})"
    )

    from eval.summary import write_latency_summary

    try:
        print(f"  wrote {plot(directory).name}")
    except ImportError as exc:
        print(f"  plot skipped: {exc}")
    print(f"  wrote {write_latency_summary(directory).name}")
    return directory


def plot(run_dir: str | Path) -> Path:
    """Draw the latency curve from a committed ``latency.json``.

    Args:
        run_dir: A ``results/<run_id>`` directory.

    Returns:
        The path written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(run_dir)
    payload = json.loads((directory / "latency.json").read_text(encoding="utf-8"))
    points = payload["points"]
    n = [p["n_questions"] for p in points]
    median = [p["median_ms"] for p in points]
    prefill = [p["median_prefill_ms"] for p in points]
    separate = [p["median_separate_ms"] for p in points]
    baseline = median[0]

    fig, (ax, ax_ratio) = plt.subplots(
        2, 1, figsize=(6.6, 6.8), height_ratios=[3, 2], constrained_layout=True
    )

    ax.plot(n, separate, "s--", color="#c1666b", lw=1.5, ms=6, label="one call per question")
    ax.plot(n, median, "o-", color="#2a6f97", lw=1.8, ms=6, label="one bundled call")
    ax.plot(n, prefill, ":", color="#8d99ae", lw=1.5, label="prefill only (shared)")
    ax.fill_between(
        n, [p["p10_ms"] for p in points], [p["p90_ms"] for p in points], color="#2a6f97", alpha=0.15
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([str(x) for x in n])
    ax.set_ylabel("wall time (ms, log)")
    ax.grid(alpha=0.15, which="both")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    meta = payload["meta"]
    ax.set_title(
        f"Latency vs questions — {meta['model'].rsplit('/', 1)[-1]} {meta['quantization']}\n"
        f"{meta.get('prefix_tokens', meta['state_tokens_actual'])}-token prefill · {meta['gpu']} "
        f"· median of {meta['repeats']}",
        fontsize=10,
    )

    ax_ratio.axhline(2.0, color="#c1666b", ls="--", lw=1.2, label="2x target")
    ax_ratio.axhline(1.0, color="#8d99ae", ls=":", lw=1)
    ax_ratio.plot(
        n,
        [m / baseline for m in median],
        "o-",
        color="#2a6f97",
        lw=1.8,
        ms=6,
        label="bundled / 1-question",
    )
    ax_ratio.plot(
        n,
        [s / baseline for s in separate],
        "s--",
        color="#c1666b",
        lw=1.2,
        ms=5,
        alpha=0.6,
        label="separate / 1-question",
    )
    ax_ratio.set_xscale("log", base=2)
    ax_ratio.set_yscale("log")
    ax_ratio.set_xticks(n)
    ax_ratio.set_xticklabels([str(x) for x in n])
    ax_ratio.set_xlabel("questions in the call")
    ax_ratio.set_ylabel("x 1-question call (log)")
    ax_ratio.grid(alpha=0.15, which="both")
    ax_ratio.legend(frameon=False, fontsize=9, loc="upper left")

    target = payload["target"]
    ax_ratio.set_title(
        f"64/1 = {target['ratio_64_over_1']:.2f}x — target {'met' if target['met'] else 'missed'}",
        fontsize=10,
    )

    path = directory / "latency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Command line for ``uv run task bench``."""
    parser = argparse.ArgumentParser(description="Benchmark latency versus question count")
    parser.add_argument("--model", default=BenchConfig.model)
    parser.add_argument("--quantization", default=BenchConfig.quantization)
    parser.add_argument("--state-tokens", type=int, default=BenchConfig.state_tokens)
    parser.add_argument("--counts", default="1,4,16,64")
    parser.add_argument("--repeats", type=int, default=BenchConfig.repeats)
    parser.add_argument("--warmup", type=int, default=BenchConfig.warmup)
    parser.add_argument("--separate-repeats", type=int, default=BenchConfig.separate_repeats)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--plot", default=None, help="redraw the plot for an existing run dir")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.plot:
        from eval.summary import write_latency_summary

        print(f"wrote {plot(Path(args.plot))}")
        print(f"wrote {write_latency_summary(Path(args.plot))}")
        return 0

    run(
        BenchConfig(
            model=args.model,
            quantization=args.quantization,
            state_tokens=args.state_tokens,
            question_counts=tuple(int(x) for x in args.counts.split(",")),
            repeats=args.repeats,
            warmup=args.warmup,
            separate_repeats=args.separate_repeats,
            run_id=args.run_id,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
