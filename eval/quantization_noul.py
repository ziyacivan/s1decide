"""Does 4-bit quantization move a `Noul` answer, the way it moves a `Score` label?

`docs/research/quantization-label-disagreement-2026-09-18.md` measured bitsandbytes nf4 against
llama.cpp Q4_K_M on the same model and the same rows, for a five-level `Score` judgement produced
by hundreds of decode steps: 78% exact agreement, 94% within one level. That note is explicit
that it may not generalise, and the hypothesis worth testing is that it does not.

A `Noul` is one masked logit read from one forward pass. There is no decode loop to accumulate
divergence, so it should be far more stable. If it is, the 22% disagreement is a fact about
*generation* and not about the primitive we publish. If it is not, then 4-bit moves our published
answers and the model card has to say so in those words.

The two runtimes are scored on identical rendered prompts and identical option token ids, checked
rather than assumed, and llama.cpp is scored sequentially because its continuous batching is not
reproducible — see `engine/llamacpp_client.score_many`.

Run in two passes because a 27B does not fit twice on a 24 GiB card:

    uv run task quant-noul --runtime llamacpp --limit 300     # llama-server must be up
    uv run task quant-noul --runtime hf --limit 300           # after the server is stopped
    uv run task quant-noul --compare
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "RUN_DIR",
    "compare",
    "draw_offset",
    "main",
    "sample_noul_rows",
    "score_hf",
    "score_llamacpp",
]

#: Where both passes and the comparison land.
RUN_DIR = "results/quant-noul"

#: Seeded so the two runtimes are handed the same rows without having to pass a manifest between
#: processes, and so a re-run months later is the same experiment.
SEED = 20260923


def sample_noul_rows(
    rows: Sequence[dict[str, Any]], limit: int, seed: int = SEED
) -> list[dict[str, Any]]:
    """Take a balanced sample of `Noul` rows, half genuine and half stage-1.

    The two kinds are scored the same way — one masked logit — so a difference between them
    would itself be a finding, and a sample drawn without regard to the split would be 99%
    stage-1 because the fan-out dwarfs everything else.

    Args:
        rows: Rows of an evaluation split.
        limit: How many rows in total.
        seed: Sampling seed.

    Returns:
        Rows, genuine first, deterministic in ``seed``.
    """
    noul = [r for r in rows if r.get("qtype") == "noul"]
    stage1 = [r for r in noul if str(r.get("stage")) == "1"]
    genuine = [r for r in noul if str(r.get("stage")) != "1"]
    rng = random.Random(seed)
    half = limit // 2
    picked = rng.sample(genuine, min(half, len(genuine)))
    picked += rng.sample(stage1, min(limit - len(picked), len(stage1)))
    return picked


def _render_rows(rows: Sequence[dict[str, Any]]) -> list[tuple[str, str, tuple[str, ...]]]:
    """Render each row to its prefix, its suffix and its option labels.

    The prefix and suffix are kept apart rather than concatenated because `HFEngine.score` wants
    them separately and the split has to fall on a token boundary. Splitting a joined prompt at
    the last *character* does not: this prompt ends ``…</think>\\n\\n`` which is one token
    (``271``), and cutting the final newline off makes it two (``198, 198``). The first version
    of this script did exactly that, so the two runtimes were scoring different token sequences
    and every row appeared to shift in the same direction.

    `render()`'s own boundary is safe — prefix and suffix tokenized separately give the identical
    89 tokens the joined string does, which is the property the whole broadcast path rests on.
    """
    from s1decide.primitives import Noul, Question
    from s1decide.prompt import render

    out = []
    for row in rows:
        question = Question(name="q", spec=Noul(instructions=row["instructions"]))
        rendered = render(row["state"], [question])
        out.append((rendered.prefix, rendered.suffixes[0], rendered.labels[0]))
    return out


def score_llamacpp(
    rows: Sequence[dict[str, Any]], base_url: str = "http://127.0.0.1:8080"
) -> dict[str, Any]:
    """Score every row through a running ``llama-server``.

    Raises:
        RuntimeError: If the server is not up. Failing here beats scoring half a sample.
        TokenizerMismatch: If the server's option tokens differ from the reference.
    """
    from transformers import AutoTokenizer

    from s1decide.engine.llamacpp_client import LlamaServerClient
    from s1decide.tokens import allowed_token_ids

    client = LlamaServerClient(base_url=base_url)
    if not client.health():
        raise RuntimeError(f"no llama-server at {base_url}; start it with the Q4_K_M gguf first")

    rendered = _render_rows(rows)
    labels = list(rendered[0][2])
    tokenizer = AutoTokenizer.from_pretrained(
        "unsloth/Qwen3.8-27B-unsloth-bnb-4bit", local_files_only=True
    )
    reference = dict(zip(labels, allowed_token_ids(tokenizer, labels), strict=True))
    token_ids = client.check_label_tokens(labels, reference=reference)

    scores = client.score_many(
        [prefix + suffix for prefix, suffix, _ in rendered],
        [list(ls) for _, _, ls in rendered],
        token_ids=token_ids,
    )
    return {
        "runtime": "llamacpp-q4_k_m",
        "labels": labels,
        "token_ids": token_ids,
        "ids": [r["id"] for r in rows],
        "answer_idx": [r["answer_idx"] for r in rows],
        "stage": [str(r.get("stage")) for r in rows],
        "p_yes": [s[labels.index("yes")] for s in scores],
    }


def score_hf(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Score every row through `HFEngine` at bitsandbytes nf4.

    One prefill per row: the states differ, so there is no shared prefix to broadcast over and
    the cache-broadcast path has nothing to amortise here. That is a property of this experiment,
    not of the engine.
    """
    import math

    from s1decide.engine.hf import HFEngine

    engine = HFEngine.from_pretrained("unsloth/Qwen3.8-27B-unsloth-bnb-4bit")
    rendered = _render_rows(rows)
    labels = list(rendered[0][2])
    yes = labels.index("yes")
    p_yes = []
    for prefix, suffix, row_labels in rendered:
        out = engine.score(prefix, [suffix], [list(row_labels)])
        logits = out.logits[0]
        top = max(logits)
        exp = [math.exp(v - top) for v in logits]
        total = sum(exp)
        p_yes.append(exp[yes] / total)
    return {
        "runtime": "hf-nf4",
        "labels": labels,
        "ids": [r["id"] for r in rows],
        "answer_idx": [r["answer_idx"] for r in rows],
        "stage": [str(r.get("stage")) for r in rows],
        "p_yes": p_yes,
    }


def compare(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Agreement between two runtimes over the same rows.

    Reports what the `Score` note reports, so the two are readable side by side: exact agreement
    on the decided answer, the marginal each runtime produces, the direction of disagreement with
    a sign test, and the mean probability shift.

    Args:
        first: Output of :func:`score_hf`.
        second: Output of :func:`score_llamacpp`.

    Returns:
        The comparison.

    Raises:
        ValueError: If the two passes did not score the same rows in the same order.
    """
    if first["ids"] != second["ids"]:
        raise ValueError("the two runtimes scored different rows; re-run both with the same seed")

    a, b = first["p_yes"], second["p_yes"]
    n = len(a)
    decided_a = [p >= 0.5 for p in a]
    decided_b = [p >= 0.5 for p in b]
    exact = sum(1 for x, y in zip(decided_a, decided_b, strict=True) if x == y)
    shifts = [y - x for x, y in zip(a, b, strict=True)]
    up = sum(1 for s in shifts if s > 0)
    down = sum(1 for s in shifts if s < 0)

    # Two-sided sign test against a fair coin, exact rather than normal-approximated: at a few
    # hundred rows the approximation is fine, but the exact form costs nothing and does not have
    # to be caveated.
    from math import comb

    trials = up + down
    smaller = min(up, down)
    tail = sum(comb(trials, k) for k in range(smaller + 1)) / (2**trials) if trials else 0.5
    p_value = min(1.0, 2 * tail)

    def accuracy(probabilities: list[float]) -> float:
        return sum(
            1
            for p, truth in zip(probabilities, first["answer_idx"], strict=True)
            if (p >= 0.5) == (truth == 1)
        ) / max(1, n)

    return {
        "n": n,
        "first": first["runtime"],
        "second": second["runtime"],
        "exact_agreement": exact / n,
        "disagreements": n - exact,
        "mean_p_first": sum(a) / n,
        "mean_p_second": sum(b) / n,
        "mean_shift": sum(shifts) / n,
        "mean_absolute_shift": sum(abs(s) for s in shifts) / n,
        "max_absolute_shift": max(abs(s) for s in shifts),
        "shift_up": up,
        "shift_down": down,
        "sign_test_p": p_value,
        "accuracy_first": accuracy(a),
        "accuracy_second": accuracy(b),
        "by_stage": {
            stage: {
                "n": sum(1 for s in first["stage"] if s == stage),
                "exact_agreement": (
                    sum(
                        1
                        for s, x, y in zip(first["stage"], decided_a, decided_b, strict=True)
                        if s == stage and x == y
                    )
                    / max(1, sum(1 for s in first["stage"] if s == stage))
                ),
            }
            for stage in sorted(set(first["stage"]))
        },
    }


def draw_offset(root: Path) -> Path:
    """Draw the log-odds of one runtime against the other.

    The figure is the argument. A cloud scattered about the diagonal would be quantization noise;
    a cloud on a line parallel to the diagonal is a *shift*, and a shift is the one thing a
    temperature cannot remove. Everything else in the note is that picture in numbers.

    Args:
        root: Repository root.

    Returns:
        The path written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from eval.figures import DPI, FIGSIZE, _style

    out_dir = root / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    first = json.loads((root / RUN_DIR / "hf.json").read_text(encoding="utf-8"))
    second = json.loads((root / RUN_DIR / "llamacpp.json").read_text(encoding="utf-8"))

    def log_odds(p: float) -> float:
        p = min(max(p, 1e-9), 1 - 1e-9)
        return math.log(p / (1 - p))

    x = [log_odds(p) for p in first["p_yes"]]
    y = [log_odds(p) for p in second["p_yes"]]
    offset = sum(b - a for a, b in zip(x, y, strict=True)) / len(x)

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    lo = min(min(x), min(y)) - 0.5
    hi = max(max(x), max(y)) + 0.5
    ax.plot([lo, hi], [lo, hi], linewidth=1.2, linestyle="--", label="no change")
    ax.plot(
        [lo, hi],
        [lo + offset, hi + offset],
        linewidth=1.2,
        label=f"constant shift of {offset:+.2f} nats",
    )
    ax.scatter(x, y, s=12, alpha=0.55, edgecolors="none", label=f"{len(x)} Noul questions")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("log-odds of yes — bitsandbytes nf4")
    ax.set_ylabel("log-odds of yes — llama.cpp Q4_K_M")
    ax.set_title("Two 4-bit quantizations of one model, same questions")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    _style(ax)
    path = out_dir / "quant-noul-offset.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Command line for ``uv run task quant-noul``."""
    from s1decide.tasks import repo_root

    parser = argparse.ArgumentParser(prog="task quant-noul")
    parser.add_argument("--runtime", choices=("hf", "llamacpp"), help="score one runtime")
    parser.add_argument("--compare", action="store_true", help="compare two saved passes")
    parser.add_argument("--figure", action="store_true", help="draw the log-odds figure")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = repo_root()
    out_dir = root / RUN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.figure:
        print(f"wrote {draw_offset(root)}")
        if not args.compare:
            return 0

    if args.compare:
        first = json.loads((out_dir / "hf.json").read_text(encoding="utf-8"))
        second = json.loads((out_dir / "llamacpp.json").read_text(encoding="utf-8"))
        payload = compare(first, second)
        (out_dir / "comparison.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(json.dumps(payload, indent=2))
        return 0

    if not args.runtime:
        parser.error("pass --runtime or --compare")

    rows = [
        json.loads(line)
        for line in (root / "data/processed" / f"{args.split}.jsonl")
        .read_text(encoding="utf-8")
        .split("\n")
        if line.strip()
    ]
    sample = sample_noul_rows(rows, args.limit)
    print(f"scoring {len(sample)} rows through {args.runtime}", flush=True)

    result = (
        score_llamacpp(sample, args.base_url) if args.runtime == "llamacpp" else score_hf(sample)
    )
    result["split"] = args.split
    result["seed"] = SEED
    name = "llamacpp.json" if args.runtime == "llamacpp" else "hf.json"
    (out_dir / name).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out_dir.name}/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
