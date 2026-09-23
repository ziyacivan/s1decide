"""Token lengths of the training rows as the trainer renders them, per primitive and stage.

The memory envelope depends on sequence length, and the trainer *skips* rows longer than
``max_seq_len`` rather than truncating them. So the distribution decides two things at once: how
much memory a given ``max_seq_len`` needs, and how many rows that cap silently removes. This
reports both, from the same render the trainer uses (`s1decide.prompt.render`, prefix + suffix).

    uv run python -m eval.token_lengths --out results/token-lengths-<date>
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["CAPS", "group_key", "length_stats", "main"]

#: Sequence caps the report counts rows above.
CAPS: tuple[int, ...] = (512, 1024, 1536, 2048)


def group_key(row: dict[str, Any]) -> str:
    """``choice``, ``noul``, ``noul/stage1`` or ``score/<hard|soft>``."""
    if row.get("stage") == 1:
        return f"{row['qtype']}/stage1"
    if row["qtype"] == "score":
        return f"score/{row.get('target_type', 'hard')}"
    return str(row["qtype"])


def length_stats(lengths: Sequence[int], caps: Sequence[int] = CAPS) -> dict[str, Any]:
    """p50/p95/p99/max, and how many rows each cap would skip.

    Args:
        lengths: Token counts.
        caps: Sequence caps to count rows above.

    Returns:
        ``rows``, ``p50``, ``p95``, ``p99``, ``max``, ``mean`` and ``over`` — for each cap, the
        count and share of rows longer than it.
    """
    a = np.asarray(lengths, dtype=np.int64)
    return {
        "rows": len(a),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": int(a.max()),
        "over": {
            str(cap): {"rows": int((a > cap).sum()), "share": float((a > cap).mean())}
            for cap in caps
        },
    }


def _render_length(row: dict[str, Any], tokenizer: Any) -> int:
    from s1decide.primitives import Choice, Noul, Question, Score
    from s1decide.prompt import render

    options = tuple(row["options"])
    if row["qtype"] == "noul":
        spec: Any = Noul(instructions=row["instructions"])
    elif row["qtype"] == "score":
        spec = Score(instructions=row["instructions"], levels=options)
    else:
        spec = Choice(instructions=row["instructions"], options=options)
    rendered = render(row["state"], [Question(name="q", spec=spec)])
    return len(tokenizer.encode(rendered.prefix + rendered.suffixes[0], add_special_tokens=False))


def main(argv: Sequence[str] | None = None) -> int:
    """Measure every training row and write ``lengths.json``."""
    from transformers import AutoTokenizer

    from s1decide.tasks import repo_root

    parser = argparse.ArgumentParser(prog="eval.token_lengths")
    parser.add_argument("--model", default="unsloth/Qwen3.8-27B-unsloth-bnb-4bit")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = repo_root()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    groups: dict[str, list[int]] = {}
    for line in (
        (root / f"data/processed/{args.split}.jsonl").read_text(encoding="utf-8").split("\n")
    ):
        if line.strip():
            row = json.loads(line)
            groups.setdefault(group_key(row), []).append(_render_length(row, tokenizer))

    everything = [n for lengths in groups.values() for n in lengths]
    payload = {
        "model": args.model,
        "split": args.split,
        "render": "s1decide.prompt.render, prefix + suffix, no special tokens (as the trainer)",
        "all": length_stats(everything),
        "by_group": {key: length_stats(groups[key]) for key in sorted(groups)},
    }
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "lengths.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
