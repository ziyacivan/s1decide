"""Audit, offline, whether each teacher's chat template applied the effort its runs record.

Runs `data.build.teach_run.effort_check` against the local tokenizers of the named teachers,
with the first item of the batch each run labelled, and compares the fingerprint of the
requested render against nothing — the check needs only the tokenizer, not the model.

    uv run python -m data.build.effort_audit qwen-low-1024 gptoss-medium-1024 \\
        --out results/effort-audit-2026-09-23
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Check each named teacher setting and write ``effort.json``; non-zero if any fails."""
    from data.build.teach import build_prompt
    from data.build.teach_run import TEACHERS, EffortNotApplied, effort_check
    from transformers import AutoTokenizer

    from s1decide.tasks import repo_root

    parser = argparse.ArgumentParser(prog="data.build.effort_audit")
    parser.add_argument("teachers", nargs="+", choices=sorted(TEACHERS))
    parser.add_argument("--items", default="data/processed/teach_items_val.jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = repo_root()
    item = json.loads((root / args.items).read_text(encoding="utf-8").split("\n")[0])
    prompt = build_prompt(item["state"], item["question"])
    results: dict[str, dict] = {}
    failed = False
    for label in args.teachers:
        setting = TEACHERS[label]
        tokenizer = AutoTokenizer.from_pretrained(setting.model, local_files_only=True)
        try:
            check = effort_check(tokenizer, setting, prompt)
            results[label] = {"model": setting.model, "applied": True, "check": check}
        except EffortNotApplied as exc:
            failed = True
            results[label] = {"model": setting.model, "applied": False, "error": str(exc)}
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "effort.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
