"""Run both teachers back to back, unattended.

Teacher 1 takes most of a day and teacher 2 a couple of hours, so chaining them into one
detached process means nobody has to be awake between them. Each teacher keeps its own run
directory, checkpoints and resume state — the chain adds sequencing and nothing else, so a
crash in the second leg cannot cost the first.

Launched by ``uv run task teach-overnight --detach``. Progress for either leg is read with
``uv run task teach --status <run_id>``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["OVERNIGHT_PLAN", "main", "run_chain"]

#: The approved plan: teacher 1 then teacher 2, over the same source rows.
#:
#: Batch sizes are not cosmetic. The 27B at nf4 leaves so little headroom that batch 16 OOMs
#: mid-generation, which is why teacher 1 runs at 4; gpt-oss at MXFP4 is small enough for 16.
OVERNIGHT_PLAN: tuple[dict[str, Any], ...] = (
    {"teacher": "qwen-low-1024", "batch_size": 4, "run_id": "teach-qwen-low-1024"},
    {"teacher": "gptoss-medium-1024", "batch_size": 16, "run_id": "teach-gptoss-medium-1024"},
)


def run_chain(
    plan: Sequence[dict[str, Any]],
    items_path: Path,
    limit: int | None,
    root: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Run each leg in order, recording what happened to all of them.

    A failing leg stops the chain and is re-raised, having already written its own ``FAILED``
    file. The legs that finished keep their results: that is the point of separate run
    directories.

    Args:
        plan: Leg descriptions.
        items_path: The state/question pairs to label.
        limit: Optional cap, for a rehearsal.
        root: Repository root.
        force: Resume legs whose configuration has changed since their rows on disk.

    Returns:
        A summary of every leg that ran.
    """
    from data.build.teach_run import TEACHERS, load_items, run

    items = load_items(items_path, limit)
    legs: list[dict[str, Any]] = []
    chain_dir = root / "results" / "teach-overnight"
    chain_dir.mkdir(parents=True, exist_ok=True)

    def note(payload: dict[str, Any]) -> None:
        (chain_dir / "chain.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
        )

    note(
        {"started": datetime.now(UTC).isoformat(timespec="seconds"), "plan": list(plan), "legs": []}
    )
    for leg in plan:
        setting = TEACHERS[leg["teacher"]]
        directory = root / "results" / leg["run_id"]
        print(f"=== {leg['teacher']} -> {directory.name}", flush=True)
        summary = run(setting, items, directory, leg["batch_size"], force=force)
        legs.append({**leg, "summary": summary})
        note(
            {
                "updated": datetime.now(UTC).isoformat(timespec="seconds"),
                "plan": list(plan),
                "legs": legs,
                "rows": len(items),
            }
        )
    return {"legs": legs, "rows": len(items)}


def main(argv: Sequence[str] | None = None) -> int:
    """Command line for ``uv run task teach-overnight``."""
    from s1decide.jobs import spawn_detached
    from s1decide.tasks import repo_root

    parser = argparse.ArgumentParser(prog="task teach-overnight")
    parser.add_argument("--items", default="data/processed/teach_items.jsonl")
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="resume legs even if the configuration changed since the rows on disk",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = repo_root()
    if args.detach:
        child = [
            sys.executable,
            "-m",
            "data.build.teach_overnight",
            "--items",
            args.items,
            "--limit",
            str(args.limit),
        ]
        if args.force:
            child.append("--force")
        pid = spawn_detached(child, root / "results" / "teach-overnight")
        print(f"detached overnight chain (pid {pid})")
        for leg in OVERNIGHT_PLAN:
            print(f"  status: uv run task teach --status {leg['run_id']}")
        return 0

    payload = run_chain(OVERNIGHT_PLAN, root / args.items, args.limit, root, force=args.force)
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
