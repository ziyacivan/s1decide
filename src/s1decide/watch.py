"""Watch a detached training run and exit with one line when it needs attention.

Replaces the ad-hoc watcher used for S1 (2026-09-24), which raised a false "rebooted" alarm: it
compared ``psutil.boot_time()`` for exact equality, and on Windows that value is derived from
uptime and drifts by a second or so. The run was healthy and went unwatched until re-armed.

Alerts, each a terminal state for the watcher:

- ``DONE`` / ``FAILED`` / ``STOPPED`` — the run wrote its marker;
- ``REBOOTED`` — boot time moved by more than :data:`BOOT_TOLERANCE_S`;
- ``GONE`` — the training process exited without writing a marker;
- ``STALE`` — no heartbeat for :data:`STALE_AFTER_S` (evaluations beat every 100 batches);
- ``SLOWDOWN`` — the last training interval ran below half the median rate, excluding intervals
  that contain an evaluation (on this card the memory cliff shows up as speed, not as an error).

    uv run task watch results/s1-3090 --eval-every 5000 --pid 980
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

__all__ = ["BOOT_TOLERANCE_S", "STALE_AFTER_S", "check_run", "interval_rates", "main"]

#: Boot-time drift tolerated before a reboot is declared.
BOOT_TOLERANCE_S = 60.0
#: Heartbeat age that counts as stale.
STALE_AFTER_S = 15 * 60.0


def interval_rates(progress: Sequence[dict[str, Any]], eval_every: int) -> list[float]:
    """Rows per second between consecutive progress lines, skipping intervals with an evaluation."""
    rates = []
    for a, b in pairwise(progress):
        spans_eval = bool(eval_every) and a["rows"] // eval_every != b["rows"] // eval_every
        dt = b["elapsed_seconds"] - a["elapsed_seconds"]
        if not spans_eval and dt > 0 and b["rows"] > a["rows"]:
            rates.append((b["rows"] - a["rows"]) / dt)
    return rates


def check_run(
    run: Path,
    *,
    eval_every: int,
    boot_at_start: float | None,
    boot_now: float | None,
    pid_alive: bool | None,
    now: float,
) -> str | None:
    """One pass of the watcher: the alert line, or ``None`` if all is well.

    Args:
        run: The run directory.
        eval_every: The run's evaluation interval in rows (0 for none).
        boot_at_start: Boot time when watching began.
        boot_now: Boot time now.
        pid_alive: Whether the training process is alive; ``None`` if not tracked.
        now: Current time (seconds since the epoch).
    """
    for marker in ("DONE", "FAILED", "STOPPED"):
        if (run / marker).exists():
            return f"{marker}: {(run / marker).read_text(encoding='utf-8')[-600:].strip()}"
    drifted = (
        boot_at_start is not None
        and boot_now is not None
        and abs(boot_now - boot_at_start) > BOOT_TOLERANCE_S
    )
    if drifted:
        return f"REBOOTED: boot time moved by {boot_now - boot_at_start:.0f} s"
    if pid_alive is False:
        return "GONE: the training process exited without DONE, FAILED or STOPPED"
    heartbeat = run / "heartbeat"
    if heartbeat.exists() and now - heartbeat.stat().st_mtime > STALE_AFTER_S:
        return f"STALE: no heartbeat for {(now - heartbeat.stat().st_mtime) / 60:.1f} min"
    progress_path = run / "progress.jsonl"
    if progress_path.exists():
        lines = [
            json.loads(x)
            for x in progress_path.read_text(encoding="utf-8").split("\n")
            if x.strip()
        ]
        rates = interval_rates(lines, eval_every)
        if len(rates) >= 6:
            typical = statistics.median(rates[:-1])
            if rates[-1] < 0.5 * typical:
                return (
                    f"SLOWDOWN: {rates[-1]:.3f} rows/s against a median of {typical:.3f} at "
                    f"{lines[-1]['rows']} rows, peak VRAM {lines[-1].get('vram_peak_gib')}"
                )
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Poll until an alert, print it, exit 0."""
    import argparse

    from s1decide.jobs import boot_time

    parser = argparse.ArgumentParser(prog="task watch")
    parser.add_argument("run")
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--poll", type=float, default=60.0)
    args = parser.parse_args(list(argv) if argv is not None else None)

    def alive() -> bool | None:
        if args.pid is None:
            return None
        import psutil

        return psutil.pid_exists(args.pid)

    start = boot_time()
    while True:
        alert = check_run(
            Path(args.run),
            eval_every=args.eval_every,
            boot_at_start=start,
            boot_now=boot_time(),
            pid_alive=alive(),
            now=time.time(),
        )
        if alert:
            print(alert, flush=True)
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
