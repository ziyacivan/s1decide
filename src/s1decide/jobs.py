"""Long-running, detached, resumable jobs.

The teacher-labelling run takes most of a night, and nobody is going to sit in front of it. That
imposes four requirements that ordinary scripts do not have.

**It must survive this session ending.** Started detached, in its own process group, so closing
the terminal or ending a Claude Code session does not take it with it.

**It must be resumable.** Re-running the same command with the same ``run_id`` continues from
the last checkpoint and never recomputes a finished row. An eight-hour job that cannot resume is
an eight-hour job you get one attempt at.

**It must be inspectable from outside.** A ``heartbeat`` file touched every batch, a
``progress.jsonl`` line per batch, and a ``log.txt``. Asking "is it still alive?" must not
require attaching to the process, because there is nothing to attach to.

**It must fail loudly and stop.** On an unhandled error it writes ``FAILED`` with the traceback
and exits. It never retries in a loop: a job that quietly restarts after an OOM produces hours
of nothing and a directory that looks busy.

The state files are deliberately plain text and JSONL. Whatever goes wrong at 3am, the recovery
path should be readable with `cat`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "CHECKPOINT_EVERY",
    "JobPaths",
    "JobStatus",
    "checkpoint_rows",
    "job_status",
    "read_done_ids",
    "run_job",
    "spawn_detached",
]

#: Rows between checkpoint flushes. Small enough that a crash costs minutes, large enough that
#: the writes are not the bottleneck.
CHECKPOINT_EVERY = 200

#: A heartbeat older than this means the job is wedged rather than slow. Generous: one batch of
#: reasoning traces on a 27B at 4-bit can legitimately take several minutes.
STALE_HEARTBEAT_SECONDS = 900


@dataclass(frozen=True)
class JobPaths:
    """Where a job keeps its state.

    Attributes:
        directory: ``results/<run_id>``.
    """

    directory: Path

    @property
    def rows(self) -> Path:
        """Completed rows, appended as JSONL. The resume point."""
        return self.directory / "rows.jsonl"

    @property
    def progress(self) -> Path:
        """One line per batch: counts, throughput, timing."""
        return self.directory / "progress.jsonl"

    @property
    def heartbeat(self) -> Path:
        """Touched every batch. Its mtime is the liveness signal."""
        return self.directory / "heartbeat"

    @property
    def log(self) -> Path:
        """Everything the job printed."""
        return self.directory / "log.txt"

    @property
    def done(self) -> Path:
        """Written once, at the end, with the summary."""
        return self.directory / "DONE"

    @property
    def failed(self) -> Path:
        """Written once, on an unhandled error, with the traceback."""
        return self.directory / "FAILED"

    def ensure(self) -> JobPaths:
        """Create the directory."""
        self.directory.mkdir(parents=True, exist_ok=True)
        return self


def read_done_ids(paths: JobPaths) -> set[str]:
    """Ids already completed, read from the checkpoint file.

    A truncated final line — the normal result of killing a process mid-write — is skipped
    rather than raising. Losing one row to a resume is correct; refusing to resume is not.

    Args:
        paths: The job's paths.

    Returns:
        The set of completed row ids.
    """
    if not paths.rows.is_file():
        return set()
    done: set[str] = set()
    for line in paths.rows.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            done.add(str(json.loads(line)["id"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def checkpoint_rows(paths: JobPaths, rows: Sequence[dict[str, Any]]) -> None:
    """Append completed rows and flush to disk.

    Flushed and fsynced: the point of a checkpoint is to survive a power cut, not merely a
    clean exit.
    """
    if not rows:
        return
    with paths.rows.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@dataclass
class JobStatus:
    """What ``--status`` prints.

    Attributes:
        run_id: The run's directory name.
        state: ``running``, ``done``, ``failed``, ``stale`` or ``missing``.
        done_rows: Rows completed.
        total_rows: Rows planned, when the job recorded a total.
        rows_per_second: Throughput over the recorded progress lines.
        heartbeat_age: Seconds since the heartbeat was touched.
        eta_seconds: Estimated seconds remaining.
        errors: Lines from ``FAILED``, if any.
    """

    run_id: str
    state: str
    done_rows: int = 0
    total_rows: int | None = None
    rows_per_second: float | None = None
    heartbeat_age: float | None = None
    eta_seconds: float | None = None
    errors: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """The human-readable report."""
        lines = [f"run {self.run_id}: {self.state.upper()}"]
        if self.total_rows:
            remaining = max(0, self.total_rows - self.done_rows)
            pct = self.done_rows / self.total_rows if self.total_rows else 0.0
            lines.append(f"  rows      {self.done_rows:,} / {self.total_rows:,} ({pct:.1%})")
            lines.append(f"  remaining {remaining:,}")
        else:
            lines.append(f"  rows      {self.done_rows:,}")
        if self.rows_per_second:
            lines.append(f"  rate      {self.rows_per_second:.2f} rows/s")
        if self.eta_seconds is not None:
            lines.append(f"  eta       {self.eta_seconds / 3600:.1f} h")
        if self.heartbeat_age is not None:
            note = "  <-- STALE" if self.heartbeat_age > STALE_HEARTBEAT_SECONDS else ""
            lines.append(f"  heartbeat {self.heartbeat_age:.0f}s ago{note}")
        for key, value in self.extra.items():
            lines.append(f"  {key:9} {value}")
        if self.errors:
            lines.append("  errors:")
            lines += [f"    {line}" for line in self.errors.strip().split("\n")[-12:]]
        return "\n".join(lines)


def job_status(directory: Path) -> JobStatus:
    """Inspect a job directory from outside the process.

    Args:
        directory: ``results/<run_id>``.

    Returns:
        A :class:`JobStatus`. ``missing`` when the directory does not exist, which is a normal
        answer to "is this run_id in use?" rather than an error.
    """
    run_id = directory.name
    if not directory.is_dir():
        return JobStatus(run_id=run_id, state="missing")

    paths = JobPaths(directory)
    done_rows = len(read_done_ids(paths))

    total_rows: int | None = None
    rate: float | None = None
    extra: dict[str, Any] = {}
    if paths.progress.is_file():
        entries = []
        for line in paths.progress.read_text(encoding="utf-8").split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if entries:
            last = entries[-1]
            total_rows = last.get("total_rows")
            elapsed = last.get("elapsed_seconds")
            completed = last.get("rows_this_run")
            if elapsed is not None and completed:
                rate = completed / max(float(elapsed), 1e-6)
            for key in ("teacher", "setting", "truncated", "dropped"):
                if key in last:
                    extra[key] = last[key]

    heartbeat_age = None
    if paths.heartbeat.is_file():
        heartbeat_age = max(0.0, time.time() - paths.heartbeat.stat().st_mtime)

    if paths.failed.is_file():
        state = "failed"
    elif paths.done.is_file():
        state = "done"
    elif heartbeat_age is not None and heartbeat_age > STALE_HEARTBEAT_SECONDS:
        state = "stale"
    else:
        state = "running"

    eta = None
    if rate and total_rows:
        eta = max(0, total_rows - done_rows) / rate

    return JobStatus(
        run_id=run_id,
        state=state,
        done_rows=done_rows,
        total_rows=total_rows,
        rows_per_second=rate,
        heartbeat_age=heartbeat_age,
        eta_seconds=eta,
        errors=paths.failed.read_text(encoding="utf-8") if paths.failed.is_file() else "",
        extra=extra,
    )


def spawn_detached(argv: Sequence[str], directory: Path) -> int:
    """Start a command in its own process group, detached from this session.

    On Windows this means ``CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS``: the child does not
    receive the console's Ctrl-C and does not die when the console closes. On POSIX it means
    ``start_new_session``. Either way the job outlives whoever started it, which is the point.

    Args:
        argv: The command to run.
        directory: Job directory; stdout and stderr are appended to its ``log.txt``.

    Returns:
        The child's process id, also written to ``pid`` in the job directory.
    """
    paths = JobPaths(directory).ensure()
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        kwargs: dict[str, Any] = {"creationflags": flags}
    else:
        kwargs = {"start_new_session": True}

    with paths.log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== spawned {datetime.now(UTC).isoformat(timespec='seconds')}: {argv}\n")
        handle.flush()
        process = subprocess.Popen(
            list(argv),
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=str(directory.parents[1]) if len(directory.parents) > 1 else None,
            **kwargs,
        )
    (directory / "pid").write_text(str(process.pid), encoding="utf-8", newline="\n")
    return process.pid


def run_job(
    directory: Path,
    items: Sequence[Any],
    work: Callable[[Sequence[Any]], Iterable[dict[str, Any]]],
    *,
    key: Callable[[Any], str],
    batch_size: int = 16,
    checkpoint_every: int = CHECKPOINT_EVERY,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a batched, checkpointed, resumable job in the foreground.

    The caller supplies the work; this supplies the bookkeeping. Rows already in the checkpoint
    are skipped before ``work`` is ever called, so resuming is not merely fast, it is free.

    Args:
        directory: Job directory.
        items: Everything to process.
        work: Called with a batch of items, returns finished rows, each carrying an ``id``.
        key: Maps an item to the id used for resume.
        batch_size: Items per call to ``work``.
        checkpoint_every: Rows between flushes.
        meta: Recorded once in ``meta.json``.

    Returns:
        A summary dict, also written to ``DONE``.

    Raises:
        Exception: Re-raised after writing ``FAILED``. The job stops; it does not retry.
    """
    paths = JobPaths(directory).ensure()
    if paths.failed.is_file():
        paths.failed.unlink()
    if paths.done.is_file():
        paths.done.unlink()
    if meta:
        (directory / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    done = read_done_ids(paths)
    pending = [item for item in items if key(item) not in done]
    started = time.perf_counter()
    buffer: list[dict[str, Any]] = []
    processed = 0

    def flush() -> None:
        nonlocal buffer
        checkpoint_rows(paths, buffer)
        buffer = []

    try:
        for index in range(0, len(pending), batch_size):
            batch = pending[index : index + batch_size]
            rows = list(work(batch))
            buffer.extend(rows)
            processed += len(rows)
            paths.heartbeat.write_text(
                datetime.now(UTC).isoformat(timespec="seconds"), encoding="utf-8"
            )
            elapsed = time.perf_counter() - started
            # A row may attach a `progress` dict; it is surfaced in the progress line so that
            # `--status` can show job-specific counters (truncations, drops) without this
            # module knowing what a teacher run is.
            extra = rows[-1].get("progress") if rows else None
            entry = {
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
                "rows_this_run": processed,
                "rows_done": len(done) + processed,
                "total_rows": len(items),
                # Not rounded to 2dp: a fast batch rounds to 0.0 and the rate becomes
                # undefined, which reads as "no throughput data" rather than "very fast".
                "elapsed_seconds": round(elapsed, 6),
                "rows_per_second": round(processed / max(elapsed, 1e-9), 4),
            }
            if isinstance(extra, dict):
                entry.update(extra)
            with paths.progress.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            if len(buffer) >= checkpoint_every:
                flush()
        flush()
    except BaseException:
        flush()
        paths.failed.write_text(
            f"{datetime.now(UTC).isoformat(timespec='seconds')}\n{traceback.format_exc()}",
            encoding="utf-8",
            newline="\n",
        )
        raise

    summary = {
        "run_id": directory.name,
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows_total": len(items),
        "rows_this_run": processed,
        "rows_resumed": len(done),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    paths.done.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return summary


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Read a JSONL file, skipping blank and truncated lines."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue
