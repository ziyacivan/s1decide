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
import shutil
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
    "VOLATILE_META_PATHS",
    "JobPaths",
    "JobStatus",
    "MetaMismatch",
    "check_resume_meta",
    "checkpoint_rows",
    "compare_meta",
    "flatten_meta",
    "job_status",
    "read_done_ids",
    "run_job",
    "spawn_detached",
]

#: Rows between checkpoint flushes, and therefore the worst-case loss when the machine goes
#: away without warning.
#:
#: 50, lowered from 200 after a Windows Update reboot cost 144 rows — about 40 minutes at the
#: teacher run's 0.060 rows/s. At 50 the worst case is ~14 minutes. The write is a few hundred
#: kilobytes appended and fsynced, so at these rates it is far below the noise: one flush per
#: ~14 minutes of GPU work.
#:
#: The number that matters is *time*, not rows. A job running at 2 rows/s checkpoints every 25
#: seconds at this setting, which is more often than it needs but costs nothing either.
CHECKPOINT_EVERY = 50

#: A heartbeat older than this means the job is wedged rather than slow. Generous: one batch of
#: reasoning traces on a 27B at 4-bit can legitimately take several minutes.
STALE_HEARTBEAT_SECONDS = 900


def boot_time() -> float | None:
    """When the machine last booted, as a Unix timestamp.

    Reported next to a stale heartbeat because the two together identify a reboot, and nothing
    else does. Windows Update restarted this machine mid-run at 04:58 UTC on 2026-09-18 and the
    job simply stopped: no ``FAILED`` file, because nothing caught anything — the process was
    killed with the operating system. A stale heartbeat alone looks the same as a wedged job,
    and the two need different responses.

    Returns:
        The boot timestamp, or ``None`` when psutil cannot supply it.
    """
    try:
        import psutil

        return float(psutil.boot_time())
    except Exception:
        return None


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
        booted_after_heartbeat: True when the machine booted *after* the last heartbeat, which
            means the job died with the machine rather than wedging.
    """

    run_id: str
    state: str
    done_rows: int = 0
    total_rows: int | None = None
    rows_per_second: float | None = None
    heartbeat_age: float | None = None
    eta_seconds: float | None = None
    errors: str = ""
    booted_after_heartbeat: bool = False
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
        booted = boot_time()
        if booted is not None:
            stamp = datetime.fromtimestamp(booted, UTC).strftime("%Y-%m-%d %H:%M UTC")
            age = (time.time() - booted) / 3600
            flag = "  <-- REBOOTED MID-RUN" if self.booted_after_heartbeat else ""
            lines.append(f"  booted    {stamp} ({age:.1f} h ago){flag}")
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
    # The checkpoint only flushes every `CHECKPOINT_EVERY` rows, so a short run reports 0 done
    # until the very end even while it is plainly working. The progress file is written every
    # batch, so prefer its count when it is ahead — otherwise `--status` on a pilot says
    # nothing is happening, which is exactly when someone is watching.
    unflushed = 0

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
            unflushed = int(last.get("rows_done") or 0)
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

    done_rows = max(done_rows, unflushed)
    eta = None
    if rate and total_rows:
        eta = max(0, total_rows - done_rows) / rate

    booted = boot_time()
    rebooted = bool(
        booted is not None
        and paths.heartbeat.is_file()
        and booted > paths.heartbeat.stat().st_mtime
    )

    return JobStatus(
        run_id=run_id,
        state=state,
        done_rows=done_rows,
        total_rows=total_rows,
        rows_per_second=rate,
        heartbeat_age=heartbeat_age,
        eta_seconds=eta,
        errors=paths.failed.read_text(encoding="utf-8") if paths.failed.is_file() else "",
        booted_after_heartbeat=rebooted,
        extra=extra,
    )


def spawn_detached(argv: Sequence[str], directory: Path) -> int:
    """Start a command that outlives the shell, the terminal and this session.

    On Windows this goes through **PowerShell ``Start-Process``**, which re-parents the child
    under the PowerShell host. ``subprocess.Popen`` with
    ``CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS`` is not enough and the difference is not
    theoretical: a run launched that way died 2.5 minutes later with

        forrtl: error (200): program aborting due to window-CLOSE event

    the Intel Fortran runtime inside numpy's MKL reacting to a console control event. The child
    stayed associated with the launching console — and, on a tool-driven shell, with its job
    object — so when that went away it was terminated. The first such run only survived five
    hours because the launching session happened to stay busy that long, which is luck, not
    detachment.

    ``CREATE_BREAKAWAY_FROM_JOB`` is tried first since it is cheaper and needs no shell; a job
    configured to forbid breakaway makes ``Popen`` fail, and then the PowerShell path is used.

    Args:
        argv: The command to run.
        directory: Job directory; stdout and stderr are appended to its ``log.txt``.

    Returns:
        The child's process id, also written to ``pid`` in the job directory.

    Raises:
        RuntimeError: If the process could not be started detached at all. Launching a
            multi-hour job that will die with the terminal is worse than not launching it.
    """
    paths = JobPaths(directory).ensure()
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    with paths.log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== spawned {stamp}: {list(argv)}\n")

    if sys.platform != "win32":
        with paths.log.open("a", encoding="utf-8") as handle:
            process = subprocess.Popen(
                list(argv),
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        (directory / "pid").write_text(str(process.pid), encoding="utf-8", newline="\n")
        return process.pid

    pid = _spawn_windows_breakaway(argv, paths) or _spawn_windows_powershell(argv, paths)
    if pid is None:
        raise RuntimeError(
            "could not start the job detached; it would die with this terminal, so it was not "
            "started at all"
        )
    (directory / "pid").write_text(str(pid), encoding="utf-8", newline="\n")
    return pid


def _spawn_windows_breakaway(argv: Sequence[str], paths: JobPaths) -> int | None:
    """Try to break the child out of this process's job object."""
    flags = (
        subprocess.CREATE_NEW_PROCESS_GROUP
        | subprocess.DETACHED_PROCESS
        | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000)
    )
    try:
        with paths.log.open("a", encoding="utf-8") as handle:
            process = subprocess.Popen(
                list(argv),
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=flags,
            )
        return process.pid
    except OSError:
        # The job forbids breakaway. Not fatal — PowerShell re-parents instead.
        return None


def _spawn_windows_powershell(argv: Sequence[str], paths: JobPaths) -> int | None:
    """Start the child from the PowerShell host, which is not in this shell's job."""
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:  # pragma: no cover - PowerShell ships with Windows
        return None

    def quote(value: str) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    # Start-Process cannot send stdout and stderr to one file, so stderr gets its own and the
    # status report reads both.
    arguments = ",".join(quote(a) for a in argv[1:])
    script = (
        f"$p = Start-Process -FilePath {quote(argv[0])} "
        + (f"-ArgumentList {arguments} " if len(argv) > 1 else "")
        + f"-WorkingDirectory {quote(str(paths.directory.parents[1]))} "
        f"-RedirectStandardOutput {quote(str(paths.log))} "
        f"-RedirectStandardError {quote(str(paths.directory / 'log.err.txt'))} "
        "-WindowStyle Hidden -PassThru; $p.Id"
    )
    completed = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    for line in reversed(completed.stdout.strip().splitlines()):
        if line.strip().isdigit():
            return int(line.strip())
    return None


#: Meta fields that legitimately differ between a run and its resume, and are therefore not
#: compared.
#:
#: ``started`` is the wall-clock time this process began: different by definition on a resume.
#: ``kernels.notes`` is prose explaining the configuration, not the configuration itself — it
#: changes when someone improves the wording, which is not a reason to refuse a resume.
VOLATILE_META_PATHS = frozenset({"started", "kernels.notes"})


class MetaMismatch(RuntimeError):
    """A resume was attempted under a different configuration than the rows on disk.

    Raised rather than warned about. A job that resumes under upgraded kernels, a different
    batch size or a regenerated input file produces one corpus containing two populations, and
    overwrites the ``meta.json`` that was the only record of the first — so the damage is both
    silent and unrecoverable after the fact.
    """


def flatten_meta(meta: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested metadata to dotted paths, so a difference can be named precisely.

    Lists are compared whole: order matters in every list we record (a rubric, a set of notes),
    and a per-element diff would report noise rather than a cause.

    Args:
        meta: The metadata.
        prefix: Path prefix, used in recursion.

    Returns:
        A mapping of dotted path to value, e.g. ``{"kernels.attention": "sdpa"}``.
    """
    flat: dict[str, Any] = {}
    for key, value in meta.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat |= flatten_meta(value, f"{path}.")
        else:
            flat[path] = value
    return flat


def compare_meta(
    stored: dict[str, Any],
    incoming: dict[str, Any],
    ignore: frozenset[str] = VOLATILE_META_PATHS,
) -> list[str]:
    """Differences between the metadata on disk and the metadata of a resuming run.

    A path present in ``incoming`` but missing from ``stored`` is **not** a difference. That is
    deliberate: it is what happens when a new field is added to the metadata while a job is
    already half finished, and refusing to resume a healthy run because the code learned to
    record one more thing would make this guard a liability on the day it shipped.

    Args:
        stored: Metadata read from the job directory.
        incoming: Metadata this run would write.
        ignore: Dotted paths that legitimately change.

    Returns:
        One human-readable line per difference, empty when the configurations agree.
    """
    on_disk, now = flatten_meta(stored), flatten_meta(incoming)
    differences = []
    for path in sorted(on_disk):
        if path in ignore or path not in now:
            continue
        if on_disk[path] != now[path]:
            differences.append(f"{path}: {on_disk[path]!r} on disk, {now[path]!r} now")
    return differences


def check_resume_meta(
    directory: Path, meta: dict[str, Any] | None, *, force: bool = False
) -> list[str]:
    """Refuse to resume a job whose configuration has changed since it last ran.

    Call this *before* loading a model: the point is to fail in a second rather than after a
    five-minute load. ``run_job`` calls it too, so a caller that forgets is still covered.

    A job with no rows yet is not a resume, whatever else its directory contains.

    Args:
        directory: Job directory.
        meta: The metadata this run would write; ``None`` skips the check.
        force: Report the differences and continue anyway.

    Returns:
        The differences found, empty when there are none or when this is a fresh run.

    Raises:
        MetaMismatch: When the configuration differs and ``force`` is not set.
    """
    stored_path = directory / "meta.json"
    if meta is None or not stored_path.is_file() or not read_done_ids(JobPaths(directory)):
        return []
    try:
        stored = json.loads(stored_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    differences = compare_meta(stored, meta)
    if differences and not force:
        raise MetaMismatch(
            f"{directory.name} has rows produced under a different configuration:"
            + "".join(f"\n  {line}" for line in differences)
            + "\n\nResuming would mix two populations in one corpus and overwrite the record "
            "of the first. Start a new run_id, or pass --force if the difference is known to "
            "be harmless."
        )
    return differences


def run_job(
    directory: Path,
    items: Sequence[Any],
    work: Callable[[Sequence[Any]], Iterable[dict[str, Any]]],
    *,
    key: Callable[[Any], str],
    batch_size: int = 16,
    checkpoint_every: int = CHECKPOINT_EVERY,
    meta: dict[str, Any] | None = None,
    force: bool = False,
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
        force: Resume even when the stored metadata disagrees with ``meta``.

    Returns:
        A summary dict, also written to ``DONE``.

    Raises:
        MetaMismatch: When resuming under a changed configuration and ``force`` is not set.
            Checked before anything on disk is touched, so a refused resume changes nothing.
        Exception: Re-raised after writing ``FAILED``. The job stops; it does not retry.
    """
    check_resume_meta(directory, meta, force=force)
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
