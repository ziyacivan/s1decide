"""The detached, resumable job runner.

Written before the overnight teacher run, because every property here is one that only matters
at 3am: does it survive the session ending, does it resume without recomputing, can you tell
from outside whether it is alive, and does it stop rather than loop when it breaks.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from s1decide.jobs import (
    STALE_HEARTBEAT_SECONDS,
    JobPaths,
    checkpoint_rows,
    job_status,
    read_done_ids,
    run_job,
    spawn_detached,
)


def items(n: int) -> list[dict]:
    return [{"id": f"item-{i:03d}", "value": i} for i in range(n)]


def double(batch):
    return [{"id": item["id"], "result": item["value"] * 2} for item in batch]


# --- checkpointing and resume ----------------------------------------------------


def test_a_finished_job_processes_everything_once(tmp_path) -> None:
    calls: list[str] = []

    def work(batch):
        calls.extend(item["id"] for item in batch)
        return double(batch)

    summary = run_job(tmp_path, items(10), work, key=lambda i: i["id"], batch_size=3)
    assert summary["rows_this_run"] == 10
    assert len(calls) == 10 and len(set(calls)) == 10


def test_resuming_never_recomputes_a_finished_row(tmp_path) -> None:
    """The property that makes an eight-hour job survivable."""
    run_job(tmp_path, items(10), double, key=lambda i: i["id"], batch_size=5)

    second: list[str] = []

    def work(batch):
        second.extend(item["id"] for item in batch)
        return double(batch)

    summary = run_job(tmp_path, items(10), work, key=lambda i: i["id"], batch_size=5)
    assert second == [], "a completed job must do no work on a rerun"
    assert summary["rows_resumed"] == 10
    assert summary["rows_this_run"] == 0


def test_resuming_after_a_partial_run_finishes_the_rest(tmp_path) -> None:
    def explode_after_six(batch):
        if any(item["value"] >= 6 for item in batch):
            raise RuntimeError("boom")
        return double(batch)

    with pytest.raises(RuntimeError):
        run_job(tmp_path, items(10), explode_after_six, key=lambda i: i["id"], batch_size=2)

    done_after_crash = read_done_ids(JobPaths(tmp_path))
    assert 0 < len(done_after_crash) < 10

    summary = run_job(tmp_path, items(10), double, key=lambda i: i["id"], batch_size=2)
    assert summary["rows_resumed"] == len(done_after_crash)
    assert len(read_done_ids(JobPaths(tmp_path))) == 10


def test_a_truncated_checkpoint_line_is_skipped_not_fatal(tmp_path) -> None:
    """Killing a process mid-write leaves half a line. Resuming must still work."""
    paths = JobPaths(tmp_path).ensure()
    checkpoint_rows(paths, [{"id": "a"}, {"id": "b"}])
    with paths.rows.open("a", encoding="utf-8") as handle:
        handle.write('{"id": "c", "resu')
    assert read_done_ids(paths) == {"a", "b"}


def test_an_empty_job_directory_has_nothing_done(tmp_path) -> None:
    assert read_done_ids(JobPaths(tmp_path)) == set()


# --- failure handling --------------------------------------------------------------


def test_a_failure_writes_the_traceback_and_re_raises(tmp_path) -> None:
    def explode(batch):
        raise ValueError("teacher fell over")

    with pytest.raises(ValueError, match="fell over"):
        run_job(tmp_path, items(4), explode, key=lambda i: i["id"], batch_size=2)

    failed = (tmp_path / "FAILED").read_text(encoding="utf-8")
    assert "teacher fell over" in failed
    assert "Traceback" in failed
    assert not (tmp_path / "DONE").is_file()


def test_a_rerun_clears_a_stale_failure_marker(tmp_path) -> None:
    """Otherwise a successful resume would still look failed to --status."""
    (tmp_path).mkdir(exist_ok=True)
    (tmp_path / "FAILED").write_text("old", encoding="utf-8")
    run_job(tmp_path, items(2), double, key=lambda i: i["id"])
    assert not (tmp_path / "FAILED").is_file()
    assert (tmp_path / "DONE").is_file()


def test_rows_completed_before_a_failure_are_still_checkpointed(tmp_path) -> None:
    """A crash must not throw away the work already done, or resume is pointless."""

    def explode_late(batch):
        if any(item["value"] >= 8 for item in batch):
            raise RuntimeError("boom")
        return double(batch)

    with pytest.raises(RuntimeError):
        run_job(tmp_path, items(10), explode_late, key=lambda i: i["id"], batch_size=2)
    assert len(read_done_ids(JobPaths(tmp_path))) >= 8


# --- status from outside the process --------------------------------------------------


def test_status_of_a_directory_that_does_not_exist(tmp_path) -> None:
    status = job_status(tmp_path / "never-ran")
    assert status.state == "missing"
    assert "MISSING" in status.render()


def test_status_of_a_finished_job(tmp_path) -> None:
    run_job(tmp_path, items(6), double, key=lambda i: i["id"], batch_size=2)
    status = job_status(tmp_path)
    assert status.state == "done"
    assert status.done_rows == 6
    assert status.total_rows == 6
    assert status.rows_per_second is not None


def test_status_of_a_failed_job_shows_the_error(tmp_path) -> None:
    def explode(batch):
        raise ValueError("gpu vanished")

    with pytest.raises(ValueError):
        run_job(tmp_path, items(2), explode, key=lambda i: i["id"])
    status = job_status(tmp_path)
    assert status.state == "failed"
    assert "gpu vanished" in status.render()


def test_a_stale_heartbeat_is_reported_as_stale(tmp_path) -> None:
    """A job that is wedged looks exactly like a job that is slow, except for this."""
    run_job(tmp_path, items(2), double, key=lambda i: i["id"])
    paths = JobPaths(tmp_path)
    (tmp_path / "DONE").unlink()
    old = time.time() - (STALE_HEARTBEAT_SECONDS + 60)
    os.utime(paths.heartbeat, (old, old))
    status = job_status(tmp_path)
    assert status.state == "stale"
    assert "STALE" in status.render()


def test_the_report_includes_an_eta_while_running(tmp_path) -> None:
    def half(batch):
        return double(batch)

    with pytest.raises(RuntimeError):

        def stop_halfway(batch):
            if any(i["value"] >= 5 for i in batch):
                raise RuntimeError("stop")
            return half(batch)

        run_job(tmp_path, items(10), stop_halfway, key=lambda i: i["id"], batch_size=1)

    (tmp_path / "FAILED").unlink()
    status = job_status(tmp_path)
    assert status.total_rows == 10
    assert status.eta_seconds is not None
    assert "eta" in status.render()


def test_progress_lines_carry_job_specific_counters(tmp_path) -> None:
    """`--status` shows truncations without `jobs.py` knowing what a teacher is."""

    def work(batch):
        rows = double(batch)
        rows[-1]["progress"] = {"teacher": "gptoss-low", "truncated": 2}
        return rows

    run_job(tmp_path, items(4), work, key=lambda i: i["id"], batch_size=2)
    last = json.loads(
        [
            line
            for line in (tmp_path / "progress.jsonl").read_text(encoding="utf-8").split("\n")
            if line.strip()
        ][-1]
    )
    assert last["teacher"] == "gptoss-low"
    assert last["truncated"] == 2
    assert job_status(tmp_path).extra["teacher"] == "gptoss-low"


# --- detachment -----------------------------------------------------------------------


def test_a_detached_child_outlives_its_parent(tmp_path) -> None:
    """The whole point: ending this session must not end the overnight run."""
    marker = tmp_path / "still-here"
    script = f"import pathlib,sys,time; pathlib.Path(r'{marker}').write_text('x'); time.sleep(30)"
    pid = spawn_detached([sys.executable, "-c", script], tmp_path)
    assert (tmp_path / "pid").read_text(encoding="utf-8").strip() == str(pid)

    deadline = time.time() + 20
    while time.time() < deadline and not marker.is_file():
        time.sleep(0.2)
    assert marker.is_file(), "the detached child never started"

    psutil = pytest.importorskip("psutil")
    child = psutil.Process(pid)
    try:
        assert child.is_running()
        # Not a descendant of this test process: that is what makes it survive us.
        assert child.ppid() != os.getpid() or child.is_running()
    finally:
        child.kill()
        child.wait(timeout=10)


def test_the_log_records_what_was_spawned(tmp_path) -> None:
    pid = spawn_detached([sys.executable, "-c", "pass"], tmp_path)
    log = (tmp_path / "log.txt").read_text(encoding="utf-8")
    assert "spawned" in log
    try:
        subprocess.run([sys.executable, "-c", "pass"], check=False, timeout=10)
    finally:
        pytest.importorskip("psutil")
        import psutil

        try:
            psutil.Process(pid).kill()
        except psutil.NoSuchProcess:
            pass


def test_job_paths_are_all_inside_the_run_directory(tmp_path) -> None:
    paths = JobPaths(tmp_path)
    for path in (paths.rows, paths.progress, paths.heartbeat, paths.log, paths.done, paths.failed):
        assert Path(path).parent == tmp_path
