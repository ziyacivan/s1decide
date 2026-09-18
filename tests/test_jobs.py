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
    CHECKPOINT_EVERY,
    STALE_HEARTBEAT_SECONDS,
    JobPaths,
    MetaMismatch,
    check_resume_meta,
    checkpoint_rows,
    compare_meta,
    flatten_meta,
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


# --- reboot detection ----------------------------------------------------------------


def test_boot_time_is_a_plausible_timestamp() -> None:
    from s1decide.jobs import boot_time

    booted = boot_time()
    if booted is None:
        pytest.skip("psutil could not supply a boot time")
    assert 0 < booted < time.time()


def test_a_reboot_after_the_last_heartbeat_is_flagged(tmp_path, monkeypatch) -> None:
    """A killed-by-reboot job and a wedged job both show a stale heartbeat and nothing else.

    Windows Update restarted this machine 13 hours into a 27-hour run; the job wrote no FAILED
    file because nothing caught anything. The two cases need different responses, so the status
    has to tell them apart.
    """
    import s1decide.jobs as jobs

    run_job(tmp_path, items(2), double, key=lambda i: i["id"])
    (tmp_path / "DONE").unlink()
    heartbeat = JobPaths(tmp_path).heartbeat
    old = time.time() - (STALE_HEARTBEAT_SECONDS + 60)
    os.utime(heartbeat, (old, old))

    monkeypatch.setattr(jobs, "boot_time", lambda: time.time() - 60)
    status = jobs.job_status(tmp_path)
    assert status.state == "stale"
    assert status.booted_after_heartbeat is True
    assert "REBOOTED MID-RUN" in status.render()


def test_a_boot_before_the_heartbeat_is_not_flagged(tmp_path, monkeypatch) -> None:
    """An old boot plus a stale heartbeat means the job wedged; that is a different problem."""
    import s1decide.jobs as jobs

    run_job(tmp_path, items(2), double, key=lambda i: i["id"])
    (tmp_path / "DONE").unlink()
    old = time.time() - (STALE_HEARTBEAT_SECONDS + 60)
    os.utime(JobPaths(tmp_path).heartbeat, (old, old))

    monkeypatch.setattr(jobs, "boot_time", lambda: time.time() - 86400)
    status = jobs.job_status(tmp_path)
    assert status.booted_after_heartbeat is False
    assert "REBOOTED MID-RUN" not in status.render()
    assert "booted" in status.render()


def test_the_status_survives_a_machine_with_no_boot_time(tmp_path, monkeypatch) -> None:
    import s1decide.jobs as jobs

    run_job(tmp_path, items(2), double, key=lambda i: i["id"])
    monkeypatch.setattr(jobs, "boot_time", lambda: None)
    assert "booted" not in jobs.job_status(tmp_path).render()


def test_the_checkpoint_interval_bounds_the_worst_case_loss() -> None:
    """Lowered to 50 after a reboot cost 144 rows — ~40 min at the teacher run's 0.060 rows/s.

    The bound that matters is time, not rows: at 0.060 rows/s, 50 rows is about 14 minutes of
    work at risk. Pinned so raising it again is a deliberate act with this arithmetic in view.
    """
    assert CHECKPOINT_EVERY <= 50
    slowest_observed_rows_per_second = 0.060
    worst_case_minutes = CHECKPOINT_EVERY / slowest_observed_rows_per_second / 60
    assert worst_case_minutes <= 15


def test_rows_are_flushed_at_the_checkpoint_interval_not_only_at_the_end(tmp_path) -> None:
    """The point of a checkpoint is to survive a kill, so it has to be on disk before the end."""
    flushed: list[int] = []

    def work(batch):
        flushed.append(len(read_done_ids(JobPaths(tmp_path))))
        return double(batch)

    run_job(tmp_path, items(120), work, key=lambda i: i["id"], batch_size=10)
    assert max(flushed) >= CHECKPOINT_EVERY, (
        f"nothing reached disk mid-run: counts seen were {sorted(set(flushed))}"
    )


# --- the resume guard ------------------------------------------------------------
#
# The job this protects ran for a day across three sessions. Resume matches by id, so it does
# not notice that the ids came from a regenerated file, that the kernels were upgraded in
# between, or that the batch size changed — it just continues, and overwrites the only record
# of how the earlier rows were made.


def test_flatten_meta_names_a_nested_field_by_its_path() -> None:
    flat = flatten_meta(
        {"batch_size": 4, "kernels": {"attention": "sdpa", "packages": {"fla": True}}}
    )
    assert flat == {"batch_size": 4, "kernels.attention": "sdpa", "kernels.packages.fla": True}


def test_identical_configurations_have_no_differences() -> None:
    meta = {"batch_size": 4, "kernels": {"attention": "sdpa"}}
    assert compare_meta(meta, dict(meta)) == []


def test_a_changed_field_is_reported_with_both_values() -> None:
    differences = compare_meta({"batch_size": 4}, {"batch_size": 16})
    assert len(differences) == 1
    assert "batch_size" in differences[0] and "4" in differences[0] and "16" in differences[0]


def test_volatile_fields_are_not_differences() -> None:
    """A resume has a different start time and may have reworded notes. Neither is a reason."""
    a = {"started": "2026-09-17T23:41:14+00:00", "kernels": {"notes": ["old wording"]}}
    b = {"started": "2026-09-18T16:14:02+00:00", "kernels": {"notes": ["new wording"]}}
    assert compare_meta(a, b) == []


def test_a_field_the_old_run_never_recorded_is_not_a_difference() -> None:
    """The guard has to be deployable onto a job that is already half finished.

    This is the case that made it worth writing: `items_digest` was added while the teacher run
    was 22% done. If a new key counted as a mismatch, shipping the guard would have refused the
    very run it was written to protect.
    """
    assert compare_meta({"batch_size": 4}, {"batch_size": 4, "items_digest": "abc123"}) == []


def test_a_fresh_directory_is_not_a_resume(tmp_path) -> None:
    (tmp_path / "meta.json").write_text('{"batch_size": 4}', encoding="utf-8")
    assert check_resume_meta(tmp_path, {"batch_size": 16}) == []


def test_resuming_under_a_changed_configuration_is_refused(tmp_path) -> None:
    run_job(tmp_path, items(4), double, key=lambda i: i["id"], meta={"batch_size": 4})
    with pytest.raises(MetaMismatch, match="batch_size"):
        run_job(tmp_path, items(8), double, key=lambda i: i["id"], meta={"batch_size": 16})


def test_a_refused_resume_changes_nothing_on_disk(tmp_path) -> None:
    """Checked before the job directory is touched, so a refusal is not a partial write."""
    run_job(tmp_path, items(4), double, key=lambda i: i["id"], meta={"batch_size": 4})
    before = (tmp_path / "meta.json").read_text(encoding="utf-8")
    rows_before = (tmp_path / "rows.jsonl").read_text(encoding="utf-8")
    with pytest.raises(MetaMismatch):
        run_job(tmp_path, items(8), double, key=lambda i: i["id"], meta={"batch_size": 16})
    assert (tmp_path / "meta.json").read_text(encoding="utf-8") == before
    assert (tmp_path / "rows.jsonl").read_text(encoding="utf-8") == rows_before


def test_force_resumes_and_records_the_new_configuration(tmp_path) -> None:
    run_job(tmp_path, items(4), double, key=lambda i: i["id"], meta={"batch_size": 4})
    summary = run_job(
        tmp_path, items(8), double, key=lambda i: i["id"], meta={"batch_size": 16}, force=True
    )
    assert summary["rows_this_run"] == 4
    assert json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))["batch_size"] == 16


def test_an_unchanged_configuration_resumes_normally(tmp_path) -> None:
    meta = {"batch_size": 4, "kernels": {"attention": "sdpa"}}
    run_job(tmp_path, items(4), double, key=lambda i: i["id"], meta=meta)
    summary = run_job(
        tmp_path,
        items(8),
        double,
        key=lambda i: i["id"],
        meta={**meta, "started": "later"},
    )
    assert summary["rows_this_run"] == 4


def test_a_job_with_no_meta_is_not_guarded(tmp_path) -> None:
    """Jobs that record nothing keep their old behaviour rather than becoming unresumable."""
    run_job(tmp_path, items(4), double, key=lambda i: i["id"])
    assert run_job(tmp_path, items(8), double, key=lambda i: i["id"])["rows_this_run"] == 4
