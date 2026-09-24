"""The run watcher: one alert per trouble, and no false reboot from boot-time drift."""

from __future__ import annotations

import json
import os
import time

from s1decide.watch import check_run, interval_rates


def run_dir(tmp_path, progress=None, heartbeat_age=0.0):
    run = tmp_path / "run"
    run.mkdir()
    hb = run / "heartbeat"
    hb.write_text("x", encoding="utf-8")
    past = time.time() - heartbeat_age
    os.utime(hb, (past, past))
    if progress:
        (run / "progress.jsonl").write_text(
            "".join(json.dumps(p) + "\n" for p in progress), encoding="utf-8"
        )
    return run


def check(run, **kw):
    base = dict(
        eval_every=5000, boot_at_start=1000.0, boot_now=1000.0, pid_alive=True, now=time.time()
    )
    return check_run(run, **{**base, **kw})


def test_a_healthy_run_raises_nothing(tmp_path) -> None:
    assert check(run_dir(tmp_path)) is None


def test_boot_time_drift_of_a_second_is_not_a_reboot(tmp_path) -> None:
    """The false alarm of 2026-09-24: exact comparison of a value Windows derives from uptime."""
    assert check(run_dir(tmp_path), boot_now=1001.3) is None


def test_a_real_reboot_is_reported(tmp_path) -> None:
    assert check(run_dir(tmp_path), boot_now=90000.0).startswith("REBOOTED")


def test_markers_win_over_everything(tmp_path) -> None:
    run = run_dir(tmp_path)
    (run / "STOPPED").write_text("stopped at 10000 rows\nreason", encoding="utf-8")
    assert check(run, pid_alive=False).startswith("STOPPED")


def test_a_vanished_process_without_a_marker_is_reported(tmp_path) -> None:
    assert check(run_dir(tmp_path), pid_alive=False).startswith("GONE")


def test_a_stale_heartbeat_is_reported(tmp_path) -> None:
    assert check(run_dir(tmp_path, heartbeat_age=20 * 60)).startswith("STALE")


def test_evaluation_intervals_do_not_count_as_slowdowns(tmp_path) -> None:
    progress = [{"rows": 200 * i, "elapsed_seconds": 180.0 * i} for i in range(8)]
    # an interval that crosses row 5,000 includes a 22-minute evaluation
    progress += [
        {"rows": 5000 + 200 * i, "elapsed_seconds": 1440 + 1320 + 180 * i} for i in range(1, 4)
    ]
    assert check(run_dir(tmp_path, progress=progress)) is None
    assert len(interval_rates(progress, 5000)) == len(progress) - 2


def test_a_real_slowdown_is_reported(tmp_path) -> None:
    progress = [{"rows": 200 * i, "elapsed_seconds": 180.0 * i} for i in range(8)]
    progress.append({"rows": 1800, "elapsed_seconds": progress[-1]["elapsed_seconds"] + 800})
    assert check(run_dir(tmp_path, progress=progress)).startswith("SLOWDOWN")
