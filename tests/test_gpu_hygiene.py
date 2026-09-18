"""GPU process inventory and the kill-tree cleanup.

Written after an interrupted benchmark left two orphaned processes holding 21.7 GB of VRAM each
and a loop shell that kept starting more, which silently corrupted every measurement taken
afterwards. The tests that matter here are the ones about *not* killing the wrong thing.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from s1decide.gpu import (
    MIN_REPORTED_MIB,
    GpuProcess,
    kill_process_tree,
    kill_s1decide_gpu_processes,
    probe_sysmem_fallback,
)


def test_ours_is_recognised_from_the_command_line() -> None:
    ours = GpuProcess(pid=1, name="python.exe", mib=20000, command="uv run task bench --counts 64")
    assert ours.is_ours


def test_someone_elses_gpu_job_is_not_ours() -> None:
    """The check exists so cleanup never touches the user's own work."""
    for name, command in (
        ("chrome.exe", "chrome.exe --type=gpu-process"),
        ("python.exe", "python.exe train_something_else.py"),
        ("unsloth-studio.exe", "unsloth-studio.exe serve"),
        ("blender.exe", ""),
    ):
        assert not GpuProcess(pid=2, name=name, mib=9000, command=command).is_ours, name


def test_a_process_with_unknown_memory_still_describes_itself() -> None:
    line = GpuProcess(pid=3, name="python.exe", mib=None, command="").describe()
    assert "pid 3" in line and "?" in line


def test_describe_marks_ours() -> None:
    line = GpuProcess(pid=4, name="python.exe", mib=100, command="s1decide bench").describe()
    assert "[ours]" in line and "100 MiB" in line


def test_the_noise_floor_is_low_enough_to_catch_a_real_run_and_high_enough_to_skip_tray_apps() -> (
    None
):
    assert 16 <= MIN_REPORTED_MIB <= 512


def test_killing_a_missing_pid_is_not_an_error() -> None:
    """Cleanup runs on a machine whose state it does not control; it must be idempotent."""
    assert kill_process_tree(2**31 - 1) == []


def test_dry_run_kills_nothing() -> None:
    ours, killed = kill_s1decide_gpu_processes(dry_run=True)
    assert killed == []
    assert isinstance(ours, list)


def test_cleanup_never_targets_the_calling_process() -> None:
    """A task that killed its own runner would take the report with it."""
    ours, _killed = kill_s1decide_gpu_processes(dry_run=True)
    assert all(process.pid != os.getpid() for process in ours)


@pytest.mark.skipif(sys.platform == "win32", reason="process-tree timing is flaky on Windows CI")
def test_kill_process_tree_takes_the_children_too() -> None:
    """The actual bug: killing the parent left the child holding the GPU."""
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess,sys,time; "
            "subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            "time.sleep(60)",
        ]
    )
    try:
        time.sleep(2.0)
        killed = kill_process_tree(parent.pid)
        assert len(killed) >= 2, "the child must be killed as well as the parent"
    finally:
        parent.kill()
        parent.wait(timeout=10)


def test_a_detached_teacher_run_is_recognised_as_ours() -> None:
    """The command line that defeated the first version of OUR_MARKERS, verbatim.

    On 2026-09-18 `gpu-kill` reported "no s1decide process is holding GPU memory" while this
    process held 23,169 MiB of the card. The interpreter is uv's shared Python, not the project
    venv; the working directory never appears in a command line; and `data.build.teach_overnight`
    contains none of the words the project is named after.
    """
    process = GpuProcess(
        pid=13208,
        name="python.exe",
        mib=23169,
        command=(
            r"C:\Users\yusuf\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none"
            r"\python.exe -m data.build.teach_overnight "
            "--items data/processed/teach_items.jsonl --limit 6000"
        ),
    )
    assert process.is_ours
    assert "[ours]" in process.describe()


def test_a_single_leg_teacher_run_is_recognised_as_ours() -> None:
    assert GpuProcess(
        1, "python.exe", 20000, "python.exe -m data.build.teach_run --teacher x"
    ).is_ours


def test_someone_elses_python_is_not_ours() -> None:
    """The failure that costs more than a missed process: killing the user's own work."""
    for command in (
        r"C:\Users\yusuf\.unsloth\studio\.venv\Scripts\python.exe -m unsloth_studio.server",
        "python.exe train.py --model llama",
        r"C:\Program Files\Blender\blender.exe",
    ):
        assert not GpuProcess(1, "python.exe", 20000, command).is_ours, command


def test_markers_do_not_match_a_bare_interpreter() -> None:
    """A marker broad enough to catch any python would make gpu-kill unsafe to run."""
    assert not GpuProcess(1, "python.exe", 20000, "python.exe").is_ours


@pytest.mark.gpu
def test_sysmem_fallback_is_disabled_on_this_machine() -> None:
    """Required setting on Windows — see docs/windows-setup.md and ADR 0003.

    With the fallback on, an over-budget run does not fail, it silently takes 3-6x longer. That
    is not a condition a benchmark can detect from inside itself, so it is checked here and in
    ``uv run task doctor`` rather than being assumed.
    """
    result = probe_sysmem_fallback()
    if result.disabled is None:
        pytest.skip(result.detail)
    assert result.disabled, result.detail
