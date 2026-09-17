"""Tests for the cross-platform task runner."""

from __future__ import annotations

import pytest

from s1decide import tasks
from s1decide.tasks import CheckResult, Status

# Every task named in CLAUDE.md "Commands". `lint` and `fmt` are extras we add.
REQUIRED_TASKS = {
    "doctor",
    "setup",
    "data",
    "smoke",
    "train",
    "eval",
    "bench",
    "serve",
    "test",
    "test-gpu",
    "gguf",
}

# Tasks that must fail loudly until the work behind them exists.
NOT_IMPLEMENTED_TASKS = {"data", "smoke", "train", "bench", "serve", "gguf"}


def test_every_documented_task_is_registered() -> None:
    missing = REQUIRED_TASKS - set(tasks.TASKS)
    assert not missing, f"tasks missing from the registry: {sorted(missing)}"


def test_task_names_are_shell_safe() -> None:
    for name in tasks.TASKS:
        assert name == name.lower()
        assert " " not in name


@pytest.mark.parametrize("name", sorted(NOT_IMPLEMENTED_TASKS))
def test_unimplemented_tasks_raise_rather_than_succeed(name: str) -> None:
    with pytest.raises(NotImplementedError, match="not implemented"):
        tasks.run_task(name)


@pytest.mark.parametrize("name", sorted(NOT_IMPLEMENTED_TASKS))
def test_unimplemented_tasks_exit_non_zero_through_main(name: str) -> None:
    assert tasks.main([name]) != 0


@pytest.mark.parametrize("name", sorted(NOT_IMPLEMENTED_TASKS))
def test_unimplemented_tasks_are_labelled_in_the_listing(name: str) -> None:
    assert "not implemented" in tasks.TASKS[name].summary


def test_unknown_task_is_a_usage_error() -> None:
    assert tasks.main(["definitely-not-a-task"]) == 2


def test_no_arguments_is_a_usage_error() -> None:
    assert tasks.main([]) == 2


def test_help_succeeds() -> None:
    assert tasks.main(["--help"]) == 0
    assert tasks.main(["--list"]) == 0


def test_run_task_rejects_unknown_names() -> None:
    with pytest.raises(KeyError):
        tasks.run_task("definitely-not-a-task")


def test_repo_root_contains_pyproject() -> None:
    assert (tasks.repo_root() / "pyproject.toml").is_file()


def test_usage_lists_every_task() -> None:
    usage = tasks._usage()
    for name in tasks.TASKS:
        assert name in usage


# --- doctor checks that are safe without a GPU -------------------------------


@pytest.mark.parametrize(
    "check",
    [
        tasks.check_platform,
        tasks.check_no_flash_attn,
        tasks.check_utf8,
        tasks.check_hf_cache,
        tasks.check_long_paths,
    ],
)
def test_cpu_safe_checks_return_a_result(check) -> None:
    result = check()
    assert isinstance(result, CheckResult)
    assert isinstance(result.status, Status)
    assert result.detail


def test_flash_attn_is_not_installed() -> None:
    """CLAUDE.md forbids flash-attn outright; this guards against dependency creep."""
    assert tasks.check_no_flash_attn().status is Status.OK


def test_hf_cache_dir_is_absolute() -> None:
    assert tasks.hf_cache_dir().is_absolute()


def test_doctor_checks_are_unique() -> None:
    names = [check.__name__ for check in tasks.DOCTOR_CHECKS]
    assert len(names) == len(set(names))
