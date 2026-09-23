"""`task push`: main moves only to a commit whose `ci` check is green, and only by fast-forward."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from s1decide.push import check_state, failure_messages, parse_github_remote, push_through_gate

SHA = "a" * 40
BASE = "b" * 40


@pytest.mark.parametrize(
    "url",
    [
        "git@github.com:ziyacivan/s1decide.git",
        "https://github.com/ziyacivan/s1decide",
        "https://github.com/ziyacivan/s1decide.git",
        "ssh://git@github.com/ziyacivan/s1decide.git",
    ],
)
def test_remote_urls_resolve_to_owner_and_repo(url: str) -> None:
    assert parse_github_remote(url) == ("ziyacivan", "s1decide")


def test_a_non_github_remote_is_refused() -> None:
    with pytest.raises(ValueError):
        parse_github_remote("git@gitlab.com:someone/thing.git")


def run(name: str, status: str, conclusion: str | None = None, id_: int = 1) -> dict:
    return {"id": id_, "name": name, "status": status, "conclusion": conclusion}


@pytest.mark.parametrize(
    ("runs", "state"),
    [
        ([], "pending"),
        ([run("test (ubuntu-latest)", "completed", "success")], "pending"),
        ([run("ci", "in_progress")], "pending"),
        ([run("ci", "completed", "success")], "success"),
        ([run("ci", "completed", "failure")], "failure"),
        # Anything but green keeps the gate shut.
        ([run("ci", "completed", "cancelled")], "failure"),
        ([run("ci", "completed", "skipped")], "failure"),
        # A re-run supersedes the earlier result.
        ([run("ci", "completed", "failure", 1), run("ci", "completed", "success", 2)], "success"),
    ],
)
def test_check_state(runs: list[dict], state: str) -> None:
    assert check_state({"check_runs": runs}) == state


def test_failure_messages_read_the_annotations_of_failed_runs() -> None:
    payload = {"check_runs": [run("test (ubuntu-latest)", "completed", "failure", 7)]}

    def fetch(url: str):
        assert url.endswith("/check-runs/7/annotations")
        return [
            {"annotation_level": "warning", "message": "Node 20 is deprecated"},
            {"annotation_level": "failure", "message": "FAILED tests/test_x.py::test_y"},
        ]

    lines = failure_messages("o", "r", payload, fetch)
    assert lines == ["test (ubuntu-latest): failure", "  FAILED tests/test_x.py::test_y"]


class FakeGit:
    """Records every git call; answers the ones the flow reads."""

    def __init__(self, *, head: str = SHA, fast_forward: bool = True) -> None:
        self.calls: list[list[str]] = []
        self.head = head
        self.fast_forward = fast_forward

    def __call__(self, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        args = list(args)
        self.calls.append(args)
        out, code = "", 0
        if args[:2] == ["remote", "get-url"]:
            out = "git@github.com:ziyacivan/s1decide.git"
        elif args == ["rev-parse", "HEAD"]:
            out = self.head
        elif args[0] == "rev-parse":
            out = BASE
        elif args[0] == "merge-base":
            code = 0 if self.fast_forward else 1
        return subprocess.CompletedProcess(args, code, out, "")

    def pushes(self) -> list[list[str]]:
        return [c for c in self.calls if c[0] == "push"]


def gate(git: FakeGit, states: list[str], **kwargs):
    """Run the flow with `ci` reporting `states` on successive polls."""
    polls = iter(states)
    conclusion = {"success": "success", "failure": "failure"}

    def fetch(url: str):
        if "annotations" in url:
            return []
        state = next(polls, states[-1])
        if state == "pending":
            return {"check_runs": [run("ci", "in_progress")]}
        return {"check_runs": [run("ci", "completed", conclusion[state])]}

    clock = iter(range(0, 10_000, 45))
    return push_through_gate(
        Path(),
        git=git,
        fetch=fetch,
        sleep=lambda _s: None,
        clock=lambda: next(clock),
        echo=lambda _m: None,
        **kwargs,
    )


def test_green_ci_pushes_the_lane_first_and_then_main_by_sha() -> None:
    git = FakeGit()
    assert gate(git, ["pending", "pending", "success"]) == 0
    assert git.pushes() == [
        ["push", "--force", "origin", f"{SHA}:refs/heads/wip"],
        ["push", "origin", f"{SHA}:refs/heads/main"],
    ]


def test_red_ci_never_touches_main() -> None:
    git = FakeGit()
    assert gate(git, ["pending", "failure"]) == 1
    assert git.pushes() == [["push", "--force", "origin", f"{SHA}:refs/heads/wip"]]


def test_a_check_that_never_finishes_times_out_without_touching_main() -> None:
    git = FakeGit()
    assert gate(git, ["pending"], timeout=200) == 1
    assert all("main" not in " ".join(c) for c in git.pushes())


def test_a_commit_that_does_not_fast_forward_main_is_refused_before_any_push() -> None:
    git = FakeGit(fast_forward=False)
    assert gate(git, ["success"]) == 2
    assert git.pushes() == []


def test_nothing_to_do_when_main_is_already_there() -> None:
    git = FakeGit(head=BASE)
    assert gate(git, ["success"]) == 0
    assert git.pushes() == []


def test_a_token_in_the_process_environment_wins(monkeypatch) -> None:
    from s1decide.push import github_token

    monkeypatch.setenv("GITHUB_TOKEN", "from-process")
    assert github_token() == "from-process"
