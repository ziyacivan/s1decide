"""Push through the CI gate: ``wip`` first, ``main`` only once ``ci`` is green on that commit.

`main` is protected: the `ci` check is required and there is no admin bypass, so a direct push of
an unchecked commit is rejected. The flow this module automates (``uv run task push``):

1. refuse unless ``HEAD`` fast-forwards ``origin/main``;
2. push ``HEAD`` to the scratch branch ``wip`` (forced — ``wip`` is a lane, not history);
3. poll the GitHub check-runs API for the ``ci`` check on that exact commit;
4. on success, push that same commit to ``main`` by SHA, so the commit that was checked is the
   commit that lands; on failure or timeout, print the failing tests' annotations and stop.

No ``gh``: the repository is public, so reads need no token. ``GITHUB_TOKEN`` or ``GH_TOKEN`` is
used when set, which lifts the unauthenticated limit of 60 requests an hour.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "CHECK_NAME",
    "check_state",
    "failure_messages",
    "fetch_json",
    "github_token",
    "parse_github_remote",
    "push_through_gate",
]

#: The one required check (see `.github/workflows/ci.yml`: the aggregate job).
CHECK_NAME = "ci"

API = "https://api.github.com"

GitRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
Fetcher = Callable[[str], Any]


def parse_github_remote(url: str) -> tuple[str, str]:
    """Owner and repository name from a GitHub remote URL.

    Args:
        url: ``git@github.com:o/r.git``, ``https://github.com/o/r(.git)`` or
            ``ssh://git@github.com/o/r.git``.

    Returns:
        ``(owner, repo)``.

    Raises:
        ValueError: If the URL is not a GitHub remote.
    """
    match = re.search(r"github\.com[:/]+([^/]+)/([^/]+?)(?:\.git)?/?$", url.strip())
    if not match:
        raise ValueError(f"not a GitHub remote: {url!r}")
    return match[1], match[2]


def check_state(payload: dict[str, Any], name: str = CHECK_NAME) -> str:
    """Reduce a check-runs response to ``pending``, ``success`` or ``failure``.

    Args:
        payload: ``GET /repos/{o}/{r}/commits/{sha}/check-runs`` JSON.
        name: The check to read. When it ran more than once on the commit, the latest counts.

    Returns:
        ``pending`` until the check exists and has completed; then ``success`` only for a
        ``success`` conclusion. Every other conclusion (``failure``, ``cancelled``,
        ``timed_out``, ``skipped``, ...) is ``failure``: the gate opens on green, not on
        "not red".
    """
    runs = [run for run in payload.get("check_runs", []) if run.get("name") == name]
    if not runs:
        return "pending"
    latest = max(runs, key=lambda run: run.get("id", 0))
    if latest.get("status") != "completed":
        return "pending"
    return "success" if latest.get("conclusion") == "success" else "failure"


def github_token() -> str | None:
    """``GITHUB_TOKEN`` or ``GH_TOKEN``, from the process or, on Windows, the user environment.

    A variable set with ``setx`` or the Settings dialog reaches only processes started after
    it, so a long-lived shell or agent session never sees it in ``os.environ``. The user-scope
    value in ``HKCU\\Environment`` is read as a fallback. The token is never printed.
    """
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                for name in ("GITHUB_TOKEN", "GH_TOKEN"):
                    try:
                        value, _ = winreg.QueryValueEx(key, name)
                    except FileNotFoundError:
                        continue
                    if value:
                        return str(value)
        except OSError:
            return None
    return None


def fetch_json(url: str) -> Any:
    """GET a GitHub API URL and decode it, authenticated if a token is available."""
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    token = github_token()
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def failure_messages(owner: str, repo: str, payload: dict[str, Any], fetch: Fetcher) -> list[str]:
    """The failure annotations of every failed check run on the commit.

    CI raises each failing test as an annotation (see the workflow), so this is what broke,
    readable without signing in.
    """
    messages: list[str] = []
    for run in payload.get("check_runs", []):
        if run.get("conclusion") not in {"failure", "timed_out", "cancelled"}:
            continue
        messages.append(f"{run.get('name')}: {run.get('conclusion')}")
        for note in fetch(f"{API}/repos/{owner}/{repo}/check-runs/{run['id']}/annotations"):
            if note.get("annotation_level") == "failure":
                messages.append(f"  {note.get('message', '')[:500]}")
    return messages


def _git(root: Path) -> GitRunner:
    def run(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)

    return run


def push_through_gate(
    root: Path,
    *,
    remote: str = "origin",
    lane: str = "wip",
    target: str = "main",
    timeout: float = 45 * 60,
    interval: float = 45.0,
    git: GitRunner | None = None,
    fetch: Fetcher = fetch_json,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    echo: Callable[[str], None] = print,
) -> int:
    """Push ``HEAD`` to ``lane``, wait for ``ci`` on it, then fast-forward ``target`` to it.

    Args:
        root: Repository root.
        remote: Remote name.
        lane: Scratch branch CI runs on before ``target`` moves.
        target: The protected branch.
        timeout: Seconds to wait for the check before giving up (``target`` untouched).
        interval: Seconds between polls.
        git, fetch, sleep, clock, echo: Injected for tests.

    Returns:
        0 when ``target`` now points at ``HEAD`` (or already did); non-zero otherwise, with the
        reason printed.
    """
    git = git or _git(root)

    def must(args: Sequence[str]) -> str:
        done = git(args)
        if done.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
        return done.stdout.strip()

    try:
        owner, repo = parse_github_remote(must(["remote", "get-url", remote]))
        must(["fetch", remote, target])
        sha = must(["rev-parse", "HEAD"])
        base = must(["rev-parse", f"{remote}/{target}"])
    except (RuntimeError, ValueError) as exc:
        echo(f"push: {exc}")
        return 2
    if sha == base:
        echo(f"push: {target} is already at {sha[:7]}; nothing to do")
        return 0
    if git(["merge-base", "--is-ancestor", base, sha]).returncode != 0:
        echo(
            f"push: HEAD {sha[:7]} does not fast-forward {remote}/{target} {base[:7]}; "
            "rebase onto it first"
        )
        return 2
    if git(["status", "--porcelain"]).stdout.strip():
        echo("push: note — uncommitted changes are not part of what is being pushed")

    echo(f"push: {sha[:7]} -> {remote}/{lane}")
    try:
        must(["push", "--force", remote, f"{sha}:refs/heads/{lane}"])
    except RuntimeError as exc:
        echo(f"push: {exc}")
        return 1

    url = f"{API}/repos/{owner}/{repo}/commits/{sha}/check-runs?per_page=100"
    deadline = clock() + timeout
    state, payload = "pending", {}
    while True:
        try:
            payload = fetch(url)
            state = check_state(payload)
        except Exception as exc:  # a transient API error is a pending poll, not a verdict
            echo(f"push: poll failed ({type(exc).__name__}); retrying")
            state = "pending"
        if state != "pending" or clock() >= deadline:
            break
        echo(f"push: waiting for `{CHECK_NAME}` on {sha[:7]} ...")
        sleep(interval)

    if state == "pending":
        echo(f"push: `{CHECK_NAME}` did not finish within {timeout:.0f}s; {target} not moved")
        return 1
    if state == "failure":
        echo(f"push: `{CHECK_NAME}` failed on {sha[:7]}; {target} not moved")
        try:
            for line in failure_messages(owner, repo, payload, fetch):
                echo(line)
        except Exception as exc:  # pragma: no cover - diagnostics only
            echo(f"push: could not read annotations ({type(exc).__name__})")
        return 1

    echo(f"push: `{CHECK_NAME}` green on {sha[:7]}; fast-forwarding {target}")
    try:
        must(["push", remote, f"{sha}:refs/heads/{target}"])
    except RuntimeError as exc:
        echo(f"push: {exc}")
        return 1
    echo(f"push: {remote}/{target} is now {sha[:7]}")
    return 0
