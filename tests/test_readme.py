"""The README's numbers must come from `results/`, like every other number in the repository.

A README is the page people read first and the one nobody re-checks. It is therefore exactly
where a stale number survives longest. The marked blocks below are wired to the same committed
runs the model card uses, and these tests fail if a rescore moves a figure the README still
claims.
"""

from __future__ import annotations

import json
import re

import pytest
from eval.baselines import REFERENCE_RUNS

from s1decide.tasks import repo_root

#: Blocks in the README whose contents are backed by a committed run.
SLOTS = ("zeroshot", "latency")


@pytest.fixture(scope="module")
def readme() -> str:
    return (repo_root() / "README.md").read_text(encoding="utf-8")


def slot(readme: str, name: str) -> str:
    """The text between the markers for one metrics block."""
    match = re.search(rf"<!--metrics:{name}-->(.*?)<!--/metrics:{name}-->", readme, re.S)
    assert match, f"the README has no <!--metrics:{name}--> block"
    return match.group(1)


@pytest.mark.parametrize("name", SLOTS)
def test_every_declared_slot_exists_and_is_not_empty(readme: str, name: str) -> None:
    assert slot(readme, name).strip()


def test_the_zero_shot_block_matches_the_committed_reference_run(readme: str) -> None:
    payload = json.loads(
        (repo_root() / "results" / REFERENCE_RUNS["zeroshot"].run_id / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    block = slot(readme, "zeroshot")
    calibrated = payload["overall_calibrated"]
    for metric in ("accuracy", "ece"):
        assert f"{calibrated[metric]:.4f}" in block, (
            f"README zero-shot {metric} does not match {REFERENCE_RUNS['zeroshot'].run_id}: "
            f"expected {calibrated[metric]:.4f}"
        )
    bss = payload["skill_calibrated"]["base_rate"]["brier_multiclass"]
    assert f"+{bss:.4f}" in block
    selective = payload["risk_coverage_calibrated"]["selective_accuracy"]["0.8"]
    assert f"{selective:.4f}" in block


def test_the_latency_block_matches_a_committed_bench_run(readme: str) -> None:
    """Whichever bench run the README quotes must exist and still say these numbers."""
    block = slot(readme, "latency")
    directory = repo_root() / "results"
    runs = sorted(p for p in directory.glob("*-latency") if (p / "latency.json").is_file())
    assert runs, "no committed latency run to check the README against"

    matching = []
    for run in runs:
        payload = json.loads((run / "latency.json").read_text(encoding="utf-8"))
        if all(f"{p['median_ms']:,.0f} ms" in block for p in payload["points"]):
            matching.append(run)
    assert matching, (
        "the README's latency table matches no committed run; re-run `uv run task bench` and "
        f"update it, or point it at one of {[r.name for r in runs]}"
    )


def test_the_readme_says_there_is_no_trained_model_yet(readme: str) -> None:
    """The single most important claim on the page, and the easiest to forget to remove."""
    assert "no trained model yet" in readme.lower()
    assert "pre-alpha" in readme.lower()


def test_the_readme_does_not_overclaim_flat_latency(readme: str) -> None:
    """ADR 0003 retired that claim; the README must not quietly restore it."""
    assert "not flat" in readme.lower()


def test_the_readme_puts_skill_before_ece(readme: str) -> None:
    """Reporting ECE without the control beside it is the failure mode the project exists to avoid."""
    block = slot(readme, "zeroshot")
    assert block.index("Brier skill") < block.index("ECE")
    assert "base rate" in readme.lower()


def test_every_linked_document_exists(readme: str) -> None:
    root = repo_root()
    missing = [
        target
        for target in re.findall(r"\]\(([^)]+)\)", readme)
        if not target.startswith(("http://", "https://", "#"))
        and not (root / target.split("#")[0]).exists()
    ]
    assert not missing, f"README links to files that do not exist: {missing}"


def test_every_task_the_readme_advertises_is_registered(readme: str) -> None:
    from s1decide.tasks import TASKS

    advertised = set(re.findall(r"uv run task ([a-z-]+)", readme))
    unknown = sorted(advertised - set(TASKS))
    assert not unknown, f"README advertises tasks that do not exist: {unknown}"


def test_the_readme_names_the_tasks_that_are_not_implemented(readme: str) -> None:
    """Promising a command that raises NotImplementedError is worse than omitting it."""
    from s1decide.tasks import TASKS

    not_implemented = sorted(
        name for name, task in TASKS.items() if "not implemented" in task.summary
    )
    for name in ("smoke", "train"):
        if name in not_implemented:
            assert name in readme, f"`{name}` is a placeholder and the README should say so"
