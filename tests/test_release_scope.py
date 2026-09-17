"""ADR 0006 — the name is final, and the model card's numbers are not typed by hand.

Two rules that are cheap to state and easy to break by accident, so both are tests rather than
notes. The naming rule matters most where it is hardest to undo: a published package name, a
Hub model id, a repository URL. The number rule matters because a model card is the one document
people quote, and a hand-typed figure drifts silently the moment a run is redone.
"""

from __future__ import annotations

import json
import re
import tomllib

import pytest

from s1decide.tasks import repo_root

#: Names we never call ourselves. Referring to them in prose and comparisons is expected and is
#: a deliverable; naming an artefact after them is not (CLAUDE.md hard rules, ADR 0006).
FORBIDDEN_IN_IDENTIFIERS = ("jev", "typesafe", "system one model", "systemonemodel")

#: The one name, everywhere.
PROJECT_NAME = "s1decide"


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads((repo_root() / "pyproject.toml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def model_card() -> str:
    return (repo_root() / "docs" / "model-card.md").read_text(encoding="utf-8")


# --- decision 1: the name ------------------------------------------------------


def test_the_package_is_named_s1decide(pyproject: dict) -> None:
    assert pyproject["project"]["name"] == PROJECT_NAME


def test_no_forbidden_name_appears_in_packaging_metadata(pyproject: dict) -> None:
    """Package name, description, keywords, classifiers and URLs — the permanent surfaces."""
    project = pyproject["project"]
    surfaces = [
        project["name"],
        project.get("description", ""),
        *project.get("keywords", []),
        *project.get("classifiers", []),
        *project.get("urls", {}).values(),
    ]
    haystack = " ".join(surfaces).lower()
    for forbidden in FORBIDDEN_IN_IDENTIFIERS:
        assert forbidden not in haystack, f"{forbidden!r} must not appear in packaging metadata"


def test_the_repository_url_uses_the_project_name(pyproject: dict) -> None:
    for url in pyproject["project"].get("urls", {}).values():
        assert PROJECT_NAME in url.lower(), url


def test_the_model_card_does_not_name_us_after_anyone(model_card: str) -> None:
    """Comparison prose is fine; an artefact id or a title is not.

    Checked on headings and on anything that looks like an identifier, rather than on the whole
    document — the head-to-head section legitimately names the system it compares against.
    """
    headings = [line for line in model_card.splitlines() if line.startswith("#")]
    for heading in headings:
        for forbidden in FORBIDDEN_IN_IDENTIFIERS:
            assert forbidden not in heading.lower(), f"{forbidden!r} in heading: {heading}"

    # Backtick-quoted identifiers: model ids, package names, paths.
    for identifier in re.findall(r"`([^`]+)`", model_card):
        for forbidden in FORBIDDEN_IN_IDENTIFIERS:
            assert forbidden not in identifier.lower(), f"{forbidden!r} in identifier: {identifier}"


# --- decision 2: scope ---------------------------------------------------------


def test_the_definition_of_done_is_scoped_to_a_pre_release() -> None:
    claude_md = (repo_root() / "CLAUDE.md").read_text(encoding="utf-8")
    assert "Definition of done (v0.1 — **pre-release**" in claude_md
    assert "Deferred to v0.2" in claude_md


def test_the_deferred_items_are_listed_rather_than_deleted() -> None:
    """Moving work out of scope must leave a trace, or it becomes work nobody remembers."""
    claude_md = (repo_root() / "CLAUDE.md").read_text(encoding="utf-8")
    deferred = claude_md.split("### Deferred to v0.2")[1]
    assert "BF16 row" in deferred
    assert "S3 GRPO" in deferred


def test_the_model_card_states_what_was_not_measured(model_card: str) -> None:
    """ADR 0006 requires these in plain words, next to the numbers, not in a footnote."""
    assert "BF16 was not measured" in model_card
    assert "not measured" in model_card.split("## Quantization")[1].split("##")[0]
    assert "post-hoc" in model_card


def test_the_model_card_declares_itself_a_pre_release(model_card: str) -> None:
    assert "pre-release" in model_card.lower()


# --- the card's numbers come from committed runs -------------------------------


def _metrics(run_id: str) -> dict:
    path = repo_root() / "results" / run_id / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _card_has(model_card: str, value: float, places: int = 4) -> bool:
    return f"{value:.{places}f}" in model_card


def test_the_zero_shot_column_matches_the_committed_reference_run(model_card: str) -> None:
    """The card quotes the baseline; a rescore must not be able to make the card wrong."""
    from eval.baselines import REFERENCE_RUNS

    payload = _metrics(REFERENCE_RUNS["zeroshot"].run_id)
    calibrated = payload["overall_calibrated"]
    for metric in ("accuracy", "ece", "brier_top_label", "nll"):
        assert _card_has(model_card, calibrated[metric]), (
            f"model card's zero-shot {metric} does not match "
            f"{REFERENCE_RUNS['zeroshot'].run_id}: expected {calibrated[metric]:.4f}"
        )


def test_the_card_quotes_the_committed_brier_skill_score(model_card: str) -> None:
    from eval.baselines import REFERENCE_RUNS

    payload = _metrics(REFERENCE_RUNS["zeroshot"].run_id)
    bss = payload["skill_calibrated"]["base_rate"]["brier_multiclass"]
    assert f"+{bss:.4f}" in model_card, f"expected BSS +{bss:.4f} in the model card"


def test_the_card_quotes_the_committed_selective_accuracies(model_card: str) -> None:
    from eval.baselines import REFERENCE_RUNS

    payload = _metrics(REFERENCE_RUNS["zeroshot"].run_id)
    curve = payload["risk_coverage_calibrated"]
    for coverage in ("0.8", "0.9"):
        assert _card_has(model_card, curve["selective_accuracy"][coverage]), coverage
        assert _card_has(model_card, curve["selective_threshold"][coverage]), coverage


def test_the_card_quotes_the_committed_latency_run(model_card: str) -> None:
    """Latency figures in the card must come from the run the card names."""
    match = re.search(r"results/(\S*latency)/latency\.json", model_card)
    assert match, "the model card must name the latency run it quotes"
    payload = json.loads(
        (repo_root() / "results" / match.group(1) / "latency.json").read_text(encoding="utf-8")
    )
    for point in payload["points"]:
        rendered = f"{point['median_ms']:,.0f} ms"
        assert rendered in model_card, f"n={point['n_questions']}: expected {rendered}"
    speedup = payload["targets"]["speedup_64"]["measured"]
    assert f"{speedup:.2f}" in model_card


def test_every_unfilled_slot_is_marked(model_card: str) -> None:
    """A draft is honest only while its holes are visible."""
    assert model_card.count("TBD") > 20, "the draft should still be mostly placeholders"
    assert "DRAFT" in model_card
