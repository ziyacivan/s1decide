"""Figures are generated artefacts and are held to the same rule as the numbers.

A README showing last week's picture next to this week's number is worse than one showing
neither, so `uv run task readme` refuses to run while a figure is older than the JSON behind it,
and these tests check the same thing in CI.
"""

from __future__ import annotations

import json

import pytest
from eval.figures import FIGURES, figure_captions, resolve_sources
from eval.readme_slots import StaleFigure, check_figures_fresh, update_readme

from s1decide.tasks import repo_root


@pytest.fixture(scope="module")
def root():
    return repo_root()


def test_every_declared_figure_exists(root) -> None:
    for spec in FIGURES:
        path = root / "docs" / "figures" / f"{spec.name}.png"
        assert path.is_file(), f"{spec.name}.png is missing; run `uv run task figures`"
        assert path.stat().st_size > 10_000, f"{spec.name}.png looks empty"


def test_no_figure_is_older_than_the_json_behind_it(root) -> None:
    stale = check_figures_fresh(root)
    assert not stale, f"stale figures: {stale} — run `uv run task figures`"


def test_every_figure_source_is_a_committed_artefact(root) -> None:
    sources = resolve_sources(root)
    for spec in FIGURES:
        path = root / spec.source.format(**sources)
        assert path.is_file(), f"{spec.name} claims to come from {path}, which does not exist"


def test_every_figure_has_alt_text() -> None:
    """A figure nobody can read is not a figure."""
    for spec in FIGURES:
        assert spec.alt.strip(), f"{spec.name} has no alt text"
        assert len(spec.alt) > 40, f"{spec.name}'s alt text is too short to describe it"


def test_every_figure_is_shown_in_the_readme_with_its_alt_text(root) -> None:
    """Image line and caption are both generated, so the alt text cannot drift from the spec."""
    readme = (root / "README.md").read_text(encoding="utf-8")
    for spec in FIGURES:
        assert f"<!--figure:{spec.name}-->" in readme, f"{spec.name} has no block in the README"
        assert f"![{spec.alt}](docs/figures/{spec.name}.png)" in readme, (
            f"{spec.name}'s image line is stale; run `uv run task readme`"
        )


def test_every_caption_names_its_run_and_hardware(root) -> None:
    """A figure lifted out of the page must still say where it came from."""
    captions = figure_captions(root)
    assert set(captions) == {spec.name for spec in FIGURES}
    for name, caption in captions.items():
        assert "Generated from" in caption, name
        assert "`" in caption, f"{name}'s caption names no run id"


def test_the_readme_carries_the_generated_captions(root) -> None:
    readme = (root / "README.md").read_text(encoding="utf-8")
    for name, caption in figure_captions(root).items():
        assert f"<!--figure:{name}-->" in readme, f"{name} has no block"
        assert caption in readme, f"{name}'s caption is stale; run `uv run task readme`"


def test_updating_the_readme_refuses_while_a_figure_is_stale(root, tmp_path, monkeypatch) -> None:
    """The gate itself, exercised rather than assumed."""
    import eval.readme_slots as slots

    monkeypatch.setattr(slots, "check_figures_fresh", lambda _root: ["latency (older than x)"])
    with pytest.raises(StaleFigure, match="out of date"):
        update_readme(root)


def test_the_figure_sources_file_records_what_each_was_built_from(root) -> None:
    recorded = json.loads((root / "docs" / "figures" / "sources.json").read_text(encoding="utf-8"))
    assert set(recorded) == {spec.name for spec in FIGURES}
    for name, source in recorded.items():
        assert (root / source).is_file(), f"{name} records a source that does not exist: {source}"


def test_the_benchmark_issue_template_asks_for_the_json_not_a_paste(root) -> None:
    template = (root / ".github" / "ISSUE_TEMPLATE" / "benchmark.md").read_text(encoding="utf-8")
    for required in ("latency.json", "GPU", "Commit hash", "doctor"):
        assert required in template, f"the benchmark template does not ask for {required}"


def test_every_figure_records_the_digest_of_its_source(root) -> None:
    """Freshness is judged by content; a figure without a recorded digest falls back to file
    times, which cannot tell a regenerated-but-identical figure from a stale one."""
    import json

    from eval.figures import DIGESTS_FILE, FIGURES

    digests = json.loads((root / "docs" / "figures" / DIGESTS_FILE).read_text(encoding="utf-8"))
    assert set(digests) == {spec.name for spec in FIGURES}
    assert all(len(value) == 64 for value in digests.values())
