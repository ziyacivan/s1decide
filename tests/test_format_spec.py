"""Assert the committed format spec is current.

`docs/format-spec.md` is generated from the renderer. If someone changes the prompt
format and forgets to regenerate the document, this fails — which is the point: the spec
is a contract, and a stale contract is worse than none.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from s1decide.prompt import FORMAT_VERSION
from s1decide.tasks import repo_root

sys.path.insert(0, str(repo_root() / "scripts"))

from render_format_spec import OUTPUT, build


@pytest.fixture(scope="module")
def spec_path() -> Path:
    return repo_root() / OUTPUT


def test_format_spec_exists(spec_path: Path) -> None:
    assert spec_path.is_file(), "run: uv run python scripts/render_format_spec.py"


def test_format_spec_is_up_to_date(spec_path: Path) -> None:
    committed = spec_path.read_text(encoding="utf-8")
    assert committed == build(), (
        "docs/format-spec.md is stale — regenerate it with "
        "`uv run python scripts/render_format_spec.py`"
    )


def test_format_spec_states_the_current_version(spec_path: Path) -> None:
    assert f"version {FORMAT_VERSION}" in spec_path.read_text(encoding="utf-8")


def test_format_spec_covers_every_primitive(spec_path: Path) -> None:
    text = spec_path.read_text(encoding="utf-8")
    for heading in ("`Choice`", "`Score`", "`Noul`", "four-question mixed call"):
        assert heading in text
