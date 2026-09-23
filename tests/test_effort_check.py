"""The reasoning effort a teacher run records must be the one its chat template applied."""

from __future__ import annotations

import pytest
from data.build.teach import TeacherSetting
from data.build.teach_run import EffortNotApplied, _render, effort_check


class Template:
    """A fake tokenizer whose chat template behaves one of the ways real ones do."""

    def __init__(self, mode: str, default: str = "medium") -> None:
        self.mode = mode
        self.default = default

    def apply_chat_template(self, messages, reasoning_effort=None, **kwargs):
        if self.mode == "rejects" and reasoning_effort is not None:
            raise TypeError("unexpected keyword argument 'reasoning_effort'")
        effort = reasoning_effort or self.default
        if self.mode == "ignores":
            effort = self.default
        return f"system\nReasoning: {effort}\nuser\n{messages[0]['content']}\nassistant"


def setting(effort: str | None) -> TeacherSetting:
    return TeacherSetting(model="m", label="t", effort=effort)


def test_a_template_that_applies_the_knob_passes_and_is_fingerprinted() -> None:
    check = effort_check(Template("reads", default="xhigh"), setting("low"), "p")
    assert check["requested"] == "low"
    assert check["matches_template_default"] is False
    assert check["effort_lines"] == ["Reasoning: low"]
    assert len(check["diff_vs_contrast_sha256"]) == 64


def test_asking_for_the_template_default_is_in_effect_not_an_error() -> None:
    """gpt-oss defaults to medium: with and without render identically, and that is fine."""
    check = effort_check(Template("reads", default="medium"), setting("medium"), "p")
    assert check["matches_template_default"] is True
    assert check["diff_vs_default_sha256"] is None


def test_a_template_that_rejects_the_kwarg_stops_the_run() -> None:
    with pytest.raises(EffortNotApplied, match="TypeError"):
        effort_check(Template("rejects"), setting("low"), "p")


def test_a_template_that_ignores_the_knob_stops_the_run() -> None:
    with pytest.raises(EffortNotApplied, match="render identically"):
        effort_check(Template("ignores"), setting("low"), "p")


def test_no_effort_means_nothing_to_check() -> None:
    assert effort_check(Template("reads"), setting(None), "p") is None


def test_render_no_longer_retries_without_the_effort() -> None:
    """The old `except TypeError: pass` dropped the effort silently; now it propagates."""
    with pytest.raises(TypeError):
        _render(Template("rejects"), setting("low"), "p")
