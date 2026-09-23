"""Laya's answers must land on our options, in our order, or the comparison is not like for like."""

from __future__ import annotations

import pytest
from eval.external_laya import from_laya_answer, to_laya_question


def row(qtype: str, options: list[str], stage: int | None = None) -> dict:
    return {"id": "r", "qtype": qtype, "options": options, "instructions": "q?", "stage": stage}


def test_a_noul_row_is_asked_as_noul_and_yes_is_index_one() -> None:
    r = row("noul", ["no", "yes"])
    assert to_laya_question(r) == {"type": "noul", "instructions": "q?"}
    assert from_laya_answer(r, {"type": "noul", "noul": 0.8}) == pytest.approx([0.2, 0.8])


def test_a_stage1_row_keeps_our_instructions_and_is_asked_as_noul() -> None:
    r = {**row("noul", ["no", "yes"], stage=1), "instructions": "Which intent?\nCandidate: x"}
    assert to_laya_question(r)["instructions"] == "Which intent?\nCandidate: x"


def test_choice_probabilities_follow_our_option_order_not_laya_s() -> None:
    r = row("choice", ["b", "a", "c"])
    answer = {"type": "choice", "probabilities": {"a": 0.2, "b": 0.7, "c": 0.1}}
    assert from_laya_answer(r, answer) == pytest.approx([0.7, 0.2, 0.1])
    assert to_laya_question(r)["criteria"] == {"b": "b", "a": "a", "c": "c"}


def test_score_levels_map_by_index() -> None:
    r = row("score", ["none", "slight", "moderate"])
    answer = {"type": "score", "probabilities": {"0": 0.5, "1": 0.3, "2": 0.2}}
    assert from_laya_answer(r, answer) == pytest.approx([0.5, 0.3, 0.2])
    assert to_laya_question(r)["criteria"] == ["none", "slight", "moderate"]


def test_a_missing_option_is_an_error_not_a_zero() -> None:
    r = row("choice", ["a", "b"])
    with pytest.raises(ValueError, match="lacks options"):
        from_laya_answer(r, {"type": "choice", "probabilities": {"a": 1.0}})


def test_a_type_mismatch_is_an_error() -> None:
    with pytest.raises(ValueError, match="expected a noul"):
        from_laya_answer(row("noul", ["no", "yes"]), {"type": "choice", "probabilities": {}})


def test_systemone_choice_options_default_to_null_descriptions() -> None:
    from eval.external_http import to_systemone_question

    r = row("choice", ["b", "a"])
    assert to_systemone_question(r)["criteria"] == {"b": None, "a": None}
    assert to_systemone_question(r, describe_options=True)["criteria"] == {"b": "b", "a": "a"}
    assert to_systemone_question(row("noul", ["no", "yes"]))["type"] == "noul"


def test_the_http_predictor_writes_our_option_order(tmp_path, monkeypatch) -> None:
    import json

    import eval.external_http as http

    slice_path = tmp_path / "slice-val.jsonl"
    rows = [
        {**row("choice", ["b", "a"]), "id": "c1", "state": "s"},
        {**row("noul", ["no", "yes"]), "id": "n1", "state": "s"},
    ]
    slice_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def fake_post(url, payload, timeout):
        question = payload["questions"]["q"]
        if question["type"] == "choice":
            return {"answers": {"q": {"type": "choice", "probabilities": {"a": 0.25, "b": 0.75}}}}
        return {"answers": {"q": {"type": "noul", "noul": 0.9}}}

    monkeypatch.setattr(http, "_post", fake_post)
    out = tmp_path / "predictions-val.jsonl"
    http.main(
        ["--url", "http://x", "--slice", str(slice_path), "--out", str(out), "--model-id", "m"]
    )
    got = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert got[0]["probabilities"] == pytest.approx([0.75, 0.25])
    assert got[1]["probabilities"] == pytest.approx([0.1, 0.9])
    assert json.loads(out.with_suffix(".meta.json").read_text())["model_id"] == "m"
