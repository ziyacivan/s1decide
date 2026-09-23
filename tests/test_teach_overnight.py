"""A second teacher batch must not land in the first batch's run directories."""

from __future__ import annotations

import json

import pytest
from data.build.teach_overnight import OVERNIGHT_PLAN, plan_with_suffix, run_chain


def test_a_suffix_renames_every_leg_and_changes_nothing_else() -> None:
    plan = plan_with_suffix("eval")
    assert [leg["run_id"] for leg in plan] == [f"{leg['run_id']}-eval" for leg in OVERNIGHT_PLAN]
    for new, old in zip(plan, OVERNIGHT_PLAN, strict=True):
        assert {k: v for k, v in new.items() if k != "run_id"} == {
            k: v for k, v in old.items() if k != "run_id"
        }


def test_no_suffix_is_the_original_plan() -> None:
    assert plan_with_suffix("") == OVERNIGHT_PLAN


def test_colliding_ids_across_files_are_refused_before_any_model_loads(tmp_path) -> None:
    row = {"id": "teach-val-00000-x", "state": "s", "question": "q"}
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    for path in (a, b):
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="collide"):
        run_chain(plan_with_suffix("t"), [a, b], None, tmp_path)
