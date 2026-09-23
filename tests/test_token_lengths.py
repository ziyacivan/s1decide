"""Token-length report: grouping and the per-cap skip counts."""

from __future__ import annotations

import pytest
from eval.token_lengths import group_key, length_stats


@pytest.mark.parametrize(
    ("row", "key"),
    [
        ({"qtype": "choice"}, "choice"),
        ({"qtype": "noul"}, "noul"),
        ({"qtype": "noul", "stage": 1}, "noul/stage1"),
        ({"qtype": "score", "target_type": "soft"}, "score/soft"),
        ({"qtype": "score"}, "score/hard"),
    ],
)
def test_group_key(row: dict, key: str) -> None:
    assert group_key(row) == key


def test_length_stats_counts_rows_over_each_cap() -> None:
    stats = length_stats([100, 600, 1100, 3000], caps=(512, 1024, 2048))
    assert stats["max"] == 3000
    assert stats["over"]["512"]["rows"] == 3
    assert stats["over"]["1024"]["rows"] == 2
    assert stats["over"]["2048"] == {"rows": 1, "share": 0.25}
