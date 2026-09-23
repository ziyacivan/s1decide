"""An envelope point is taken only from a run that trained the full adapter."""

from __future__ import annotations

import pytest
from eval.memory_envelope import envelope_point


def summary(adapted: int, expected: int = 256) -> dict:
    return {
        "run_id": "r",
        "config": {"model": "m", "max_seq_len": 1024, "rank": 8, "batch_size": 1, "grad_accum": 1},
        "rows_seen": 200,
        "adapter_layout": {"adapted": adapted, "expected": expected},
        "loader": {"path": "unsloth"},
        "vram_peak_gib": 20.9,
        "tokens_per_second": 118.5,
        "elapsed_seconds": 224.0,
        "kernels": {"gated_deltanet": {}},
    }


def test_a_full_adapter_run_is_a_point() -> None:
    point = envelope_point(summary(256))
    assert point["vram_peak_gib"] == 20.9
    assert point["seconds_per_row"] == pytest.approx(1.12)


def test_a_partial_adapter_run_is_refused() -> None:
    """Attempt 2's 20.55 GiB was measured on 64 of 256 modules; it is not an envelope point."""
    with pytest.raises(ValueError, match="incomplete"):
        envelope_point(summary(64))
