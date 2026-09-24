"""S2 on the comparison slices: fitted on val, applied to test, written in the shared format."""

from __future__ import annotations

import json
import random

import pytest


def _write(path, rows) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_calibrate_fits_on_val_and_writes_both_splits(tmp_path) -> None:
    pytest.importorskip("numpy")
    from eval.adapter_slice import _calibrate

    raw, out = tmp_path / "raw", tmp_path / "s2"
    raw.mkdir()
    rng = random.Random(0)
    for split in ("val", "test"):
        rows, preds = [], []
        for i in range(120):
            answer = rng.randrange(2)
            rows.append(
                {
                    "id": f"{split}{i}",
                    "family": "f",
                    "qtype": "noul",
                    "options": ["no", "yes"],
                    "answer_idx": answer,
                    "eval_weight": 1.0,
                }
            )
            # Overconfident logits: a temperature above 1 should be fitted.
            sign = 1 if rng.random() < 0.7 else -1
            logits = [0.0, 6.0 * sign] if answer == 1 else [6.0 * sign, 0.0]
            preds.append({"id": f"{split}{i}", "logits": logits})
        _write(raw / f"slice-{split}.jsonl", rows)
        (raw / f"slice-{split}.json").write_text("{}", encoding="utf-8")
        _write(raw / f"predictions-{split}.jsonl", preds)
    (raw / "predictions-val.meta.json").write_text(json.dumps({"model": "m"}), encoding="utf-8")

    _calibrate(raw, out)
    assert (out / "calibration.json").is_file()
    for split in ("val", "test"):
        meta = json.loads((out / f"predictions-{split}.meta.json").read_text(encoding="utf-8"))
        assert meta["calibration_fit_on_this_split"] is (split == "val")
        for line in (out / f"predictions-{split}.jsonl").read_text(encoding="utf-8").splitlines():
            probs = json.loads(line)["probabilities"]
            assert sum(probs) == pytest.approx(1.0)
            assert max(probs) < 0.999  # the fitted temperature softened the overconfident logits
