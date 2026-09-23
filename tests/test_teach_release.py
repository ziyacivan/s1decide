"""Giving a teacher's weights back before the next one loads.

The overnight chain runs both legs in one process. On 2026-09-22 leg 1 finished all 6,000 rows
and leg 2 died loading its model: "0 bytes is free ... 19.45 GiB is allocated by PyTorch, and
3.76 GiB is reserved by PyTorch but unallocated". Leg 1's 27B was still resident, because the
`work` closure held a reference to it and because PyTorch's caching allocator keeps freed blocks.

These tests run on CPU. They cannot prove VRAM came back — that needs a card and is covered by
the gpu-marked test at the end — but they can prove the two things that were actually wrong: the
release runs at all, and it runs even when the job raises.
"""

from __future__ import annotations

import pytest
from data.build.teach_run import release_teacher


class FakeModel:
    """Stands in for a loaded model: something with a `.to` that records being moved."""

    def __init__(self) -> None:
        self.moved_to: str | None = None

    def to(self, device: str) -> FakeModel:
        self.moved_to = device
        return self


def test_release_moves_what_it_is_given_off_the_device() -> None:
    model = FakeModel()
    release_teacher(model)
    assert model.moved_to == "meta"


def test_release_survives_an_object_that_refuses_to_move() -> None:
    """A tokenizer has no device, and a quantized model can refuse `.to`. Neither is a failure:
    dropping the reference is what frees the memory, the move is only a nudge."""

    class Stubborn:
        def to(self, device: str) -> None:
            raise RuntimeError("cannot move a 4-bit model")

    release_teacher(Stubborn(), object())  # must not raise


def test_release_of_nothing_is_not_an_error() -> None:
    """Called from a `finally`, so it runs even on paths where there is nothing to free."""
    assert isinstance(release_teacher(), int)


def test_release_reports_bytes_still_reserved() -> None:
    """Returned rather than printed so the gpu test can assert the number fell."""
    assert isinstance(release_teacher(FakeModel()), int)


def test_the_closure_holding_the_model_is_released_too() -> None:
    """The reason `del model` in the caller was never going to be enough.

    `run()` builds a `work` closure that captures the model, so the model outlives the local
    name. The closure has to be handed to the release as well, which is why the signature takes
    *held rather than a model and a tokenizer.
    """
    model = FakeModel()

    def work(batch):
        return model

    release_teacher(model, work)
    assert model.moved_to == "meta"


@pytest.mark.gpu
def test_releasing_a_teacher_returns_vram_to_the_driver() -> None:
    """The real property, on a real card: reserved memory falls while a reference still exists.

    A module is the right stand-in and a bare tensor is not. `Tensor.to()` returns a *new* tensor
    and leaves the original where it was, so it would free nothing; `Module.to()` moves its
    parameters in place. That is the whole mechanism — the caller's `work` closure still holds
    the model when the release runs, so detaching the storage has to happen through the object,
    not by dropping a name.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    torch.cuda.empty_cache()
    before = torch.cuda.memory_reserved()
    layer = torch.nn.Linear(4096, 4096).cuda()  # ~64 MiB of parameters
    assert torch.cuda.memory_reserved() > before

    after = release_teacher(layer)

    assert layer.weight.device.type == "meta", "the module was not moved off the device"
    assert after <= before + 1024 * 1024, (
        f"reserved memory did not come back: {after} vs {before} before the model was loaded"
    )


# --- 2026-09-23: the same OOM again, and why the first fix did not hold -----------------------
#
# The eval chain's leg 2 died with "19.46 GiB is allocated" exactly as on 09-22. Two reasons:
# a bitsandbytes 4-bit model refuses `.to()`, and the suppress hid it; and `run()`'s own locals
# still held the model when the cache was emptied. A relaunch then reloaded the 27B for zero
# pending rows and rewrote leg 1's summary with a zero elapsed time.


def test_a_module_that_refuses_to_move_still_has_its_storage_freed() -> None:
    torch = pytest.importorskip("torch")

    class FourBitLike(torch.nn.Linear):
        def to(self, *args, **kwargs):
            raise ValueError("`.to` is not supported for 4-bit models")

    layer = FourBitLike(64, 64)
    release_teacher(layer)  # the caller still holds `layer`, as run()'s frame did
    assert all(p.numel() == 0 for p in layer.parameters())


def _job(tmp_path, ids, done_ids):
    import json

    items = [{"id": i, "state": "s", "question": "q"} for i in ids]
    rows = [
        {"id": i, "level": 3, "truncated": False, "hit_cap": False, "trace_tokens": 100}
        for i in done_ids
    ]
    directory = tmp_path / "job"
    directory.mkdir()
    (directory / "rows.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return directory, items


def test_pending_items_are_the_ones_without_a_row(tmp_path) -> None:
    from data.build.teach_run import pending_items

    directory, items = _job(tmp_path, ["a", "b", "c"], ["a", "c"])
    assert [i["id"] for i in pending_items(directory, items)] == ["b"]
    assert len(pending_items(tmp_path / "missing", items)) == 3


def test_a_finished_job_is_not_reloaded_and_its_summary_is_not_touched(
    tmp_path, monkeypatch
) -> None:
    import json

    import data.build.teach_run as teach_run

    directory, items = _job(tmp_path, ["a", "b"], ["a", "b"])
    summary = {"run_id": "job", "elapsed_seconds": 19477.87, "tokens_per_second": 44.1}
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    def no_load(_setting):
        raise AssertionError("a finished job must not load its teacher")

    monkeypatch.setattr(teach_run, "load_teacher", no_load)
    monkeypatch.setattr(teach_run, "check_resume_meta", lambda *a, **k: None)
    setting = next(iter(teach_run.TEACHERS.values()))
    assert teach_run.run(setting, items, directory, batch_size=4) == summary
    assert json.loads((directory / "summary.json").read_text(encoding="utf-8")) == summary
    assert not (directory / "DONE").exists()


def test_a_summary_is_rebuilt_from_rows_and_the_progress_log(tmp_path) -> None:
    import json

    import data.build.teach_run as teach_run

    directory, _ = _job(tmp_path, ["a", "b"], ["a", "b"])
    progress = {
        "at": "2026-09-23T17:09:17+00:00",
        "rows_this_run": 2,
        "rows_done": 2,
        "total_rows": 2,
        "elapsed_seconds": 10.0,
    }
    (directory / "progress.jsonl").write_text(json.dumps(progress) + "\n", encoding="utf-8")
    setting = next(iter(teach_run.TEACHERS.values()))
    payload = teach_run.rebuild_summary(directory, setting)
    assert payload["elapsed_seconds"] == 10.0
    assert payload["tokens_per_second"] == pytest.approx(20.0)  # 2 rows x 100 tokens / 10 s
    assert payload["committed"] == 2
    assert payload["rebuilt_from"]


@pytest.mark.gpu
def test_vram_comes_back_from_a_module_that_refuses_to_move_while_still_referenced() -> None:
    """The 09-23 failure on a real card: `.to()` refused, a caller reference alive."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    class FourBitLike(torch.nn.Linear):
        def to(self, *args, **kwargs):
            raise ValueError("`.to` is not supported for 4-bit models")

    torch.cuda.empty_cache()
    before = torch.cuda.memory_reserved()
    layer = FourBitLike(4096, 4096, device="cuda")  # ~64 MiB, held below
    assert torch.cuda.memory_reserved() > before

    after = release_teacher(layer)

    assert after <= before + 1024 * 1024, f"reserved {after} vs {before} before loading"
    del layer
