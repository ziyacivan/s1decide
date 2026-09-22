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
