"""ADR 0007's loss, checked against its properties rather than remembered numbers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from train.ordinal_loss import (  # noqa: E402
    DEFAULT_LAMBDA,
    expected_level,
    level_distance,
    ordinal_loss,
    unimodality_penalty,
)


def logits(*rows: list[float]) -> torch.Tensor:
    return torch.tensor(list(rows), dtype=torch.float32)


# --- the property that makes it ordinal ------------------------------------------


def test_a_near_miss_costs_less_than_a_far_one() -> None:
    """The whole reason this exists. Plain cross-entropy charges the same for both."""
    target = torch.tensor([2])
    near = ordinal_loss(logits([0.0, 0.0, 0.0, 4.0, 0.0]), target)
    far = ordinal_loss(logits([0.0, 0.0, 0.0, 0.0, 4.0]), target)
    assert near < far


def test_plain_cross_entropy_does_not_tell_them_apart() -> None:
    """Same two predictions at lambda 0, to show the difference is the distance term and not
    something else about the shapes."""
    target = torch.tensor([2])
    near = ordinal_loss(logits([0.0, 0.0, 0.0, 4.0, 0.0]), target, lambda_distance=0.0)
    far = ordinal_loss(logits([0.0, 0.0, 0.0, 0.0, 4.0]), target, lambda_distance=0.0)
    assert near == pytest.approx(far)


def test_being_right_costs_least() -> None:
    target = torch.tensor([2])
    right = ordinal_loss(logits([0.0, 0.0, 6.0, 0.0, 0.0]), target)
    near = ordinal_loss(logits([0.0, 0.0, 0.0, 6.0, 0.0]), target)
    assert right < near


def test_the_distance_term_scales_with_lambda() -> None:
    row, target = logits([0.0, 0.0, 0.0, 0.0, 4.0]), torch.tensor([0])
    base = ordinal_loss(row, target, lambda_distance=0.0)
    small = ordinal_loss(row, target, lambda_distance=0.1)
    large = ordinal_loss(row, target, lambda_distance=1.0)
    assert base < small < large


# --- soft targets ------------------------------------------------------------------


def test_a_hard_label_and_its_one_hot_are_the_same_loss() -> None:
    """One code path, so the two cannot drift apart."""
    row = logits([0.5, 1.0, 0.2, 0.0, -1.0])
    index = ordinal_loss(row, torch.tensor([1]))
    one_hot = ordinal_loss(row, torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]]))
    assert index == pytest.approx(float(one_hot), abs=1e-6)


def test_a_soft_target_is_best_served_by_predicting_it() -> None:
    """1,805 of the teacher rows are 0.5/0.5 over two adjacent levels. The loss should prefer a
    model that says so to one that commits to either side."""
    target = torch.tensor([[0.0, 0.5, 0.5, 0.0, 0.0]])
    honest = ordinal_loss(logits([-9.0, 1.0, 1.0, -9.0, -9.0]), target)
    committed = ordinal_loss(logits([-9.0, 6.0, -9.0, -9.0, -9.0]), target)
    assert honest < committed


def test_a_soft_target_that_does_not_sum_to_one_is_refused() -> None:
    """Silently accepted, it is a per-row reweighting of the loss wearing a label's clothes."""
    with pytest.raises(ValueError, match="sum to 1"):
        ordinal_loss(logits([0.0, 0.0, 0.0]), torch.tensor([[0.5, 0.2, 0.0]]))


def test_a_soft_target_is_not_collapsed_to_its_mean() -> None:
    """A 0.5/0.5 target over levels 1 and 2 has mean 1.5, a level the scale does not have.
    Distance is taken as the expectation over both distributions instead."""
    soft = torch.tensor([[0.0, 0.5, 0.5, 0.0, 0.0]])
    probabilities = torch.tensor([[0.0, 0.5, 0.5, 0.0, 0.0]])
    # Predicting the target exactly still costs 0.5: half the mass is one level away.
    assert float(level_distance(probabilities, soft)) == pytest.approx(0.5)


# --- the pieces ---------------------------------------------------------------------


def test_expected_level_of_a_point_mass_is_that_level() -> None:
    p = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
    assert float(expected_level(p)) == pytest.approx(2.0)


def test_expected_level_sits_between_two_modes() -> None:
    p = torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.5]])
    assert float(expected_level(p)) == pytest.approx(2.0)


def test_distance_is_zero_only_when_both_agree_on_one_level() -> None:
    one_hot = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
    assert float(level_distance(one_hot, one_hot)) == pytest.approx(0.0)


def test_a_unimodal_distribution_is_not_penalised() -> None:
    p = torch.tensor([[0.05, 0.15, 0.6, 0.15, 0.05]])
    assert float(unimodality_penalty(p)) == pytest.approx(0.0, abs=1e-6)


def test_a_two_humped_distribution_is_penalised() -> None:
    """ "Probably 0, possibly 4, definitely not 2" makes an expected level describe neither."""
    p = torch.tensor([[0.45, 0.02, 0.02, 0.06, 0.45]])
    two_humped = float(unimodality_penalty(p))
    unimodal = float(unimodality_penalty(torch.tensor([[0.1, 0.2, 0.4, 0.2, 0.1]])))
    assert two_humped > unimodal


def test_the_shape_penalty_is_off_by_default() -> None:
    """A prior that is on by default is one nobody measured."""
    row = logits([4.0, -2.0, -2.0, -2.0, 4.0])
    target = torch.tensor([0])
    assert ordinal_loss(row, target) == pytest.approx(
        float(ordinal_loss(row, target, mu_unimodality=0.0))
    )
    assert ordinal_loss(row, target, mu_unimodality=1.0) > ordinal_loss(row, target)


# --- shapes and plumbing --------------------------------------------------------------


@pytest.mark.parametrize("reduction,expected_shape", [("none", (3,)), ("mean", ()), ("sum", ())])
def test_reductions(reduction, expected_shape) -> None:
    out = ordinal_loss(torch.zeros(3, 5), torch.tensor([0, 1, 2]), reduction=reduction)
    assert tuple(out.shape) == expected_shape


def test_an_unknown_reduction_is_refused() -> None:
    with pytest.raises(ValueError, match="reduction"):
        ordinal_loss(torch.zeros(1, 3), torch.tensor([0]), reduction="average")


def test_the_loss_has_a_gradient() -> None:
    row = torch.zeros(2, 5, requires_grad=True)
    ordinal_loss(row, torch.tensor([[0.0, 0.5, 0.5, 0.0, 0.0]] * 2)).backward()
    assert row.grad is not None and torch.isfinite(row.grad).all()


def test_the_default_lambda_is_the_adr_value() -> None:
    assert DEFAULT_LAMBDA == 0.3
