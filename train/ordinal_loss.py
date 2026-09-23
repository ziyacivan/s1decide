"""The ordinal loss for `Score` — ADR 0007.

Cross-entropy charges the same for every wrong answer. On a five-level rubric that is wrong about
the task: answering "slight" when the truth is "moderate" is a near miss, and answering "none"
when the truth is "decisive" is a different judgement, and a loss that cannot tell them apart
will not learn the scale.

    L = CE(p, y) + lambda * E_{k~p}[ |k - y| ]   ( + mu * unimodality )

The distance term is taken under the model's *own* distribution, so it pushes probability mass
toward the target rather than merely penalising the argmax. Both terms accept a soft target,
because 1,805 of the 4,804 teacher-labelled rows are two teachers one level apart and that is
genuine ambiguity rather than a missing label.

Neighbour label smoothing is deliberately not the default; ADR 0007 records why, and it stays an
A/B arm for S1 proper.

Nothing here imports the trainer. It takes logits and targets and returns a scalar, so it can be
tested on CPU with numbers whose right answer is known by hand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

__all__ = [
    "DEFAULT_LAMBDA",
    "DEFAULT_UNIMODALITY",
    "expected_level",
    "level_distance",
    "loss_parts",
    "ordinal_loss",
    "unimodality_penalty",
]

#: Weight on the distance term. ADR 0007's starting point, swept on `val` against QWK and MAE
#: **together with** ECE: a lambda that improves the ordinal metrics by making the model
#: overconfident has not improved anything, and only the third metric shows it.
DEFAULT_LAMBDA = 0.3

#: Weight on the unimodality penalty. Off by default — it is a shape prior, and a prior that is
#: on by default is one nobody measured.
DEFAULT_UNIMODALITY = 0.0


def _as_distribution(target: Any, num_levels: int, device: Any, dtype: Any) -> torch.Tensor:
    """Accept either class indices or a distribution, and return a distribution.

    A hard label is the one-hot case of a soft one, so there is one code path rather than two
    that have to be kept in step.

    Args:
        target: ``(batch,)`` integer indices, or ``(batch, levels)`` probabilities.
        num_levels: Size of the scale.
        device: Device to build on.
        dtype: Floating dtype for the result.

    Returns:
        ``(batch, levels)`` probabilities.

    Raises:
        ValueError: If a supplied distribution does not sum to one, which is the shape a
            half-built soft target has and would otherwise train quietly on a scaled loss.
    """
    import torch

    if target.dim() == 1:
        return torch.nn.functional.one_hot(target.long(), num_levels).to(device=device, dtype=dtype)
    distribution = target.to(device=device, dtype=dtype)
    sums = distribution.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-4):
        worst = float((sums - 1).abs().max())
        raise ValueError(
            f"soft targets must sum to 1; worst row is off by {worst:.4g}. A target that does not "
            "is a per-row reweighting of the loss wearing a label's clothes."
        )
    return distribution


def expected_level(probabilities: torch.Tensor) -> torch.Tensor:
    """The expected level under a distribution, in level units.

    Args:
        probabilities: ``(batch, levels)``.

    Returns:
        ``(batch,)`` expected level.
    """
    import torch

    levels = torch.arange(
        probabilities.shape[-1], device=probabilities.device, dtype=probabilities.dtype
    )
    return (probabilities * levels).sum(dim=-1)


def level_distance(probabilities: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``E_{k~p}[ |k - y| ]`` per row, with ``y`` itself possibly a distribution.

    Computed as the expectation over both distributions rather than against the target's mean,
    so that a soft target spread across two levels is not silently replaced by the level halfway
    between them — which for a 0.5/0.5 target over levels 1 and 2 would be 1.5, a level the scale
    does not have.

    Args:
        probabilities: ``(batch, levels)`` predicted.
        target: ``(batch, levels)`` target distribution.

    Returns:
        ``(batch,)`` expected absolute distance.
    """
    import torch

    levels = torch.arange(
        probabilities.shape[-1], device=probabilities.device, dtype=probabilities.dtype
    )
    gap = (levels[:, None] - levels[None, :]).abs()
    return torch.einsum("bi,ij,bj->b", probabilities, gap, target)


def unimodality_penalty(probabilities: torch.Tensor) -> torch.Tensor:
    """Penalise probability mass in a local maximum that is not adjacent to the global one.

    "Probably level 1, possibly level 4, definitely not 2 or 3" is a shape an ordinal judgement
    rarely means. It is usually a symptom — two modes in the training data, or a model hedging
    across two unrelated readings of the question — and it makes an expected level a number
    describing neither mode.

    Off by default. It is a prior about answers, and it should have to earn its weight on `val`.

    Args:
        probabilities: ``(batch, levels)``.

    Returns:
        ``(batch,)`` penalty, zero for a unimodal row.
    """
    import torch

    if probabilities.shape[-1] < 3:
        return torch.zeros(probabilities.shape[0], device=probabilities.device)
    # Pad below zero so the two endpoints can be peaks. They usually are: on a five-level scale
    # the common two-humped shape is mass at "none" and at "decisive" and a hollow in between,
    # and an interior-only test scores that a perfect zero.
    pad = torch.full_like(probabilities[:, :1], -1.0)
    padded = torch.cat([pad, probabilities, pad], dim=-1)
    is_peak = (probabilities > padded[:, :-2]) & (probabilities > padded[:, 2:])
    peaks = probabilities * is_peak
    # Every peak but the largest is a second mode. A unimodal row therefore scores zero, and a
    # plateau scores zero too, because a tie is not a strict maximum.
    return peaks.sum(dim=-1) - peaks.max(dim=-1).values


def ordinal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    lambda_distance: float = DEFAULT_LAMBDA,
    mu_unimodality: float = DEFAULT_UNIMODALITY,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross-entropy plus a distance penalty, over the level distribution (ADR 0007).

    Args:
        logits: ``(batch, levels)`` masked option logits, unnormalised.
        target: ``(batch,)`` indices or ``(batch, levels)`` probabilities. Soft targets are the
            point: an exact-agreement row is one-hot, a one-level disagreement is 0.5/0.5.
        lambda_distance: Weight on ``E[|k - y|]``. Zero gives plain cross-entropy, which is the
            baseline the sweep is measured against.
        mu_unimodality: Weight on the shape penalty. Zero by default.
        reduction: ``"mean"``, ``"sum"`` or ``"none"``.

    Returns:
        The loss, reduced as asked.

    Raises:
        ValueError: On an unknown reduction, or a soft target that does not sum to one.
    """
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"unknown reduction {reduction!r}")

    per_row = loss_parts(
        logits, target, lambda_distance=lambda_distance, mu_unimodality=mu_unimodality
    )["total"]
    if reduction == "mean":
        return per_row.mean()
    if reduction == "sum":
        return per_row.sum()
    return per_row


def loss_parts(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    lambda_distance: float = DEFAULT_LAMBDA,
    mu_unimodality: float = DEFAULT_UNIMODALITY,
) -> dict[str, torch.Tensor]:
    """The loss per row, and every piece of it, so a curve can be read against its floor.

    A soft target has a floor the model cannot go below. For a 0.5/0.5 split, cross-entropy
    bottoms out at ``ln 2`` and ``E|k - y|`` at 0.5 — the latter for *any* prediction on the two
    levels, not just the target. So mean loss over `Score` rows moves with the hard/soft mix of
    whatever window it is averaged over, and the first smoke reports read that as learning going
    backwards. ``kl`` and ``distance_excess`` are the parts that are zero at the optimum.

    Args:
        logits: ``(batch, levels)`` masked option logits.
        target: ``(batch,)`` indices or ``(batch, levels)`` probabilities.
        lambda_distance: Weight on the distance term.
        mu_unimodality: Weight on the shape penalty.

    Returns:
        ``(batch,)`` tensors: ``total`` (what is optimised), ``cross_entropy``, ``entropy`` of the
        target, ``kl`` = cross_entropy - entropy, ``distance``, ``distance_floor`` (its minimum
        over predictions), ``distance_excess``, ``unimodality``, and ``floor`` — the lowest
        ``total`` any prediction can reach, ``entropy + lambda * distance_floor``.
    """
    import torch

    log_probabilities = torch.log_softmax(logits, dim=-1)
    probabilities = log_probabilities.exp()
    distribution = _as_distribution(
        target, logits.shape[-1], logits.device, log_probabilities.dtype
    )

    cross_entropy = -(distribution * log_probabilities).sum(dim=-1)
    entropy = -torch.special.xlogy(distribution, distribution).sum(dim=-1)
    distance = level_distance(probabilities, distribution)
    # E_{k~p}[|k - y|] is linear in p, so its minimum is at a point mass: the best single level.
    levels = torch.arange(logits.shape[-1], device=logits.device, dtype=distribution.dtype)
    gap = (levels[:, None] - levels[None, :]).abs()
    distance_floor = (distribution @ gap).min(dim=-1).values
    unimodality = unimodality_penalty(probabilities)

    total = cross_entropy
    if lambda_distance:
        total = total + lambda_distance * distance
    if mu_unimodality:
        total = total + mu_unimodality * unimodality
    return {
        "total": total,
        "cross_entropy": cross_entropy,
        "entropy": entropy,
        "kl": cross_entropy - entropy,
        "distance": distance,
        "distance_floor": distance_floor,
        "distance_excess": distance - distance_floor,
        "unimodality": unimodality,
        "floor": entropy + lambda_distance * distance_floor,
    }
