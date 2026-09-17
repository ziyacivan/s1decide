"""Decision metrics: accuracy, calibration, sharpness, discrimination.

The house rules (CLAUDE.md, AGENTS.md) that shape this module:

* **ECE uses 15 equal-mass bins**, not equal-width. Equal-width bins on a confident model
  put almost every prediction in the top bin and report a number driven by a handful of
  examples elsewhere.
* **ECE is never reported alone.** :func:`summarize` always returns Brier and NLL beside it,
  and :func:`base_rate_probabilities` exists so every report can carry the negative control —
  a predictor that ignores the state entirely can post an excellent ECE, and saying so is the
  point of publishing calibration at all.
* Metrics take **probabilities**, and runs store **logits**, so temperature scaling can be
  re-fitted and everything re-scored without re-running the model.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from s1decide.calibrate import OPTION_COUNT_BUCKETS, option_count_bucket

__all__ = [
    "DEFAULT_BINS",
    "OPTION_COUNT_BUCKETS",
    "Prediction",
    "ReliabilityBin",
    "accuracy",
    "auroc",
    "base_rate_probabilities",
    "brier_multiclass",
    "brier_skill_score",
    "brier_top_label",
    "ece",
    "equal_mass_bins",
    "mce",
    "nll",
    "option_count_bucket",
    "risk_coverage",
    "summarize",
    "summarize_by",
    "uniform_probabilities",
]

#: Number of equal-mass bins used for ECE and reliability diagrams.
DEFAULT_BINS = 15

#: Coverages at which selective accuracy is reported in every summary. A model that knows when
#: it does not know should be markedly more accurate on the 80% of questions it is most
#: confident about than on all of them.
SELECTIVE_COVERAGES: tuple[float, ...] = (0.8, 0.9)

#: Coverage grid for the risk-coverage curve. Starts at 0.05 because accuracy on the top 5% of
#: a 1,500-question set is already noisy, and ends at 1.0, where it is plain accuracy.
RISK_COVERAGE_GRID: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 21))

_EPS = 1e-12


@dataclass(frozen=True)
class Prediction:
    """One scored question, stored as logits so it can be re-calibrated later.

    Attributes:
        id: Stable identifier of the source row.
        family: Task family, used for held-out-family analysis and the base-rate control.
        qtype: ``choice``, ``score`` or ``noul``.
        logits: Raw masked logits, one per option, in the question's option order.
        answer_idx: Index of the correct option.
        weight: Importance weight. 1.0 for a question that represents itself; larger for one
            that stands in for others under case-control sampling, where every stage-1 positive
            is kept and the negatives are sampled. Weighting them back is what makes a 30-minute
            evaluation report the same numbers as a 3.3-hour one.
    """

    id: str
    family: str
    qtype: str
    logits: tuple[float, ...]
    answer_idx: int
    weight: float = 1.0

    def __post_init__(self) -> None:
        if len(self.logits) < 2:
            raise ValueError(f"{self.id}: a prediction needs at least 2 logits")
        # Coerce to a true int: a bool would index a numpy array as a *mask* rather than a
        # position, which silently corrupts the base-rate control instead of failing.
        object.__setattr__(self, "answer_idx", int(self.answer_idx))
        if not 0 <= self.answer_idx < len(self.logits):
            raise ValueError(
                f"{self.id}: answer_idx {self.answer_idx} out of range for "
                f"{len(self.logits)} options"
            )
        object.__setattr__(self, "weight", float(self.weight))
        if not self.weight > 0 or not math.isfinite(self.weight):
            raise ValueError(f"{self.id}: weight must be finite and positive, got {self.weight}")

    @property
    def n_options(self) -> int:
        """Number of options this question offered."""
        return len(self.logits)

    def probabilities(self, temperature: float = 1.0) -> np.ndarray:
        """Softmax the stored logits at ``temperature``.

        Args:
            temperature: Divides the logits first; 1.0 is the model's own distribution.

        Returns:
            Probabilities in option order.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        z = np.asarray(self.logits, dtype=np.float64) / temperature
        z -= z.max()
        w = np.exp(z)
        return w / w.sum()

    def to_json(self) -> dict[str, Any]:
        """Serialise for ``predictions.jsonl``."""
        return asdict(self) | {"logits": list(self.logits)}

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> Prediction:
        """Rebuild from a ``predictions.jsonl`` row."""
        return cls(
            id=str(row["id"]),
            family=str(row["family"]),
            qtype=str(row["qtype"]),
            logits=tuple(float(x) for x in row["logits"]),
            answer_idx=int(row["answer_idx"]),
            weight=float(row.get("weight", 1.0)),
        )


@dataclass(frozen=True)
class ReliabilityBin:
    """One equal-mass bin of a reliability diagram.

    Attributes:
        count: Predictions in the bin.
        lo: Lowest confidence in the bin.
        hi: Highest confidence in the bin.
        mean_confidence: Mean predicted confidence, weighted.
        accuracy: Fraction actually correct, weighted. A calibrated model has this equal to
            ``mean_confidence``.
        weight: Total importance weight in the bin. Equals ``count`` for an unweighted run;
            under case-control sampling it is the size of the population the bin stands for,
            and it is what ECE averages over.
    """

    count: int
    lo: float
    hi: float
    mean_confidence: float
    accuracy: float
    weight: float = 0.0

    def __post_init__(self) -> None:
        # An unweighted caller should not have to know this field exists.
        if self.weight == 0.0:
            object.__setattr__(self, "weight", float(self.count))

    @property
    def gap(self) -> float:
        """Signed calibration gap: positive means over-confident."""
        return self.mean_confidence - self.accuracy


def _weights(weights: Sequence[float] | None, n: int) -> np.ndarray:
    """Validate importance weights, defaulting to uniform.

    Args:
        weights: One positive weight per prediction, or ``None`` for unweighted.
        n: How many predictions there are.

    Returns:
        A float array of length ``n``.

    Raises:
        ValueError: If the length is wrong or a weight is not finite and positive. A zero
            weight would silently delete a prediction, which is not the same as not having it.
    """
    if weights is None:
        return np.ones(n, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.size != n:
        raise ValueError(f"{w.size} weights for {n} predictions")
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("weights must be finite and positive")
    return w


def _as_arrays(
    probabilities: Sequence[Sequence[float]], labels: Sequence[int]
) -> tuple[list[np.ndarray], np.ndarray]:
    """Validate and normalise ragged probability rows plus their labels."""
    if len(probabilities) != len(labels):
        raise ValueError(f"{len(probabilities)} probability rows but {len(labels)} labels")
    if not probabilities:
        raise ValueError("no predictions to score")
    rows = [np.asarray(p, dtype=np.float64) for p in probabilities]
    for i, (row, label) in enumerate(zip(rows, labels)):
        if row.ndim != 1 or row.size < 2:
            raise ValueError(f"row {i}: expected at least 2 probabilities, got shape {row.shape}")
        if not np.isfinite(row).all():
            raise ValueError(f"row {i}: non-finite probability")
        if abs(row.sum() - 1.0) > 1e-6:
            raise ValueError(f"row {i}: probabilities sum to {row.sum()!r}, not 1")
        if not 0 <= label < row.size:
            raise ValueError(f"row {i}: label {label} out of range for {row.size} options")
    return rows, np.asarray(labels, dtype=np.int64)


def _confidence_and_correct(
    rows: Sequence[np.ndarray], labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Top-label confidence and whether the top label was right.

    Ties resolve to the lowest index, matching ``Result.argmax_index``.
    """
    predicted = np.array([int(np.argmax(r)) for r in rows])
    confidence = np.array([float(r[p]) for r, p in zip(rows, predicted)])
    return confidence, (predicted == labels).astype(np.float64)


def accuracy(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    weights: Sequence[float] | None = None,
) -> float:
    """Top-1 accuracy, importance-weighted when ``weights`` is given."""
    rows, y = _as_arrays(probabilities, labels)
    w = _weights(weights, len(rows))
    return float(np.average(_confidence_and_correct(rows, y)[1], weights=w))


def nll(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    weights: Sequence[float] | None = None,
) -> float:
    """Mean negative log-likelihood of the correct option, in nats."""
    rows, y = _as_arrays(probabilities, labels)
    w = _weights(weights, len(rows))
    per_row = np.array([-math.log(max(float(r[t]), _EPS)) for r, t in zip(rows, y)])
    return float(np.average(per_row, weights=w))


def brier_multiclass(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    weights: Sequence[float] | None = None,
) -> float:
    """Multiclass Brier score: mean squared error against the one-hot answer.

    Summed over options, so it lies in ``[0, 2]``. Reported alongside
    :func:`brier_top_label` because this version depends on the option count, which varies
    across the eval set.
    """
    rows, y = _as_arrays(probabilities, labels)
    w = _weights(weights, len(rows))
    per_row = np.empty(len(rows), dtype=np.float64)
    for i, (row, target) in enumerate(zip(rows, y)):
        onehot = np.zeros_like(row)
        onehot[target] = 1.0
        per_row[i] = float(np.sum((row - onehot) ** 2))
    return float(np.average(per_row, weights=w))


def brier_top_label(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    weights: Sequence[float] | None = None,
) -> float:
    """Brier score of the top-label confidence against whether it was correct.

    In ``[0, 1]`` and comparable across questions with different option counts. This is the
    quantity the reliability diagram decomposes.
    """
    rows, y = _as_arrays(probabilities, labels)
    w = _weights(weights, len(rows))
    confidence, correct = _confidence_and_correct(rows, y)
    return float(np.average((confidence - correct) ** 2, weights=w))


def equal_mass_bins(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    n_bins: int = DEFAULT_BINS,
    weights: Sequence[float] | None = None,
) -> list[ReliabilityBin]:
    """Bin predictions into ``n_bins`` groups of (almost) equal **weight**, by confidence.

    Unweighted, equal mass is equal count. With importance weights it is equal weight, which is
    the thing ECE actually needs: under case-control sampling one retained negative may stand
    for ninety, and binning it as a single observation would put the bin edges in the wrong
    place and misreport the gap.

    Args:
        probabilities: One row per question.
        labels: Correct option index per question.
        n_bins: Number of bins. Bins are as equal in weight as the data allows.
        weights: Importance weight per question.

    Returns:
        Non-empty bins, in increasing confidence order.

    Note:
        Predictions with identical confidence can land in adjacent bins; with equal-mass
        binning that is unavoidable and affects the bin edges, not the ECE weighting.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    rows, y = _as_arrays(probabilities, labels)
    w = _weights(weights, len(rows))
    confidence, correct = _confidence_and_correct(rows, y)
    order = np.argsort(confidence, kind="mergesort")

    # Cut at cumulative-weight quantiles rather than at equal counts. A heavy prediction has to
    # fill a bin the way the population it stands for would; with uniform weights this reduces
    # to equal counts, which is what it must do. (A greedy "close the bin once it is full" pass
    # does not: it overshoots on every bin and leaves the remainder in the last one.)
    ordered_weight = w[order]
    cumulative = np.cumsum(ordered_weight)
    total = float(cumulative[-1])
    count = min(n_bins, len(order))
    edges = [
        int(np.searchsorted(cumulative, total * (i + 1) / count, side="left")) + 1
        for i in range(count)
    ]

    chunks: list[np.ndarray] = []
    start = 0
    for edge in edges:
        stop = min(max(edge, start), len(order))
        if stop > start:
            chunks.append(order[start:stop])
            start = stop
    if start < len(order):
        chunks.append(order[start:])

    bins: list[ReliabilityBin] = []
    for chunk in chunks:
        if chunk.size == 0:  # pragma: no cover - defensive
            continue
        c = confidence[chunk]
        cw = w[chunk]
        bins.append(
            ReliabilityBin(
                count=int(chunk.size),
                lo=float(c.min()),
                hi=float(c.max()),
                mean_confidence=float(np.average(c, weights=cw)),
                accuracy=float(np.average(correct[chunk], weights=cw)),
                weight=float(cw.sum()),
            )
        )
    return bins


def ece(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    n_bins: int = DEFAULT_BINS,
    weights: Sequence[float] | None = None,
) -> float:
    """Expected calibration error over equal-mass bins.

    The count-weighted mean absolute gap between confidence and accuracy. Zero is perfect;
    it says nothing about whether the model is *right*, which is why it is always reported
    with Brier, accuracy and the base-rate control.
    """
    bins = equal_mass_bins(probabilities, labels, n_bins, weights)
    total = sum(b.weight for b in bins)
    return sum(b.weight * abs(b.gap) for b in bins) / total


def mce(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    n_bins: int = DEFAULT_BINS,
    weights: Sequence[float] | None = None,
) -> float:
    """Maximum calibration error: the worst single-bin gap."""
    return max(abs(b.gap) for b in equal_mass_bins(probabilities, labels, n_bins, weights))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks starting at 1, ties sharing their mean rank (scipy's ``rankdata``, no scipy)."""
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < ordered.size:
        j = i
        while j + 1 < ordered.size and ordered[j + 1] == ordered[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auroc(
    scores: Sequence[float],
    positive: Sequence[float],
    weights: Sequence[float] | None = None,
) -> float | None:
    """Area under the ROC curve, with ties handled exactly.

    Computed as the weighted probability that a random positive outscores a random negative,
    counting ties as half — the definition the rank formula implements, written out so that
    importance weights fit into it.

    Args:
        scores: Higher means more likely positive.
        positive: 1 for a positive case, 0 for a negative one.
        weights: Importance weight per case.

    Returns:
        The AUROC, or ``None`` when one class is absent and the quantity is undefined.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(positive, dtype=np.float64)
    if s.size != y.size:
        raise ValueError(f"{s.size} scores but {y.size} labels")
    w = _weights(weights, s.size)
    pos_mass = float(w[y == 1].sum())
    neg_mass = float(w[y == 0].sum())
    if pos_mass == 0 or neg_mass == 0:
        return None

    # Sweep the distinct scores in increasing order, accumulating negative mass below each.
    # Each positive contributes the negative mass strictly beneath it, plus half of any tied
    # negative mass — the standard tie convention, weighted.
    order = np.argsort(s, kind="mergesort")
    s_sorted, y_sorted, w_sorted = s[order], y[order], w[order]
    total = 0.0
    below = 0.0
    index = 0
    while index < s_sorted.size:
        end = index
        while end < s_sorted.size and s_sorted[end] == s_sorted[index]:
            end += 1
        block = slice(index, end)
        tie_pos = float(w_sorted[block][y_sorted[block] == 1].sum())
        tie_neg = float(w_sorted[block][y_sorted[block] == 0].sum())
        total += tie_pos * (below + 0.5 * tie_neg)
        below += tie_neg
        index = end
    return float(total / (pos_mass * neg_mass))


def brier_skill_score(brier: float, reference_brier: float) -> float | None:
    """Skill of a Brier score against a reference: ``1 - brier / reference_brier``.

    1.0 is perfect, 0.0 is no better than the reference, negative is worse.

    This exists because ECE alone can be won by a model that never commits. The base-rate
    control on our own eval set scores ECE 0.034 against the calibrated model's 0.045 while
    being 33 accuracy points worse — it is "well calibrated" precisely by predicting the
    training frequencies and nothing else. A proper scoring rule cannot be gamed that way, and
    stating it as skill *against that control* makes the comparison the headline rather than a
    footnote.

    Args:
        brier: The model's Brier score, lower is better.
        reference_brier: The control's Brier score on the same predictions.

    Returns:
        The skill score, or ``None`` when the reference is 0 and skill is undefined.
    """
    if reference_brier <= 0:
        return None
    return float(1.0 - brier / reference_brier)


def risk_coverage(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    coverages: Sequence[float] = RISK_COVERAGE_GRID,
    weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Accuracy on the retained set as a function of how much of it is retained.

    Questions are ranked by top-label confidence and the least confident are abstained on
    first. At coverage ``c`` the most confident ``ceil(c * n)`` are kept and scored. This is
    the quantity a caller actually deploys against: "if I route the least confident 20% to a
    human, how good is what is left?".

    It depends only on the *order* of the confidences, which makes it a complement to ECE
    rather than a restatement of it. Note that our temperatures are fitted **per option-count
    bucket**, so calibration does move this curve: a single global temperature is monotone and
    would leave the ranking untouched, but different temperatures per bucket re-rank questions
    *across* buckets. That is a real effect worth reading — if per-bucket scaling improves the
    curve, the buckets were miscalibrated relative to each other. Ties break on the sort's
    stable order.

    Args:
        probabilities: One row per question.
        labels: Correct option index per question.
        coverages: Coverage grid, each in ``(0, 1]``.

    Returns:
        ``{"curve": [{"coverage", "n_kept", "accuracy", "min_confidence"}, ...],
        "selective_accuracy": {"0.8": ..., "0.9": ...}, "selective_threshold": {...},
        "aurc": ...}``. ``selective_threshold`` is the confidence at which to stop answering to
        reach that coverage — the setting an operator configures, where the accuracy is what
        they get for it. ``aurc`` is the area under the *risk* (1 - accuracy) curve over the
        grid, lower being better.

    Raises:
        ValueError: If any coverage is outside ``(0, 1]``.
    """
    rows, y = _as_arrays(probabilities, labels)
    w = _weights(weights, len(rows))
    confidence, correct = _confidence_and_correct(rows, y)
    order = np.argsort(-confidence, kind="stable")
    ranked_correct = correct[order]
    ranked_confidence = confidence[order]
    ranked_weight = w[order]
    cumulative = np.cumsum(ranked_weight)
    total_weight = float(cumulative[-1])

    def _at(coverage: float) -> tuple[int, float, float]:
        """How many to keep for a coverage, and the accuracy and threshold it buys.

        Coverage is a fraction of *weight*, not of rows: under case-control sampling answering
        "80% of questions" means 80% of the population the sample stands for.
        """
        keep = int(np.searchsorted(cumulative, coverage * total_weight, side="left") + 1)
        keep = min(max(keep, 1), len(rows))
        kept_accuracy = float(np.average(ranked_correct[:keep], weights=ranked_weight[:keep]))
        return keep, kept_accuracy, float(ranked_confidence[keep - 1])

    curve: list[dict[str, Any]] = []
    for coverage in coverages:
        if not 0 < coverage <= 1:
            raise ValueError(f"coverage must be in (0, 1], got {coverage}")
        keep, kept_accuracy, threshold = _at(coverage)
        curve.append(
            {
                "coverage": float(coverage),
                "n_kept": keep,
                "accuracy": kept_accuracy,
                "min_confidence": threshold,
            }
        )

    # Accuracy is what a coverage buys; the threshold is what an operator actually configures
    # to buy it — "answer when confidence >= 0.62" is the deployable form of "answer 80%".
    selective: dict[str, float] = {}
    selective_threshold: dict[str, float] = {}
    for c in SELECTIVE_COVERAGES:
        _keep, kept_accuracy, threshold = _at(c)
        selective[f"{c:g}"] = kept_accuracy
        selective_threshold[f"{c:g}"] = threshold
    # Trapezoid over the grid; only comparable between runs scored on the same grid.
    xs = [point["coverage"] for point in curve]
    risks = [1.0 - point["accuracy"] for point in curve]
    aurc = float(np.trapezoid(risks, xs) / (xs[-1] - xs[0])) if len(xs) > 1 else None
    return {
        "curve": curve,
        "selective_accuracy": selective,
        "selective_threshold": selective_threshold,
        "aurc": aurc,
    }


def summarize(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    n_bins: int = DEFAULT_BINS,
    weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Compute the full metric set for one group of predictions.

    Returns:
        A JSON-serialisable mapping with ``n``, ``accuracy``, ``ece``, ``mce``,
        ``brier_multiclass``, ``brier_top_label``, ``nll``, ``auroc_confidence``,
        ``mean_confidence``, ``mean_options`` and the reliability ``bins``.
        ``auroc_confidence`` is ``None`` when every prediction is right or every one is wrong.
    """
    rows, y = _as_arrays(probabilities, labels)
    w = _weights(weights, len(rows))
    confidence, correct = _confidence_and_correct(rows, y)
    bins = equal_mass_bins(probabilities, labels, n_bins, weights)
    total = sum(b.weight for b in bins)
    coverage = risk_coverage(probabilities, labels, weights=weights)
    return {
        "n": len(rows),
        "effective_n": float(w.sum()),
        "accuracy": float(np.average(correct, weights=w)),
        "ece": sum(b.weight * abs(b.gap) for b in bins) / total,
        "mce": max(abs(b.gap) for b in bins),
        "brier_multiclass": brier_multiclass(probabilities, labels, w),
        "brier_top_label": brier_top_label(probabilities, labels, w),
        "nll": nll(probabilities, labels, w),
        "auroc_confidence": auroc(confidence, correct, w),
        "mean_confidence": float(np.average(confidence, weights=w)),
        "mean_options": float(np.average([r.size for r in rows], weights=w)),
        "selective_accuracy": coverage["selective_accuracy"],
        "selective_threshold": coverage["selective_threshold"],
        "aurc": coverage["aurc"],
        "n_bins": len(bins),
        "bins": [asdict(b) | {"gap": b.gap} for b in bins],
    }


def summarize_by(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    groups: Sequence[str],
    n_bins: int = DEFAULT_BINS,
    min_count: int = 1,
    weights: Sequence[float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run :func:`summarize` separately for each group label.

    Args:
        probabilities: One row per question.
        labels: Correct option index per question.
        groups: Group key per question, e.g. the option-count bucket or the family.
        n_bins: Bins per group. Small groups get fewer, since bins cannot exceed the count.
        min_count: Groups smaller than this are skipped rather than reported on noise.

    Returns:
        Group label to metric mapping, in sorted key order.
    """
    if not (len(probabilities) == len(labels) == len(groups)):
        raise ValueError("probabilities, labels and groups must be the same length")
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(groups):
        buckets[key].append(i)
    out: dict[str, dict[str, Any]] = {}
    for key in sorted(buckets):
        idx = buckets[key]
        if len(idx) < min_count:
            continue
        out[key] = summarize(
            [probabilities[i] for i in idx],
            [labels[i] for i in idx],
            n_bins,
            None if weights is None else [weights[i] for i in idx],
        )
    return out


# --- negative controls -------------------------------------------------------


def uniform_probabilities(n_options: Sequence[int]) -> list[np.ndarray]:
    """The dumbest control: ``1/n`` over every option.

    Its accuracy is the floor any real model must clear, and its ECE is a reminder that a
    model can be well calibrated and useless at the same time.
    """
    return [np.full(n, 1.0 / n) for n in n_options]


def base_rate_probabilities(
    fit: Sequence[Prediction],
    evaluate: Sequence[Prediction],
    alpha: float = 1.0,
) -> list[np.ndarray]:
    """The base-rate negative control: predict the label frequencies, ignore the state.

    For each ``(family, n_options)`` group, the empirical distribution of answer indices in
    ``fit`` is used as a constant prediction for every question of that group in
    ``evaluate``. Groups unseen in ``fit`` fall back to uniform.

    This control matters because it can score a *better* ECE than a real model
    (`shamazharikh/qwen-rlcd` warns of exactly this): it is perfectly calibrated to the
    label marginal by construction while knowing nothing. A calibration claim is only
    meaningful when it beats this row on accuracy and AUROC too.

    Args:
        fit: Predictions whose labels define the base rates — the *validation* split, never
            the split being evaluated.
        evaluate: Predictions to produce base-rate probabilities for.
        alpha: Laplace smoothing added to every count, so no option gets probability zero.

    Returns:
        One probability row per entry of ``evaluate``, in order.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    counts: dict[tuple[str, int], np.ndarray] = {}
    for p in fit:
        key = (p.family, p.n_options)
        if key not in counts:
            counts[key] = np.full(p.n_options, alpha)
        # Weighted, so the control learns the *population* label frequencies rather than the
        # sample's. Under case-control sampling those differ by two orders of magnitude, and an
        # unweighted control would be a much easier baseline than the real one.
        counts[key][int(p.answer_idx)] += float(p.weight)
    rows = []
    for p in evaluate:
        c = counts.get((p.family, p.n_options))
        rows.append(np.full(p.n_options, 1.0 / p.n_options) if c is None else c / c.sum())
    return rows


@dataclass(frozen=True)
class Report:
    """A complete evaluation of one set of predictions.

    Attributes:
        overall: Metrics over every prediction.
        by_option_count: Metrics per option-count bucket.
        by_family: Metrics per task family.
        by_qtype: Metrics per primitive.
        controls: Metrics for the uniform and base-rate negative controls.
        skill: Brier skill against each control — the headline calibration number, because
            unlike ECE it cannot be won by refusing to commit.
        risk_coverage: The full risk-coverage curve over every prediction.
        meta: Provenance — run id, model, quantization, temperature, dataset, commit.
    """

    overall: dict[str, Any]
    by_option_count: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_family: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_qtype: dict[str, dict[str, Any]] = field(default_factory=dict)
    controls: dict[str, dict[str, Any]] = field(default_factory=dict)
    skill: dict[str, Any] = field(default_factory=dict)
    risk_coverage: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Serialise for ``metrics.json``."""
        return {
            "meta": self.meta,
            "overall": self.overall,
            "by_option_count": self.by_option_count,
            "by_family": self.by_family,
            "by_qtype": self.by_qtype,
            "controls": self.controls,
            "skill": self.skill,
            "risk_coverage": self.risk_coverage,
        }


def build_report(
    predictions: Sequence[Prediction],
    *,
    temperature: float | Mapping[str, float] = 1.0,
    control_fit: Sequence[Prediction] | None = None,
    n_bins: int = DEFAULT_BINS,
    meta: Mapping[str, Any] | None = None,
) -> Report:
    """Score predictions every way the project reports them.

    Args:
        predictions: What the model produced, as stored logits.
        temperature: A single temperature, or a mapping from option-count bucket to
            temperature (what :mod:`s1decide.calibrate` fits).
        control_fit: Predictions the base-rate control learns its frequencies from —
            normally the validation split. Omit to skip that control.
        n_bins: Equal-mass bins.
        meta: Provenance recorded verbatim in the report.

    Returns:
        The :class:`Report`.
    """
    if not predictions:
        raise ValueError("no predictions to score")

    def temp_for(p: Prediction) -> float:
        if isinstance(temperature, Mapping):
            return float(temperature.get(option_count_bucket(p.n_options), 1.0))
        return float(temperature)

    probs = [p.probabilities(temp_for(p)) for p in predictions]
    labels = [p.answer_idx for p in predictions]
    weights = [p.weight for p in predictions]

    controls: dict[str, dict[str, Any]] = {
        "uniform": summarize(
            uniform_probabilities([p.n_options for p in predictions]), labels, n_bins, weights
        )
    }
    if control_fit:
        controls["base_rate"] = summarize(
            base_rate_probabilities(control_fit, predictions), labels, n_bins, weights
        )

    overall = summarize(probs, labels, n_bins, weights)
    # Skill against every control we computed. `base_rate` is the one that matters: it is the
    # strategy that beats us on ECE, so beating it on a proper scoring rule is the claim.
    skill = {
        name: {
            "brier_multiclass": brier_skill_score(
                overall["brier_multiclass"], control["brier_multiclass"]
            ),
            "brier_top_label": brier_skill_score(
                overall["brier_top_label"], control["brier_top_label"]
            ),
            "accuracy_gain": overall["accuracy"] - control["accuracy"],
        }
        for name, control in controls.items()
    }

    return Report(
        overall=overall,
        by_option_count=summarize_by(
            probs,
            labels,
            [option_count_bucket(p.n_options) for p in predictions],
            n_bins,
            weights=weights,
        ),
        by_family=summarize_by(
            probs, labels, [p.family for p in predictions], n_bins, weights=weights
        ),
        by_qtype=summarize_by(
            probs, labels, [p.qtype for p in predictions], n_bins, weights=weights
        ),
        controls=controls,
        skill=skill,
        risk_coverage=risk_coverage(probs, labels, weights=weights),
        meta=dict(meta or {}),
    )
