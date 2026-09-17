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
    "brier_top_label",
    "ece",
    "equal_mass_bins",
    "mce",
    "nll",
    "option_count_bucket",
    "summarize",
    "summarize_by",
    "uniform_probabilities",
]

#: Number of equal-mass bins used for ECE and reliability diagrams.
DEFAULT_BINS = 15

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
    """

    id: str
    family: str
    qtype: str
    logits: tuple[float, ...]
    answer_idx: int

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
        )


@dataclass(frozen=True)
class ReliabilityBin:
    """One equal-mass bin of a reliability diagram.

    Attributes:
        count: Predictions in the bin.
        lo: Lowest confidence in the bin.
        hi: Highest confidence in the bin.
        mean_confidence: Mean predicted confidence.
        accuracy: Fraction actually correct. A calibrated model has this equal to
            ``mean_confidence``.
    """

    count: int
    lo: float
    hi: float
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        """Signed calibration gap: positive means over-confident."""
        return self.mean_confidence - self.accuracy


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


def accuracy(probabilities: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Top-1 accuracy."""
    rows, y = _as_arrays(probabilities, labels)
    return float(_confidence_and_correct(rows, y)[1].mean())


def nll(probabilities: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Mean negative log-likelihood of the correct option, in nats."""
    rows, y = _as_arrays(probabilities, labels)
    return float(-np.mean([math.log(max(float(r[t]), _EPS)) for r, t in zip(rows, y)]))


def brier_multiclass(probabilities: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Multiclass Brier score: mean squared error against the one-hot answer.

    Summed over options, so it lies in ``[0, 2]``. Reported alongside
    :func:`brier_top_label` because this version depends on the option count, which varies
    across the eval set.
    """
    rows, y = _as_arrays(probabilities, labels)
    total = 0.0
    for row, target in zip(rows, y):
        onehot = np.zeros_like(row)
        onehot[target] = 1.0
        total += float(np.sum((row - onehot) ** 2))
    return total / len(rows)


def brier_top_label(probabilities: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Brier score of the top-label confidence against whether it was correct.

    In ``[0, 1]`` and comparable across questions with different option counts. This is the
    quantity the reliability diagram decomposes.
    """
    rows, y = _as_arrays(probabilities, labels)
    confidence, correct = _confidence_and_correct(rows, y)
    return float(np.mean((confidence - correct) ** 2))


def equal_mass_bins(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    n_bins: int = DEFAULT_BINS,
) -> list[ReliabilityBin]:
    """Bin predictions into ``n_bins`` groups of (almost) equal count, sorted by confidence.

    Args:
        probabilities: One row per question.
        labels: Correct option index per question.
        n_bins: Number of bins. Bins are as equal in size as the count allows.

    Returns:
        Non-empty bins, in increasing confidence order.

    Note:
        Predictions with identical confidence can land in adjacent bins; with equal-mass
        binning that is unavoidable and affects the bin edges, not the ECE weighting.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    rows, y = _as_arrays(probabilities, labels)
    confidence, correct = _confidence_and_correct(rows, y)
    order = np.argsort(confidence, kind="mergesort")
    bins: list[ReliabilityBin] = []
    for chunk in np.array_split(order, min(n_bins, len(order))):
        if chunk.size == 0:  # pragma: no cover - array_split only empties when n_bins > n
            continue
        c = confidence[chunk]
        bins.append(
            ReliabilityBin(
                count=int(chunk.size),
                lo=float(c.min()),
                hi=float(c.max()),
                mean_confidence=float(c.mean()),
                accuracy=float(correct[chunk].mean()),
            )
        )
    return bins


def ece(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    n_bins: int = DEFAULT_BINS,
) -> float:
    """Expected calibration error over equal-mass bins.

    The count-weighted mean absolute gap between confidence and accuracy. Zero is perfect;
    it says nothing about whether the model is *right*, which is why it is always reported
    with Brier, accuracy and the base-rate control.
    """
    bins = equal_mass_bins(probabilities, labels, n_bins)
    total = sum(b.count for b in bins)
    return sum(b.count * abs(b.gap) for b in bins) / total


def mce(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    n_bins: int = DEFAULT_BINS,
) -> float:
    """Maximum calibration error: the worst single-bin gap."""
    return max(abs(b.gap) for b in equal_mass_bins(probabilities, labels, n_bins))


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


def auroc(scores: Sequence[float], positive: Sequence[float]) -> float | None:
    """Area under the ROC curve, computed from ranks so ties are handled exactly.

    Args:
        scores: Higher means more likely positive.
        positive: 1 for a positive case, 0 for a negative one.

    Returns:
        The AUROC, or ``None`` when one class is absent and the quantity is undefined.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(positive, dtype=np.float64)
    if s.size != y.size:
        raise ValueError(f"{s.size} scores but {y.size} labels")
    n_pos = float(y.sum())
    n_neg = float(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _average_ranks(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def summarize(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    n_bins: int = DEFAULT_BINS,
) -> dict[str, Any]:
    """Compute the full metric set for one group of predictions.

    Returns:
        A JSON-serialisable mapping with ``n``, ``accuracy``, ``ece``, ``mce``,
        ``brier_multiclass``, ``brier_top_label``, ``nll``, ``auroc_confidence``,
        ``mean_confidence``, ``mean_options`` and the reliability ``bins``.
        ``auroc_confidence`` is ``None`` when every prediction is right or every one is wrong.
    """
    rows, y = _as_arrays(probabilities, labels)
    confidence, correct = _confidence_and_correct(rows, y)
    bins = equal_mass_bins(probabilities, labels, n_bins)
    total = sum(b.count for b in bins)
    return {
        "n": len(rows),
        "accuracy": float(correct.mean()),
        "ece": sum(b.count * abs(b.gap) for b in bins) / total,
        "mce": max(abs(b.gap) for b in bins),
        "brier_multiclass": brier_multiclass(probabilities, labels),
        "brier_top_label": brier_top_label(probabilities, labels),
        "nll": nll(probabilities, labels),
        "auroc_confidence": auroc(confidence, correct),
        "mean_confidence": float(confidence.mean()),
        "mean_options": float(np.mean([r.size for r in rows])),
        "n_bins": len(bins),
        "bins": [asdict(b) | {"gap": b.gap} for b in bins],
    }


def summarize_by(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    groups: Sequence[str],
    n_bins: int = DEFAULT_BINS,
    min_count: int = 1,
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
        out[key] = summarize([probabilities[i] for i in idx], [labels[i] for i in idx], n_bins)
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
        counts[key][int(p.answer_idx)] += 1.0
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
        meta: Provenance — run id, model, quantization, temperature, dataset, commit.
    """

    overall: dict[str, Any]
    by_option_count: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_family: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_qtype: dict[str, dict[str, Any]] = field(default_factory=dict)
    controls: dict[str, dict[str, Any]] = field(default_factory=dict)
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

    controls: dict[str, dict[str, Any]] = {
        "uniform": summarize(
            uniform_probabilities([p.n_options for p in predictions]), labels, n_bins
        )
    }
    if control_fit:
        controls["base_rate"] = summarize(
            base_rate_probabilities(control_fit, predictions), labels, n_bins
        )

    return Report(
        overall=summarize(probs, labels, n_bins),
        by_option_count=summarize_by(
            probs, labels, [option_count_bucket(p.n_options) for p in predictions], n_bins
        ),
        by_family=summarize_by(probs, labels, [p.family for p in predictions], n_bins),
        by_qtype=summarize_by(probs, labels, [p.qtype for p in predictions], n_bins),
        controls=controls,
        meta=dict(meta or {}),
    )
