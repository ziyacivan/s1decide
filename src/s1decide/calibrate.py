"""Temperature scaling, fitted per option-count bucket and per deployment quantization.

Temperature scaling divides the logits by one positive number before the softmax. It cannot
change which option wins, so **accuracy is untouched**; it only moves confidence. That makes
it the right first calibration step: any ECE it removes was pure over- or under-confidence.

Two rules from CLAUDE.md are baked in here:

* **One temperature per option-count bucket.** A model that is well calibrated over two
  options is usually over-confident over seventy-seven; a single global temperature averages
  those apart.
* **Fitted at the deployment quantization.** Q4 logits are not BF16 logits, so a calibration
  fitted in BF16 and shipped with a Q4 model is fitted to a model nobody runs. The
  quantization is recorded in the file and :meth:`Calibration.check_matches` refuses a
  silent mismatch.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CALIBRATION_METHODS",
    "CALIBRATION_VERSION",
    "GGUF_QUANTIZATION_MARKERS",
    "MIN_VECTOR_SAMPLES_PER_PARAMETER",
    "OPTION_COUNT_BUCKETS",
    "Calibration",
    "CalibrationSet",
    "default_method",
    "fit_calibration",
    "fit_temperature",
    "fit_vector_scaling",
    "is_gguf",
    "nll_at_temperature",
    "nll_with_vector",
    "option_count_bucket",
]

#: Schema version of ``calibration.json``.
CALIBRATION_VERSION = "0.1"

#: Option-count buckets from AGENTS.md, plus an open bucket above the documented range.
#: Calibration is fitted per bucket and the eval reports per bucket, so the definition lives
#: here — in the installable library — and `eval.metrics` imports it, never the other way round.
OPTION_COUNT_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("2", 2, 2),
    ("3-5", 3, 5),
    ("6-16", 6, 16),
    ("17-77", 17, 77),
    ("78+", 78, 1_000_000),
)


def option_count_bucket(n_options: int) -> str:
    """Return the reporting and calibration bucket for a question with ``n_options`` options.

    Args:
        n_options: Number of options.

    Returns:
        The bucket label, e.g. ``"3-5"``.

    Raises:
        ValueError: If ``n_options`` is below 2.
    """
    if n_options < 2:
        raise ValueError(f"a question needs at least 2 options, got {n_options}")
    for label, lo, hi in OPTION_COUNT_BUCKETS:
        if lo <= n_options <= hi:
            return label
    raise ValueError(f"no bucket for {n_options} options")  # pragma: no cover - buckets are total


#: Search bounds for the temperature. Wide enough to express "far too confident" (T >> 1)
#: and "far too timid" (T << 1) without letting a degenerate fit run away.
TEMPERATURE_BOUNDS = (0.05, 20.0)

#: The calibration methods a run may fit and report.
#:
#: ``temperature`` divides every logit by one scalar per option-count bucket. It can only make a
#: distribution sharper or flatter; it cannot change which option wins, so accuracy is identical
#: before and after by construction.
#:
#: ``vector`` adds a per-position bias as well: ``softmax(z / T + b)``. It **can** change the
#: winner, and that is the point — the option→token mapping is shuffled per example, so a
#: position that is still systematically favoured reflects the model preferring a *slot* rather
#: than an answer. Correcting that is a real gain; it is also a real risk, which is why both
#: methods are fitted, both are reported, and the deployed one is named in `calibration.json`.
CALIBRATION_METHODS: tuple[str, ...] = ("temperature", "vector")

#: A bias vector has one free parameter per option, so a bucket spanning 3-5 options cannot
#: share one. Vector scaling is therefore fitted per **exact** option count, and only where
#: there is enough data for `n + 1` parameters not to be fitted to noise.
MIN_VECTOR_SAMPLES_PER_PARAMETER = 20

_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


def _validate_weights(weights: Sequence[float] | None, n: int) -> np.ndarray:
    """Validate importance weights, defaulting to uniform.

    Args:
        weights: One positive weight per question, or ``None``.
        n: How many questions there are.

    Returns:
        A float array of length ``n``.

    Raises:
        ValueError: If the length is wrong, or a weight is not finite and positive. Zero is
            refused rather than treated as "ignore this row": dropping a question and giving it
            no weight are different statements, and only one of them is honest.
    """
    if weights is None:
        return np.ones(n, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.size != n:
        raise ValueError(f"{w.size} weights for {n} questions")
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("weights must be finite and positive")
    return w


def nll_at_temperature(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    temperature: float,
    weights: Sequence[float] | None = None,
) -> float:
    """Mean negative log-likelihood of the correct option at a given temperature.

    Args:
        logits: Raw masked logits, one row per question (rows may differ in length).
        labels: Index of the correct option per question.
        temperature: Positive scalar to divide the logits by.
        weights: Importance weight per question. Under case-control sampling these are what
            make the fitted temperature the population's rather than the sample's — a
            temperature fitted on a 6:1 sample and deployed at 96:1 is fitted to the wrong
            distribution, which is the whole reason weights exist here.

    Returns:
        Weighted mean NLL in nats.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    w = _validate_weights(weights, len(labels))
    total = 0.0
    for row, target, weight in zip(logits, labels, w):
        z = np.asarray(row, dtype=np.float64) / temperature
        total += weight * float(np.logaddexp.reduce(z) - z[target])
    return total / float(w.sum())


def fit_temperature(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    bounds: tuple[float, float] = TEMPERATURE_BOUNDS,
    tolerance: float = 1e-4,
    weights: Sequence[float] | None = None,
) -> float:
    """Find the temperature minimising NLL, by golden-section search on the inverse temperature.

    NLL is convex in ``beta = 1/T`` — it is ``logsumexp(beta z) - beta z_y``, a convex function
    plus a linear one — so a bracketed 1-D search finds the global optimum. No scipy, no
    gradients, and the same answer on every platform.

    Args:
        logits: Raw masked logits, one row per question.
        labels: Index of the correct option per question.
        bounds: ``(min_temperature, max_temperature)``.
        tolerance: Absolute tolerance on ``beta``.
        weights: Importance weight per question. A weighted NLL is still convex in ``beta`` —
            a non-negative combination of convex functions — so the search is unchanged.

    Returns:
        The fitted temperature. A value at a bound means the optimum lies outside the search
        range, which is worth noticing rather than silently extrapolating.

    Raises:
        ValueError: If there is nothing to fit or the bounds are invalid.
    """
    if len(logits) != len(labels):
        raise ValueError(f"{len(logits)} logit rows but {len(labels)} labels")
    if not logits:
        raise ValueError("no data to fit a temperature on")
    low_t, high_t = bounds
    if not 0 < low_t < high_t:
        raise ValueError(f"invalid bounds {bounds}")

    rows = [np.asarray(r, dtype=np.float64) for r in logits]
    targets = np.asarray(labels, dtype=np.int64)
    w = _validate_weights(weights, len(rows))
    mass = float(w.sum())

    def objective(beta: float) -> float:
        total = 0.0
        for row, target, weight in zip(rows, targets, w):
            z = row * beta
            total += weight * float(np.logaddexp.reduce(z) - z[target])
        return total / mass

    lo, hi = 1.0 / high_t, 1.0 / low_t
    b, c = hi - _GOLDEN * (hi - lo), lo + _GOLDEN * (hi - lo)
    fb, fc = objective(b), objective(c)
    while hi - lo > tolerance:
        if fb < fc:
            hi, c, fc = c, b, fb
            b = hi - _GOLDEN * (hi - lo)
            fb = objective(b)
        else:
            lo, b, fb = b, c, fc
            c = lo + _GOLDEN * (hi - lo)
            fc = objective(c)
    return float(1.0 / (0.5 * (lo + hi)))


def fit_vector_scaling(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    weights: Sequence[float] | None = None,
    iterations: int = 500,
    tolerance: float = 1e-7,
) -> tuple[float, list[float]]:
    """Fit ``softmax(z / T + b)`` by gradient descent on the weighted NLL.

    Every row must have the same number of options: a bias vector is indexed by position, so
    there is nothing to share between a 3-option and a 5-option question.

    The objective is convex in ``(beta, b)`` where ``beta = 1/T`` — a log-sum-exp of an affine
    function minus a linear term — so a plain descent with a backtracking step finds the global
    optimum, and does it identically on every platform. No scipy, no gradients library.

    ``b`` is centred to sum zero after every step. Softmax is shift-invariant, so without that
    the bias would drift along a flat direction forever and two equivalent fits would serialise
    as different numbers.

    Args:
        logits: Raw masked logits, one row per question, all the same length.
        labels: Index of the correct option per question.
        weights: Importance weight per question, for case-control sampling.
        iterations: Maximum descent steps.
        tolerance: Stop when the objective improves by less than this.

    Returns:
        ``(temperature, bias)``.

    Raises:
        ValueError: If there is nothing to fit, or the rows differ in length.
    """
    if not logits:
        raise ValueError("no data to fit vector scaling on")
    # Checked before np.asarray: numpy raises on ragged input too, but with a message about
    # inhomogeneous shapes that says nothing about what the caller did wrong.
    widths = {len(row) for row in logits}
    if len(widths) != 1:
        raise ValueError(
            f"vector scaling needs rows of equal length; got widths {sorted(widths)} — "
            "group by option count first"
        )
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if z.shape[0] != y.size:
        raise ValueError(f"{z.shape[0]} logit rows but {y.size} labels")
    w = _validate_weights(weights, z.shape[0])
    mass = float(w.sum())
    n_options = z.shape[1]
    onehot = np.zeros_like(z)
    onehot[np.arange(y.size), y] = 1.0

    def objective(beta: float, bias: np.ndarray) -> float:
        scores = beta * z + bias
        shifted = scores - scores.max(axis=1, keepdims=True)
        log_norm = np.log(np.exp(shifted).sum(axis=1)) + scores.max(axis=1)
        return float(np.sum(w * (log_norm - scores[np.arange(y.size), y])) / mass)

    beta = 1.0
    bias = np.zeros(n_options, dtype=np.float64)
    step = 0.5
    current = objective(beta, bias)

    for _ in range(iterations):
        scores = beta * z + bias
        shifted = scores - scores.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        residual = probabilities - onehot
        grad_beta = float(np.sum(w * np.sum(residual * z, axis=1)) / mass)
        grad_bias = (w[:, None] * residual).sum(axis=0) / mass
        grad_bias -= grad_bias.mean()

        # Backtracking: halve the step until the objective actually falls. A convex objective
        # plus a step that never increases it cannot diverge, which is what makes the fixed
        # iteration count safe.
        improved = False
        for _ in range(40):
            candidate_beta = max(beta - step * grad_beta, 1.0 / TEMPERATURE_BOUNDS[1])
            candidate_beta = min(candidate_beta, 1.0 / TEMPERATURE_BOUNDS[0])
            candidate_bias = bias - step * grad_bias
            candidate_bias -= candidate_bias.mean()
            value = objective(candidate_beta, candidate_bias)
            if value <= current:
                improved = value < current - tolerance
                beta, bias, current = candidate_beta, candidate_bias, value
                step *= 1.1
                break
            step *= 0.5
        else:
            break
        if not improved:
            break

    return float(1.0 / beta), [float(x) for x in bias]


def nll_with_vector(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    temperature: float,
    bias: Sequence[float],
    weights: Sequence[float] | None = None,
) -> float:
    """Weighted mean NLL under ``softmax(z / T + b)``, for reporting the fit."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    z = np.asarray(logits, dtype=np.float64) / temperature + np.asarray(bias, dtype=np.float64)
    w = _validate_weights(weights, z.shape[0])
    y = np.asarray(labels, dtype=np.int64)
    shifted = z - z.max(axis=1, keepdims=True)
    log_norm = np.log(np.exp(shifted).sum(axis=1)) + z.max(axis=1)
    return float(np.sum(w * (log_norm - z[np.arange(y.size), y])) / float(w.sum()))


@dataclass(frozen=True)
class Calibration:
    """Fitted temperatures, one per option-count bucket, plus the context they are valid in.

    Attributes:
        temperatures: Bucket label to temperature, e.g. ``{"2": 1.8, "3-5": 2.1}``.
        default: Temperature for a bucket that was never fitted.
        quantization: What the model was running as when this was fitted, e.g. ``"nf4-bf16"``.
            Compared by :meth:`check_matches`.
        vectors: Per **exact** option count, ``{"3": {"temperature": T, "bias": [...]}}`` —
            the vector-scaling fit. Keyed by count rather than bucket because a bias vector is
            indexed by position and a 3-option question has nothing to share with a 5-option one.
        primitive: Which primitive this fit is for — ``"choice"``, ``"score"``, ``"noul"``, or
            ``"all"`` for a fit shared across them. A bias vector is indexed by option position,
            so a two-option `Noul` fit has nothing to lend a five-level `Score`.
        method: Which method :meth:`apply` uses. Both are always fitted and reported; this names
            the one that is deployed. :func:`default_method` chooses it from the quantization.
        meta: Provenance — model, dataset, split, run id, commit, fit NLL before and after.
        version: Schema version.
    """

    temperatures: dict[str, float]
    default: float = 1.0
    quantization: str = "unknown"
    vectors: dict[str, dict[str, Any]] = field(default_factory=dict)
    primitive: str = "all"
    method: str = "temperature"
    meta: dict[str, Any] = field(default_factory=dict)
    version: str = CALIBRATION_VERSION

    def __post_init__(self) -> None:
        for bucket, t in self.temperatures.items():
            if not math.isfinite(t) or t <= 0:
                raise ValueError(f"bucket {bucket!r}: temperature must be positive, got {t}")
        if not math.isfinite(self.default) or self.default <= 0:
            raise ValueError(f"default temperature must be positive, got {self.default}")
        if self.method not in CALIBRATION_METHODS:
            raise ValueError(f"unknown method {self.method!r}; known: {CALIBRATION_METHODS}")
        for count, fit in self.vectors.items():
            bias = fit.get("bias", [])
            if len(bias) != int(count):
                raise ValueError(
                    f"option count {count}: bias vector has {len(bias)} entries, expected {count}"
                )
            if not math.isfinite(fit.get("temperature", 0.0)) or fit["temperature"] <= 0:
                raise ValueError(f"option count {count}: temperature must be positive")

    def temperature_for(self, n_options: int) -> float:
        """Temperature for a question with ``n_options`` options."""
        return float(self.temperatures.get(option_count_bucket(n_options), self.default))

    def vector_for(self, n_options: int) -> tuple[float, list[float]] | None:
        """The vector-scaling fit for this exact option count, if one was fitted."""
        fit = self.vectors.get(str(n_options))
        if not fit:
            return None
        return float(fit["temperature"]), [float(x) for x in fit["bias"]]

    def apply(self, logits: Sequence[float], method: str | None = None) -> np.ndarray:
        """Softmax one row of logits under the chosen calibration method.

        Args:
            logits: Raw masked logits for one question.
            method: Override the deployed method, for reporting both side by side.

        Returns:
            Calibrated probabilities in option order.

        Raises:
            ValueError: If ``method`` is not a known method.
        """
        chosen = method or self.method
        if chosen not in CALIBRATION_METHODS:
            raise ValueError(f"unknown method {chosen!r}; known: {CALIBRATION_METHODS}")
        z = np.asarray(logits, dtype=np.float64)
        fit = self.vector_for(len(logits)) if chosen == "vector" else None
        if fit is not None:
            # Vector scaling where it was fitted; a count with too little data falls back to the
            # temperature rather than to an unfitted bias of zeros, which would silently be a
            # different method under the same name.
            temperature, bias = fit
            z = z / temperature + np.asarray(bias, dtype=np.float64)
        else:
            z = z / self.temperature_for(len(logits))
        z -= z.max()
        w = np.exp(z)
        return w / w.sum()

    def with_method(self, method: str) -> Calibration:
        """A copy that deploys ``method``. Both fits are carried either way."""
        if method not in CALIBRATION_METHODS:
            raise ValueError(f"unknown method {method!r}; known: {CALIBRATION_METHODS}")
        return Calibration(
            temperatures=dict(self.temperatures),
            default=self.default,
            quantization=self.quantization,
            vectors=dict(self.vectors),
            primitive=self.primitive,
            method=method,
            meta=dict(self.meta),
            version=self.version,
        )

    def check_matches(self, quantization: str) -> None:
        """Raise if this calibration was not fitted at the quantization now in use.

        Args:
            quantization: The quantization the model is currently running as.

        Raises:
            ValueError: On a mismatch. Shipping a BF16-fitted temperature with a Q4 model is
                a documented anti-pattern (AGENTS.md), so it fails loudly rather than warning —
                and `docs/research/quantization-label-disagreement-2026-09-18.md` measures why:
                two 4-bit quantizations of one model disagree on 22% of an ordinal judgement.
        """
        if self.quantization != quantization:
            raise ValueError(
                f"calibration was fitted at {self.quantization!r} but the model is running as "
                f"{quantization!r}; re-fit at the deployment quantization"
            )

    def to_json(self) -> dict[str, Any]:
        """Serialise for ``calibration.json``."""
        return {
            "version": self.version,
            "quantization": self.quantization,
            "primitive": self.primitive,
            "method": self.method,
            "default": self.default,
            "temperatures": dict(sorted(self.temperatures.items())),
            "vectors": {k: self.vectors[k] for k in sorted(self.vectors, key=int)},
            "meta": self.meta,
        }

    def save(self, path: str | Path) -> Path:
        """Write ``calibration.json``. Returns the path written."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_json(), indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        return target

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> Calibration:
        """Build one fit from its serialised form."""
        if raw.get("version") != CALIBRATION_VERSION:
            raise ValueError(
                f"calibration entry is version {raw.get('version')!r}, expected "
                f"{CALIBRATION_VERSION!r}"
            )
        return cls(
            temperatures={str(k): float(v) for k, v in raw["temperatures"].items()},
            default=float(raw.get("default", 1.0)),
            quantization=str(raw.get("quantization", "unknown")),
            # Absent in files written before vector scaling existed; those load as
            # temperature-only rather than failing, which is what they are.
            vectors={str(k): dict(v) for k, v in raw.get("vectors", {}).items()},
            primitive=str(raw.get("primitive", "all")),
            method=str(raw.get("method", "temperature")),
            meta=dict(raw.get("meta", {})),
            version=str(raw["version"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        """Read a single-fit ``calibration.json`` written by :meth:`save`."""
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def identity(cls, quantization: str = "unknown") -> Calibration:
        """An uncalibrated calibration: temperature 1 everywhere. The baseline to beat."""
        return cls(temperatures={}, default=1.0, quantization=quantization)


def fit_calibration(
    predictions: Sequence[Any],
    *,
    quantization: str,
    primitive: str = "all",
    min_count: int = 30,
    meta: Mapping[str, Any] | None = None,
    method: str | None = None,
) -> Calibration:
    """Fit one temperature per option-count bucket on a held-out split.

    Args:
        predictions: :class:`eval.metrics.Prediction` objects from the **validation** split.
            Fitting on the split you then report is self-evaluation, not calibration.
        quantization: What the model was running as, recorded and later enforced. It also
            chooses the deployed method unless ``method`` overrides it — see
            :func:`default_method`.
        primitive: Which primitive these predictions are, recorded on the fit.
        min_count: Buckets with fewer predictions than this are left at the default rather
            than fitted to noise; which ones were skipped is recorded in ``meta``.
        meta: Extra provenance to record.
        method: Force the deployed method instead of taking it from the quantization.

    Returns:
        The fitted :class:`Calibration`. **Both** methods are fitted: one temperature per
        option-count bucket, and one temperature-plus-bias per exact option count where there
        is enough data. Which one is *deployed* comes from the quantization — vector scaling for
        a GGUF artefact, temperature otherwise — and the eval reports both either way, so the
        comparison is visible rather than assumed.

        The fit target is always the validation split's own labels. It is never agreement with
        another runtime: nf4 is a diagnostic reference for measuring drift, not a judge of what
        the right answer is, and on the `Noul` study it was in fact the *less* accurate of the
        two runtimes being compared.
    """
    by_bucket: dict[str, list[Any]] = {}
    for p in predictions:
        by_bucket.setdefault(option_count_bucket(p.n_options), []).append(p)

    temperatures: dict[str, float] = {}
    fit_info: dict[str, Any] = {}
    for bucket, group in sorted(by_bucket.items()):
        logits = [p.logits for p in group]
        labels = [p.answer_idx for p in group]
        if len(group) < min_count:
            fit_info[bucket] = {"n": len(group), "skipped": f"fewer than {min_count} examples"}
            continue
        temperature = fit_temperature(logits, labels)
        temperatures[bucket] = temperature
        fit_info[bucket] = {
            "n": len(group),
            "temperature": temperature,
            "nll_before": nll_at_temperature(logits, labels, 1.0),
            "nll_after": nll_at_temperature(logits, labels, temperature),
        }

    # Vector scaling, per exact option count. A bias vector has one parameter per option, so
    # the data requirement scales with the option count — fitting 26 biases from 40 examples
    # would produce a confident-looking artefact of the sample.
    by_count: dict[int, list[Any]] = {}
    for prediction in predictions:
        by_count.setdefault(prediction.n_options, []).append(prediction)

    vectors: dict[str, dict[str, Any]] = {}
    vector_info: dict[str, Any] = {}
    for count, group in sorted(by_count.items()):
        needed = MIN_VECTOR_SAMPLES_PER_PARAMETER * (count + 1)
        logits = [item.logits for item in group]
        labels = [item.answer_idx for item in group]
        weights = [getattr(item, "weight", 1.0) for item in group]
        if len(group) < needed:
            vector_info[str(count)] = {
                "n": len(group),
                "skipped": f"fewer than {needed} examples for {count + 1} parameters",
            }
            continue
        temperature, bias = fit_vector_scaling(logits, labels, weights=weights)
        vectors[str(count)] = {"temperature": temperature, "bias": bias, "n": len(group)}
        vector_info[str(count)] = {
            "n": len(group),
            "temperature": temperature,
            "max_abs_bias": max(abs(b) for b in bias),
            "nll_temperature_only": nll_at_temperature(
                logits, labels, fit_temperature(logits, labels, weights=weights), weights
            ),
            "nll_vector": nll_with_vector(logits, labels, temperature, bias, weights),
        }

    deployed = method or default_method(quantization)
    if deployed == "vector" and not vectors:
        # A GGUF row with too little data for any bias vector. Deploying "vector" would silently
        # fall back to the temperature on every question; saying so is better than implying a fit
        # that does not exist.
        deployed = "temperature"
        vector_info["deployed"] = "temperature: no option count had enough data for a bias vector"

    return Calibration(
        vectors=vectors,
        temperatures=temperatures,
        default=1.0,
        quantization=quantization,
        primitive=primitive,
        method=deployed,
        meta=dict(meta or {})
        | {
            "fitted_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "n_predictions": len(predictions),
            "min_count": min_count,
            "primitive": primitive,
            "method_default_for_quantization": default_method(quantization),
            "fit_target": "validation labels; never agreement with another runtime",
            "buckets": fit_info,
            "vector_scaling": vector_info,
        },
    )


#: Quantization names that mean a GGUF artefact, matched case-insensitively as a substring.
#:
#: The list is the llama.cpp naming scheme rather than a guess: a k-quant, a legacy quant or an
#: 8-bit dump all arrive through the same runtime.
GGUF_QUANTIZATION_MARKERS: tuple[str, ...] = (
    "gguf",
    "q2_k",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
    "q8_0",
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
)


def is_gguf(quantization: str) -> bool:
    """Whether a quantization label names a GGUF artefact.

    Args:
        quantization: A label such as ``"nf4-bf16"`` or ``"q4_k_m"``.

    Returns:
        True for GGUF artefacts.
    """
    lowered = quantization.lower()
    return any(marker in lowered for marker in GGUF_QUANTIZATION_MARKERS)


def default_method(quantization: str) -> str:
    """Which calibration method a quantization deploys by default.

    **GGUF artefacts deploy vector scaling; everything else deploys temperature.** Measured
    reason, not a preference: switching from bitsandbytes nf4 to llama.cpp Q4_K_M moved the
    yes-versus-no log-odds of all 300 rows of the `Noul` stability study in the same direction,
    by a near-constant -2.075 nats. A temperature is monotone, so it cannot move a two-option
    decision at all — the best temperature recovered 0.887 agreement against 0.887 uncorrected,
    exactly nothing — while a per-option bias recovered 0.930 and the two together 0.953.

    A shift that one extra parameter removes should not be reported as a cost of the
    quantization. Temperature is still fitted and still reported beside it, because the
    comparison is the point: it shows whether a row's cost is confidence or position.

    See `docs/research/noul-quantization-offset-2026-09-23.md`.

    Args:
        quantization: The deployment quantization label.

    Returns:
        ``"vector"`` or ``"temperature"``.
    """
    return "vector" if is_gguf(quantization) else "temperature"


@dataclass
class CalibrationSet:
    """Every fit a release carries, keyed by quantization and primitive.

    One number per release is not enough. A quantization changes the logits, and a primitive
    changes what a logit *is* — a `Noul` is two options and a `Score` is an ordered five, so a
    bias vector fitted on one says nothing about the other. The quantization table reports a row
    per quantization, and each row has to name the fit it deployed.

    Attributes:
        entries: ``{"q4_k_m/noul": Calibration, ...}``. Keys are ``quantization/primitive``.
        version: Schema version.
    """

    entries: dict[str, Calibration] = field(default_factory=dict)
    version: str = CALIBRATION_VERSION

    @staticmethod
    def key(quantization: str, primitive: str) -> str:
        """The lookup key for a (quantization, primitive) pair."""
        return f"{quantization}/{primitive}"

    def add(self, calibration: Calibration) -> CalibrationSet:
        """Record a fit under its own quantization and primitive. Returns self, for chaining."""
        self.entries[self.key(calibration.quantization, calibration.primitive)] = calibration
        return self

    def get(self, quantization: str, primitive: str) -> Calibration:
        """The fit for a pair, or the identity.

        Falls back to a fit for the same quantization across all primitives before giving up, so
        a release that fitted one calibration per quantization still resolves. It does **not**
        fall back across quantizations: deploying a fit from a different quantization is the
        specific mistake the whole study exists to prevent, and an identity calibration is
        honestly uncalibrated where a borrowed one is quietly wrong.

        Args:
            quantization: Deployment quantization.
            primitive: ``"choice"``, ``"score"`` or ``"noul"``.

        Returns:
            The fitted calibration, or an identity for this quantization.
        """
        exact = self.entries.get(self.key(quantization, primitive))
        if exact is not None:
            return exact
        shared = self.entries.get(self.key(quantization, "all"))
        if shared is not None:
            return shared
        return Calibration.identity(quantization)

    def to_json(self) -> dict[str, Any]:
        """Serialise every entry, each naming its own method and parameters."""
        return {
            "version": self.version,
            "entries": {k: self.entries[k].to_json() for k in sorted(self.entries)},
        }

    def save(self, path: str | Path) -> Path:
        """Write ``calibration.json``. Returns the path written."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_json(), indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> CalibrationSet:
        """Read a ``calibration.json``, in either the set form or the single-fit form.

        A file written before this schema existed holds one fit and no primitive; it loads as a
        single ``all`` entry rather than failing, because those files are still valid fits.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if "entries" not in raw:
            single = Calibration.load(path)
            return cls(entries={cls.key(single.quantization, single.primitive): single})
        entries = {}
        for key, payload in raw["entries"].items():
            entries[key] = Calibration.from_json(payload)
        return cls(entries=entries, version=str(raw.get("version", CALIBRATION_VERSION)))
