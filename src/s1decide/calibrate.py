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
    "CALIBRATION_VERSION",
    "OPTION_COUNT_BUCKETS",
    "Calibration",
    "fit_calibration",
    "fit_temperature",
    "nll_at_temperature",
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

_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


def nll_at_temperature(
    logits: Sequence[Sequence[float]], labels: Sequence[int], temperature: float
) -> float:
    """Mean negative log-likelihood of the correct option at a given temperature.

    Args:
        logits: Raw masked logits, one row per question (rows may differ in length).
        labels: Index of the correct option per question.
        temperature: Positive scalar to divide the logits by.

    Returns:
        Mean NLL in nats.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    total = 0.0
    for row, target in zip(logits, labels):
        z = np.asarray(row, dtype=np.float64) / temperature
        total += float(np.logaddexp.reduce(z) - z[target])
    return total / len(labels)


def fit_temperature(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    bounds: tuple[float, float] = TEMPERATURE_BOUNDS,
    tolerance: float = 1e-4,
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

    def objective(beta: float) -> float:
        total = 0.0
        for row, target in zip(rows, targets):
            z = row * beta
            total += float(np.logaddexp.reduce(z) - z[target])
        return total / len(rows)

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


@dataclass(frozen=True)
class Calibration:
    """Fitted temperatures, one per option-count bucket, plus the context they are valid in.

    Attributes:
        temperatures: Bucket label to temperature, e.g. ``{"2": 1.8, "3-5": 2.1}``.
        default: Temperature for a bucket that was never fitted.
        quantization: What the model was running as when this was fitted, e.g. ``"nf4-bf16"``.
            Compared by :meth:`check_matches`.
        meta: Provenance — model, dataset, split, run id, commit, fit NLL before and after.
        version: Schema version.
    """

    temperatures: dict[str, float]
    default: float = 1.0
    quantization: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)
    version: str = CALIBRATION_VERSION

    def __post_init__(self) -> None:
        for bucket, t in self.temperatures.items():
            if not math.isfinite(t) or t <= 0:
                raise ValueError(f"bucket {bucket!r}: temperature must be positive, got {t}")
        if not math.isfinite(self.default) or self.default <= 0:
            raise ValueError(f"default temperature must be positive, got {self.default}")

    def temperature_for(self, n_options: int) -> float:
        """Temperature for a question with ``n_options`` options."""
        return float(self.temperatures.get(option_count_bucket(n_options), self.default))

    def apply(self, logits: Sequence[float]) -> np.ndarray:
        """Softmax one row of logits at its bucket's temperature.

        Args:
            logits: Raw masked logits for one question.

        Returns:
            Calibrated probabilities in option order.
        """
        z = np.asarray(logits, dtype=np.float64) / self.temperature_for(len(logits))
        z -= z.max()
        w = np.exp(z)
        return w / w.sum()

    def check_matches(self, quantization: str) -> None:
        """Raise if this calibration was not fitted at the quantization now in use.

        Args:
            quantization: The quantization the model is currently running as.

        Raises:
            ValueError: On a mismatch. Shipping a BF16-fitted temperature with a Q4 model is
                a documented anti-pattern (AGENTS.md), so it fails loudly rather than warning.
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
            "default": self.default,
            "temperatures": dict(sorted(self.temperatures.items())),
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
    def load(cls, path: str | Path) -> Calibration:
        """Read a ``calibration.json`` written by :meth:`save`."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("version") != CALIBRATION_VERSION:
            raise ValueError(
                f"calibration.json is version {raw.get('version')!r}, expected "
                f"{CALIBRATION_VERSION!r}"
            )
        return cls(
            temperatures={str(k): float(v) for k, v in raw["temperatures"].items()},
            default=float(raw.get("default", 1.0)),
            quantization=str(raw.get("quantization", "unknown")),
            meta=dict(raw.get("meta", {})),
            version=str(raw["version"]),
        )

    @classmethod
    def identity(cls, quantization: str = "unknown") -> Calibration:
        """An uncalibrated calibration: temperature 1 everywhere. The baseline to beat."""
        return cls(temperatures={}, default=1.0, quantization=quantization)


def fit_calibration(
    predictions: Sequence[Any],
    *,
    quantization: str,
    min_count: int = 30,
    meta: Mapping[str, Any] | None = None,
) -> Calibration:
    """Fit one temperature per option-count bucket on a held-out split.

    Args:
        predictions: :class:`eval.metrics.Prediction` objects from the **validation** split.
            Fitting on the split you then report is self-evaluation, not calibration.
        quantization: What the model was running as, recorded and later enforced.
        min_count: Buckets with fewer predictions than this are left at the default rather
            than fitted to noise; which ones were skipped is recorded in ``meta``.
        meta: Extra provenance to record.

    Returns:
        The fitted :class:`Calibration`, with before/after NLL per bucket in ``meta``.
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

    return Calibration(
        temperatures=temperatures,
        default=1.0,
        quantization=quantization,
        meta=dict(meta or {})
        | {
            "fitted_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "n_predictions": len(predictions),
            "min_count": min_count,
            "buckets": fit_info,
        },
    )
