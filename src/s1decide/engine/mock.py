"""A deterministic, dependency-free engine for CPU tests and CI.

Logits are derived from a hash of ``(seed, prefix, suffix, label)``, so the same call always
returns the same numbers, different labels get different numbers, and nothing about the
output depends on torch, a GPU or a download. It knows nothing about language; it exists so
the whole API above the engine — rendering, masking, softmax, `Result` construction, the
fuzz test — can be exercised end to end.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from s1decide.engine.base import EngineOutput

__all__ = ["MockEngine"]


class MockEngine:
    """Hash-based stand-in for a real model.

    Args:
        seed: Changes every logit; two mocks with different seeds disagree.
        scale: Logits are uniform in ``[-scale, scale]``. Larger values make the mock more
            "confident" (peakier softmax), which is useful for calibration tests.
    """

    name = "mock"

    def __init__(self, seed: int = 0, scale: float = 3.0) -> None:
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.seed = seed
        self.scale = scale

    def _logit(self, prefix: str, suffix: str, label: str) -> float:
        digest = hashlib.blake2b(
            f"{self.seed}\x1f{prefix}\x1f{suffix}\x1f{label}".encode(), digest_size=8
        ).digest()
        unit = int.from_bytes(digest, "big") / 2**64  # [0, 1)
        return (2.0 * unit - 1.0) * self.scale

    def score(
        self, prefix: str, suffixes: Sequence[str], labels: Sequence[Sequence[str]]
    ) -> EngineOutput:
        """Return hash-derived logits shaped exactly like a real engine's output."""
        if len(suffixes) != len(labels):
            raise ValueError(f"{len(suffixes)} suffixes but {len(labels)} label sets")
        rows = tuple(
            tuple(self._logit(prefix, suffix, label) for label in row_labels)
            for suffix, row_labels in zip(suffixes, labels)
        )
        return EngineOutput(
            logits=rows,
            meta={"engine": self.name, "seed": self.seed, "rows": len(rows), "passes": 1},
        )
