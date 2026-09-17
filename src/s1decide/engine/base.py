"""The engine contract every backend implements.

An engine does exactly one thing: given a shared prefix, N suffixes and N label sets, return
the raw logits of each label at each suffix's answer position. It never decodes, never
samples and never sees a `Question` — the string rendering above it and the probability
maths below it are shared, so engines stay small and interchangeable, and the same
contract tests run against all of them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = ["Engine", "EngineError", "EngineOutput", "validate_output"]


class EngineError(RuntimeError):
    """An engine returned something that does not match the request."""


@dataclass(frozen=True)
class EngineOutput:
    """Raw answer-position logits, one row per suffix, one value per label.

    Logits, not probabilities, on purpose: temperature scaling (`s1decide.calibrate`) must
    operate before the softmax, and the eval needs the unnormalised values too.

    Attributes:
        logits: ``logits[i][j]`` is the logit of label ``j`` for suffix ``i``.
        meta: Engine-specific diagnostics — token counts, timings, how many forward passes
            were needed. Never required for correctness.
    """

    logits: tuple[tuple[float, ...], ...]
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of suffixes scored."""
        return len(self.logits)


@runtime_checkable
class Engine(Protocol):
    """Score N suffixes over a shared prefix in as few forward passes as the backend allows."""

    @property
    def name(self) -> str:
        """Short backend identifier, e.g. ``"hf"`` or ``"mock"``. Recorded in results."""
        ...

    def score(
        self,
        prefix: str,
        suffixes: Sequence[str],
        labels: Sequence[Sequence[str]],
    ) -> EngineOutput:
        """Return the answer-position logits of every label for every suffix.

        Args:
            prefix: The shared text, rendered by :func:`s1decide.prompt.render`. Backends
                that can, prefill it once.
            suffixes: One per question. Each ends exactly at the answer position.
            labels: ``labels[i]`` are the single-token answer labels for ``suffixes[i]``.

        Returns:
            An :class:`EngineOutput` with ``len(logits) == len(suffixes)`` and
            ``len(logits[i]) == len(labels[i])``.
        """
        ...


def validate_output(
    output: EngineOutput, suffixes: Sequence[str], labels: Sequence[Sequence[str]]
) -> None:
    """Check that an engine answered the request it was given.

    Args:
        output: What the engine returned.
        suffixes: The suffixes that were requested.
        labels: The label sets that were requested.

    Raises:
        EngineError: If the row count or any row's label count is wrong, or a logit is not
            a finite number.
    """
    if len(output.logits) != len(suffixes):
        raise EngineError(f"engine returned {len(output.logits)} rows for {len(suffixes)} suffixes")
    for i, (row, row_labels) in enumerate(zip(output.logits, labels)):
        if len(row) != len(row_labels):
            raise EngineError(
                f"engine returned {len(row)} logits for {len(row_labels)} labels on suffix {i}"
            )
        for value in row:
            if (
                not isinstance(value, int | float)
                or value != value
                or value in (float("inf"), float("-inf"))
            ):
                raise EngineError(f"non-finite logit {value!r} on suffix {i}")
