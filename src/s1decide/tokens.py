"""Mapping options to single-token labels, and checking that they really are single tokens.

The whole approach in ADR 0001 rests on one property: every label the model may answer
with must occupy exactly one token, so that a single logit vector at a single position
carries the full distribution. That property belongs to the *tokenizer*, not to us, so
it is measured, not assumed — see :func:`check_single_token` and the tests.

Measured on ``Qwen/Qwen3.8-27B`` (2026-09-17): ``A``-``Z`` are single tokens both bare
and with a leading space, but **digits are not** — ``" 0"`` encodes to ``[220, 15]``.
That is why the label alphabet is letters only, and why the single-token ceiling is 26.
"""

from __future__ import annotations

import string
from collections.abc import Sequence
from typing import Any, Protocol

from s1decide.primitives import NOUL_OPTIONS, Question

__all__ = [
    "LETTER_LABELS",
    "MAX_SINGLE_TOKEN_OPTIONS",
    "NOUL_LABELS",
    "allowed_token_ids",
    "check_single_token",
    "labels_for_count",
    "labels_for_question",
    "token_id",
]


class _Tokenizer(Protocol):
    """The slice of a Hugging Face tokenizer this module needs."""

    def encode(self, text: str, add_special_tokens: bool = ..., **kwargs: Any) -> list[int]: ...


#: Answer labels for Choice and Score, in index order. Letters only: digits are not
#: single tokens with a leading space on the Qwen3.8 tokenizer.
LETTER_LABELS: tuple[str, ...] = tuple(string.ascii_uppercase)

#: Answer labels for Noul, in index order. Index 1 is the true case.
NOUL_LABELS: tuple[str, ...] = NOUL_OPTIONS

#: How many options one question can address with single-token labels. Above this a
#: two-stage scheme is required (ADR 0001); it is not implemented yet.
MAX_SINGLE_TOKEN_OPTIONS = len(LETTER_LABELS)


def labels_for_count(count: int) -> tuple[str, ...]:
    """Return the answer labels for a question with ``count`` options.

    Args:
        count: Number of options, at least 2.

    Returns:
        The first ``count`` letter labels, in index order.

    Raises:
        ValueError: If ``count`` is below 2.
        NotImplementedError: If ``count`` exceeds :data:`MAX_SINGLE_TOKEN_OPTIONS`.
            High-cardinality questions need the two-stage path from ADR 0001, which
            does not exist yet; failing here is deliberate, so that no caller can
            silently receive a truncated option set.
    """
    if count < 2:
        raise ValueError(f"a question needs at least 2 options, got {count}")
    if count > MAX_SINGLE_TOKEN_OPTIONS:
        raise NotImplementedError(
            f"{count} options exceeds the {MAX_SINGLE_TOKEN_OPTIONS} single-token labels "
            "available; the two-stage high-cardinality path from ADR 0001 is not implemented yet"
        )
    return LETTER_LABELS[:count]


def labels_for_question(question: Question) -> tuple[str, ...]:
    """Return the answer labels for a question.

    Noul answers with ``no``/``yes`` directly; Choice and Score answer with letters that
    stand for their options.

    Args:
        question: The question to label.

    Returns:
        The answer labels, aligned index-for-index with ``question.labels``.

    Raises:
        NotImplementedError: If the question has more options than there are
            single-token labels.
    """
    if question.qtype == "noul":
        return NOUL_LABELS
    return labels_for_count(len(question.labels))


def token_id(tokenizer: _Tokenizer, label: str, *, leading_space: bool = False) -> int:
    """Return the single token id for a label.

    Args:
        tokenizer: A Hugging Face tokenizer.
        label: The answer label, e.g. ``"A"`` or ``"yes"``.
        leading_space: Encode ``" " + label`` instead of ``label``. Useful for
            verifying the property holds in both positions, since a prompt that ends
            with a newline and one that ends with a space tokenize differently.

    Returns:
        The token id.

    Raises:
        ValueError: If the label does not encode to exactly one token.
    """
    text = f" {label}" if leading_space else label
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"label {text!r} is not a single token, encodes to {ids}")
    return ids[0]


def check_single_token(
    tokenizer: _Tokenizer, labels: Sequence[str]
) -> dict[str, dict[str, list[int]]]:
    """Find labels that are not single tokens.

    Checks each label bare and with a leading space, because which of the two applies
    depends on the exact byte the prompt ends on.

    Args:
        tokenizer: A Hugging Face tokenizer.
        labels: The labels to check.

    Returns:
        A mapping of offending label to its encodings, keyed ``"bare"`` and
        ``"leading_space"``. Empty when every label is a single token both ways.
    """
    offenders: dict[str, dict[str, list[int]]] = {}
    for label in labels:
        bare = tokenizer.encode(label, add_special_tokens=False)
        spaced = tokenizer.encode(f" {label}", add_special_tokens=False)
        if len(bare) != 1 or len(spaced) != 1:
            offenders[label] = {"bare": list(bare), "leading_space": list(spaced)}
    return offenders


def allowed_token_ids(
    tokenizer: _Tokenizer, labels: Sequence[str], *, leading_space: bool = False
) -> tuple[int, ...]:
    """Return the token ids a question's answer position may take.

    This is the mask: the engine keeps exactly these logits and softmaxes over them, so
    an off-schema answer is impossible by construction rather than by parsing.

    Args:
        tokenizer: A Hugging Face tokenizer.
        labels: The answer labels, in index order.
        leading_space: Encode each label with a leading space.

    Returns:
        Token ids aligned index-for-index with ``labels``.

    Raises:
        ValueError: If any label is not a single token, or if two labels collide on the
            same token id — which would make their probabilities indistinguishable.
    """
    ids = tuple(token_id(tokenizer, label, leading_space=leading_space) for label in labels)
    if len(set(ids)) != len(ids):
        raise ValueError(f"labels {list(labels)} collide on token ids {list(ids)}")
    return ids
