"""The public API: ``decide(state, questions, engine=...)``.

Render once, score once, build one validated :class:`Result` per question. Every step is
type-safe by construction — the engine only ever sees single-token labels, and a `Result`
cannot be built with the wrong number of probabilities or a mass that does not sum to one —
so nothing here parses text and nothing can return an option the caller did not offer.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from s1decide.engine.base import Engine, EngineOutput, validate_output
from s1decide.primitives import Question, Result, Spec
from s1decide.prompt import DEFAULT_TEMPLATE, FORMAT_VERSION, ChatTemplate, render

__all__ = ["Response", "decide", "softmax"]


def softmax(logits: Sequence[float], temperature: float = 1.0) -> tuple[float, ...]:
    """Numerically stable softmax with a temperature.

    Args:
        logits: Raw scores.
        temperature: Divides the logits first. ``1.0`` is the model's own distribution;
            calibration fits a value per option-count bucket (`s1decide.calibrate`).

    Returns:
        Probabilities in the same order, renormalised so they sum to one exactly enough for
        :class:`~s1decide.primitives.Result` to accept them.

    Raises:
        ValueError: If ``logits`` is empty or ``temperature`` is not positive.
    """
    if not logits:
        raise ValueError("softmax of an empty sequence")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    scaled = [x / temperature for x in logits]
    peak = max(scaled)
    weights = [math.exp(x - peak) for x in scaled]
    total = math.fsum(weights)
    return tuple(w / total for w in weights)


@dataclass(frozen=True)
class Response:
    """Everything one ``decide`` call returned.

    Attributes:
        results: One per question, in the order the questions were given.
        engine: Which engine produced the logits.
        format_version: The prompt format that was rendered.
        temperature: The temperature the softmax used.
        meta: The engine's diagnostics, untouched.
    """

    results: tuple[Result, ...]
    engine: str
    format_version: str
    temperature: float
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterator[Result]:
        return iter(self.results)

    def __getitem__(self, name: str) -> Result:
        """Look a result up by question name."""
        for result in self.results:
            if result.name == name:
                return result
        raise KeyError(name)

    def _of(self, qtype: str) -> dict[str, Result]:
        return {r.name: r for r in self.results if r.qtype == qtype}

    @property
    def choices(self) -> dict[str, Result]:
        """Results of every ``Choice`` question, by name."""
        return self._of("choice")

    @property
    def scores(self) -> dict[str, Result]:
        """Results of every ``Score`` question, by name."""
        return self._of("score")

    @property
    def nouls(self) -> dict[str, Result]:
        """Results of every ``Noul`` question, by name."""
        return self._of("noul")


def _as_questions(questions: Sequence[Question] | Mapping[str, Spec]) -> list[Question]:
    """Accept either a sequence of ``Question`` or a ``{name: spec}`` mapping."""
    if isinstance(questions, Mapping):
        return [Question(name=name, spec=spec) for name, spec in questions.items()]
    return list(questions)


def results_from_output(
    questions: Sequence[Question], output: EngineOutput, temperature: float
) -> tuple[Result, ...]:
    """Turn engine logits into validated results.

    Args:
        questions: The questions, in the order they were scored.
        output: The engine's logits, one row per question.
        temperature: Softmax temperature.

    Returns:
        One :class:`Result` per question.
    """
    results = []
    for question, row in zip(questions, output.logits):
        results.append(
            Result(
                name=question.name,
                qtype=question.qtype,
                options=question.labels,
                probabilities=softmax(row, temperature),
            )
        )
    return tuple(results)


def decide(
    state: str | Mapping[str, Any],
    questions: Sequence[Question] | Mapping[str, Spec],
    *,
    engine: Engine,
    temperature: float = 1.0,
    template: ChatTemplate = DEFAULT_TEMPLATE,
) -> Response:
    """Answer typed questions about a piece of state in one engine call.

    Args:
        state: Free text or a JSON-serialisable mapping.
        questions: ``Question`` objects, or a ``{name: Choice | Score | Noul}`` mapping.
        engine: Any :class:`~s1decide.engine.base.Engine`.
        temperature: Softmax temperature; ``1.0`` until calibration is fitted.
        template: Chat template to render with. Use
            :meth:`~s1decide.prompt.ChatTemplate.from_tokenizer` for a non-default model.

    Returns:
        A :class:`Response` holding one validated :class:`~s1decide.primitives.Result` per
        question, in input order.

    Raises:
        ValueError: If there are no questions, names repeat, or the temperature is not
            positive.
        EngineError: If the engine's output does not match the request.
        NotImplementedError: If a question has more options than single-token labels.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    question_list = _as_questions(questions)
    rendered = render(state, question_list, template=template)
    output = engine.score(rendered.prefix, rendered.suffixes, rendered.labels)
    validate_output(output, rendered.suffixes, rendered.labels)
    return Response(
        results=results_from_output(question_list, output, temperature),
        engine=engine.name,
        format_version=FORMAT_VERSION,
        temperature=temperature,
        meta=dict(output.meta),
    )
