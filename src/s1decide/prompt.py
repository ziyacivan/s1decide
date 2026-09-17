"""The canonical prompt format: one shared prefix, one suffix per question.

The split is the contract the whole engine rests on. Everything that does not depend on
the question — chat header, system prompt, the state — goes in the **prefix**, which is
prefilled exactly once. Everything that does goes in a **suffix**, which ends precisely
at the position whose logits are the answer. One prefill, N one-row suffixes, one
forward pass (locked decision 2 in CLAUDE.md).

The format is versioned. Changing any rendered byte means bumping
:data:`FORMAT_VERSION`, updating ``docs/format-spec.md`` and re-running the smoke test,
because a model trained on one format will quietly lose accuracy on another.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from s1decide.primitives import Question
from s1decide.tokens import labels_for_question

__all__ = [
    "DEFAULT_TEMPLATE",
    "FORMAT_VERSION",
    "SYSTEM_PROMPT",
    "ChatTemplate",
    "RenderedPrompt",
    "assert_no_open_thinking",
    "render",
    "render_question_block",
    "render_state_block",
]

#: Version of the rendered format. Bump on any change to the rendered bytes.
FORMAT_VERSION = "0.1"

#: Instruction given once, in the system turn, and shared by every question in a call.
SYSTEM_PROMPT = (
    "You are a decision engine. You read a piece of state, then answer one question "
    "about it.\n"
    "Answer with exactly one label from the allowed set. Output the label and nothing "
    "else: no punctuation, no explanation, no restatement of the question."
)


@dataclass(frozen=True)
class ChatTemplate:
    """The three literal pieces of a chat template that surround our content.

    A rendered two-turn chat prompt is always ``head + system + mid + user + tail``.
    Splitting it this way lets us build one prefix and many suffixes without re-running
    the template, and keeps the renderer independent of any particular model.

    Attributes:
        head: Everything before the system message content.
        mid: Everything between the system content and the user content.
        tail: Everything after the user content, including the generation prompt. The
            answer token is whatever comes immediately after this.
        name: Identifier for diagnostics and for the format spec.
    """

    head: str
    mid: str
    tail: str
    name: str = "chatml"

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: Any,
        *,
        template_kwargs: Mapping[str, Any] | None = None,
        name: str | None = None,
    ) -> ChatTemplate:
        """Derive a template by probing a tokenizer's own chat template.

        Renders a two-turn conversation with sentinel contents and splits the result
        around them, so the model's real control tokens are used rather than ours.

        The default ``template_kwargs`` disable reasoning. On Qwen3.8 that is
        ``enable_thinking=False``: it drops the reasoning-effort preamble and closes the
        thinking block immediately, so the very next token is the answer. Note that the
        *string* ``<think>`` still appears — as an empty, closed ``<think>\\n\\n</think>``
        block. What matters, and what :func:`assert_no_open_thinking` checks, is that the
        prompt does not end inside one.

        Args:
            tokenizer: A tokenizer with ``apply_chat_template``.
            template_kwargs: Extra keyword arguments for ``apply_chat_template``.
                Defaults to ``{"enable_thinking": False}``; if the template rejects
                them, they are dropped and the render is retried bare.
            name: Identifier to record; defaults to the tokenizer's class name.

        Returns:
            The derived template.

        Raises:
            ValueError: If the rendered prompt does not contain both sentinels exactly
                once, which means the template is shaped in a way this split cannot
                represent.
        """
        sentinel_system = "\x00S1DECIDE_SYSTEM\x00"
        sentinel_user = "\x00S1DECIDE_USER\x00"
        messages = [
            {"role": "system", "content": sentinel_system},
            {"role": "user", "content": sentinel_user},
        ]
        kwargs = dict({"enable_thinking": False} if template_kwargs is None else template_kwargs)
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kwargs
            )
        except Exception:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        if text.count(sentinel_system) != 1 or text.count(sentinel_user) != 1:
            raise ValueError(
                "chat template did not reproduce both sentinels exactly once; "
                "cannot split it into head/mid/tail"
            )
        head, rest = text.split(sentinel_system, 1)
        mid, tail = rest.split(sentinel_user, 1)
        return cls(head=head, mid=mid, tail=tail, name=name or type(tokenizer).__name__)


#: Qwen3.8 ChatML with reasoning disabled, as measured on 2026-09-17. Used when no
#: tokenizer is available (CPU tests, the mock engine, the format spec).
DEFAULT_TEMPLATE = ChatTemplate(
    head="<|im_start|>system\n",
    mid="<|im_end|>\n<|im_start|>user\n",
    tail="<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    name="qwen3.8-chatml-nothink",
)


@dataclass(frozen=True)
class RenderedPrompt:
    """One prefill prefix and one suffix per question.

    Attributes:
        format_version: The :data:`FORMAT_VERSION` this was rendered with.
        template_name: Which :class:`ChatTemplate` produced it.
        prefix: The shared block: chat header, system prompt and state. Prefilled once.
        suffixes: One per question, in the order the questions were given. Each ends
            exactly at the answer position.
        names: Question names, aligned with ``suffixes``.
        labels: Answer labels per question, aligned with ``suffixes``. These are what
            the engine masks the logits to.
    """

    format_version: str
    template_name: str
    prefix: str
    suffixes: tuple[str, ...]
    names: tuple[str, ...]
    labels: tuple[tuple[str, ...], ...]

    def __len__(self) -> int:
        """Number of questions."""
        return len(self.suffixes)

    def full(self, index: int) -> str:
        """Return the complete single-question prompt for one question.

        This is what an unbatched, un-broadcast run would send. The equality
        ``full(i) == prefix + suffixes[i]`` is what makes KV broadcast valid, and it is
        asserted in the tests.

        Args:
            index: Question index.

        Returns:
            The full prompt text.
        """
        return self.prefix + self.suffixes[index]


def render_state_block(state: str | Mapping[str, Any]) -> str:
    """Render the shared state.

    Args:
        state: Free text, or a JSON-serialisable mapping. Mappings are rendered as
            JSON with **sorted keys**, so that two callers who pass equal content get
            byte-identical prompts (and therefore the same prefill cache), regardless of
            the order they happened to build the dict in.

    Returns:
        The rendered state block, ending with a blank line.

    Raises:
        TypeError: If ``state`` is neither a string nor a mapping.
    """
    if isinstance(state, str):
        body = state.strip()
    elif isinstance(state, Mapping):
        body = json.dumps(state, indent=2, sort_keys=True, ensure_ascii=False)
    else:
        raise TypeError(f"state must be a string or a mapping, got {type(state).__name__}")
    return f"### State\n{body}\n\n"


def render_question_block(question: Question, labels: Sequence[str]) -> str:
    """Render one question, including its allowed answers.

    Args:
        question: The question to render.
        labels: The answer labels, aligned with ``question.labels``.

    Returns:
        The rendered question block, ending with a newline.

    Raises:
        ValueError: If ``labels`` is not aligned with the question's options.
    """
    if len(labels) != len(question.labels):
        raise ValueError(
            f"question {question.name!r} has {len(question.labels)} options "
            f"but {len(labels)} labels"
        )

    lines = [f"### Question\n{question.instructions.strip()}\n"]

    if question.qtype == "noul":
        lines.append(
            "\n### Answer\nIs the statement above true? Reply with exactly one of: yes, no\n"
        )
        return "".join(lines)

    heading = "Levels (lowest to highest)" if question.qtype == "score" else "Options"
    lines.append(f"\n### {heading}\n")
    lines.extend(f"{label}. {option}\n" for label, option in zip(labels, question.labels))
    lines.append(f"\n### Answer\nReply with exactly one of: {', '.join(labels)}\n")
    return "".join(lines)


def render(
    state: str | Mapping[str, Any],
    questions: Sequence[Question],
    *,
    template: ChatTemplate = DEFAULT_TEMPLATE,
    system_prompt: str = SYSTEM_PROMPT,
) -> RenderedPrompt:
    """Render a decision call into one shared prefix and one suffix per question.

    Args:
        state: The shared state, free text or a mapping.
        questions: The questions to ask about it. Names must be unique.
        template: The chat template to wrap with. Defaults to Qwen3.8 ChatML with
            reasoning disabled; use :meth:`ChatTemplate.from_tokenizer` for any other
            model.
        system_prompt: The shared instruction. Part of the prefix, so it costs one
            prefill regardless of question count.

    Returns:
        The rendered prompt.

    Raises:
        ValueError: If ``questions`` is empty or two questions share a name.
        NotImplementedError: If a question has more options than there are single-token
            labels.
    """
    if not questions:
        raise ValueError("render() needs at least one question")

    names = tuple(q.name for q in questions)
    if len(set(names)) != len(names):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"question names must be unique, repeated: {duplicates}")

    prefix = template.head + system_prompt + template.mid + render_state_block(state)

    labels = tuple(labels_for_question(q) for q in questions)
    suffixes = tuple(
        render_question_block(q, ls) + template.tail for q, ls in zip(questions, labels)
    )

    return RenderedPrompt(
        format_version=FORMAT_VERSION,
        template_name=template.name,
        prefix=prefix,
        suffixes=suffixes,
        names=names,
        labels=labels,
    )


def assert_no_open_thinking(text: str) -> None:
    """Assert that a rendered prompt does not leave a thinking block open.

    CLAUDE.md requires reasoning to be off in every prompt. On Qwen3.8 "off" does not
    mean the string ``<think>`` is absent — ``enable_thinking=False`` emits an empty,
    immediately closed ``<think>\\n\\n</think>`` block, and that is the form the model was
    trained on, so we keep it. What must never happen is the prompt *ending* inside a
    thinking block, because then the first generated token is reasoning, not an answer,
    and the logits we read are meaningless.

    Args:
        text: A rendered prompt.

    Raises:
        AssertionError: If a ``<think>`` is unclosed, or if any ``<think>`` appears after
            the last ``</think>``.
    """
    opens = text.count("<think>")
    closes = text.count("</think>")
    if opens != closes:
        raise AssertionError(f"unbalanced thinking block: {opens} <think> vs {closes} </think>")
    if closes and "<think>" in text.rsplit("</think>", 1)[1]:
        raise AssertionError("prompt ends inside a thinking block")
