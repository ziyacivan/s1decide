"""Tests for the prompt renderer: determinism, the prefix/suffix contract, reasoning off."""

from __future__ import annotations

import pytest

from s1decide.primitives import Choice, Question
from s1decide.prompt import (
    DEFAULT_TEMPLATE,
    FORMAT_VERSION,
    ChatTemplate,
    RenderedPrompt,
    assert_no_open_thinking,
    render,
    render_state_block,
)


def test_format_version_is_set() -> None:
    assert FORMAT_VERSION == "0.1"


# --- the prefix / suffix contract --------------------------------------------


def test_prefix_is_shared_by_every_question(ticket_state, tone, urgency, billing) -> None:
    """The whole KV-broadcast design rests on this: one prefix, N suffixes."""
    rendered = render(ticket_state, [tone, urgency, billing])
    assert len(rendered) == 3
    for i in range(3):
        assert rendered.full(i).startswith(rendered.prefix)
        assert rendered.full(i) == rendered.prefix + rendered.suffixes[i]


def test_prefix_does_not_depend_on_the_questions(ticket_state, tone, urgency) -> None:
    one = render(ticket_state, [tone])
    two = render(ticket_state, [urgency])
    three = render(ticket_state, [tone, urgency])
    assert one.prefix == two.prefix == three.prefix


def test_suffix_does_not_depend_on_sibling_questions(ticket_state, tone, urgency) -> None:
    """Questions are answered in isolation, so a suffix must not shift when others change."""
    alone = render(ticket_state, [tone])
    together = render(ticket_state, [tone, urgency])
    assert alone.suffixes[0] == together.suffixes[0]


def test_suffix_ends_at_the_answer_position(ticket_state, tone) -> None:
    rendered = render(ticket_state, [tone])
    assert rendered.suffixes[0].endswith(DEFAULT_TEMPLATE.tail)


def test_state_appears_once_in_the_prefix_only(ticket_state, tone, urgency) -> None:
    rendered = render(ticket_state, [tone, urgency])
    assert ticket_state in rendered.prefix
    for suffix in rendered.suffixes:
        assert ticket_state not in suffix


# --- determinism -------------------------------------------------------------


def test_rendering_is_deterministic(ticket_state, tone, urgency, billing) -> None:
    first = render(ticket_state, [tone, urgency, billing])
    second = render(ticket_state, [tone, urgency, billing])
    assert first == second


def test_mapping_state_is_key_order_independent(tone) -> None:
    """Equal content must give byte-identical prompts, so the prefill cache can be reused."""
    a = render({"subject": "billing", "body": "charged twice"}, [tone])
    b = render({"body": "charged twice", "subject": "billing"}, [tone])
    assert a.prefix == b.prefix


def test_mapping_state_renders_as_sorted_json() -> None:
    block = render_state_block({"b": 2, "a": 1})
    assert block.index('"a"') < block.index('"b"')


def test_state_must_be_text_or_a_mapping() -> None:
    with pytest.raises(TypeError, match="must be a string or a mapping"):
        render_state_block(["not", "valid"])  # type: ignore[arg-type]


def test_non_ascii_state_survives_rendering(tone) -> None:
    rendered = render({"note": "ödeme iki kez alındı"}, [tone])
    assert "ödeme iki kez alındı" in rendered.prefix


# --- questions ---------------------------------------------------------------


def test_render_rejects_no_questions(ticket_state) -> None:
    with pytest.raises(ValueError, match="at least one question"):
        render(ticket_state, [])


def test_render_rejects_duplicate_question_names(ticket_state, tone) -> None:
    with pytest.raises(ValueError, match="must be unique"):
        render(ticket_state, [tone, tone])


def test_choice_labels_are_letters(ticket_state, tone) -> None:
    rendered = render(ticket_state, [tone])
    assert rendered.labels == (("A", "B", "C"),)
    assert "A. calm" in rendered.suffixes[0]
    assert "Reply with exactly one of: A, B, C" in rendered.suffixes[0]


def test_score_is_presented_as_ordered(ticket_state, urgency) -> None:
    suffix = render(ticket_state, [urgency]).suffixes[0]
    assert "Levels (lowest to highest)" in suffix
    assert suffix.index("A. can wait") < suffix.index("C. today")


def test_noul_answers_yes_or_no(ticket_state, billing) -> None:
    rendered = render(ticket_state, [billing])
    assert rendered.labels == (("no", "yes"),)
    assert "yes, no" in rendered.suffixes[0]


def test_labels_are_aligned_with_options(ticket_state, tone, urgency, billing) -> None:
    rendered = render(ticket_state, [tone, urgency, billing])
    for question, labels in zip([tone, urgency, billing], rendered.labels):
        assert len(labels) == len(question.labels)


def test_too_many_options_fails_loudly(ticket_state) -> None:
    """27 options is valid as a schema but has no single-token labelling yet."""
    wide = Question(
        name="wide",
        spec=Choice(instructions="q", options=tuple(f"option-{i}" for i in range(27))),
    )
    with pytest.raises(NotImplementedError, match="two-stage"):
        render(ticket_state, [wide])


# --- reasoning off -----------------------------------------------------------


def test_default_template_does_not_end_inside_a_thinking_block(ticket_state, tone) -> None:
    assert_no_open_thinking(render(ticket_state, [tone]).full(0))


def test_assert_no_open_thinking_catches_an_open_block() -> None:
    with pytest.raises(AssertionError, match="unbalanced"):
        assert_no_open_thinking("<|im_start|>assistant\n<think>\n")


def test_assert_no_open_thinking_catches_a_reopened_block() -> None:
    with pytest.raises(AssertionError, match="unbalanced"):
        assert_no_open_thinking("<think></think><think>")


def test_assert_no_open_thinking_accepts_a_closed_empty_block() -> None:
    assert_no_open_thinking("<|im_start|>assistant\n<think>\n\n</think>\n\n")


# --- ChatTemplate ------------------------------------------------------------


def test_from_tokenizer_matches_the_hardcoded_default(tokenizer) -> None:
    """The committed default must stay equal to what the real tokenizer produces."""
    derived = ChatTemplate.from_tokenizer(tokenizer)
    assert (derived.head, derived.mid, derived.tail) == (
        DEFAULT_TEMPLATE.head,
        DEFAULT_TEMPLATE.mid,
        DEFAULT_TEMPLATE.tail,
    )


def test_real_chat_template_does_not_end_inside_a_thinking_block(
    tokenizer, ticket_state, tone
) -> None:
    derived = ChatTemplate.from_tokenizer(tokenizer)
    assert_no_open_thinking(render(ticket_state, [tone], template=derived).full(0))


def test_reasoning_preamble_is_absent_when_thinking_is_disabled(tokenizer) -> None:
    """With reasoning on, Qwen3.8 injects a 'Reasoning effort is set to' system preamble."""
    derived = ChatTemplate.from_tokenizer(tokenizer)
    assert "Reasoning effort" not in derived.head + derived.mid + derived.tail


def test_from_tokenizer_rejects_a_template_it_cannot_split() -> None:
    class Dropping:
        def apply_chat_template(self, messages, **kwargs) -> str:
            return "no sentinels here"

    with pytest.raises(ValueError, match="sentinels"):
        ChatTemplate.from_tokenizer(Dropping())


def test_from_tokenizer_falls_back_when_kwargs_are_rejected() -> None:
    """Templates that do not know `enable_thinking` must still work."""

    class Picky:
        def apply_chat_template(
            self, messages, tokenize=True, add_generation_prompt=False, **kwargs
        ):
            if kwargs:
                raise TypeError("unexpected keyword")
            return f"H{messages[0]['content']}M{messages[1]['content']}T"

    template = ChatTemplate.from_tokenizer(Picky())
    assert (template.head, template.mid, template.tail) == ("H", "M", "T")


def test_rendered_prompt_records_its_provenance(ticket_state, tone) -> None:
    rendered = render(ticket_state, [tone])
    assert isinstance(rendered, RenderedPrompt)
    assert rendered.format_version == FORMAT_VERSION
    assert rendered.template_name == DEFAULT_TEMPLATE.name
    assert rendered.names == ("tone",)


# --- tokenization boundary -------------------------------------------------


def test_prefix_and_suffix_tokenize_independently(tokenizer, tone, urgency, billing) -> None:
    """The engine encodes prefix and suffix separately and concatenates the ids.

    That is only equivalent to encoding the full prompt if no BPE merge crosses the
    boundary. The prefix ends on a "\n\n" token and every suffix starts on "###", so it
    holds; this test keeps it that way when the format changes.
    """
    for state in [
        "I was charged twice.",
        {"subject": "refund", "body": "please refund"},
        "ends with newline\n",
    ]:
        rendered = render(state, [tone, urgency, billing])
        prefix_ids = tokenizer.encode(rendered.prefix, add_special_tokens=False)
        for i, suffix in enumerate(rendered.suffixes):
            separate = prefix_ids + tokenizer.encode(suffix, add_special_tokens=False)
            joint = tokenizer.encode(rendered.full(i), add_special_tokens=False)
            assert separate == joint, (
                f"BPE merge across the prefix/suffix boundary for {rendered.names[i]}"
            )
