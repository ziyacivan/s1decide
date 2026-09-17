"""Generate docs/format-spec.md from the renderer itself.

Every example in the spec is produced by :mod:`s1decide.prompt`, so the document cannot
drift from the code. ``tests/test_format_spec.py`` asserts the committed file is current.

Run with::

    uv run python scripts/render_format_spec.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from s1decide.primitives import Choice, Noul, Question, Score
from s1decide.prompt import (
    DEFAULT_TEMPLATE,
    FORMAT_VERSION,
    needs_two_stage,
    render,
    render_stage1,
    render_stage2,
)
from s1decide.tokens import LETTER_LABELS, MAX_SINGLE_TOKEN_OPTIONS, NOUL_LABELS

OUTPUT = Path("docs/format-spec.md")

STATE = (
    "Subject: charged twice\n"
    "From: dana@example.com\n\n"
    "I was charged twice for my subscription this month. I have been a customer for "
    "three years and this has never happened before. Please fix this today."
)

TONE = Question(
    name="tone",
    spec=Choice(
        instructions="What is the customer's tone?",
        options=("calm", "frustrated", "angry"),
    ),
)
URGENCY = Question(
    name="urgency",
    spec=Score(
        instructions="How urgent is this ticket?",
        levels=("can wait", "this week", "today"),
    ),
)
BILLING = Question(
    name="billing",
    spec=Noul(instructions="This ticket is about billing."),
)
REFUND = Question(
    name="refund_requested",
    spec=Noul(instructions="The customer is asking for a refund."),
)


def fence(text: str) -> str:
    """Wrap text in a fenced block, making trailing whitespace and newlines visible.

    Args:
        text: The rendered text.

    Returns:
        A fenced code block.
    """
    return "```text\n" + text + "\n```"


def build() -> str:
    """Render the whole document.

    Returns:
        The full Markdown source of the format spec.
    """
    wide = Question(
        name="intent",
        spec=Choice(
            instructions="Which banking intent does this message have?",
            options=tuple(f"intent {i:02d}" for i in range(40)),
        ),
    )
    assert needs_two_stage(wide)
    stage1 = render_stage1(STATE, wide)
    stage2 = render_stage2(STATE, wide, [7, 2, 31])

    single = render(STATE, [TONE])
    score = render(STATE, [URGENCY])
    noul = render(STATE, [BILLING])
    mixed = render(STATE, [TONE, URGENCY, BILLING, REFUND])

    parts: list[str] = []
    add = parts.append

    add(f"""# Prompt format specification — version {FORMAT_VERSION}

<!-- GENERATED FILE — do not edit by hand.
     Produced by scripts/render_format_spec.py from s1decide.prompt.
     Regenerate with:  uv run python scripts/render_format_spec.py
     tests/test_format_spec.py fails if this file is stale. -->

This document is the contract between the data pipeline, the training scripts and the
inference engines. A model trained against one version of this format will quietly lose
accuracy under another, so the version is part of the released artefact.

**Changing any rendered byte requires** bumping `FORMAT_VERSION` in
`src/s1decide/prompt.py`, regenerating this file, and re-running `uv run task smoke`.

## 1. Shape

A call is one `state` and N questions. It renders to **one prefix and N suffixes**:

```
prefix   = chat header + system prompt + state block      (prefilled once)
suffix_i = question block_i + generation prompt           (one batch row each)
full_i   = prefix + suffix_i
```

The prefix is byte-identical across every question in a call, and independent of which
questions are asked. That is what makes the KV-cache broadcast in locked decision 2
valid: prefill the prefix once, expand the cache to one row per question, run all N
suffixes in a single forward pass. Each suffix ends **exactly** at the answer position —
the next token is the answer, and nothing is ever decoded beyond it.

Questions are answered in isolation. A suffix never depends on its siblings, so adding a
question cannot change another question's answer.

## 2. Chat template

Rendered with reasoning **disabled**. The default below is Qwen3.8 ChatML as measured on
2026-09-17; `ChatTemplate.from_tokenizer()` derives the same three pieces from any
model's own template by probing it with sentinels.

| Piece | Value |
|---|---|""")

    for label, value in (
        ("`head`", DEFAULT_TEMPLATE.head),
        ("`mid`", DEFAULT_TEMPLATE.mid),
        ("`tail`", DEFAULT_TEMPLATE.tail),
    ):
        # Escape pipes: the chat control tokens contain `|`, which would split the cell.
        add(f"| {label} | `{repr(value).replace('|', chr(92) + '|')}` |")

    add(f"""
Template name: `{DEFAULT_TEMPLATE.name}`.

### A note on `<think>`

Qwen3.8 defaults to `reasoning_effort=xhigh` and opens an unterminated `<think>` block in
the generation prompt. `enable_thinking=False` removes the reasoning-effort system
preamble and emits an **empty, immediately closed** block instead:

```text
<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n
```

So the literal string `<think>` is still present, and it is kept deliberately: that is the
form the model was trained on for non-thinking mode. `reasoning_effort="none"` is not a
valid value — the template rejects it, accepting only `xhigh`, `medium` and `low`.

What must never happen is the prompt *ending* inside a thinking block, because then the
first token is reasoning rather than an answer and the logits we read are meaningless.
`prompt.assert_no_open_thinking()` enforces exactly that, and the tests apply it to both
the committed default and the live tokenizer.

## 3. Answer labels

| Primitive | Labels | Index meaning |
|---|---|---|
| `Choice` | `{", ".join(LETTER_LABELS[:4])}, ...` up to `{LETTER_LABELS[MAX_SINGLE_TOKEN_OPTIONS - 1]}` | caller's option order |
| `Score` | same letters | lowest level first |
| `Noul` | `{", ".join(NOUL_LABELS)}` | index 1 is the true case |

Letters only, and the ceiling is **{MAX_SINGLE_TOKEN_OPTIONS} options**, for a measured
reason. On the `Qwen/Qwen3.8-27B` tokenizer every letter `A`-`Z` is a single token both
bare and with a leading space, but digits are not: `" 0"` encodes to `[220, 15]`. Since
the answer position may follow either a newline or a space depending on the template,
labels must be single tokens in both forms. `tests/test_tokens.py` asserts this against
the real tokenizer and will fail if a future base model breaks it.

Questions may declare up to 255 options — the schema limit in `primitives.py` — but
rendering more than {MAX_SINGLE_TOKEN_OPTIONS} raises `NotImplementedError` rather than
truncating, until the two-stage high-cardinality path from ADR 0001 exists.

## 4. State block

Free text is inserted verbatim (stripped). A mapping is rendered as JSON with **sorted
keys**, indent 2, `ensure_ascii=False`, so two callers who pass equal content get
byte-identical prompts and therefore share a prefill cache.

## 5. Rendered examples

All examples below use this state:

{fence(STATE)}

### 5.1 `Choice`

{fence(single.full(0))}

Answer position masked to: `{", ".join(single.labels[0])}`.

### 5.2 `Score`

Only the suffix is shown; the prefix is identical to 5.1.

{fence(score.suffixes[0])}

Answer position masked to: `{", ".join(score.labels[0])}`. The result reports the full
distribution, the argmax level, and the probability-weighted `expected_level`, because a
bimodal distribution over an ordinal rubric has an argmax that hides the disagreement.

### 5.3 `Noul`

Only the suffix is shown.

{fence(noul.suffixes[0])}

Answer position masked to: `{", ".join(noul.labels[0])}`. `Result.noul` is P(`yes`).

### 5.4 A four-question mixed call

One prefix, four suffixes, one forward pass.

**Shared prefix** ({len(mixed.prefix)} chars, prefilled once):

{fence(mixed.prefix)}""")

    for i, name in enumerate(mixed.names):
        add(
            f"\n**Suffix {i} — `{name}`** "
            f"({len(mixed.suffixes[i])} chars, masked to `{', '.join(mixed.labels[i])}`):\n\n"
            + fence(mixed.suffixes[i])
        )

    prefix_share = len(mixed.prefix) / (len(mixed.prefix) + sum(len(s) for s in mixed.suffixes))
    add(f"""
The prefix is {prefix_share:.0%} of the total rendered characters for this call, and it is
prefilled once no matter how many questions are asked. That ratio is the whole latency
argument, and `uv run task bench` measures whether it holds in practice.

## 6. Two-stage rendering for high-cardinality questions

A question with more options than there are single-token labels ({MAX_SINGLE_TOKEN_OPTIONS})
cannot be asked in one pass: there is no letter left to stand for option 27.
`needs_two_stage(question)` decides, and ADR 0001's scheme handles it.

**Stage 1 — score every option independently.** One suffix per option, all sharing the same
state prefix, so the whole stage is a single broadcast call. Each asks a yes/no question, which
needs only the two fixed labels and so has no ceiling.

{fence(stage1.suffixes[0])}

There are {len(stage1)} suffixes like this one, named `{stage1.names[0]}` through
`{stage1.names[-1]}`, each masked to `no, yes`.

**These are not a distribution.** The options are scored independently, so their P(yes) values do
not sum to one. Stage 1 produces a *shortlist*, not an answer.

**Stage 2 — one Choice over the survivors**, which is an ordinary Choice and is calibrated the
same way every other Choice is:

{fence(stage2.suffixes[0])}

Masked to `{", ".join(stage2.labels[0])}`. Results map back through the candidate indices that
were passed in: stage 2's label `A` means `question.labels[candidates[0]]`, never the raw letter.

## 7. Versioning

| Version | Date | Change |
|---|---|---|
| 0.1 | 2026-09-17 | Initial format. |
| {FORMAT_VERSION} | 2026-09-17 | ADR 0003 option B: dropped the `### Question` / `### Options` headers and the verbose `### Answer` block in favour of a one-line answer cue. ADR 0001: added two-stage rendering for questions above {MAX_SINGLE_TOKEN_OPTIONS} options. |
""")

    return "\n".join(parts).rstrip() + "\n"


def main() -> int:
    """Write the spec to disk.

    Returns:
        Process exit code.
    """
    root = Path(__file__).resolve().parent.parent
    target = root / OUTPUT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build(), encoding="utf-8", newline="\n")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
