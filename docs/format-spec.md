# Prompt format specification — version 0.1

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
|---|---|
| `head` | `'<\|im_start\|>system\n'` |
| `mid` | `'<\|im_end\|>\n<\|im_start\|>user\n'` |
| `tail` | `'<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n\n</think>\n\n'` |

Template name: `qwen3.8-chatml-nothink`.

### A note on `<think>`

Qwen3.8 defaults to `reasoning_effort=xhigh` and opens an unterminated `<think>` block in
the generation prompt. `enable_thinking=False` removes the reasoning-effort system
preamble and emits an **empty, immediately closed** block instead:

```text
<|im_start|>assistant\n<think>\n\n</think>\n\n
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
| `Choice` | `A, B, C, D, ...` up to `Z` | caller's option order |
| `Score` | same letters | lowest level first |
| `Noul` | `no, yes` | index 1 is the true case |

Letters only, and the ceiling is **26 options**, for a measured
reason. On the `Qwen/Qwen3.8-27B` tokenizer every letter `A`-`Z` is a single token both
bare and with a leading space, but digits are not: `" 0"` encodes to `[220, 15]`. Since
the answer position may follow either a newline or a space depending on the template,
labels must be single tokens in both forms. `tests/test_tokens.py` asserts this against
the real tokenizer and will fail if a future base model breaks it.

Questions may declare up to 255 options — the schema limit in `primitives.py` — but
rendering more than 26 raises `NotImplementedError` rather than
truncating, until the two-stage high-cardinality path from ADR 0001 exists.

## 4. State block

Free text is inserted verbatim (stripped). A mapping is rendered as JSON with **sorted
keys**, indent 2, `ensure_ascii=False`, so two callers who pass equal content get
byte-identical prompts and therefore share a prefill cache.

## 5. Rendered examples

All examples below use this state:

```text
Subject: charged twice
From: dana@example.com

I was charged twice for my subscription this month. I have been a customer for three years and this has never happened before. Please fix this today.
```

### 5.1 `Choice`

```text
<|im_start|>system
You are a decision engine. You read a piece of state, then answer one question about it.
Answer with exactly one label from the allowed set. Output the label and nothing else: no punctuation, no explanation, no restatement of the question.<|im_end|>
<|im_start|>user
### State
Subject: charged twice
From: dana@example.com

I was charged twice for my subscription this month. I have been a customer for three years and this has never happened before. Please fix this today.

### Question
What is the customer's tone?

### Options
A. calm
B. frustrated
C. angry

### Answer
Reply with exactly one of: A, B, C
<|im_end|>
<|im_start|>assistant
<think>

</think>


```

Answer position masked to: `A, B, C`.

### 5.2 `Score`

Only the suffix is shown; the prefix is identical to 5.1.

```text
### Question
How urgent is this ticket?

### Levels (lowest to highest)
A. can wait
B. this week
C. today

### Answer
Reply with exactly one of: A, B, C
<|im_end|>
<|im_start|>assistant
<think>

</think>


```

Answer position masked to: `A, B, C`. The result reports the full
distribution, the argmax level, and the probability-weighted `expected_level`, because a
bimodal distribution over an ordinal rubric has an argmax that hides the disagreement.

### 5.3 `Noul`

Only the suffix is shown.

```text
### Question
This ticket is about billing.

### Answer
Is the statement above true? Reply with exactly one of: yes, no
<|im_end|>
<|im_start|>assistant
<think>

</think>


```

Answer position masked to: `no, yes`. `Result.noul` is P(`yes`).

### 5.4 A four-question mixed call

One prefix, four suffixes, one forward pass.

**Shared prefix** (494 chars, prefilled once):

```text
<|im_start|>system
You are a decision engine. You read a piece of state, then answer one question about it.
Answer with exactly one label from the allowed set. Output the label and nothing else: no punctuation, no explanation, no restatement of the question.<|im_end|>
<|im_start|>user
### State
Subject: charged twice
From: dana@example.com

I was charged twice for my subscription this month. I have been a customer for three years and this has never happened before. Please fix this today.


```

**Suffix 0 — `tone`** (185 chars, masked to `A, B, C`):

```text
### Question
What is the customer's tone?

### Options
A. calm
B. frustrated
C. angry

### Answer
Reply with exactly one of: A, B, C
<|im_end|>
<|im_start|>assistant
<think>

</think>


```

**Suffix 1 — `urgency`** (205 chars, masked to `A, B, C`):

```text
### Question
How urgent is this ticket?

### Levels (lowest to highest)
A. can wait
B. this week
C. today

### Answer
Reply with exactly one of: A, B, C
<|im_end|>
<|im_start|>assistant
<think>

</think>


```

**Suffix 2 — `billing`** (171 chars, masked to `no, yes`):

```text
### Question
This ticket is about billing.

### Answer
Is the statement above true? Reply with exactly one of: yes, no
<|im_end|>
<|im_start|>assistant
<think>

</think>


```

**Suffix 3 — `refund_requested`** (178 chars, masked to `no, yes`):

```text
### Question
The customer is asking for a refund.

### Answer
Is the statement above true? Reply with exactly one of: yes, no
<|im_end|>
<|im_start|>assistant
<think>

</think>


```

The prefix is 40% of the total rendered characters for this call, and it is
prefilled once no matter how many questions are asked. That ratio is the whole latency
argument, and `uv run task bench` measures whether it holds in practice.

## 6. Versioning

| Version | Date | Change |
|---|---|---|
| 0.1 | 2026-09-17 | Initial format. |
