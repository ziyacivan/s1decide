# ADR 0003 — Retire the "64 questions < 2x one question" target

- **Status:** **accepted** (2026-09-17)
- **Date:** 2026-09-17, Phase 0 Step 6
- **Deciders:** project owner (human), inference-engineer role

## The finding, stated plainly

**KV broadcast amortises the state, not the questions.** Prefilling once and broadcasting the
cache removes the cost of re-reading the state; it does nothing about the cost of reading the
questions, which still traverse every layer. Latency is flat in question count only when

```
prefix_tokens  >~  61 x tokens_per_question
```

Below that threshold, latency is dominated by the questions and grows close to linearly in
their number. The original "< 2x" rule encoded this threshold accidentally, so it measured the
benchmark's state size at least as much as the engine. It has been replaced (see Decision).
- **Evidence:** `results/20260917-120323-qwen38-27b-unsloth-bnb-4bit-nf4-bf16-latency/`
  (`latency.json` is the source of truth; `LATENCY.md` and `latency.png` are generated from it).
  Measured twice, half an hour apart, agreeing to within 0.3%.

## Context

`CLAUDE.md`'s definition of done says: *"Latency at 1 / 4 / 16 / 64 questions on the 3090
measured and plotted; 64-question call < 2x the 1-question call."* Step 6 measured it on the
target model at the deployment quantization, 20 timed runs per point after 2 warm-ups, on a
fixed ~1,500-token state.

**The result is 4.31x. The target is missed, and it is missed for a reason that no amount of
engineering removes.**

## What the measurement says

Numbers below are in `latency.json`; see `LATENCY.md` for the full table.

The suffix phase fits `a * passes + b * suffix_tokens` with **R² = 0.999**:

- **a ≈ 159 ms per forward pass** — weight streaming and launch overhead.
- **b ≈ 0.98 ms per suffix token** — the marginal cost of question text.

And the decisive comparison: **a prefix token costs ≈ 0.87 ms; a suffix token costs ≈ 0.98 ms.**
They are the same price. Batching the broadcast rows buys no per-token discount, because a
27B forward pass on a 3090 is already compute-bound at this size.

That is the whole story. "Prefill once, answer everything in one pass" saves the *prefill*, and
nothing else. The question text still has to go through all 64 layers, and 64 questions of ~58
tokens is **3,710 suffix tokens against a 1,617-token prefix** — the questions are 2.3x the
state. Under those conditions latency cannot be flat in the number of questions; it is
dominated by the questions.

### Why it cannot be engineered away at this state size

Writing the target out with the fitted model, for one pass and `s` tokens per question:

```
prefill + a + 64·b·s  <  2·(prefill + a + b·s)
        =>  62·b·s  <  prefill + a
        =>  s  <  (prefill + a) / 61     i.e.  prefix_tokens  >~  61 · s
```

**The 2x target at 64 questions requires the shared prefix to be about 61x the per-question
suffix length.** At the measured 58 tokens per question that needs a ~3,600-token prefix; we
have 1,617.

Projections from the fitted model, all at the current 1,617-token prefix:

| Change | suffix tok/question | passes | projected 64-question | ratio |
|---|---|---|---|---|
| (measured today) | 58 | 10 | 6,674 ms | **4.31x** |
| Trim our format boilerplate | ~40 | 10 | ~5,530 ms | ~3.6x |
| Fit all 64 rows in one pass | 58 | 1 | ~5,250 ms | ~3.4x |
| Both | ~40 | 1 | ~4,100 ms | ~2.7x |
| Both, Noul-only workload | ~25 | 1 | ~3,170 ms | ~2.05x |

**Even doing everything, a mixed workload does not reach 2x at this state size.**

### What the boilerplate actually costs (measured)

Per-question suffix tokens, broken down against the Qwen3.8 tokenizer:

| primitive | suffix | chat tail | `### Answer` block | `### Question` | instruction | options | boilerplate |
|---|---|---|---|---|---|---|---|
| Noul | 44 | 9 | 19 | 3 | 12 | 1 | **70%** |
| Choice (4) | 62 | 9 | 17 | 3 | 11 | 22 | **47%** |
| Score (4) | 66 | 9 | 17 | 3 | 10 | 27 | **44%** |

Mean 57.3 tokens, **52% of it format boilerplate**. The 9-token chat tail is the model's
(`<|im_end|>…<think>\n\n</think>`); the ~20 tokens of `### Question` header plus
`### Answer\nReply with exactly one of: …` are ours and could be cut to ~3.

## The result that is not in the target

The target compares 64 questions to 1 question. That is not the comparison a user faces. A user
chooses between **asking 64 questions in one call** and **asking them in 64 calls**:

| questions | one bundled call | one call each | speedup | ms per question |
|---|---|---|---|---|
| 4 | 1,867 ms | 6,356 ms | **3.40x** | 467 |
| 16 | 2,864 ms | 25,669 ms | **8.96x** | 179 |
| 64 | 6,674 ms | 102,878 ms | **15.41x** | **104** |

The KV-broadcast design works. It turns 103 seconds into 6.7, and per-question latency from
1,548 ms to 104 ms. The 2x rule is simply the wrong way to express that.

## Options

**A — Restate the target as a curve plus the crossover rule.** Keep the measured curve and the
bundling speedup as the published latency claim; replace the "< 2x" line in the definition of
done with the condition it actually encodes (`prefix_tokens ≳ 61 × tokens_per_question`, or a
stated state size at which 2x holds). Costs nothing, claims nothing false.

**B — Trim the format boilerplate** (`FORMAT_VERSION` 0.1 → 0.2). Drops ~18 tokens per question,
worth ~1.1 s at 64 questions. Requires re-running the Step 5 baseline to confirm accuracy and
calibration do not move, since it changes every prompt the model sees.

**C — Fit more broadcast rows per pass.** Today VRAM caps it at 7 rows (~250 MB of cache per row
at this state length, of which ~151 MB is the fp32 gated-deltanet recurrent state, which is
*independent of sequence length*). Storing that state in bf16, or freeing VRAM by quantizing the
GDN input projections the checkpoint leaves in 16-bit, would roughly double rows per pass. Worth
~0.8–1.4 s at 64 questions, and it is a pure engineering win with no prompt change — but it
needs its own numerical validation against the Step 4 contract test.

**D — Chase 2x anyway** by specifying a longer benchmark state (~3,600 tokens). The number would
turn green without anything improving. Rejected: that is measuring to the target rather than
targeting the measurement.

## Decision

**A now, C next, B in Phase 1. D rejected.**

- **A** — adopted. `CLAUDE.md`'s definition of done no longer contains the "< 2x" rule. It is
  replaced by four targets, all produced by `eval/latency_bench.py` and written to
  `results/<run_id>/latency.json`:

  | target | threshold | status at adoption |
  |---|---|---|
  | bundled vs N separate calls, 16 questions | at least 8x | met |
  | bundled vs N separate calls, 64 questions | at least 12x | met |
  | marginal cost per question at 64, `(median(64) - median(1)) / 63` | at most 150 ms | met |
  | `format_overhead_controllable` — overhead excluding the model's chat tail | at most 25% | **met at format 0.2** (21.95%, from ~41% at 0.1) |
  | `format_overhead` — all-in, including the chat tail | reported, not a bar | 43.9% at 0.2, from 51.7% at 0.1 |

  **Amended 2026-09-17:** the target applies to the *controllable* figure. The all-in number
  cannot reach 25% by any change of ours: a zero-overhead Noul suffix is ~12 instruction
  tokens plus the 9-token chat tail, so the tail alone is ~43% of the minimum possible
  suffix. Removing it would mean abandoning the `enable_thinking=False` form the model was
  trained on. Both figures are reported in every bench output and in `LATENCY.md`, with the
  difference named, so nothing is hidden by the change.

  The fitted cost model (`a * passes + b * suffix_tokens`, with R²) is re-fitted and committed
  per release, so a regression in either term is visible rather than inferred.

- **C** — next, owned by `inference-engineer`. Raise rows-per-pass above the current 7 by
  reducing per-row cache cost (bf16 gated-deltanet recurrent state, and/or freeing VRAM by
  quantizing the GDN input projections the checkpoint leaves in 16-bit).
  **Conditions on any such change, both required:**
  1. the `N-row broadcast == N independent runs` contract test stays green
     (`tests/test_engine_gpu_27b.py`, and the CPU equivalent), and
  2. **no ECE regression** on the Step 5 eval — re-run `uv run task eval` and compare
     `results/<run_id>/metrics.json` against the committed zero-shot baseline.

  A latency win that moves calibration is not a win; the project's differentiator is the
  calibration, not the milliseconds.

- **B** — **done** (2026-09-17, Phase 1 Step 2a). `FORMAT_VERSION` 0.2 dropped the
  `### Question` / `### Options` headers and the verbose `### Answer` block for a one-line
  answer cue. Mean suffix tokens 57.3 → 41.0; controllable overhead ~41% → **21.95%**
  (target met); all-in 51.7% → 43.9%. Accuracy and calibration under 0.2 are re-measured once
  the Phase 1 data pipeline lands, and the 0.1 → 0.2 delta is itself reported.

- **D** — rejected. Specifying a longer benchmark state would turn the number green without
  anything improving.

## Consequences

- `CLAUDE.md`'s definition of done and the `inference-engineer` checklist in `AGENTS.md` have
  both been updated; the "< 2x" rule appears nowhere as a live requirement. `latency.json`
  still reports it under `retired_target` so the historical number stays traceable.
- Any published latency claim must state the state size and the question mix. "Latency barely
  changes with more questions" is true only when the state dominates; we say where the
  crossover is rather than repeat a competitor's framing.
- The crossover rule has a second consequence worth chasing: it suggests why a competitor can
  claim flat latency when we cannot. See
  `docs/research/latency-amortisation-2026-09-17.md`.
- Options B and C each get their own measurement before adoption; neither is assumed.
