# ab-buffer-on

<!-- GENERATED from latency.json by eval/summary.py — do not edit by hand. -->

- **Model**: `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` · `nf4-bf16` · NVIDIA GeForce RTX 3090
- **State**: 1554 tokens (1617-token prefill including the system prompt) · median of 8 timed runs after 2 warm-ups
- **Created**: 2026-09-17T14:08:55+00:00 · torch 2.10.0+cu130

## Targets

| target | measured | threshold | met |
|---|---|---|---|
| bundled vs 16 separate calls | 9.079 | at least 8.0 | yes |
| bundled vs 64 separate calls | 14.762 | at least 12.0 | yes |
| marginal ms per question at 64 | 84.121 | at most 150.0 | yes |
| format overhead excluding the model's chat tail | 0.220 | at most 0.25 | yes |

**All targets met: yes.** Amortised cost at 64 questions: 106 ms/question.

The two format-overhead rows differ by the base model's chat tail (`<|im_end|>...<think></think>`), which its template imposes and no change to our format removes. The target applies to the controllable figure (ADR 0003).

The retired rule (`64-question call < 2x the 1-question call (retired by ADR 0003)`) would read **4.51x**. ADR 0003 explains why it was replaced: it is met only when `prefix_tokens >~ 61 x tokens_per_question`, which measures the benchmark's state size more than the engine.

## Measurements

| questions | median (ms) | p10-p90 | prefill | suffix | passes x rows | suffix tok | ms/suffix tok | peak GiB | vs 1-question | one call each (ms) | bundling speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1508 | 1497-1511 | 1391 | 113 | 1 x 4 | 28 | 4.052 | 20.64 | 1.00x | 1516 | 1.01x |
| 16 | 2742 | 2728-2746 | 1413 | 1324 | 4 x 4 | 649 | 2.041 | 21.63 | 1.82x | 24893 | 9.08x |
| 64 | 6808 | 6803-6813 | 1427 | 5373 | 16 x 4 | 2665 | 2.016 | 21.63 | 4.51x | 100498 | 14.76x |

## Where the time goes

Least-squares fit of the suffix phase, `suffix_ms ~ a * passes + b * suffix_tokens`:

- **69.7 ms per forward pass** (weight streaming and launch overhead)
- **1.598 ms per suffix token** (marginal cost of question text)
- R² = 1.0000

A prefix token costs 0.860 ms, so a suffix token costs essentially the same as a prefix token: batching the rows buys no per-token discount at this batch size, because the pass is already compute-bound.

## Format overhead

Boilerplate is what the renderer adds; content is the caller's instruction and option labels, which no format change can remove. Boilerplate splits into the chat tail, which the base model's template imposes, and **ours** — the headers and answer block, which is the only part ADR 0003 option B can shrink.

| primitive | suffix tok | content | ours | chat tail | boilerplate | controllable |
|---|---|---|---|---|---|---|
| noul | 28 | 12 | 7 | 9 | 57% | 25% |
| choice | 45 | 29 | 7 | 9 | 36% | 16% |
| score | 50 | 28 | 13 | 9 | 44% | 26% |

## Files

- `latency.json` — the source of truth for every number above.
- `latency.png` — drawn from it by `uv run task bench --plot results/<run_id>`.
