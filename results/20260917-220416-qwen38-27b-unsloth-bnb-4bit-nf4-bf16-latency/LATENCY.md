# 20260917-220416-qwen38-27b-unsloth-bnb-4bit-nf4-bf16-latency

<!-- GENERATED from latency.json by eval/summary.py — do not edit by hand. -->

- **Model**: `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` · `nf4-bf16` · NVIDIA GeForce RTX 3090
- **State**: 1554 tokens (1617-token prefill including the system prompt) · median of 20 timed runs after 2 warm-ups
- **Created**: 2026-09-17T22:13:20+00:00 · torch 2.10.0+cu130

- **Kernels**: attention `sdpa` · accelerated: `chunk_gated_delta_rule`, `fused_recurrent_gated_delta_rule` · torch: `causal_conv1d_fn`, `causal_conv1d_update`

## Targets

| target | measured | threshold | met |
|---|---|---|---|
| bundled vs 16 separate calls | 9.524 | at least 8.0 | yes |
| bundled vs 64 separate calls | 16.947 | at least 12.0 | yes |
| marginal ms per question at 64 | 68.762 | at most 150.0 | yes |
| format overhead excluding the model's chat tail | 0.220 | at most 0.25 | yes |

**All targets met: yes.** Amortised cost at 64 questions: 91 ms/question.

The two format-overhead rows differ by the base model's chat tail (`<|im_end|>...<think></think>`), which its template imposes and no change to our format removes. The target applies to the controllable figure (ADR 0003).

The retired rule (`64-question call < 2x the 1-question call (retired by ADR 0003)`) would read **3.86x**. ADR 0003 explains why it was replaced: it is met only when `prefix_tokens >~ 61 x tokens_per_question`, which measures the benchmark's state size more than the engine.

## Measurements

| questions | median (ms) | p10-p90 | prefill | suffix | passes x rows | suffix tok | ms/suffix tok | peak GiB | vs 1-question | one call each (ms) | bundling speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1515 | 1493-1581 | 1402 | 108 | 1 x 6 | 28 | 3.865 | 20.50 | 1.00x | 1550 | 1.02x |
| 4 | 1761 | 1737-1777 | 1427 | 331 | 1 x 7 | 151 | 2.191 | 21.06 | 1.16x | 6207 | 3.53x |
| 16 | 2614 | 2610-2677 | 1427 | 1182 | 3 x 7 | 649 | 1.821 | 21.69 | 1.73x | 24899 | 9.52x |
| 64 | 5847 | 5845-5849 | 1418 | 4421 | 11 x 6 | 2665 | 1.659 | 21.69 | 3.86x | 99092 | 16.95x |

## Where the time goes

Least-squares fit of the suffix phase, `suffix_ms ~ a * passes + b * suffix_tokens`:

- **113.6 ms per forward pass** (weight streaming and launch overhead)
- **1.196 ms per suffix token** (marginal cost of question text)
- R² = 0.9994

A prefix token costs 0.867 ms, so a suffix token costs essentially the same as a prefix token: batching the rows buys no per-token discount at this batch size, because the pass is already compute-bound.

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
