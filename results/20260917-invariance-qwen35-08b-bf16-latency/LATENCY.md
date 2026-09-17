# 20260917-invariance-qwen35-08b-bf16-latency

<!-- GENERATED from latency.json by eval/summary.py — do not edit by hand. -->

- **Model**: `Qwen/Qwen3.5-0.8B` · `bf16` · NVIDIA GeForce RTX 3090
- **State**: 1554 tokens (1617-token prefill including the system prompt) · median of 20 timed runs after 2 warm-ups
- **Created**: 2026-09-17T12:43:34+00:00 · torch 2.10.0+cu130

## Targets

| target | measured | threshold | met |
|---|---|---|---|
| bundled vs 16 separate calls | 13.296 | at least 8.0 | yes |
| bundled vs 64 separate calls | 21.280 | at least 12.0 | yes |
| marginal ms per question at 64 | 3.098 | at most 150.0 | yes |
| format boilerplate as a fraction of suffix tokens | 0.517 | at most 0.25 | **no** |

**All targets met: no.** Amortised cost at 64 questions: 5 ms/question.

The retired rule (`64-question call < 2x the 1-question call (retired by ADR 0003)`) would read **3.04x**. ADR 0003 explains why it was replaced: it is met only when `prefix_tokens >~ 61 x tokens_per_question`, which measures the benchmark's state size more than the engine.

## Measurements

| questions | median (ms) | p10-p90 | prefill | suffix | passes x rows | suffix tok | ms/suffix tok | peak GiB | vs 1-question | one call each (ms) | bundling speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 96 | 95-97 | 59 | 34 | 1 x 16 | 44 | 0.767 | 1.55 | 1.00x | 96 | 1.00x |
| 4 | 97 | 96-99 | 60 | 34 | 1 x 16 | 216 | 0.159 | 1.62 | 1.01x | 387 | 3.99x |
| 16 | 119 | 119-136 | 60 | 55 | 1 x 16 | 910 | 0.061 | 2.15 | 1.25x | 1589 | 13.30x |
| 64 | 291 | 290-293 | 60 | 222 | 4 x 16 | 3710 | 0.060 | 2.15 | 3.04x | 6190 | 21.28x |

## Where the time goes

Least-squares fit of the suffix phase, `suffix_ms ~ a * passes + b * suffix_tokens`:

- **30.8 ms per forward pass** (weight streaming and launch overhead)
- **0.027 ms per suffix token** (marginal cost of question text)
- R² = 0.9997

A prefix token costs 0.037 ms, so a suffix token costs essentially the same as a prefix token: batching the rows buys no per-token discount at this batch size, because the pass is already compute-bound.

## Format overhead

Boilerplate is what the renderer adds (chat tail, `### Question` header, `### Answer` block); content is the caller's instruction and option labels, which no format change can remove. ADR 0003 option B is the work of shrinking the former.

| primitive | suffix tok | tail | header | answer block | content | boilerplate |
|---|---|---|---|---|---|---|
| noul | 44 | 9 | 3 | 19 | 13 | 70% |
| choice | 62 | 9 | 3 | 17 | 33 | 47% |
| score | 66 | 9 | 3 | 17 | 37 | 44% |

## Files

- `latency.json` — the source of truth for every number above.
- `latency.png` — drawn from it by `uv run task bench --plot results/<run_id>`.
