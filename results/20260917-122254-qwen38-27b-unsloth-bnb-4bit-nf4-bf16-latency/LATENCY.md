# 20260917-122254-qwen38-27b-unsloth-bnb-4bit-nf4-bf16-latency

<!-- GENERATED from latency.json by eval/summary.py — do not edit by hand. -->

- **Model**: `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` · `nf4-bf16` · NVIDIA GeForce RTX 3090
- **State**: 1554 tokens (1617-token prefill including the system prompt) · median of 20 timed runs after 2 warm-ups
- **Created**: 2026-09-17T12:32:29+00:00 · torch 2.10.0+cu130

## Targets

| target | measured | threshold | met |
|---|---|---|---|
| bundled vs 16 separate calls | 9.003 | at least 8.0 | yes |
| bundled vs 64 separate calls | 15.382 | at least 12.0 | yes |
| marginal ms per question at 64 | 81.188 | at most 150.0 | yes |
| format boilerplate as a fraction of suffix tokens | 0.517 | at most 0.25 | **no** |

**All targets met: no.** Amortised cost at 64 questions: 104 ms/question.

The retired rule (`64-question call < 2x the 1-question call (retired by ADR 0003)`) would read **4.31x**. ADR 0003 explains why it was replaced: it is met only when `prefix_tokens >~ 61 x tokens_per_question`, which measures the benchmark's state size more than the engine.

## Measurements

| questions | median (ms) | p10-p90 | prefill | suffix | passes x rows | suffix tok | ms/suffix tok | peak GiB | vs 1-question | one call each (ms) | bundling speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1547 | 1539-1551 | 1404 | 140 | 1 x 7 | 44 | 3.184 | 20.50 | 1.00x | 1560 | 1.01x |
| 4 | 1864 | 1855-1868 | 1423 | 437 | 1 x 7 | 216 | 2.023 | 21.07 | 1.20x | 6342 | 3.40x |
| 16 | 2848 | 2843-2871 | 1429 | 1414 | 3 x 7 | 910 | 1.554 | 21.71 | 1.84x | 25639 | 9.00x |
| 64 | 6662 | 6657-6672 | 1435 | 5217 | 10 x 7 | 3710 | 1.406 | 21.71 | 4.31x | 102476 | 15.38x |

## Where the time goes

Least-squares fit of the suffix phase, `suffix_ms ~ a * passes + b * suffix_tokens`:

- **156.4 ms per forward pass** (weight streaming and launch overhead)
- **0.989 ms per suffix token** (marginal cost of question text)
- R² = 0.9994

A prefix token costs 0.868 ms, so a suffix token costs essentially the same as a prefix token: batching the rows buys no per-token discount at this batch size, because the pass is already compute-bound.

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
