# 20260917-120323-qwen38-27b-unsloth-bnb-4bit-nf4-bf16-latency

<!-- GENERATED from latency.json by eval/summary.py — do not edit by hand. -->

- **Model**: `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` · `nf4-bf16` · NVIDIA GeForce RTX 3090
- **State**: 1554 tokens (1617-token prefill including the system prompt) · median of 20 timed runs after 2 warm-ups
- **Created**: 2026-09-17T12:12:59+00:00 · torch 2.10.0+cu130

## Target: 64-question call < 2x the 1-question call

**4.31x — MISSED.**

## Measurements

| questions | median (ms) | p10-p90 | prefill | suffix | passes x rows | suffix tok | ms/suffix tok | peak GiB | vs 1-question | one call each (ms) | bundling speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1548 | 1534-1557 | 1404 | 140 | 1 x 7 | 44 | 3.186 | 20.50 | 1.00x | 1557 | 1.01x |
| 4 | 1867 | 1857-1871 | 1425 | 439 | 1 x 7 | 216 | 2.032 | 21.07 | 1.21x | 6356 | 3.40x |
| 16 | 2864 | 2848-2867 | 1437 | 1422 | 3 x 7 | 910 | 1.562 | 21.71 | 1.85x | 25669 | 8.96x |
| 64 | 6674 | 6672-6679 | 1440 | 5225 | 10 x 7 | 3710 | 1.408 | 21.71 | 4.31x | 102878 | 15.41x |

## Where the time goes

Least-squares fit of the suffix phase, `suffix_ms ~ a * passes + b * suffix_tokens`:

- **159.3 ms per forward pass** (weight streaming and launch overhead)
- **0.983 ms per suffix token** (marginal cost of question text)
- R² = 0.9993

A prefix token costs 0.869 ms, so a suffix token costs essentially the same as a prefix token: batching the rows buys no per-token discount at this batch size, because the pass is already compute-bound.

## Files

- `latency.json` — the source of truth for every number above.
- `latency.png` — drawn from it by `uv run task bench --plot results/<run_id>`.
