# One model, two quantizations, different answers — nf4 vs Q4_K_M

**Date:** 2026-09-18
**Measured by:** `results/pilot1024-qwen-low-1024/` (bitsandbytes nf4) and
`results/pilot1024-llamacpp-qwen-low/` (llama.cpp Q4_K_M)
**Why it exists:** it is the measured reason calibration must be fitted at the deployment
quantization, and the reason a 5.18x speed-up was declined.

## What was compared

The **same model** (`Qwen3.8-27B`), the **same 100 rows**, the **same rubric and prompt**, the
same reasoning setting (`effort=low`), the same 1,024-token cap, and greedy decoding on both
sides. The only difference is how the weights are stored and which runtime multiplies them:

| | teacher run | experiment |
|---|---|---|
| quantization | bitsandbytes **nf4**, bf16 compute | llama.cpp **Q4_K_M** |
| runtime | transformers | `llama-server`, continuous batching, `--parallel 16` |
| committed a level | 100/100 | 100/100 |
| hit the 1,024 cap | 0% | 0% |
| mean reasoning trace | 570 tokens | 529 tokens |
| throughput | 0.060 rows/s | 0.312 rows/s (**5.18x**) |

Both runtimes produce a complete, parseable judgement every time. Neither is broken. They simply
do not agree.

## The disagreement

| | |
|---|---|
| exact match | **78.0%** (78/100) |
| within ±1 level | 94.0% |
| **beyond ±1** | **6.0%** |

Adjacent-level confusion is where an ordinal scale is always weakest, so 94% within ±1 sounds
reassuring. It is worth noticing that **6 of 100 differ by two levels or more** — those are not
boundary cases, they are different readings of the same text.

## It is not symmetric noise

This is the part that matters, and it is not visible in the 78% figure.

| | nf4 | Q4_K_M |
|---|---|---|
| level 1 (none) | 31 | **39** |
| level 2 (slight) | 24 | 20 |
| level 3 (moderate) | 14 | 13 |
| level 4 (strong) | 20 | 19 |
| level 5 (decisive) | 11 | 9 |
| **mean level** | **1.560** | **1.390** |

Of the 22 rows where they disagree, **16 go lower under Q4_K_M and 6 go higher**. The whole
distribution shifts down by **0.17 levels** on a 5-point scale: Q4_K_M reads the same text as
less emphatic.

A two-sided exact sign test on 16/22 gives **p = 0.053**. At this sample size that is
*suggestive, not established* — 22 disagreements is a small number and the honest reading is
"there is a directional effect here worth taking seriously", not "a bias of 0.17 levels has been
demonstrated". What settles the practical question is that the direction is consistent with the
marginals shifting, and that the cost of being wrong about it is silent.

## Why this decided two things

**1. The 5.18x speed-up was declined** (ADR 0005). The rule agreed beforehand required both a
2.5x speed-up and 90% exact match. Speed passed; agreement did not. Adopting Q4_K_M would have
changed what the teacher said while the dataset card went on naming one model and one revision —
and the change is concentrated exactly where the ±1 keep-rule already does the most work, so it
would have altered *which rows survive filtering*, not merely their labels.

**2. Calibration is fitted at the deployment quantization**, which was already a locked decision
(`CLAUDE.md`, stage S2) and is now a measured one rather than a precaution. If two 4-bit
quantizations of one model disagree on 22% of an ordinal judgement and shift the label marginal
by 0.17 levels, then a temperature fitted under one and served under the other is fitted to a
distribution the deployment does not have. A temperature is a single scalar correcting a
systematic over- or under-confidence; a systematic shift in what the model *says* is precisely
the thing it cannot absorb.

## What this does not say

- **Neither quantization is "right".** There is no ground truth here. The rubric levels are
  defined but the judgements are the model's, and this note measures disagreement, not error.
  Which is closer to a bf16 reference is unknown and unmeasurable on a 24 GB card (ADR 0006
  defers the BF16 row to v0.2 for exactly this reason).
- **It does not generalise to accuracy.** All of this is on a 5-level ordinal rubric with
  reasoning on. A 2-option `Noul` read from a single masked logit is a different measurement and
  may well be far more stable. That is worth testing and has not been.
- **It is not about llama.cpp.** The runtime was fast and correct. The variable that moved is
  the weight quantization.

## Reproducing

```
uv run task teach --teacher qwen-low-1024 --limit 100 --batch-size 4 --run-id pilot1024-qwen-low-1024
# then, with llama-server on Q4_K_M at --parallel 16, the client in
# src/s1decide/engine/llamacpp_client.py over the same 100 rows
```

Both row files are committed. The comparison is a join on `id` over `level`.
