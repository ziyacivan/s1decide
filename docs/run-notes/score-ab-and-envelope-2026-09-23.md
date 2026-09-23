# Before/after `Score` check and the training memory envelope, 2026-09-23

Owner's items 1 and 2, before any S1 run. Numbers come from `results/score-ab-2026-09-23/`,
`results/token-lengths-2026-09-23/` and `results/memory-envelope-2026-09-23/`.

## 1. Before/after on the 468 genuine `Score` val rows — the smoke adapter is worse

`eval/score_ab.py`: the teacher-labelled `Score` rows of `val` (teacher run 2), scored on the base
model and then with `results/smoke-27b-attempt3/adapter` injected into the same weights (256/256
modules confirmed after injection). Raw softmax, no temperature, in both columns.

| | zero-shot | + smoke adapter |
|---|---|---|
| accuracy (vs teacher 1's level) | 0.592 | 0.521 |
| accuracy, hard rows only | 0.720 | 0.713 |
| soft rows: argmax on either teacher's level | 0.709 | 0.637 |
| QWK | 0.811 | 0.764 |
| MAE, argmax | 0.483 | 0.624 |
| MAE, expected level | 0.593 | 0.667 |
| KL(target ‖ pred), mean | 0.741 | 0.764 |
| ECE (15 equal-mass bins) | 0.051 | 0.101 |
| Brier / BSS vs base rate | 0.504 / 0.323 | 0.575 / 0.226 |

**It is a shift, not uniform damage.** Per row (`rows.jsonl`), the adapter improved rows whose
answer is level 4 (KL −0.87, 22 wrong→right, none right→wrong) and level 3 (KL −0.62), and hurt
level 0 (KL +0.27, 32 right→wrong, 3 wrong→right) and level 1 (+0.43). The predicted mean level
moved from 1.04 to 1.64; the val mean is 1.32.

**Most of it is the smoke sample, not training as such.** The smoke sampler stratifies for code
coverage, not for the deployed distribution: half its `Score` rows are soft, 9 of 73 are
rule-labelled `ordinal_control`, and their target marginal has mean level 1.56 — the adapter
moved roughly onto it. The later two thirds of the pass were also higher-level than the first
(mean 1.9 and 1.8 against 1.0), which a single-row step at lr 1e-4 would track.

### Candidates, in the owner's order, with what the evidence says

1. **Learning rate for 27B QLoRA (1e-4, batch 1, no warmup or decay).** Supported: 200 single-row
   steps moved the level prior most of the way to the sample's in one pass, and toward its
   later rows. At effective batch 1 every step is one row's gradient.
2. **λ of the distance term (0.3).** Weakly supported: level 2's share of predictions rose from
   20.5% to 32.3% while level 2 was only 11.6% of the trained `Score` targets. Expected distance
   is minimised by moving mass toward the middle when uncertain.
3. **`Score` row scarcity.** Not supported *in this experiment*: `Score` was 73 of 200 rows. It
   is a real concern for S1 proper, where `Score` is 2.7% of the effective mix.

Not on the list and material: **the smoke sample's `Score` prior differs from deployment.** Any
smoke-adapter comparison inherits it; S1 proper samples by family weight instead.

### Proposed controlled change — one knob

**Learning rate 1e-4 → 2e-5, everything else identical** (same 200 rows, same order, rank 8,
λ 0.3, batch 1, one pass), then rescore the same 468 rows with `eval/score_ab.py`. If the prior
shift and the ECE/BSS loss largely disappear, the step size was the cause and S1's rate follows
from it; if they do not, λ is next, changed alone. Cost: one 4-minute run and one 6-minute
scoring pass.

## 2. Memory envelope

### Token lengths (`results/token-lengths-2026-09-23/lengths.json`)

Every training row rendered exactly as the trainer renders it, no cap applied.

| group | rows | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| all | 90,594 | 103 | 151 | 218 | 843 |
| choice | 10,413 | 141 | 183 | 330 | 843 |
| noul (genuine) | 9,922 | 108 | 223 | 381 | 802 |
| noul / stage 1 | 64,729 | 101 | 114 | 132 | 188 |
| score / hard | 3,725 | 135 | 168 | 290 | 818 |
| score / soft | 1,805 | 130 | 162 | 299 | 564 |

**No row exceeds 1024 tokens**, so `max_seq_len` 1024 skips nothing, and at batch 1 (no padding)
a 2048 cap trains on exactly the same tokens.

### Two one-knob runs against attempt 3 (`results/memory-envelope-2026-09-23/envelope.json`)

| run | seq cap | rank | adapter | peak VRAM | tokens/s |
|---|---|---|---|---|---|
| `smoke-27b-attempt3` | 1024 | 8 | 256/256 | 20.853 GiB | 118.5 |
| `smoke-27b-seq2048` | **2048** | 8 | 256/256 | **20.853 GiB** | 139.9 |
| `smoke-27b-rank16` | 1024 | **16** | 256/256 | **21.552 GiB** | 167.0 |

- **seq 2048 costs nothing** — identical peak, as the length distribution predicts.
- **rank 16 costs +0.70 GiB.** The profile's measured cliff on this card (ADR 0003): 21.90 GiB
  was fine, 22.34 GiB ran 2.8× slower with no error. Rank 16 sits 0.35 GiB under the fine point.
- **tokens/s is not comparable across these runs.** Rank 16 cannot be faster than rank 8; the
  spread is run-to-run (the first run after an environment sync pays Triton compilation and
  autotuning). Peak memory is deterministic; these throughputs are not a ranking.

**The envelope was measured on rows of at most 412 tokens** (the smoke sample's longest); the
corpus goes to 843. S1 will meet rows twice as long, and none of these peaks covers them. The
measurement that decides rank for S1 is one more run, one knob against attempt 3: the same
budget over the **longest** rows of the corpus, at rank 8. The profile's `training_envelope`
points at the JSON above, so that run extends it rather than replacing it.

## Follow-up, same day — the controlled change and the long-row envelope

### Learning rate 1e-4 → 2e-5, nothing else changed

`train/configs/smoke_27b_lr2e5.yaml` differs from attempt 3 only in `learning_rate`; the same 468
rows were rescored (`results/score-ab-lr2e5-2026-09-23/`). The 1e-4 run was rescored from its
stored logits to add the level means (`--rescore`; no other field changed).

| | zero-shot | lr 1e-4 | lr 2e-5 |
|---|---|---|---|
| accuracy | 0.592 | 0.521 | **0.645** |
| QWK | 0.811 | 0.764 | **0.841** |
| MAE, argmax | 0.483 | 0.624 | **0.419** |
| KL, mean | 0.741 | 0.764 | **0.644** |
| ECE | **0.051** | 0.101 | 0.076 |
| BSS vs base rate | 0.323 | 0.226 | **0.387** |
| mean predicted level (target 1.363) | 1.043 | 1.641 | 1.150 |
| rows whose KL fell | — | 40.6% | **76.3%** |
| right→wrong / wrong→right | — | 97 / 64 | **17 / 42** |

**The step size was the cause.** At 2e-5 the prior does not overshoot, and every accuracy and
proper-score metric beats zero-shot. ECE is the exception: mean confidence rose from 0.60 to
0.67, and raw confidence is what S2's per-bucket temperature is fitted to correct. λ was not
changed and needs no separate run on this evidence.

### The longest rows (`smoke-27b-longest`)

`selection: longest` takes the 200 longest rows of the corpus as rendered — 376 to 843 tokens,
including the corpus maximum — at rank 8, one knob against attempt 3.

| run | max row tokens | rank | peak VRAM |
|---|---|---|---|
| attempt 3 (stratified) | 412 | 8 | 20.853 GiB |
| **longest rows** | **843** | 8 | **20.985 GiB** |
| rank 16 (stratified) | 412 | 16 | 21.552 GiB |

Doubling the longest row adds 0.13 GiB at rank 8. **S1 uses rank 8; rank 16 is deferred to v0.2
on an H100** — both recorded in `results/memory-envelope-2026-09-23/envelope.json`, which the
3090 hardware profile points at. Training summaries now record `max_row_tokens` (backfilled for
these four runs from their deterministic selections, and marked as such).
