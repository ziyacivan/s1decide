# Two teachers, one rubric: how much do they actually agree?

**Date:** 2026-09-22
**Measured on:** the 100 pilot rows, both teachers, same states, same rubric, same 1,024-token cap
**Runs:** `results/pilot1024-qwen-low-1024`, `results/pilot1024-gptoss-medium-1024`
**Reproduce:** `uv run task teach-fold --first pilot1024-qwen-low-1024 --second pilot1024-gptoss-medium-1024 --dry-run`

## The numbers

| | value |
|---|---|
| rows compared | 100 |
| exact agreement | 55.0% |
| agreement expected by chance | 23.5% |
| Cohen's κ | 0.412 |
| quadratic-weighted κ | 0.656 |
| within one level | 80.0% |
| two or more apart | 20.0% |
| mean level, teacher 1 | 1.56 |
| mean level, teacher 2 | 1.32 |

Teacher 1 is `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` at native effort low; teacher 2 is
`openai/gpt-oss-20b` at effort medium. Chance agreement is 23.5% rather than a flat 20% because
both marginals are skewed, and κ is computed against that, not against uniformity.

## What it says

**The disagreements are mostly near misses.** Unweighted κ of 0.41 looks poor; the
quadratic-weighted κ of 0.66 is the honest figure for an ordinal scale, where 3-vs-4 is a near
miss and 0-vs-4 is not. Both are worth reporting, and reporting only the weighted one would be
the flattering choice.

**The two teachers have different thresholds, not just noise.** Their marginal distributions
differ in shape, not only in spread:

| level | 0 none | 1 slight | 2 moderate | 3 strong | 4 decisive |
|---|---|---|---|---|---|
| teacher 1 | 0.31 | 0.24 | 0.14 | 0.20 | 0.11 |
| teacher 2 | 0.50 | 0.11 | 0.14 | 0.07 | 0.18 |

Teacher 2 puts half of everything at "none" and is bimodal: it commits to "none" or "decisive"
and is reluctant about the middle of the scale. Teacher 1 spreads across it. On the 20 rows the
two were two or more levels apart, teacher 2 was the lower one on 14.

That is a systematic offset of about a quarter of a level, and it is the reason the default
resolution for a one-level disagreement is **teacher 1's level rather than the minimum**. Taking
the minimum would fold teacher 2's measured downward skew into every ambiguous row in the corpus.
This is a default, not a finding: both teachers' levels are written onto every kept row, so the
rule can be changed later without spending another day of GPU time re-labelling.

## What it costs

Keeping exact and ±1 agreement keeps **80% of compared rows**. Scaled to the 6,000-row overnight
job that is roughly **4,800 `Score` rows kept and 1,200 dropped**, before the separate loss to
truncated traces (1.0% of teacher 1's rows so far).

The kept set is not a uniform sample of the dropped-from set. 30 of the 100 pilot rows were
`0 -> 0`, so agreement is concentrated at the bottom of the scale, and the resulting label
distribution is `{0: 31, 1: 22, 2: 6, 3: 13, 4: 8}` — 39% of kept rows are "none", and level 2
("moderate") gets 6. A `Score` corpus that thin in the middle will train a model that is good at
recognising absence and unsure everywhere between.

That figure is also the measured cost of the resolution rule. Resolving one-level disagreements
to the *minimum* instead would have given `{0: 39, 1: 15, 2: 7, 3: 12, 4: 7}` — 49% "none"
against 39%. Ten points of skew rode on a choice that looked arbitrary.

## What this does not say

It does not say either teacher is right. There is no reference label here: two models agreeing
is not evidence of correctness, and the whole comparison is 100 rows from one pilot. A
disagreement rate is a lower bound on the error rate of the pair, nothing more.

It also does not transfer to the `Choice` and `Noul` primitives, which are not teacher-labelled.

## Consequences taken

- The one-level resolution default is teacher 1's level, recorded in `data/build/agree.py` with
  this note as its basis.
- `FoldReport` separates rows lost to disagreement (`far`) from rows lost to a teacher that never
  committed (`no_commit`). They are different failures: one is a rubric or model problem, the
  other is a generation cap.
- The label skew is a known shortfall for v0.1, alongside `Score` being 2.7% of the training mix.
  The model card has to state it rather than let a reader infer a balanced ordinal corpus.

## Open

- Whether the agreement rate holds at 6,000 rows or the pilot was easy. Measurable the moment
  leg 2 finishes, at no extra GPU cost.
- Whether a third teacher would break ties usefully, or just add a third threshold.
- Whether resolving ambiguous rows to a soft target over both levels beats picking one. The
  ordinal loss is designed to consume neighbour mass (locked decision 3), so this is testable
  without changing the data — the rows already carry both levels.
