# Two teachers, one rubric: how much do they actually agree?

**Date:** 2026-09-22, revised 2026-09-23 with the full run
**Measured on:** all 6,000 rows, both teachers, same states, same rubric, same 1,024-token cap
**Runs:** `results/teach-qwen-low-1024`, `results/teach-gptoss-medium-1024`
**Report:** `results/teach-qwen-low-1024-fold/fold.json`
**Reproduce:** `uv run task teach-fold --first teach-qwen-low-1024 --second teach-gptoss-medium-1024 --dry-run`

Teacher 1 is `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` at native effort low; teacher 2 is
`openai/gpt-oss-20b` at effort medium.

## The numbers

| | full run (n=5,928) | pilot (n=100) |
|---|---|---|
| exact agreement | **50.6%** | 55.0% |
| agreement expected by chance | 24.7% | 23.5% |
| Cohen's κ | **0.344** | 0.412 |
| quadratic-weighted κ | **0.650** | 0.656 |
| within one level (kept) | **81.0%** | 80.0% |
| two or more apart (dropped) | **19.0%** | 20.0% |
| mean level, teacher 1 | 1.47 | 1.56 |
| mean level, teacher 2 | 1.39 | 1.32 |

72 of the 6,000 rows are missing from the comparison because one teacher never committed — a
truncated trace. Chance agreement is 24.7% rather than a flat 20% because both marginals are
skewed, and κ is computed against that, not against uniformity.

## Which pilot numbers survived, and which did not

This is worth recording on its own, because the pilot was used to make decisions.

**The weighted κ was accurate to 0.006** (0.656 → 0.650) and **the keep rate to one point**
(80.0% → 81.0%). Those two carried the planning decisions and both held.

**Raw agreement and unweighted κ did not.** Exact agreement fell 4.4 points and Cohen's κ fell
from 0.412 to 0.344 — a 17% relative drop. A 100-row pilot flattered the statistic that is most
sensitive to the marginal distribution, which is the one people quote.

If only one number is carried forward from a small pilot on an ordinal task, it should be the
weighted κ.

## What it says

**The disagreements are mostly near misses.** Unweighted κ of 0.34 looks poor; the weighted κ of
0.65 is the honest figure for an ordinal scale, where 3-vs-4 is a near miss and 0-vs-4 is not.
Both are reported, and reporting only the weighted one would be the flattering choice.

**The two teachers differ in shape, not by a shift.** Their marginals:

| level | 0 none | 1 slight | 2 moderate | 3 strong | 4 decisive |
|---|---|---|---|---|---|
| teacher 1 | 0.348 | 0.226 | 0.143 | 0.176 | 0.107 |
| teacher 2 | 0.477 | 0.106 | 0.136 | 0.110 | 0.171 |

Teacher 2 is **polarised**: heavier than teacher 1 at both ends and thinner in between. Teacher 1
spreads across the scale.

**A correction to the pilot's reading.** The pilot note said teacher 2 "sits about a quarter of a
level low", from a mean gap of 0.24 and 14 of 20 far disagreements being lower. At full scale
that is wrong in size and wrong in generality:

- the mean gap is **0.08**, not 0.24 — teacher 2's extra mass at "decisive" nearly cancels its
  extra mass at "none";
- on the **±1 disagreements that we actually keep**, teacher 2 is the *higher* one more often —
  1,010 against 795;
- only on the far disagreements, which are dropped, does it lean low: 647 against 477.

So "teacher 2 scores lower" was a small-sample artifact. The durable difference is polarisation.

## The resolution rule, re-justified

The default for a one-level disagreement is **teacher 1's level**. That decision stands, but the
reason given for it in the pilot note — compounding a downward skew — was based on the figure
that did not survive. The measured case, on all 4,804 kept rows:

| rule | resulting "none" share | mean |
|---|---|---|
| **teacher 1 (chosen)** | **41.0%** | 1.28 |
| minimum | 51.3% | 1.11 |
| maximum | 38.3% | 1.49 |
| teacher 2 | 48.6% | 1.32 |

The minimum is still clearly the worst: it produces a corpus that is over half "none". The
maximum yields the smallest "none" share, but it imports teacher 2's polarisation — level 3 falls
to 0.103 while level 4 rises to 0.179 — and buys a flatter bottom by hollowing the upper middle.
Teacher 1's level keeps the distribution closest in shape to the better-spread teacher, which is
what the rubric is trying to elicit.

Every kept row carries both teachers' levels, so this remains reversible without re-labelling.

## What it costs

Keeping exact and ±1 agreement kept **4,804 of 5,928 compared rows (81.0%)**, against a planned
target band of 4,000–6,000. Every row resolved to a licensed source; none was dropped for
provenance. Licences: cc-by-4.0 1,839, apache-2.0 1,269, cc-by-3.0 905, mit 791, across 9
families.

The kept set is skewed to the bottom of the scale: `{0: 1970, 1: 1159, 2: 510, 3: 694, 4: 471}` —
41% "none" and 10.6% "moderate". A `Score` corpus this thin in the middle will train a model that
is good at recognising absence and least certain exactly where the rubric is hardest. That is a
known v0.1 shortfall alongside `Score` being a small share of the training mix, and the model
card states it rather than letting a reader assume a balanced ordinal set.

## What this does not say

It does not say either teacher is right. There is no reference label here: two models agreeing is
not evidence of correctness, and a disagreement rate is a lower bound on the error rate of the
pair, nothing more. A corpus filtered to agreement is a corpus filtered to *easy*, which is its
own bias and not a quality guarantee.

It also does not transfer to `Choice` and `Noul`, which are not teacher-labelled.

## Open

- Whether a third teacher would break ties usefully or just add a third shape.
- Whether resolving ambiguous rows to a soft target over both levels beats picking one. The
  ordinal loss is designed to consume neighbour mass (locked decision 3), so this is testable
  without re-labelling — the rows already carry both levels.
- Whether the 1,124 dropped rows are systematically harder, or systematically ambiguous in the
  rubric's wording. Reading a sample of them is cheap and has not been done.
