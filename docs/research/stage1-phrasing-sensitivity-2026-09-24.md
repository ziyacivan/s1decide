# Stage-1 phrasing sensitivity in external models — 2026-09-24

**Question.** Stage-1 rows — the per-candidate `Noul` of the two-stage many-option `Choice` — are
asked in our own text, `"<question>\nCandidate: <option>"`. External models were far below the
base rate on them (Kev-9B worst). Is that the models' decisions, or our phrasing?

**Method.** The same stage-1 rows (300 questions × 17 rows per split, case-control weighted,
val and test) re-asked as a labelled plain proposition,
`The answer to "<question>" is "<option>".` (`eval.external_laya.plain_proposition`), with
everything else fixed: same slices, same answer mapping, same report code. Only the stage-1 rows
were re-run (`--only-stage1`); `merge` spliced them into each model's native run so the two runs
differ in that one respect. Our own model is not re-run: `Candidate:` is the format S1 was trained
on, so the plain form would be out of format for it rather than fairer.

| model | native run | plain run |
|---|---|---|
| Laya (root, CPU) | `results/external-laya-2026-09-24/` | `results/external-laya-2026-09-24-plain/` |
| Laya typed-decisions (CPU) | `results/external-laya-typed-2026-09-24/` | `…-typed-2026-09-24-plain/` |
| Kev-9B (GPU, bf16) | `results/external-kev-9b-2026-09-24/` | `…-kev-9b-2026-09-24-plain/` |

Numbers: each run's `metrics.json`, `splits.<split>.model["noul/stage1"]`, against the base-rate
control in the same file.

## Result

- **Kev-9B is strongly phrasing-sensitive.** In the native form it says "yes" to about a quarter
  of candidates when about 1 in 17 is right; asked the plain proposition, its yes-rate drops by
  more than half, accuracy rises by more than ten points on both splits, and Brier skill rises
  by several units — but **stays negative**: still worse than the base rate.
- **Both Laya checkpoints barely move.** Differences are small and go in opposite directions for
  the two checkpoints; both stay below the base rate in both forms.
- Our S1 row on the same stage-1 rows (`results/s1-3090-slices/metrics.json`) is the only one
  with positive skill, in its own training format.

## What it means for the comparison

1. The native-form stage-1 numbers overstate how badly Kev does at this *task*; much of its
   deficit was the phrasing. Both forms are reported, labelled, and the comparison post uses the
   plain form as the external models' stage-1 row with the native form beside it.
2. Neither form rescues the external models to positive skill on stage 1: at a ~16:1 prior,
   saying "yes" to plausible-but-wrong candidates costs more than knowing nothing. That part is
   about calibration against a skewed prior, not wording.
3. This is one alternative phrasing, not a search over phrasings. A model could do better under
   a form neither of us tried; the claim is only that the result is phrasing-sensitive for one
   model and not for the other two.
