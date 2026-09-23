# Laya on our slices — 2026-09-24

`eval/external_laya.py`, CPU only (torch 2.14 CPU build, 6 threads, a separate environment),
while S1 held the GPU. `convaiinnovations/laya` revision `aa8c91c`, laya 0.3.11, checkpoints
`laya` and `typed-decisions`. Slices: exactly S1's `val` evaluation slice (8,329 rows) and the same
construction on `test` (8,350). Results: `results/external-laya-2026-09-24/metrics.json`,
`results/external-laya-typed-2026-09-24/metrics.json` — per primitive, with risk-coverage and the
base-rate control; the numbers are read from there, not from this note.

## What the runs show (val; test agrees in direction)

- **`Noul` is Laya's best primitive here** and the one where it beats the untrained 27B: both
  checkpoints have positive Brier skill where zero-shot Qwen3.8-27B is below its base rate.
- **`Choice` is well below the 27B** (accuracy ~0.74 against 0.94), still with clear positive skill.
- **Stage-1 rows have negative Brier skill** for both checkpoints, strongly so for
  `typed-decisions` — confident on a question it answers no better than the base rate.
- **Teacher-labelled `Score` is below the base rate** for both (accuracy ~0.27–0.30).

## Caveats that travel with every Laya number

1. **Home ground.** The slices come from our own sources and families. Neither Laya nor the
   zero-shot 27B was trained on them, but they are closer to what *our* adapter trains on.
2. **Our question text, not Laya's idiom.** Stage-1 rows are asked as `noul` with our
   "…?\nCandidate: X" instructions — not a statement, which is what a `noul` expects. That is
   the fair setting for a comparison (every model sees the same text) and a plausible reason for
   the stage-1 result; it is not a verdict on Laya's stage-1 ability.
3. **Undescribed options.** Our `Choice` rows carry labels only; Laya was given each label as its
   own description. Laya's examples use descriptions.
4. **Laya's own calibration, as shipped.** Its card applies post-hoc temperatures per question
   type and option count. On load it warned that the `choice:11+` temperature is invalid and was
   clamped to 0.5 — affecting `Choice` with 11+ options, which our slices do not contain.
5. **No temperature fitted by us.** Every row in the eventual table is compared as shipped, plus —
   for our model — with S2's temperature; fitting one for Laya on our `val` would be a separate,
   labelled row.
