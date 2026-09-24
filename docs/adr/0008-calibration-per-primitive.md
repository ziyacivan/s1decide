# ADR 0008 — S2 calibration per (primitive, option-count bucket)

**Status:** accepted 2026-09-24 (owner decision); the open question was decided the same day — option B, plus `none` as a selectable method (see the end).
**Context:** locked decision 6 makes S2 post-hoc temperature scaling "per option-count bucket".
The first S2 fit on S1 (`results/s1-3090-slices-s2/`, run note
`docs/run-notes/s1-slices-and-s2-2026-09-24.md`) was mixed: the `3-5` bucket held teacher-labelled
`Score` (five genuinely uncertain levels), rule-labelled `Score` (near-certain) and 4-option
`Choice`, and one temperature had to serve all three. Minimising NLL over the bucket minimised
nobody's ECE.

## Decision

1. **Cells are (primitive, option-count bucket).** Primitives for calibration are `choice`,
   `noul`, `score` and `stage1`. Stage-1 rows — the per-option `Noul` of the two-stage
   high-cardinality path (locked decision 4) — are their own primitive: the two-stage path knows
   at inference time that it is running stage 1, and their ~99:1 prior has nothing to lend a
   genuine `Noul`.
2. **Both methods are fitted in every cell** — temperature per bucket, vector scaling
   (temperature + per-position bias) per exact option count — and **both are reported**.
3. **The deployed method is chosen per cell, out of sample.** Choosing on the rows a method was
   fitted to always favours vector scaling (more parameters). `select_methods` splits val into
   two folds by a hash of the row id, fits on one, scores NLL on the other, pools, and picks the
   lower held-out NLL; ties go to temperature. The choice and both held-out NLLs are written into
   `calibration.json` (`meta.method_selection`), so the file says why each cell deploys what it
   deploys.
4. **Rule-labelled controls are excluded from the fit** (`--exclude-families ordinal_control`)
   and still evaluated. Their labels are exact by construction and they are not deployment
   data; letting them pull the `Score` fit toward certainty calibrates for a test, not for
   callers. The variant fitted with them is produced and reported beside it.
5. Calibration files stay backward-compatible: `method_by_bucket` is a new optional field;
   files without it deploy `method` everywhere, as before.

Implemented in `s1decide.calibrate.select_methods` / `fit_calibration_cells` and
`eval/adapter_slice.py calibrate` (`--method` forces one method everywhere for the side-by-side
columns). Tests: `tests/test_calibrate.py` (a positional shift selects vector; pure temperature
data keeps temperature; excluded families are out of the fit; the choice round-trips).

## Amendment, same day: option B and `none` (owner decision)

- **Fit and selection use the soft target where a row has one** — teacher-labelled `Score`.
  Temperature and vector scaling minimise cross-entropy against it; held-out selection scores
  against it. Accuracy against the hard label is still reported. Everything else has one-hot
  targets, where the two coincide.
- **`none` is a selectable method per cell** — the raw softmax. It is scored out of sample like
  the others, and on a tie the method with fewer parameters wins (`none`, then temperature, then
  vector), so a cell stays raw unless a fit beats it.
- **Bucket temperatures are now fitted with the case-control weights**, as vector scaling already
  was. Before this they were fitted unweighted, so the two methods were fitted to different
  populations. The fix changes stage-1 temperature fits; the runs below include it.

Tests: `tests/test_calibrate.py` (a soft-target fit recovers the generating temperature exactly
while a hard-label fit on the same rows does not; `none` wins when the raw softmax is already
the target; `none` deploys the raw softmax).

## What it did on S1 (test slice)

Runs, all from `results/s1-3090-slices` raw logits, re-fitted after the amendment:

| run | what |
|---|---|
| `s1-3090-slices-s2cells` | deployed: per-cell choice, control excluded |
| `s1-3090-slices-s2cells-temperature` | temperature in every cell |
| `s1-3090-slices-s2cells-vector` | vector in every cell |
| `s1-3090-slices-s2cells-withcontrol` | per-cell choice, control included in the fit |

Numbers in each run's `metrics.json`; the choice and all three held-out losses per cell in each
`calibration.json` (`meta.method_selection`). In short:

- **stage 1 picks vector**, the largest gain S2 produces: Brier skill and KL both clearly better.
- **`choice`**: `6-16` now picks **`none`** (no fit beats the raw softmax out of sample); `3-5`
  keeps temperature. **`noul`** picks vector by a small margin.
- **teacher `Score` picks temperature on the soft target.** KL to the teacher and ECE both
  improve; skill is flat; accuracy is unchanged (temperature cannot move the argmax). The
  hard-label vector fit that won before this amendment is gone, and so is its KL cost.
- **Rule-labelled `Score` is still worse than raw under the deployed fit.** It shares the `score`
  cell with teacher rows, because a caller cannot tell us whether a `Score` question is
  rule-labelled — a separate cell has no deployment-time key. Scored as a cell of its own on its
  92 val rows it would pick `none` (`s1-3090-slices-s2cells/excluded_cell_diagnostic.json`),
  which is the owner's expectation and the right answer for that data, but it cannot be
  deployed as such. Including the controls in the fit (`-withcontrol`) makes them less bad and
  leaves teacher `Score` no worse; the deployed variant excludes them per the decision above.
