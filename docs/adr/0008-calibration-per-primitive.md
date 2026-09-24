# ADR 0008 — S2 calibration per (primitive, option-count bucket)

**Status:** accepted 2026-09-24 (owner decision); one open question at the end.
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

## What it did on S1 (test slice)

Runs, all from `results/s1-3090-slices` raw logits:

| run | what |
|---|---|
| `s1-3090-slices-s2cells` | deployed: per-cell choice, control excluded |
| `s1-3090-slices-s2cells-temperature` | temperature in every cell |
| `s1-3090-slices-s2cells-vector` | vector in every cell |
| `s1-3090-slices-s2cells-withcontrol` | per-cell choice, control included in the fit |

The numbers are in each run's `metrics.json`; the run note reads them. In short:

- **stage 1 selects vector and it is the largest gain S2 has produced** — held-out NLL roughly
  halves, and Brier skill on stage-1 rows rises on test. A per-position bias fixes what a
  temperature cannot: a constant lean toward "no" in a 99:1 cell.
- `choice` keeps temperature in both buckets; `noul` selects vector by a small margin.
- **teacher `Score` selects vector**: accuracy, Brier skill and ECE against the hard label
  improve on test, but **KL to the teacher's soft distribution gets worse**. See the open
  question.
- **rule-labelled `Score` is worse after any S2 fit** than raw, and least bad when it is in the
  fit. Expected: it is near-certain and correctly so, and a cell fitted on genuinely uncertain
  teacher rows softens it.

## Open question (owner)

Selection and fitting both use the hard label (`answer_idx`). For teacher `Score`, the training
target is soft (1,805 of 4,804 teacher rows split across two levels), and the vector bias that
wins on hard-label NLL moves the distribution away from the soft target. Options:

- **A (current):** hard-label NLL everywhere. Simple, one rule; accepts the KL cost on teacher
  `Score` and reports it.
- **B:** for cells whose rows carry soft targets, fit and select on cross-entropy against the
  soft target. Calibrates toward the teacher's uncertainty, which is what `Score` was trained to
  express.

Recommendation: **B for `Score`**, because the soft target is the thing a `Score` distribution
claims to be; A elsewhere, where targets are one-hot and the two coincide. Not implemented until
decided.
