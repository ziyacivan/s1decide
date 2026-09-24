# Fair-ground evaluation — 2026-09-24

`eval/fair_ground.py`. The `eval` split: two held-out families (CommonsenseQA, PubMedQA), four
OOD sets (ANLI r1, SciQ, BoolQ, SNLI) and two unseen languages (MASSIVE fr-FR, ja-JP, through the
two-stage path), 18,557 rows (`slice-eval.json`). Every model scored on the same file, by the same
report code; the cross-model numbers are `results/fair-ground-2026-09-24/table.json`, each model's
full report its own `fair_ground.json`.

| row | how it ran |
|---|---|
| `zeroshot-qwen38-27b` | our loader, no-op LoRA, nf4, batch 4, 67.6 min on the 3090 |
| `s1-3090` | S1 adapter, nf4, batch 4, 67.2 min |
| `s1-3090-s2` | S1 raw logits + the deployed S2 (`results/s1-3090-slices-s2cells/calibration.json`), fitted on the comparison `val` slice — never on these rows |
| `kev-9b` | Kev's own server, bf16, its shipped temperature, 24.0 min |
| `laya`, `laya-typed-decisions` | Laya's package on CPU, as shipped, ~62 min each |

**Contamination** is marked per model and family from each model's own published sources
(`eval/contamination.json`): Kev trained on BoolQ and on MNLI (SNLI's sibling task); Laya lists
BoolQ in its mix and publishes nothing more, so its other rows are `unknown`; the zero-shot base
model's pretraining is unpublished (`unknown` everywhere, and it applies to every row alike).

**Two controls** per family and group: the training base rate (as in every table here) and the
family's own label marginal (the harder bar).

## Reading it

- S1 is ahead of zero-shot on every family, most on PubMedQA, ANLI and the stage-1 rows.
- Against the external models S1 leads everywhere except BoolQ, where Kev-9B — trained on BoolQ —
  is ahead; that row carries its mark.
- The stage-1 rows (fr/ja) are asked in our native phrasing, which Kev is sensitive to
  (`docs/research/stage1-phrasing-sensitivity-2026-09-24.md`); a plain-phrasing stage-1 row for the
  external models on this slice is not run yet.
- S2 helps stage 1 and ANLI/CommonsenseQA a little, and costs a little on BoolQ/PubMedQA/SNLI:
  the `noul` cell's vector fit, chosen on in-distribution val, does not transfer perfectly OOD.

Peak VRAM for both 27B runs was again 22.335 GiB; the runs were slower than the comparison
slices per row despite shorter rows (67 min for 18.6k rows), which is at the card's measured
slowdown point and is worth an investigation before the full fan-out.
