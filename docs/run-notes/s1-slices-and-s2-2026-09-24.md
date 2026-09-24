# S1 adapter on the comparison slices, raw and + S2 — 2026-09-24

`eval/adapter_slice.py`. The final S1 adapter (`results/s1-3090/adapter`) scored on exactly the
val and test slices every external row used, through the trainer's own loader and evaluation
path. **The val output reproduced S1's final checkpoint evaluation** (the script stops if it
does not), so the numbers below and the training-time ones are the same measurement.
Peak VRAM 22.335 GiB during scoring — at the card's measured slowdown point; it did not affect
the result but the next run of this should use `eval_batch_size: 4`.

- raw: `results/s1-3090-slices/metrics.json`
- + S2: `results/s1-3090-slices-s2/metrics.json`, `calibration.json` — temperature per
  option-count bucket, fitted on the val slice (`2`, `3-5`, `6-16` buckets; all above 1, i.e.
  softening), deployed method `temperature` as the nf4 default; vector scaling fitted beside it.

## The comparison (test; numbers in each run's `metrics.json`)

On every primitive S1 is ahead of Kev-9B and both Laya checkpoints in accuracy and Brier skill,
and it is the only row with positive skill on stage-1 rows besides the control's zero.

**This is our home ground and the table must say so.** The slices come from the families S1 was
trained on (held-out states, same distributions); Kev and Laya were trained on other data and
are run as shipped. What the table measures is how each model does on *these* decisions — the
question a caller with this kind of data would ask — not a general ranking. The Tier-1 OOD sets
and held-out families are the fairer ground and are next.

## S2 is mixed, and that is a finding

| test | ECE raw → S2 |
|---|---|
| choice | improves |
| noul | improves |
| stage 1 | ~unchanged (already ~0) |
| rule-labelled `Score` | **worsens** |
| teacher `Score` | improves |

On val — the split it was fitted on — teacher `Score` ECE *worsens*. Brier skill moves by at most
~0.015 in either direction. The likely reason is structural: the `3-5` bucket holds both teacher
`Score` (five genuinely uncertain levels) and rule-labelled `Score` (near-certain, 3–5 levels),
and one temperature has to serve both; minimising NLL over the bucket does not minimise either
group's ECE. **Proposal (ADR, not a tweak): fit S2 per primitive × option-count, not per
option-count alone**, and report both until the choice is made. Not done here.
