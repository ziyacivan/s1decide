# Zero-shot Qwen3.8-27B on the comparison slices — 2026-09-24

`eval/adapter_slice.py score` with no `--adapter`: the trainer's own loader with a freshly
initialised (no-op) LoRA, so the zero-shot model goes through exactly the read path S1 does.
`eval_batch_size` 4 (the new default). Results: `results/zeroshot-27b-slices/metrics.json`;
val 8,329 rows, test 8,350, about 22 minutes per split.

## Batch size changes the logits, slightly

The same weights on the same val slice were scored before, at batch 8, as S1's row-0
evaluation (`results/s1-3090/evals.jsonl`, first entry). Batch 4 against batch 8, per primitive:
KL moves by at most ~0.001 and accuracy by under one point (a handful of rows). The recurrent
layers see left padding, so batch composition is part of the computation. Any comparison across
runs scored at different batch sizes carries this much noise; comparisons in the tables are made
at one batch size per table.

## Peak VRAM is not the eval batch

`score_meta.json` records 22.335 GiB peak at batch 4 — the same as the S1 scoring run at batch 8.
The peak is set elsewhere (model load / first forward), not by the eval batch; halving the batch
did not move it. It sits at this card's measured slowdown point and is worth a look before the
long-context TypeSafe runs.
