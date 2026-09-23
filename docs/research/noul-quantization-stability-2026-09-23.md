# Does 4-bit quantization move a `Noul` answer?

**Date:** 2026-09-23
**Measured on:** 300 `Noul` rows from `test`, 150 genuine and 150 stage-1, seed 20260923
**Runtimes:** `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` (nf4, `HFEngine`) against
`unsloth/Qwen3.8-27B-GGUF / Qwen3.8-27B-UD-Q4_K_M.gguf` (llama-server)
**Results:** `results/quant-noul/{hf,llamacpp,comparison}.json`
**Reproduce:** `uv run task quant-noul --runtime llamacpp` → stop the server →
`--runtime hf` → `--compare`

## The question

[`quantization-label-disagreement-2026-09-18.md`](quantization-label-disagreement-2026-09-18.md)
found nf4 and Q4_K_M agreeing on only **78%** of a five-level `Score` judgement produced by
hundreds of decode steps. The hypothesis queued then was that a `Noul` — one masked logit, one
forward pass, no decode loop to accumulate divergence — would be far more stable, and that the
22% would turn out to be a fact about *generation* rather than about the primitives we publish.

The hypothesis is **half right, and the half that fails is the half that matters for
calibration.**

## The answer

| | value |
|---|---|
| decisions agreeing (p ≥ 0.5) | **88.7%** (266 of 300) |
| — on stage-1 rows | 97.3% |
| — on genuine `Noul` rows | 80.0% |
| mean p(yes), nf4 | 0.295 |
| mean p(yes), Q4_K_M | 0.169 |
| mean shift | **−0.126** |
| rows shifted down | **300 of 300** |
| sign test | p ≈ 1e-90 |
| largest single shift | 0.708 |

**The decision is more stable than `Score`'s label: 88.7% against 78%.** So a single masked logit
does survive quantization better than a judgement assembled over hundreds of decode steps, which
is what the hypothesis predicted.

**The probability is not stable at all.** Every one of the 300 rows moved in the same direction.
In log-odds the shift is **−2.075 nats on average, median −2.047, standard deviation 0.727**, with
a slope against the nf4 log-odds of only +0.12. That is not noise and it is not a temperature
difference: it is close to a **constant per-option bias** on the yes-versus-no gap.

## What it is not

Ruled out before concluding anything, because each of these would produce exactly this shape:

- **Different prompts.** Both runtimes were handed the identical rendered string.
- **A BOS token on one side.** Token counts match exactly, row by row: 102/102, 101/101, 100/100,
  105/105. Qwen's tokenizer adds no BOS and neither does the server.
- **Different option tokens.** `yes` is id 9405 and `no` is id 2083 on *both* sides, asserted by
  `check_label_tokens` before scoring rather than assumed.
- **Sampler penalties.** Re-scored with every penalty explicitly neutral: identical to eight
  decimal places. The server's defaults were already neutral.
- **llama.cpp's concurrency.** Scored sequentially. Concurrent scoring on this server spreads a
  repeated question by 0.027; sequential scoring is bit-identical across ten repeats.
- **A bug in the harness.** There was one, and it was caught: the first version split the joined
  prompt at the last *character* to get a suffix for `HFEngine.score`, which retokenizes
  `…</think>\n\n` from one token (271) into two (198, 198). Fixed by using `render()`'s own
  prefix/suffix boundary, which tokenizes identically to the joined string. The corrected run is
  the one reported here; the broken one showed 298/300 instead of 300/300, so it changed the
  numbers without changing the conclusion.

## Why it matters: temperature cannot fix this, and we already knew we would need more

Scaling a logit by temperature is monotone, so it cannot move a two-option decision across 0.5 —
`sigmoid(z/T) ≥ 0.5` exactly when `z ≥ 0`, whatever `T` is. The measurement confirms the algebra:

| correction applied to Q4_K_M | agreement with nf4's decision | mean \|Δp\| |
|---|---|---|
| none | 0.887 | 0.1257 |
| temperature only, best `T` = 1.84 | **0.887** | 0.1013 |
| per-option bias only, `b` = +2.075 | 0.930 | 0.0494 |
| vector scaling, `T` = 1.18, `b` = +1.35 | **0.953** | 0.0376 |

Temperature scaling buys **exactly zero** decision agreement. A per-option bias buys 4.3 points,
and the two together buy 6.6 and bring the probabilities 3.3× closer.

This is direct empirical support for a decision taken before it was measured: vector scaling
(temperature plus per-option bias) is a second calibration method, and the quantization table
reports **both** methods per row. A table reporting temperature alone would show a quantization's
cost as irreducible when most of it is a shift that one extra parameter removes.

## What this does not say

**It does not say nf4 is right.** Against the true labels, Q4_K_M is the *better* of the two here
— NLL 0.289 against 0.393, Brier 0.094 against 0.126. Correcting Q4_K_M toward nf4 makes it agree
more and score worse. "Agreement with nf4" and "correctness" are different targets and this
experiment measures the first.

Which of the two is closer to the model as trained cannot be settled on this machine, because the
arbiter is BF16 and a BF16 27B forward pass does not fit in 24 GiB. That is the hole v0.1 ships
with ([ADR 0006](../adr/0006-release-scope.md)), and this measurement makes it sharper rather than
smaller: two 4-bit deployments of one model disagree by two nats on every row, and we cannot say
which one drifted.

It also does not separate "nf4 versus Q4_K_M" from "Unsloth's bnb recipe versus Unsloth's GGUF
recipe". Both are dynamic quantizations of the same base by the same publisher, and both are what
a user actually downloads, so the comparison is the practically relevant one — but it is a
comparison of two artifacts, not of two algorithms in isolation.

## Side finding: our own logits arrive at bf16 resolution

`HFEngine` returns logits in the model's compute dtype. At the magnitudes these sit at, bf16's
spacing puts the yes-minus-no gap on multiples of 0.125, so 300 rows produced only **84 distinct
probabilities** — the first ten log-odds are −1.0, 1.375, −0.625, −2.375, −2.0, −1.0, 3.5, −0.375,
−1.125, 0.625, every one a multiple of an eighth. llama.cpp returns float32 and produced 300
distinct values.

This does not explain the shift — 0.125 of granularity against a 2.075 offset — but it is a real
constraint on fine-grained calibration work: an equal-mass ECE bin narrower than 0.125 in log-odds
is measuring the dtype. Worth a look before the reliability diagrams are read too closely.

## Consequences

- The `Noul` hypothesis is recorded as **partly supported**: decisions are more stable than
  `Score` labels, probabilities are not stable at all.
- The quantization table keeps both calibration methods per row, now with a measured reason.
- The model card says that calibration is fitted at the deployment quantization *and* why:
  switching 4-bit schemes shifts the log-odds by about two nats, which temperature cannot undo.
- Open: whether the offset is stable across question families and option counts, or specific to
  the two-option case. Cheap to extend — the scoring path now exists for both runtimes.
