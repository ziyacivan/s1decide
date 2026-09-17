# ADR 0001 — Read option probabilities from token logits, not a classification head

- **Status:** proposed
- **Date:** 2026-09-17
- **Deciders:** project owner (human), architect role
- **Supersedes / superseded by:** —

## Context

The product is a decision model: given a `state` and a set of typed questions, return a
probability distribution over the caller's options for each question, plus a confidence,
without generating text. Two implementations of "probability over a caller-supplied option
set" are available, and the community has already shipped both:

1. **Token-logit readout.** Map each option to a single token, place the model at one answer
   position, read the logit vector there, mask it to the allowed tokens, softmax. This is the
   SALSA recipe (arXiv 2510.22691), and it is what `rorshopping/jev-on-a-laptop` does.
2. **Classification head.** Attach a sequence-classification head over (state, question, option)
   triples and softmax its outputs. This is what `pngwn/system-one-qwen3.5-4b-scorer` does, and
   it reports a good result: temperature scaling took ECE from 0.135 to 0.044 at unchanged
   accuracy.
   (Both replicas: `docs/research/jev-landscape-2026-09-17.md` section 4.)

The decision is already recorded as locked in `CLAUDE.md` ("Locked architecture decisions", 1).
This ADR exists to write down *why*, so that the cost of revisiting it is visible.

## Decision

Use the **token-logit readout**. The released artefact stays a standard causal LM with a LoRA
adapter: no new tensors, no custom `forward`, no runtime-specific head.

Concretely:

- Options are mapped to single tokens (`A`…`Z`, digits, or `yes`/`no`), verified single-token
  against the base tokenizer both with and without a leading space.
- The mapping option↔token is generated per example and the option order is shuffled at train
  time, so the model cannot learn a positional or lexical prior over labels.
- Inference reads the logits at exactly one position, masks to the allowed token ids, and
  softmaxes. Nothing is decoded; `.generate()` with `max_new_tokens > 1` appears nowhere.
- `Noul` is a 2-option `Choice` over `yes`/`no`. `Score` is an ordinal `Choice` trained with an
  ordinal-aware loss and reported as a full distribution plus an expected level.
- Above 26 options, a two-stage scheme (independent per-option scoring, then a `Choice` over the
  top-k) keeps the single-token constraint intact.

## Rationale

**Adoption is the deciding factor.** A plain causal LM converts to GGUF and runs unchanged in
llama.cpp, vLLM, MLX, Ollama and every other runtime the community already has. A
classification head does not: it needs a custom runtime, which for a ≥ 27B model — where most
users will only ever run a quantized GGUF — means most users cannot run it at all. Since the
stated goal is "the first ≥ 27B open alternative", a format nobody can load defeats the project.

**Type safety is preserved either way.** Masking the logit vector to the allowed token ids makes
an off-schema answer impossible by construction, exactly as a head does. Neither approach makes
the model *right*; both make it *well-formed*. We should describe it that way, per the critique
in `docs/research/jev-landscape-2026-09-17.md` section 3.

**Calibration is not obviously worse.** The published head-based result calibrates well, but
post-hoc temperature scaling applies identically to masked logits, and RLCR (arXiv 2507.16806)
operates on the model's own output distribution, so the Stage-3 path stays open. This is an
assumption we must *measure*, not assert — see "Consequences".

**It composes with the KV-broadcast design.** One answer position per question means each
question is a one-row suffix over a shared prefix, which is precisely what locked decision 2
needs. A head over (state, question, option) triples multiplies the row count by the option
count instead.

## Alternatives considered

- **Classification head.** Rejected for runtime portability, as above. It remains the better
  choice for anyone who controls their own serving stack, and `pngwn`'s model is the reference
  for that route.
- **Constrained decoding of the option string.** Rejected: multi-token options mean multiple
  forward passes, latency grows with option length, and the probability of an option becomes a
  product over tokens that is not comparable across options of different lengths. It is,
  however, exactly the right thing for the *baseline* rows in the eval (`eval/baselines/`).
- **Per-option `Noul` scoring for every question.** Rejected as the default: it costs one row
  per option rather than one per question. Retained only as the first stage of the > 26-option
  path.

## Consequences

- The vocabulary imposes a hard ceiling on single-token labels, so high-cardinality questions
  need the two-stage path. `CLAUDE.md` mirrors the documented 255-option limit; the actual limit
  must be verified against current vendor docs before we publish it.
- Label tokens must be verified per base model. Changing the base model can silently break the
  single-token property — hence a test, not a comment (`tests/test_tokens.py`, Step 2).
- We inherit whatever lexical priors the base model attaches to `A`/`B`/`yes`/`no`. Shuffling
  option order per example is a mitigation, not a proof; the eval must include a base-rate
  negative control, and `shamazharikh/qwen-rlcd` warns specifically that a base-rate predictor
  can score a deceptively good ECE.
- **Open question to be closed by measurement:** whether token-logit readout calibrates as well
  as a head at equal accuracy. Step 5 produces the zero-shot calibration baseline; if the
  post-LoRA, post-temperature ECE is materially worse than the published head-based numbers on
  comparable data, this ADR gets revisited rather than quietly tolerated.
