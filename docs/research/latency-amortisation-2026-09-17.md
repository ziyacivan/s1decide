# Why "latency barely changes with more questions" is a claim about token ratios (2026-09-17)

Written at the close of Phase 0, from our own measurements in
`results/20260917-122254-qwen38-27b-unsloth-bnb-4bit-nf4-bf16-latency/latency.json`. The part
about Jev is a **hypothesis with falsifiable predictions**, not a fact about their system;
nothing about their architecture is public. Flagged as such throughout.

## What we measured

Qwen3.8-27B at nf4 with bf16 compute, RTX 3090, 1,554-token state (1,617-token prefill),
20 timed runs per point. The suffix phase fits `a * passes + b * suffix_tokens` at R² = 0.999:

- `a` ≈ 156 ms per forward pass
- `b` ≈ 0.99 ms per suffix token
- a prefix token costs ≈ 0.87 ms — **the same price as a suffix token**

So KV broadcast saves re-reading the *state*. It does not save reading the *questions*: those
tokens still traverse all 64 layers, at full price. Flatness in question count therefore needs

```
prefill + a·passes + b·N·s  ≈  prefill        i.e.   b·N·s  <<  prefill
```

which, because `b ≈` the prefill's per-token cost, reduces to a **pure token-count condition**:
the prefix must be much larger than the sum of all question suffixes. For the retired 2x rule
at N = 64 that came out as `prefix_tokens >~ 61 × tokens_per_question` (ADR 0003).

## Is `a/b` a property of the hardware rather than the model? **No — measured and refuted**

The first draft of this note argued that it was. The reasoning: per-pass cost is weight
streaming (`a ≈ bytes(weights)/bandwidth`), per-token cost is matmuls (`b ≈ 2·params/FLOPS`),
both linear in model size, so `a/b ≈ FLOPS/(2·bandwidth)` cancels the model and leaves the
accelerator's arithmetic intensity. The predicted consequence was that **a smaller model would
not be any flatter**.

That prediction was testable in ten minutes, so it was tested. It is **wrong**.

`results/20260917-invariance-qwen35-08b-bf16-latency/latency.json`, same bench, same 1,554-token
state, same machine, `Qwen/Qwen3.5-0.8B` in bf16:

| | Qwen3.8-27B (nf4) | Qwen3.5-0.8B (bf16) | ratio |
|---|---|---|---|
| `a` — ms per forward pass | 156.4 | 30.8 | 5.1x |
| `b` — ms per suffix token | 0.989 | 0.027 | 36.6x |
| **`a/b` — tokens per pass** | **158** | **1,141** | **0.14x** |
| parameters | 27B | 0.8B | 34x |
| rows per pass (VRAM) | 7 | 16 | |
| 64-question call | 6,662 ms | 291 ms | |
| marginal ms per question at 64 | 81.2 | **3.1** | |

**`b` scales almost exactly with parameter count** (36.6x measured against a 34x parameter
ratio) — that half of the theory holds. **`a` does not**: it grew only 5.1x for a 34x larger
model, far short of the ~12x its weight bytes alone would predict. So `a/b` is not invariant; it
*falls* with model size, and large models are relatively more token-bound.

Two consequences, both more useful than the claim they replace:

1. **Smaller models really are flatter.** Rewriting the crossover with each model's own
   constants, the prefix needed to satisfy the retired 2x rule at 64 questions and ~58
   tokens per question is **~3,600 tokens for the 27B but ~1,800 for the 0.8B** — and the 0.8B
   at our 1,617-token prefix lands at 3.04x, only just outside. A small model on a slightly
   larger state is comfortably flat.
2. **A large part of `a` is our own software, not the hardware.** Weight streaming accounts for
   roughly 21 ms of the 27B's 156 ms. The rest is per-layer launch overhead and — importantly —
   the `copy.deepcopy` of the prefix cache and the `.contiguous()` in `expand_cache`, both of
   which we perform once per pass. At 10 passes that is ~1.5 s of a 6.7 s call. **This is a
   concrete lead for ADR 0003 option C** and it is cheaper than changing the state dtype: reuse
   one pre-expanded cache buffer across passes instead of deep-copying per pass.

## Hypothesis about Jev's flat-latency claim

TypeSafe's documented behaviour (see `jev-landscape-2026-09-17.md` §1) is that "all questions in
one call are evaluated in parallel and in isolation against the same state; adding questions
barely changes latency", with 70–500 ms end-to-end.

Given the *corrected* analysis above, there are now two credible explanations rather than one,
and they compound.

**Model size, which the first draft wrongly dismissed.** A 0.8B model on this same benchmark has
a marginal cost of **3.1 ms per question** against the 27B's 81 ms, and reaches flatness at
roughly half the prefix length. A small decision model on datacenter hardware would look flat on
almost any realistic state. Jev's 70–500 ms end-to-end is consistent with something far smaller
than 27B, and nothing in their published material says otherwise.

**A smaller per-question token cost.** Ours is ~58 tokens per question, of which
**52% is format boilerplate** we render ourselves (`### Question` header, `### Answer` block,
chat tail; 70% for a Noul). Plausible ways their number could be much smaller:

1. **A compact per-question representation.** With a non-causal-LM architecture — a
   classification head, or learned question embeddings — a "question" need not be rendered text
   at all. If a question costs a handful of tokens instead of 58, the crossover falls from
   ~3,600 prefix tokens to a few hundred, and flatness is easy at any realistic state size.
   `pngwn`'s classification-head replica takes this route, and our ADR 0001 explicitly traded it
   away for GGUF portability.
2. **Demos and benchmarks that are state-heavy.** A 4,000-token document with eight questions
   satisfies the ratio comfortably. The published demos (a support ticket, a document, game
   state) are consistent with this without implying anything clever.
3. **Options not re-sent per question.** Our suffix re-renders the option list for every
   question. A system that encodes the option set once, or caches it, pays it once.

Note these are not mutually exclusive, and (1) and (3) are both things we could partially adopt.

### What would confirm or refute this

- **Done (2026-09-17):** the `Qwen3.5-0.8B` run above. It refuted the model-independence
  argument and put model size back on the list of explanations. Recorded rather than quietly
  dropped, because the refuted version is what the first draft of this note claimed.
- **Refutes the remaining hypothesis:** published evidence that Jev's latency is flat on a
  workload with a *short* state and many *verbose* questions. That would mean something we have
  not accounted for.
- **Supports it:** any disclosure of model size, or that questions are embedded rather than
  rendered as text.

### A caution about our own numbers

The 0.8B comparison is bf16 against the 27B's nf4, so `a` and `b` are not like-for-like in
quantization. The parameter-scaling conclusion for `b` survives that (36.6x against 34x is close
enough that dequantisation overhead cannot be doing the work), but the exact `a` ratio should not
be over-read. A cleaner comparison would quantize both identically; it was not worth the GPU time
for a secondary result.

## What we do about it

Nothing in this note changes a locked decision. Three things it does change:

1. It raises the priority of ADR 0003's **option B**: trimming format boilerplate is not merely
   a ~17% latency saving, it moves us toward the regime where the flatness claim becomes true
   for realistic states.
2. It hands **option C** a cheaper first move than the one the ADR named: cut the per-pass cache
   `deepcopy` and `.contiguous()` before touching the state dtype. `a` is largely ours.
3. It means we should **stop treating model size as irrelevant to the latency story**. A 27B is
   the right choice for the project's stated goal — the first open decision model at that scale —
   but we should say plainly that it costs flatness, and quote the 0.8B numbers beside it rather
   than implying the architecture alone delivers flat latency.

**Phase 2 idea, not to be acted on now:** a small same-family front model (e.g. Qwen3.5-0.8B,
3.1 ms marginal per question) answers every question, and the 27B is invoked only on the ones
whose calibrated confidence falls below a threshold — a cascade whose escalation rate is itself
a calibrated quantity. Recorded here so it is not lost; no work scheduled.

It also sharpens what we should claim. "Adding questions barely changes latency" is not
something we can say at a 1.5k state. What we can say, and have measured, is that bundling 64
questions is **15.4x faster than asking them separately**, at **104 ms per question amortised**
and **81 ms marginal** — and state the state size alongside it, as ADR 0003 requires.
