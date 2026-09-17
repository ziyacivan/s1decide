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

## The ratio `a/b` is a property of the hardware, not the model

This is the part that generalises, and it is worth stating because it is counter-intuitive.

- Per-pass fixed cost is dominated by streaming the weights: `a ≈ bytes(weights) / bandwidth`.
- Per-token marginal cost is dominated by the matmuls: `b ≈ 2·params / FLOPS`.

Both scale linearly with model size, so the ratio

```
a / b  ≈  FLOPS / (2 · bandwidth)
```

**cancels the model size** and leaves the accelerator's arithmetic intensity. Our measured
`a/b ≈ 158 tokens`; an RTX 3090's headline bf16 throughput over its memory bandwidth is the
same order (tens of ops per byte), which is consistent once 4-bit dequantisation overhead is
included.

**Consequence: running a smaller model does not make latency flatter.** It makes everything
faster in proportion and leaves the crossover token ratio roughly where it was. Anyone claiming
flat latency is, if this analysis holds, either running a state-heavy workload or sending very
few tokens per question.

## Hypothesis about Jev's flat-latency claim

TypeSafe's documented behaviour (see `jev-landscape-2026-09-17.md` §1) is that "all questions in
one call are evaluated in parallel and in isolation against the same state; adding questions
barely changes latency", with 70–500 ms end-to-end.

Given the analysis above, the most likely explanation is **not** that they have a better cache
trick than ours — the state is the easy part, and we already amortise it fully. It is that
**their per-question token cost is far below ours.** Ours is ~58 tokens per question, of which
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

- **Cheap, and we should do it:** run `uv run task bench` against `Qwen/Qwen3.5-0.8B`, already
  in the local cache. If `a/b` lands near the 27B's ~158 tokens, the model-independence argument
  holds and "use a smaller model" is ruled out as an explanation for flatness. If `a/b` is
  dramatically larger, the argument is wrong and this note needs revising. *Not yet run —
  proposed as a follow-up, not scheduled.*
- **Refutes the hypothesis:** any published evidence that Jev's latency is flat on a workload
  with a short state and many verbose questions. That would mean they have something we have
  not accounted for.
- **Supports it:** any disclosure that questions are embedded rather than rendered as text.

## What we do about it

Nothing in this note changes a locked decision. It does raise the priority of ADR 0003's
option B: trimming format boilerplate is not merely a ~17% latency saving, it moves us toward
the regime where the flatness claim becomes true for realistic states. That is a better
argument for doing it than the milliseconds are.

It also sharpens what we should claim. "Adding questions barely changes latency" is not
something we can say at a 1.5k state. What we can say, and have measured, is that bundling 64
questions is **15.4x faster than asking them separately**, at **104 ms per question amortised**
and **81 ms marginal** — and state the state size alongside it, as ADR 0003 requires.
