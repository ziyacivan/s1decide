# ADR 0006 — Release scope: the name is final, and v0.1 is a 3090 pre-release

- **Status:** **accepted** (2026-09-17)
- **Date:** 2026-09-17, Phase 1
- **Deciders:** project owner (human)
- **Supersedes:** the "working name" framing at the top of `CLAUDE.md`
- **Affects:** `CLAUDE.md` definition of done, `docs/model-card.md`, every artefact id we publish

## Context

Two questions had been left open and were starting to cost us. The project name was labelled a
"working name, rename freely", which is fine while nothing is published and expensive the moment
anything is: a Hugging Face model id, a PyPI package and a GitHub repo are all effectively
permanent once other people link to them. And the definition of done was written before we knew
what the 3090 could actually do, so it silently required a rented H100 for two of its items —
which meant v0.1 could not ship at all until someone spent money.

## Decision 1 — The name is `s1decide`, final

`s1decide` is the project name, the PyPI package, the GitHub repository and the Hugging Face
organisation and model prefix. **No rename.** The "rename freely" note in `CLAUDE.md` is retired
by this ADR.

Nothing in the name, the description, the model card, the repository topics or any marketing
copy references **Jev**, **TypeSafe** or **"System One Model"**. That rule already existed as a
hard rule in `CLAUDE.md`; this ADR extends it explicitly to the published artefact identifiers,
which are the places it is easiest to violate by accident and hardest to fix afterwards.

Referring to those systems in **documentation and comparisons remains expected**, and the
head-to-head evaluation is a deliverable. The distinction is between *naming ourselves after
them*, which we never do, and *measuring ourselves against them*, which is the point.

### Consequences

- `s1decide` on PyPI, `s1decide/*` on the Hub, `s1decide` on GitHub.
- A test asserts the forbidden strings appear in no packaging metadata or artefact id.
- "System-One-style decisions" stays as the expansion of the name in prose. "System One Model"
  does not, being someone else's product name.

## Decision 2 — v0.1 is the 3090-trained S1, released as a **pre-release**

v0.1 is **Stage 1 QLoRA on the 3090 plus Stage 2 post-hoc temperature scaling**, published as a
pre-release rather than a stable release. Two items move out of its definition of done and into
v0.2, and both move for the same reason: they require hardware this project does not own.

| moved to v0.2 | why |
|---|---|
| **BF16 row of the quantization table** | A BF16 27B forward pass does not fit in 24 GiB. Producing this row means renting an H100, which costs money and needs an explicit "go". |
| **S3 GRPO calibration RL** | Already marked "Linux GPU only" in `CLAUDE.md`'s locked decisions. It was never a 3090 item. |
| **Q8_0 row** (amended 2026-09-24) | A Q8_0 27B GGUF is ~29 GB and does not fit in 24 GiB; on the 3090 it would run with CPU offload, several times slower, and a partly offloaded row is not the deployment anyone would pick on this card. It joins BF16 on the H100. |

**v0.1 therefore ships the Q5_K_M / Q4_K_M quantization table only** (Q8 removed by the
2026-09-24 amendment below), each row with its own per-(primitive, bucket) S2 (ADR 0008), and the model card
says so in those words: **BF16 was not measured**, and **calibration is post-hoc temperature
scaling, not learned**. Neither is a caveat buried in a footnote; both are stated where the
numbers are.

**Everything else in the definition of done stays in v0.1**, including the two items that might
look like H100 work and are not:

- **Fair LLM baselines** — run in 4-bit on the 3090, against the same 4-bit deployment the
  quantization table covers. A baseline measured at the deployment quantization is arguably the
  more honest comparison anyway.
- **TypeSafe head-to-head** — likewise 4-bit, on the strict common subset, with the
  reference-label caveat stated verbatim.

### Why pre-release rather than 1.0

Calling it a pre-release is not modesty, it is accuracy about what is missing. The model card's
quantization table will have a visible hole in it, and the calibration is the post-hoc kind. A
stable 1.0 would imply neither is true. A pre-release lets the work be used, cited and
criticised now, which is worth more than waiting for a rented GPU.

### Consequences

- `CLAUDE.md`'s definition of done is retitled **v0.1 (pre-release)** and gains a **Deferred to
  v0.2 (H100)** section, so the moved items stay visible rather than disappearing.
- `docs/model-card.md` exists from now, with every number as a placeholder pointing at the
  `results/` path that will fill it. A slot that already exists cannot be quietly skipped, and
  no number ever gets typed by hand into the card (`CLAUDE.md` hard rule).
- The v0.2 scope is **BF16 quantization row + S3 GRPO**, both on rented H100, both needing an
  explicit "go" before anything is spent.

## What this ADR does not decide

- **When** v0.1 ships. That waits on Phase 1 Step 3 (S1 QLoRA) and the evaluation that follows.
- **Whether** we rent an H100 at all. v0.2 is a scope, not a commitment.
- The version number after a hypothetical v0.2. Not worth deciding now.

## Amendment 2026-09-24 — Q5_K_M and Q4_K_M only; the merge path; byte identity (owner decision)

- **v0.1's quantization table is Q5_K_M and Q4_K_M**, each with its own per-(primitive, bucket)
  S2 fitted on `val` at that quantization (ADR 0008). **Q8_0 moves to v0.2** with BF16.
- **Merge path.** First choice: a shard-by-shard CPU merge of the LoRA into the BF16 base with
  bounded memory, then convert and quantize. If that does not fit in this machine's 63 GB of
  RAM, the fallback is llama.cpp's LoRA-adapter route: the base quantized on its own, the
  adapter as a GGUF LoRA applied at runtime. That artefact is **numerically distinct from
  merge-then-quantize** (the adapter is not quantized with the base weights), and the model card
  says so in those words if it is what ships.
- **What ships is byte-identical to what was calibrated and evaluated.** Each released GGUF (and
  adapter GGUF, on the fallback route) is recorded by SHA-256 in the run that fitted its S2 and
  the run that evaluated it; a file whose hash is not in both is not released.
