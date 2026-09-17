# Phase 1 plan (approved 2026-09-17, with amendments A–D)

Plan of record. Phase 0 is tagged `phase-0`. Each step stops and reports before the next begins.

## Sequence

```
Step 1  licence gate + source audit            (data-engineer)      ── blocks everything
   │
   ├── Step 2a  two-stage high-cardinality path (architect, then inference-engineer)
   │            architect specifies the format FIRST; it is an input to 2b
   │
   └── Step 2b  data pipeline                   (data-engineer)
   │
Step 3  S1 QLoRA smoke run, then S1 proper      (training-engineer)
```

**Amendment A.** The two-stage path moves *before* the S1 smoke run. The 17–77 option bucket is
13.1% of the external test set and banking77 is likely our largest permissively licensed source,
so S1 must train on and be evaluated against the full distribution rather than one that silently
excludes the hardest bucket. 2a and 2b run in parallel — different owners, different files — but
**the architect specifies the two-stage prompt format in `docs/format-spec.md` before the
data-engineer emits any high-cardinality row**, because the training-data shape depends on it:

- stage 1: per-option Noul scoring (`is this option the answer?`), one row per option;
- stage 2: `Choice` over the top-k survivors.

## Step 1 — Licence gate and source audit (data-engineer)

Built before any row is fetched, so it can refuse sources rather than discover problems after
them. Specified in ADR 0004.

- `data/build/licences.py`: `TRAIN_LICENCE_ALLOWLIST`, `classify_licence(spdx)`,
  `assert_train_splits_are_licensed(rows)`.
- `tests/test_licences.py` — the gate is a test, not a comment. Runs in the normal CPU suite.
- Audit every source named in `AGENTS.md`, resolving each to an SPDX identifier, dated, in
  `docs/research/source-licences-<date>.md`.

**Amendment D.** Unknown or ambiguous ⇒ tag `eval-only` and **continue**; never stop per source.
All ambiguous cases are collected into **ADR 0005** at the end of Step 1, with a recommendation
per source. Expected casualties (Yelp, MMLU) are fine — a thin training set is a finding, not a
failure.

## Step 2a — Two-stage high-cardinality path (architect → inference-engineer)

ADR 0001's scheme, for questions above the 26 single-token labels.

1. **Architect** extends `docs/format-spec.md` with both stages and bumps `FORMAT_VERSION`
   (folding in ADR 0003 option B's boilerplate trim, so the format changes once, not twice).
   Acceptance for the trim: format overhead ≤ 25% of suffix tokens; baseline 51.7% / 70% Noul.
2. **Inference-engineer** implements it in `decide.py` + `engine/`, with the existing contract
   tests extended to cover it, and adds it as a **separate series in `eval/latency_bench.py`**
   (its cost profile differs by construction: stage 1 is one row per option).

## Step 2b — Data pipeline (data-engineer)

`data/build/{fetch,normalise,augment,split}.py` behind `uv run task data`, emitting the
`CLAUDE.md` jsonl schema into `data/processed/`. Deterministic, idempotent, seeded, `pathlib`,
UTF-8/LF.

Augmentation: shuffle option order per example, paraphrase labels, inject `other` /
`none of the above`, and **bundle 5–30 questions per state** — the external set only ever
bundled up to 4, and the engine's design assumes more. High-cardinality rows follow the Step 2a
format.

Splitting by **state hash**, with held-out *families* and ≥ 2 fully OOD sets reserved and never
touched by augmentation. `docs/dataset-card.md` regenerated from the manifest.

### Amendment B — two-tier eval reporting, always labelled

| Tier | Contents | Purpose |
|---|---|---|
| **Tier 1 (headline)** | our own held-out families + ≥ 2 OOD sets | in-distribution to our training sources |
| **Tier 2 (external)** | `pngwn/system-one-decisions`, eval-only per ADR 0004 | comparability with community replicas |

**Leakage guard.** `pngwn/system-one-decisions` is derived from banking77 and other sources we
will now train on. Every row of the Tier-2 set whose **state hash appears in any of our train
splits is dropped**, and the number dropped is reported in `metrics.json`. This is enforced by a
test, not by care.

## Step 2c — MASSIVE, locale-filtered (data-engineer)

Approved 2026-09-17. MASSIVE loads only from its parquet branch's single `default` config
(1,784,670 rows, all 51 locales), so the loader filters by `locale` and `partition`.

- **Train locales: `en-US`, `tr-TR`, `de-DE`**, capped per locale so MASSIVE cannot dominate.
  Cap: **1,200 source rows per locale**, measured 2026-09-17 against the rebuilt manifest.

  The first proposal was 3,000, and it was wrong. It compared *source* rows against banking77's
  4,000 and ignored two things the manifest makes obvious: each source row expands to ~5 rows
  through the two-stage path, and there are three locales. At 3,000 MASSIVE was **48.2% of
  training** — the largest block in the corpus by a wide margin. The comparison has to be made
  on expanded rows summed across locales.

  | source rows/locale | MASSIVE train rows | share of training |
  |---|---|---|
  | 3,000 | 36,180 | 48.4% |
  | 1,500 | 18,090 | 32.0% |
  | **1,200** | **14,335** | **27.1%** (measured) |
  | 1,000 | 12,060 | 23.8% |

  At 1,200 each locale is ~9% of training — the same weight as go_emotions, which is what the
  original rationale claimed and only now is true — and the three together (27.1%) sit just
  below banking77 alone (29.9%) and clinc_oos (30.4%). That is the intended reading of "must
  not dominate": comparable to the largest other source, not larger than all of them.

- **MASSIVE is a parallel corpus** — the same utterance is translated into all 51 locales and
  keeps its id. Taking the first N rows of each locale would give N meanings three times, not
  3N. The training locales therefore take **disjoint slices** (offsets 0 / 1,200 / 2,400).
  Verified: the first 3,000 train ids are identical across en/tr/de, and the slices now share
  no utterance. The state-hash leakage guard cannot see this — two translations of one sentence
  are different strings — so the slicing is what prevents it, not the guard.
- **Held-out locales: `fr-FR` and `ja-JP`**, never in train or val, forming an **unseen-language
  OOD set**, and taken from the **test partition** so they are different content and not
  translations of training rows (verified: zero utterance-id overlap with the training slices).
  They are parallel to *each other* on purpose, so the near/far difference is language and
  script alone. Residual text overlap with training is 1 row in 1,000 — the French word
  `silence` is also English — and the leakage guard drops it. Chosen deliberately: `fr-FR` is close to the training languages (Latin script,
  Indo-European) and `ja-JP` is far from all of them (non-Latin script, different family). Two
  points on that axis say more than two similar languages would — if accuracy holds on French
  but collapses on Japanese, the failure is script and tokenisation rather than language
  transfer, and we can see which.
- MASSIVE has 60 intents, so it is above the 26-label ceiling and goes through the same
  two-stage expansion.

## Step 2d — Stage-1 class balance (data-engineer) — **blocks S1**

Added 2026-09-17 after the first full build. Stage-1 rows are **~83% of the corpus at roughly
76:1 no:yes**, because expanding a 77- or 151-option question produces one positive and many
negatives. Training on that unmodified teaches the model to answer "no".

1. **Per-question negative subsampling, TRAIN ONLY.** Keep the positive plus `k` hard negatives,
   `k` configurable, **default 6**. "Hard" means the highest zero-shot `P(yes)` from the
   committed baseline run; fall back to uniform random when no baseline is available, and record
   which was used. **`val` and `test` keep the full stage-1 fan-out**, so evaluation still sees
   the real class balance and the real task.
2. **Family-balanced sampling weights** in the training config, with the **effective mix printed
   and saved next to the run**, not just the raw counts.
3. **Manifest and `dataset-build.md` report raw rows and effective training mix separately.**

Acceptance: a test asserting the train stage-1 `no:yes` ratio is **≤ 8:1** after subsampling.

## Step 2g — Genuine `Noul` supply (data-engineer) — **blocks S1**

Added 2026-09-17, after the per-primitive mix was measured for the first time.

A `qtype` count reported `Noul` at **78.9% of training**. The primitive we actually publish was
**9.0%**, all of it from go_emotions. The other 69.9% were stage-1 rows: expanding a 60- or
151-option question emits one `noul`-shaped row per candidate — *"is `card arrival` the intent
of this message?"* — which is option membership, not a claim about the state. The two are
indistinguishable in a `qtype` count and are not the same task.

The gap exists because every NLI source that would normally supply `Noul` — SNLI, BoolQ, FEVER,
MultiNLI — is share-alike and eval-only under ADR 0005.

**Done:**

1. **`primitive_mix()` reports every split per primitive with `stage1` and `genuine` counted
   separately**, plus a `genuine_by_family` breakdown so a primitive carried by one source is
   visible. In `manifest.json` under `by_primitive`.
2. **`mmlu_noul`**, a second genuine-`Noul` source: one true statement (the correct answer) and
   two sampled distractors per MMLU question, MIT, first-party. Deliberately the `validation`
   split — training on MMLU's `test` split would invalidate any later MMLU evaluation of these
   weights, and `auxiliary_train` aggregates ARC and RACE, which are share-alike.
   The same questions already appear as `mmlu` `Choice` rows, so one state teaches two
   primitives and the state-hash split keeps them together.
3. **go_emotions raised 6,000 → 8,000**, now that it is not the only source.
4. **A floor: `MIN_GENUINE_NOUL_SHARE = 0.15`**, asserted against the built manifest, plus
   `MIN_GENUINE_NOUL_FAMILIES = 2`. A floor rather than a target — below it the primitive is
   carried by incidental data rather than trained deliberately.

**Measured after the change:**

| primitive | total | stage-1 | genuine | genuine share |
|---|---|---|---|---|
| `choice` | 10,406 | 0 | 10,406 | 17.9% |
| `noul` | 46,886 | 36,976 | **9,910** | **17.1%** |
| `score` | 727 | 0 | 727 | 1.3% |

Genuine `Noul` now comes from go_emotions (6,340) and mmlu_noul (3,570), at 2.31:1 no:yes.
`Score` stays at 1.3% until Step 2e's teacher run; it is the primitive with no natural corpus.

When Step 2d lands, the floor moves from the raw manifest to the **effective** training mix,
since that is what the model actually sees.

## Step 2d — Stage-1 class balance — **DONE** (2026-09-17)

Implemented in `data/build/balance.py`, reported in `docs/dataset-build.md`.

**The split now happens before the expansion.** `split_by_state` depends only on role and state
hash, both of which a stage-1 row inherits from its parent, so the answer is identical either
way — but doing it first lets the expansion give each split its own fan-out, instead of
materialising 1.2M rows and discarding 94% of them.

1. **Train keeps the positive plus 6 negatives** (`STAGE1_TRAIN_NEGATIVES`), giving **6.00:1**
   no:yes against the 8:1 ceiling. Negatives are currently chosen **at random** and the manifest
   says so: the hard-negative hook takes `{parent_id: {option: P(yes)}}`, and the committed
   zero-shot baseline does not cover these rows, so there is nothing to rank by yet.
2. **`val`, `test` and `eval` keep the full fan-out** — 94.8:1, 96.0:1 and 59.0:1. That is the
   ratio the deployed two-stage path faces, and a temperature fitted on a 6:1 sample would be
   fitted to a distribution we never serve.
3. **Family-balanced sampling weights**, written to `data/processed/sampling_weights.json` and
   copied into the manifest so a run and its corpus cannot disagree. Families are equalised
   within a group; groups are steered toward a target mix and **clipped at 3x oversampling**,
   with any shortfall reported rather than restated as a result.
4. **Raw and effective mixes are reported separately**, per primitive, stage and family.

**Measured:**

| group | rows | raw share | effective share | target | oversample |
|---|---|---|---|---|---|
| `stage1` | 64,729 | 75.5% | **37.8%** | 35% | 0.46x |
| `choice` | 10,413 | 12.1% | **32.4%** | 30% | 2.47x |
| `noul` (genuine) | 9,922 | 11.6% | **27.0%** | 25% | 2.16x |
| `score` | 726 | 0.8% | **2.7%** | 10% | 3.00x (clipped) |

`score` cannot reach 10% inside the cap — it would need 11.8x — and that is the honest reading
of having 726 rows. Step 2e's teacher run is the fix, not a bigger weight.

**The floors now apply to the effective mix**, as approved: genuine `Noul` is 11.6% of rows and
**27.0%** of what the model draws, against the 15% floor from two families.

### Consequence that needs a decision

Full fan-out in `val` and `test` takes the evaluation from ~13 minutes to **~3.3 hours**
(117,767 + 114,563 questions, bundled ~96 to a state). That is the price of evaluating the
distribution we actually deploy. The cheaper alternative is **case-control sampling**: keep all
positives, sample k negatives, and importance-weight them back to the true base rate when
fitting temperature and computing ECE. It is unbiased and would cost ~30 minutes, but it needs
weighted metrics in `eval/metrics.py`. Flagged rather than chosen.

## Step 2e — Teacher-labelled `Score` (data-engineer) — approved, needs a final go

Approved 2026-09-17 including the ~30 GB download and overnight GPU time, on these terms:

- **Teachers run with reasoning ON at low effort, not single-token.** We are distilling
  deliberate System-2 judgements into a System-1 student; a teacher answering in one token is
  not doing the thing we want to copy. Recorded in ADR 0005 and the dataset card.
- **Teacher 1: `Qwen/Qwen3.8-27B` at 4-bit** (already local).
  **Teacher 2: a different family**, open-weight, permissive licence, fits the 3090 at 4-bit.
  Shortlist to verify before downloading: `microsoft/phi-4-reasoning` (MIT),
  Mistral's Apache-2.0 reasoning small model, `allenai/OLMo-2-32B-Instruct` (Apache-2.0).
  **Licence and fit are verified and proposed for approval before any download.**
- **Target 4,000–6,000 rows.** Keep exact-agreement and ±1-neighbour only; record agreement rate,
  ±1 rate, drop rate, both teacher IDs and revisions, both prompt hashes, and the reasoning
  setting.
- **Run overnight**, logging VRAM peak and wall time. **Close Unsloth Studio first.**
- Amendment B's leakage guard applies to every labelled state.

## Step 2f — Tier-2 leakage drop (eval-scientist) — blocks any Tier-2 number

Confirmed 2026-09-17. The first full build found **217 of 1,750** external rows sharing a state
hash with our training data. Before any Tier-2 figure is reported:

- drop the overlapping rows,
- **print the dropped count in every summary**, and
- state in the model card that `pngwn/system-one-decisions` is derived from sources we train on,
  so Tier 2 is **"external" in labelling, not in distribution**.

## Deferred to after the S1 smoke run — bf16 GDN recurrent state (inference-engineer)

The remaining half of ADR 0003 option C. Deferred 2026-09-17 on purpose: it changes numerics,
and the calibration it must not break is **S1's**, which does not exist yet. Checking it against
the zero-shot baseline would be protecting the wrong model. Conditions when it runs are in
ADR 0003 — accuracy and ECE for zero-shot *and* S1, and the broadcast-equivalence test with an
explicit tolerance rather than bitwise identity.

## Step 3 — S1 QLoRA (training-engineer)

`train/sft_lora.py`, `train/configs/sft_3090.yaml`, `uv run task smoke`.

ADR 0002's conditions, each an assertion rather than a comment: **no packing / no
`padding_free`** on transformers < 5.9, **bf16 compute never fp16**, LoRA on attention + MLP
only, `offload_embedding=True`, `use_gradient_checkpointing="unsloth"`. Ordinal loss for `Score`
(currently the weakest primitive). Log VRAM peak and tokens/s.

Also required: **a test that fails if packing or `padding_free` is enabled for a GDN hybrid model
on transformers < 5.9**, so the silent state-leak finding cannot be undone by a version bump.

### Planned experiment: raw-completion render (after S1 LoRA, not before)

A/B a **raw-completion render** — no chat template, no `<think></think>` block — against the
chat render, on identical data, comparing **accuracy and ECE**. The chat tail is 9 tokens per
question and is the entire remaining gap in all-in format overhead; if a raw render holds up,
Noul's minimum suffix drops from 21 tokens to 12.

Run it **after** S1 LoRA, because a trained model may tolerate a format the base model does not,
and the base model's zero-shot behaviour is not evidence about the trained one. **Do not change
the default render** on the strength of this; it is a measurement, and switching would be its own
`FORMAT_VERSION` bump with its own re-run.

### Amendment C — hardware

- **Smoke run** (pipeline correctness, ≤ 5 min): the small same-family model from ADR 0002,
  `Qwen/Qwen3.5-0.8B`, on the 3090.
- **27B QLoRA on the 3090: attempted exactly once**, at `max_seq_length=1024`, batch 1, rank 8,
  gradient checkpointing `"unsloth"`. **On OOM: stop and prepare
  `train/configs/sft_h100_linux.yaml`** rather than tuning further.
- **The H100 run waits for explicit human "go".** It costs money.

## Also approved

- The ~10 min hardware-invariance measurement on the small model, added to
  `docs/research/latency-amortisation-2026-09-17.md`.
- The packing / `padding_free` regression test (Step 3, above).
- Ordinal loss for `Score` stays in Step 3.

## Backup

Push `main` and the `phase-0` tag to a private remote **once the owner supplies the URL**.
Backup only — not a release. **Nothing goes to Hugging Face.** No push has been made.
