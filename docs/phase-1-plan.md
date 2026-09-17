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

## Step 3 — S1 QLoRA (training-engineer)

`train/sft_lora.py`, `train/configs/sft_3090.yaml`, `uv run task smoke`.

ADR 0002's conditions, each an assertion rather than a comment: **no packing / no
`padding_free`** on transformers < 5.9, **bf16 compute never fp16**, LoRA on attention + MLP
only, `offload_embedding=True`, `use_gradient_checkpointing="unsloth"`. Ordinal loss for `Score`
(currently the weakest primitive). Log VRAM peak and tokens/s.

Also required: **a test that fails if packing or `padding_free` is enabled for a GDN hybrid model
on transformers < 5.9**, so the silent state-leak finding cannot be undone by a version bump.

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
