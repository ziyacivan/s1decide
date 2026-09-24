# Open System One landscape — research snapshot (2026-09-24)

Supersedes the landscape parts of [`jev-landscape-2026-09-17.md`](jev-landscape-2026-09-17.md),
which remains the record of what was known about Jev itself on 17 Sep. Everything here was read
from public pages on **24 Sep 2026**; vendor and author numbers are theirs, not ours, and are
quoted as published. When a source disagrees with another, both are recorded.

## 1. The field in one paragraph

Nine days after TypeSafe launched Jev (15 Sep 2026), the open side of "System One" — typed
`Choice`/`Score`/`Noul` decisions in one forward pass, with probabilities — has gone from two
community replicas to a crowded field. madewithjev's directory counts **35 open alternatives**
(page updated 22 Sep) [1]; the curated *awesome-open-system-one* list, which admits only what
you can run and audit, lists **8** (5 models, 3 evaluation projects) [2]. Latent.Space counted
six clones within two days of launch (19 Sep) [3]. The count depends on what one admits: the 35
include wrappers, demos and a meetup; the 8 do not.

**What that means for this project.** Another decision model is one more row. What the field
lacks — every source below reports its own numbers on its own sets, calibration is almost always
one post-hoc temperature, and comparisons against live Jev are rare — is a **common evaluation**
with proper scoring rules, equal-mass ECE beside Brier skill and a base-rate control, risk-coverage,
and fixed splits. That is the gap `s1decide` fills; our own model is one row in its table.

## 2. The three projects the owner asked to track

### Laya — Convai Innovations [4][5][6]

- **Size and backbone.** 421M parameters: ModernBERT-large (395M) plus a decision head of "2
  transformer layers, an option-marker scorer, and an act/escalate head". A multilingual variant
  on mmBERT-base (322M, 100+ languages).
- **Checkpoints.** `convaiinnovations/laya` (English, 512-token context);
  `convaiinnovations/laya-multilingual` (1024); **`convaiinnovations/laya-typed-decisions`**
  (ModernBERT-large, 1024, fine-tuned on typed-decision workflows).
- **Training: RLCD**, "Reinforcement Learning for Calibrated Decisions" — a strictly proper
  scoring-rule reward (log + spherical, ranked probability score for ordinal questions), shipped
  with the model's package (`pip install laya`). **Also post-hoc temperature**, one per question
  type and option count: ECE 0.466 → 0.081 (`laya`), 0.314 → 0.106 (`laya-multilingual`).
- **Reported.** typed-decisions: accuracy 0.766, ECE 0.213 raw / 0.081 after temperature; AG News
  0.950; DAIR Emotion 0.595. Latency on a T4: 39.5 ms for one question, 158.6 ms for ten
  (`laya`). **CPU inference is supported**, 193–464 ms per question.
- **Licence.** Apache-2.0. Not autoregressive, so our token-logit harness does not apply; it is
  evaluated through its own package (`eval/`, see item 4b of the 24 Sep plan).
- Latent.Space describes its training as "PPO over sequence embeddings" with entropy-based
  confidence [3]; the model card's own account is RLCD [4]. Recorded as a discrepancy between a
  secondary source and the primary one; the card is taken as authoritative.

### Kev — jaredpalmer/kev [7]

- **Current generation.** Rank-16 LoRA plus a small **pointer head** on frozen **Qwen3.5** bases:
  0.8B, 4B and **9B**. The pointer head scores each option's `</opt>` hidden state against the
  question's `<decide>` state. Earlier generation on Qwen3 (0.6B/4B/8B).
  *Secondary sources describe an older Kev-0.5B on Qwen2.5-0.5B* [1][3] — a previous release, not
  a contradiction.
- **Claims verified in CI.** Published accuracy numbers are checked against the committed reports
  they come from (`docs/claims.json`, `scripts/verify_claims.py`) in CI. The same discipline as
  this repository's "no hand-typed numbers" rule, enforced mechanically.
- **External evaluations against live Jev**, as published: SemIf set (144 decisions) Kev-9B 0.917
  vs Jev 0.965; Scienthoon (900 tickets) 0.952 / 0.911 vs 0.897 / 0.914 (routing / tone);
  WANLI-v1 (256) 0.703 vs 0.758; TypeSafe-v1 public evals (102) 0.809 / 0.226 vs 0.891 / 0.125
  (agreement / distance).
- **Calibration: a single temperature fitted in distribution** (≈2.1–2.4 per checkpoint), applied
  at serving; it never changes answers.
- **Automatable share at a 5% error budget, in distribution: Kev-9B 0.45–0.57 vs Jev 0.70.** The
  README attributes the gap to "a fixed temperature can't reorder confidences" — the same
  limitation our `Noul` quantization study found (temperature buys zero decision agreement
  against a shifted prior; per-option bias and vector scaling do).
- Apache-2.0; its local server implements TypeSafe's `/v1/systemone` request format.

### SemIf (formerly "OpenJev") [3][8][9]

- Causal **Qwen3.5 4B and 35B** backbones with a **tiny three-class NLI classifier on the last
  token**. Reads the logits assigned to the permitted answers rather than generating — the same
  family of approach as this project's ADR 0001, plus an NLI head.
- An MLX 4-bit conversion of the 4B with paired-accuracy and token-speed benchmarks exists [9];
  a downstream integration reports a measured score of 0.2273 as "first calibration-shaped
  signal" [10] (context not established here).
- Its 144-decision set is one of Kev's external evaluations above.

## 3. Others worth knowing (one line each, as described by their sources)

- **von** — 395M non-autoregressive, under 15 ms per decision [1][2][11].
- **poorjev** — local-first NLI reproduction; temperature scaling plus conformal abstention;
  "ECE 0.170 to 0.071, cross-validated" [11].
- **NanoJev** — minimal educational replica, 0.6B, full distribution in one forward pass [1][2].
- **Bespoke Nimble** — LoRA on Qwen3.5-9B; "base Qwen improved from 66% to 90%, versus 93% for
  Jev" [3].
- **Evaluation projects**: jev-benchmarks, jev-baselines-eval (pre-registered), jev-eval [2].

## 4. What changes for s1decide

1. **Positioning** — the calibration-and-evaluation reference for System One models. The
   comparison table is the product; `s1decide`'s adapter is one row. (`docs/announce/`.)
2. **External models in `eval/`**, same slices and metrics for every row: Laya-421M and its
   typed-decisions checkpoint first (CPU), Kev-9B on the GPU after S1.
3. **Two points from the field that match our own findings**: a single temperature cannot fix a
   shifted prior (Kev's automatable-share gap; our `Noul` quantization study), and published
   numbers should be checked by CI against committed reports (Kev's `verify_claims.py`; our
   `results/` rule).

## Sources (read 2026-09-24)

1. madewithjev — "Is Jev open source? No, but 35 open alternatives are" (updated 22 Sep 2026): https://madewithjev.com/open-source-jev
2. awesome-open-system-one: https://github.com/MorrisZJ/awesome-open-system-one
3. Latent.Space, "[AINews] Here are 6 Clones of Jev in 2 days" (19 Sep 2026): https://www.latent.space/p/ainews-here-are-6-clones-of-jev-in
4. Laya model card: https://huggingface.co/convaiinnovations/laya
5. Laya repository: https://github.com/NandhaKishorM/laya
6. Laya site: https://laya.convaiinnovations.com/
7. Kev README: https://github.com/jaredpalmer/kev
8. SemIf 4B MLX 4-bit: https://huggingface.co/vinci00/semif-qwen3.5-4b-mlx-4bit
9. SemIf MLX benchmarks: https://github.com/VinciGit00/semif-qwen3.5-4b-mlx-4bit
10. allternit-platform PR #692 (SemIf head): https://github.com/Gizziio/allternit-platform/pull/692
11. DEV, "Open-Source Jev Alternatives: Run Typed, Calibrated LLM Decisions Locally" (20 Sep 2026): https://dev.to/rupesh_poojary_ce8e5e7994/open-source-jev-alternatives-run-typed-calibrated-llm-decisions-locally-4dfb

## Addendum (same day): training overlap and TypeSafe's public evals

Read 2026-09-24 for the fair-ground evaluation; the per-family marks are
`eval/contamination.json`, which the report copies onto every row.

**Kev** (repo @ `c9c1f85`, card @ `2629c06`). Released models use `decision-v7`: "10,000 examples
from ten public datasets, 896 generated policy examples, and 1,680 examples from 60 generated
rule structures" [7]. `evals/v7/decision-v7/manifest.json` lists `trainable_sources` agnews,
amazon, banking77, boolq, dbpedia14, imdb, mnli, sst5, trec, yelp, and `eval_only_sources` mmlu,
emotion, tweet_offensive, qnli, paws, sciq. `kev/data.py` L157–158 trains BoolQ from `train`
(dev/test from `validation`) and MNLI from `train` (eval from `validation_matched`); L285: "MMLU
and SciQ stay eval-only … SNLI (MNLI sibling) [is] excluded entirely." The 9B card: "No Jev
outputs were used for training."

**Laya** (HF @ `55cf4c4`, GitHub NandhaKishorM/laya @ `23a1752`). No `datasets:` metadata; the
full mix is not published. The GitHub README marks AG News and **BoolQ "in training mix"**, DAIR
Emotion and SST-5 held out; `bench_apps.py` marks banking77 `in_training=False`. The
`typed-decisions` checkpoint is fine-tuned on `LocalLLaMA/typed-decisions` train — a synthetic
benchmark whose card says it "is not affiliated with TypeSafe", labelled by an unnamed ~4B
teacher. **It is not TypeSafe's public evals** despite sharing the four workflow names.

**TypeSafe public workflow evals** — https://evals.typesafe.ai. Four workflows (invoice 150
cases, security 240, agent-trace 117, customer service 204), **but only 5 cases per workflow are
published (20 total)**, as `<workflow>-cases.js` data files carrying per-case reference
distributions and three models' answers (including TypeSafe's own, `typesafe:v13_snowy_elephant`).
**No licence or terms are stated** on the page or in the files. Reference-label caveat, verbatim:
"Instead of debating the correctness of the harness and labels, we assume that the code is
correct, and measure against the current smartest large models. For this eval, the reference
labels are generated via an average of the responses of GPT-6 Astra and Claude Fable 5.1, both
at high thinking, answering every question in the harness. All other models are evaluated using
the provider's default reasoning settings." Jev's published headline is *end-to-end case
accuracy* over all cases (67.8% overall; security 61.7%, agent-trace 71.6%, invoice 61.8%,
customer service 76.0%), not per question. **The live files have changed** since SemIf froze
them: Kev's copy (`evals/external/typesafe-v1/development.jsonl`, 102 questions over the 20
cases, 66 `Noul` + 36 `Choice`, no `Score`) pins snapshot hashes the live site no longer
matches. Kev reports agreement / TVD on it: live Jev 0.891 / 0.125, the published TypeSafe
answers 0.883 / 0.127, Kev-9B 0.809 / 0.226 on 89 answered rows (13 exceed its context) [7].

**Other published Jev numbers on public sets** (third-party, not TypeSafe): Kev-9B card, Kev /
Jev: QNLI 0.93 / 0.93, SciQ 0.96 / 0.99, MMLU 0.74 / 0.90; github.com/OmarMujahid/jev-decision-bench
(jev-1.13.0, ~200 items per task): BoolQ AUROC 0.97, MNLI 0.84, ANLI R3 0.66, CommonsenseQA
0.87. None found for SNLI, PubMedQA, MASSIVE fr/ja.
