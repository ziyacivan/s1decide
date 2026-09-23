# Built dataset

<!-- GENERATED from data/processed/manifest.json by data/build/report.py.
     Rebuild with: uv run task data -->

**449,281 rows**, seed `20260917`. Licence reasoning and the eval-only rules are in [dataset-card.md](dataset-card.md); this file is the counts.

## Splits

| split | rows | purpose |
|---|---|---|
| `train` | 90,594 | S1 training |
| `val` | 117,767 | calibration fitting and model selection |
| `test` | 114,563 | Tier-1 in-distribution report |
| `eval` | 126,357 | Tier-1 held-out families and OOD sets |

## Families

A family in `eval` only is either a **held-out family** (in-distribution, never trained on) or an **OOD set**. Both are Tier 1.

| family | licence | role | primitive | `train` | `val` | `test` | `eval` |
|---|---|---|---|---|---|---|---|
| anli | `cc-by-nc-4.0` | ood | choice | — | — | — | 1,000 |
| banking77 | `cc-by-4.0` | train | choice | 25,320 | 32,682 | 32,448 | — |
| clinc_oos | `cc-by-3.0` | train | choice | 25,720 | 59,736 | 59,584 | — |
| commonsense_qa | `mit` | heldout | choice | — | — | — | 1,476 |
| go_emotions | `apache-2.0` | train | noul | 6,340 | 749 | 911 | — |
| massive_de | `cc-by-4.0` | train | choice | 7,704 | 8,113 | 6,344 | — |
| massive_en | `cc-by-4.0` | train | choice | 7,600 | 8,052 | 7,198 | — |
| massive_fr | `cc-by-4.0` | ood | choice | — | — | — | 61,000 |
| massive_ja | `cc-by-4.0` | ood | choice | — | — | — | 61,000 |
| massive_tr | `cc-by-4.0` | train | choice | 7,632 | 7,625 | 7,381 | — |
| mmlu | `mit` | train | choice | 1,166 | 178 | 153 | — |
| mmlu_noul | `mit` | train | noul | 3,582 | 540 | 462 | — |
| ordinal_control | `apache-2.0` | train | score | 726 | 92 | 82 | — |
| pubmedqa | `mit` | heldout | noul | — | — | — | 890 |
| sciq | `cc-by-nc-3.0` | ood | choice | — | — | — | 991 |
| score_teacher | `per-row, inherited from the state's source` | train | score | 4,804 | — | — | — |

## Sources

| family | loaded from | audited as | rows loaded | rows kept | note |
|---|---|---|---|---|---|
| banking77 | `legacy-datasets/banking77` | `PolyAI/banking77` | 4,000 | 4,000 | 77 intents — exercises the two-stage path. PolyAI/banking77 is script-based and unloadable on datasets 4.x; this is HF's parquet mirror of the same CC-BY-4.0 data. |
| clinc_oos | `clinc/clinc_oos` | same | 4,000 | 4,000 | 151 intents including out-of-scope — the highest-cardinality source we have. |
| massive_en | `AmazonScience/massive` | same | 1,200 | 1,200 | English. Utterances 0-1,199 of the locale. The control locale: the other four are read against it. |
| massive_tr | `AmazonScience/massive` | same | 1,200 | 1,200 | Turkish. Utterances 1,200-2,399, disjoint from the other locales. Agglutinative and Latin-script; the non-English language we most want to work, and the reason MASSIVE is in the corpus at all. |
| massive_de | `AmazonScience/massive` | same | 1,200 | 1,200 | German. Utterances 2,400-3,599, disjoint from the other locales. A second Latin-script European language, so 'multilingual' is not one language plus English. |
| massive_fr | `AmazonScience/massive` | same | 1,000 | 1,000 | OOD, unseen language, NEAR: same script and family as the training locales. Parallel to massive_ja on purpose — same utterances, so the near/far difference is language and script alone. |
| massive_ja | `AmazonScience/massive` | same | 1,000 | 1,000 | OOD, unseen language, FAR: different script, no shared family, different tokenisation behaviour. Parallel to massive_fr, as above. |
| go_emotions | `google-research-datasets/go_emotions` | same | 8,000 | 8,000 | Multi-label over 28 emotions, so one message yields several bundled Nouls. |
| mmlu | `cais/mmlu` | same | 1,500 | 1,497 | 4-option knowledge questions; the question is the state. |
| mmlu_noul | `cais/mmlu` | same | 4,584 | 4,584 | Is this answer correct? One true statement plus two sampled distractors per question. Same questions as the `mmlu` Choice family, so a state teaches both. |
| commonsense_qa | `tau/commonsense_qa` | same | 1,500 | 1,476 | Held-out family: 5-option commonsense, never trained on. |
| pubmedqa | `qiaojin/PubMedQA` | same | 890 | 890 | Held-out family: biomedical yes/no over an abstract. 'maybe' rows are dropped. |
| anli | `facebook/anli` | same | 1,000 | 1,000 | OOD. Non-commercial, so the licence gate makes training on it impossible. |
| sciq | `allenai/sciq` | same | 991 | 991 | OOD. Non-commercial, as above. |
| ordinal_control | `s1decide/ordinal_control` | same | 900 | 900 | Rule-labelled ordinal control set, generated here (ADR 0005 rule c). No model in the loop, so its labels are exact. |
| score_teacher | `score_teacher` | same | 4,804 | 4,804 | Two open-weight teachers labelling a five-level rubric, kept where they agreed exactly or within one level. Exact agreement is a hard target; one level apart is a soft target split equally across the two levels. See docs/research/teacher-agreement-2026-09-22.md. |

## Two-stage expansion

**13,600 questions** had more options than the 26 single-token labels and were expanded into the ADR 0001 two-stage form: one yes/no row per sampled candidate, plus one Choice over a shortlist containing the answer. Stage-1 rows are rendered byte-identically to `render_stage1`, so training and inference see the same prompt.

## Tier-2 leakage guard

`pngwn/system-one-decisions` is eval-only (ADR 0004) but is *derived from* sources we now train on, so overlap is possible even though the datasets differ. Comparing **state hashes** against our 14,265 training states finds **207 of 1,750** external rows (11.8%) that must be dropped before any Tier-2 number is reported.

Affected families: `go_emotions`, `mmlu`.

## Effective training mix

Stage-1 questions keep the positive plus **6 negatives** in `train` only, chosen **at random**. `val` and `test` keep the **full fan-out**, because the ratio they carry (~96:1 no:yes) is the one the deployed two-stage path faces, and a temperature fitted on a subsample would be fitted to a distribution we never serve.

| group | rows | raw share | effective share | target | oversample |
|---|---|---|---|---|---|
| `choice` | 10,413 | 11.5% | **30.0%** | 30% | 2.61x |
| `noul` | 9,922 | 11.0% | **25.0%** | 25% | 2.28x |
| `score` | 5,530 | 6.1% | **10.0%** | 10% | 1.64x |
| `stage1` | 64,729 | 71.4% | **35.0%** | 35% | 0.49x |

### Per primitive, stage and family

`rows before` is the full fan-out the expansion would have produced; `rows after` is what training keeps. `effective` is the share of the weighted draw.

| primitive | stage | family | rows before | rows after | raw | effective |
|---|---|---|---|---|---|---|
| `noul` | genuine | `go_emotions` | 6,340 | 6,340 | 7.0% | **12.5%** |
| `noul` | genuine | `mmlu_noul` | 3,582 | 3,582 | 4.0% | **12.5%** |
| `noul` | stage-1 | `massive_en` | 57,000 | 6,650 | 7.3% | **7.0%** |
| `noul` | stage-1 | `massive_tr` | 57,240 | 6,678 | 7.4% | **7.0%** |
| `noul` | stage-1 | `massive_de` | 57,780 | 6,741 | 7.4% | **7.0%** |
| `noul` | stage-1 | `banking77` | 243,705 | 22,155 | 24.5% | **7.0%** |
| `noul` | stage-1 | `clinc_oos` | 485,465 | 22,505 | 24.8% | **7.0%** |
| `choice` | genuine | `banking77` | 3,165 | 3,165 | 3.5% | **5.0%** |
| `choice` | genuine | `mmlu` | 1,166 | 1,166 | 1.3% | **5.0%** |
| `choice` | genuine | `clinc_oos` | 3,215 | 3,215 | 3.5% | **5.0%** |
| `choice` | genuine | `massive_en` | 950 | 950 | 1.0% | **5.0%** |
| `choice` | genuine | `massive_tr` | 954 | 954 | 1.1% | **5.0%** |
| `choice` | genuine | `massive_de` | 963 | 963 | 1.1% | **5.0%** |
| `score` | genuine | `ordinal_control` | 726 | 726 | 0.8% | **5.0%** |
| `score` | genuine | `score_teacher` | 4,804 | 4,804 | 5.3% | **5.0%** |


## Licences present

| source | licence | verdict | rows |
|---|---|---|---|
| `AmazonScience/massive` | `CC-BY-4.0` | train | 190,122 |
| `PolyAI/banking77` | `CC-BY-4.0` | train | 91,816 |
| `allenai/sciq` | `cc-by-nc-3.0` | eval-only | 991 |
| `cais/mmlu` | `MIT` | train | 6,872 |
| `clinc/clinc_oos` | `CC-BY-3.0` | train | 145,945 |
| `facebook/anli` | `CC-BY-NC-4.0` | eval-only | 1,000 |
| `google-research-datasets/go_emotions` | `Apache-2.0` | train | 8,794 |
| `qiaojin/PubMedQA` | `MIT` | train | 890 |
| `s1decide/ordinal_control` | `Apache-2.0` | train | 1,375 |
| `tau/commonsense_qa` | `MIT` | train | 1,476 |

Every row in `train` and `val` carries a licence from the ADR 0004 allowlist; the build fails otherwise, before anything is written.
