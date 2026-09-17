# Built dataset

<!-- GENERATED from data/processed/manifest.json by data/build/report.py.
     Rebuild with: uv run task data -->

**52,754 rows**, seed `20260917`. Licence reasoning and the eval-only rules are in [dataset-card.md](dataset-card.md); this file is the counts.

## Splits

| split | rows | purpose |
|---|---|---|
| `train` | 38,527 | S1 training |
| `val` | 4,895 | calibration fitting and model selection |
| `test` | 4,975 | Tier-1 in-distribution report |
| `eval` | 4,357 | Tier-1 held-out families and OOD sets |

## Families

A family in `eval` only is either a **held-out family** (in-distribution, never trained on) or an **OOD set**. Both are Tier 1.

| family | licence | role | primitive | `train` | `val` | `test` | `eval` |
|---|---|---|---|---|---|---|---|
| anli | `cc-by-nc-4.0` | ood | choice | — | — | — | 1,000 |
| banking77 | `cc-by-4.0` | train | choice | 15,810 | 2,085 | 2,105 | — |
| clinc_oos | `cc-by-3.0` | train | choice | 16,075 | 1,970 | 1,955 | — |
| commonsense_qa | `mit` | heldout | choice | — | — | — | 1,476 |
| go_emotions | `apache-2.0` | train | noul | 4,753 | 578 | 669 | — |
| mmlu | `mit` | train | choice | 1,162 | 178 | 157 | — |
| ordinal_control | `apache-2.0` | train | score | 727 | 84 | 89 | — |
| pubmedqa | `mit` | heldout | noul | — | — | — | 890 |
| sciq | `cc-by-nc-3.0` | ood | choice | — | — | — | 991 |

## Sources

| family | loaded from | audited as | rows loaded | rows kept | note |
|---|---|---|---|---|---|
| banking77 | `legacy-datasets/banking77` | `PolyAI/banking77` | 4,000 | 4,000 | 77 intents — exercises the two-stage path. PolyAI/banking77 is script-based and unloadable on datasets 4.x; this is HF's parquet mirror of the same CC-BY-4.0 data. |
| clinc_oos | `clinc/clinc_oos` | same | 4,000 | 4,000 | 151 intents including out-of-scope — the highest-cardinality source we have. |
| go_emotions | `google-research-datasets/go_emotions` | same | 6,000 | 6,000 | Multi-label over 28 emotions, so one message yields several bundled Nouls. |
| mmlu | `cais/mmlu` | same | 1,500 | 1,497 | 4-option knowledge questions; the question is the state. |
| commonsense_qa | `tau/commonsense_qa` | same | 1,500 | 1,476 | Held-out family: 5-option commonsense, never trained on. |
| pubmedqa | `qiaojin/PubMedQA` | same | 890 | 890 | Held-out family: biomedical yes/no over an abstract. 'maybe' rows are dropped. |
| anli | `facebook/anli` | same | 1,000 | 1,000 | OOD. Non-commercial, so the licence gate makes training on it impossible. |
| sciq | `allenai/sciq` | same | 991 | 991 | OOD. Non-commercial, as above. |
| ordinal_control | `s1decide/ordinal_control` | same | 900 | 900 | Rule-labelled ordinal control set, generated here (ADR 0005 rule c). No model in the loop, so its labels are exact. |

## Two-stage expansion

**8,000 questions** had more options than the 26 single-token labels and were expanded into the ADR 0001 two-stage form: one yes/no row per sampled candidate, plus one Choice over a shortlist containing the answer. Stage-1 rows are rendered byte-identically to `render_stage1`, so training and inference see the same prompt.

## Tier-2 leakage guard

`pngwn/system-one-decisions` is eval-only (ADR 0004) but is *derived from* sources we now train on, so overlap is possible even though the datasets differ. Comparing **state hashes** against our 10,606 training states finds **217 of 1,750** external rows (12.4%) that must be dropped before any Tier-2 number is reported.

Affected families: `go_emotions`, `mmlu`.

## Licences present

| source | licence | verdict | rows |
|---|---|---|---|
| `PolyAI/banking77` | `CC-BY-4.0` | train | 20,000 |
| `allenai/sciq` | `cc-by-nc-3.0` | eval-only | 991 |
| `cais/mmlu` | `MIT` | train | 1,497 |
| `clinc/clinc_oos` | `CC-BY-3.0` | train | 20,000 |
| `facebook/anli` | `CC-BY-NC-4.0` | eval-only | 1,000 |
| `google-research-datasets/go_emotions` | `Apache-2.0` | train | 6,000 |
| `qiaojin/PubMedQA` | `MIT` | train | 890 |
| `s1decide/ordinal_control` | `Apache-2.0` | train | 900 |
| `tau/commonsense_qa` | `MIT` | train | 1,476 |

Every row in `train` and `val` carries a licence from the ADR 0004 allowlist; the build fails otherwise, before anything is written.
