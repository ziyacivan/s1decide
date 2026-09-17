# Training-data source licence audit (2026-09-17)

Phase 1 Step 1, role `data-engineer`. Every source named in `AGENTS.md` for the data pipeline,
plus replacement candidates for the gaps that opened, resolved against the **declared licence in
the Hugging Face Hub API** (`cardData.license`) and classified by `data/build/licences.py` per
ADR 0004.

**No dataset was downloaded.** This is metadata only. Ambiguous cases were tagged `eval-only` or
`refused` and collected into ADR 0005 rather than stopping the audit (Phase 1 plan, amendment D).

## Verdicts

`train` = on the ADR 0004 allowlist. `eval-only` = recognised NC/ND. `refused` = anything else,
including unknown, "other", and multi-licence declarations — refused is the default, never
permissive-by-assumption.

### Train-eligible

| Source | Declared | Purpose | Options | Trust |
|---|---|---|---|---|
| `PolyAI/banking77` | `cc-by-4.0` | Choice / banking intent | **77** | first-party |
| `clinc/clinc_oos` | `cc-by-3.0` | Choice / intent + OOS | **150** | first-party (authors) |
| `AmazonScience/massive` | `cc-by-4.0` | Choice / intent, 51 languages incl. Turkish | **60** | first-party |
| `google-research-datasets/go_emotions` | `apache-2.0` | Noul / emotion presence | 2 | first-party |
| `cais/mmlu` | `mit` | Choice / knowledge | 4 | first-party |
| `tau/commonsense_qa` | `mit` | Choice → Noul | 5 | first-party |
| `qiaojin/PubMedQA` | `mit` | Noul / yes-no-maybe | 3 | first-party (author) |
| `SetFit/amazon_reviews_multi_en` | `apache-2.0` | **Score / 1–5 stars** | 5 | **re-upload** |
| `jakartaresearch/google-play-review` | `cc-by-4.0` | **Score / ratings** | 5 | **re-upload** |

### Eval-only (non-commercial)

| Source | Declared |
|---|---|
| `pngwn/system-one-decisions` | `cc-by-nc-4.0` — external Tier-2 set, ADR 0004 |
| `facebook/anli` | `cc-by-nc-4.0` |
| `allenai/sciq` | `cc-by-nc-3.0` |

### Refused (to ADR 0005)

| Source | Declared | Why |
|---|---|---|
| `nyu-mll/multi_nli` | `cc-by-3.0, cc-by-sa-3.0, mit, other` | four licences at once, per genre |
| `fever/fever` | `cc-by-sa-3.0, gpl-3.0` | mixed; GPL on the codebase |
| `google/boolq` | `cc-by-sa-3.0` | share-alike |
| `stanfordnlp/snli` | `cc-by-sa-4.0` | share-alike |
| `allenai/ai2_arc` | `cc-by-sa-4.0` | share-alike |
| `fancyzhx/dbpedia_14` | `cc-by-sa-3.0` | share-alike |
| `Yelp/yelp_review_full` | `other` | Yelp's own terms, academic use |
| `stanfordnlp/sst`, `SetFit/sst5` | `unknown` / none | no declaration |
| `fancyzhx/ag_news` | `unknown` | no declaration |
| `allenai/openbookqa`, `Rowan/hellaswag`, `allenai/winogrande`, `allenai/scitail`, `CogComp/trec`, `SetFit/20_newsgroups`, `cardiffnlp/tweet_eval`, `sealuzh/app_reviews`, `mteb/amazon_reviews_multi` | none / `unknown` | no declaration |

## What this changes

**Two expectations were wrong, in both directions.** The plan predicted MMLU and Yelp as likely
casualties. **MMLU is MIT and survives**; **Yelp is indeed refused**. Worth recording, because it
is the reason the audit is done by API rather than from memory.

**1. The high-cardinality sources are exactly the permissive ones.** banking77 (77 options),
clinc_oos (150), MASSIVE (60) are all train-eligible, and all three are far above the 26
single-token labels. This is strong independent support for **amendment A**: the two-stage path
is not a nice-to-have for one bucket, it is the gate on the majority of our permissively licensed
training data. Without it, our three largest usable Choice sources are unusable.

**2. The Noul sources were nearly wiped out.** Every NLI/fact source named in `AGENTS.md` —
MNLI, ANLI, FEVER, BoolQ — is refused or eval-only. Survivors: `go_emotions` (Apache-2.0,
genuinely Noul-shaped) and `PubMedQA` (MIT, yes/no/maybe). `commonsense_qa` (MIT) can be
converted Choice → Noul. That is thinner than planned but not empty.

**3. Score was rescued, on weaker footing.** Both named Score sources (SST-5, Yelp) are refused.
The two replacements are **third-party re-uploads**, and a re-uploader's licence declaration is a
claim, not necessarily the rights-holder's grant. Score is already our weakest primitive
(0.579 accuracy zero-shot), so it is doubly exposed: least data, least trustworthy provenance.
See ADR 0005.

**4. Share-alike is the single largest category of loss.** SNLI, BoolQ, ARC, DBpedia, FEVER and
part of MultiNLI are all CC-BY-SA. SA permits commercial use, so these are *not* eval-only on
their face; the open question is whether model weights are a derivative work requiring
share-alike. That question is worth answering once, deliberately — ADR 0005.

## Method and its limits

- Source: `HfApi().dataset_info(repo).cardData["license"]`, 2026-09-17.
- **The declaration is the uploader's, not necessarily the rights-holder's.** For first-party
  and institutional repos (PolyAI, AmazonScience, google-research-datasets, cais, allenai, tau,
  the dataset's own authors) this is reasonable evidence. For a third-party re-upload it is
  weaker, and the two Score candidates are exactly that.
- A licence covering a *dataset* may not cover the *underlying content* (Reddit comments in
  GoEmotions, Google Play reviews). We inherit the dataset's terms, which is the normal practice
  and is what every comparable release does, but it is an assumption and is recorded as one.
- Nothing here is legal advice. The gate encodes a conservative reading; ADR 0005 asks a human
  to decide the genuinely ambiguous cases.
