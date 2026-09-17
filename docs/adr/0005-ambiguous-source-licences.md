# ADR 0005 — Ambiguous source licences: recommendations per source

- **Status:** proposed — needs a decision from the project owner
- **Date:** 2026-09-17, Phase 1 Step 1
- **Deciders:** project owner (human), data-engineer role
- **Evidence:** `docs/research/source-licences-2026-09-17.md`; gate in `data/build/licences.py`,
  enforced by `tests/test_licences.py`

## Context

ADR 0004 requires every training row to carry an allowlisted licence and refuses anything
unrecognised. The Phase 1 Step 1 audit ran that rule over every source named in `AGENTS.md` plus
replacement candidates. Per amendment D, ambiguous sources were tagged and the audit continued;
they are collected here.

**Nine sources are train-eligible and need no decision.** They are enough to start Phase 1. The
decisions below would *widen* the training set, and one of them (question 3) concerns a source we
already plan to use.

## Question 1 — Are model weights a derivative work under CC-BY-SA?

**Affected:** `google/boolq` (SA-3.0), `stanfordnlp/snli` (SA-4.0), `allenai/ai2_arc` (SA-4.0),
`fancyzhx/dbpedia_14` (SA-3.0), `fever/fever` (SA-3.0 + GPL-3.0), part of `nyu-mll/multi_nli`.
Collectively the largest block of loss, and it removes almost all of our NLI/fact data.

Share-alike permits commercial use — so these are *not* eval-only — but requires derivatives to
carry the same licence. Whether a fine-tuned weight set is a "derivative work" of its training
data is genuinely unsettled, and the industry practice of training on SA data and releasing
permissively is common but not authoritative.

**Recommendation: treat CC-BY-SA as eval-only for now, and revisit only with legal input.**
Adopting SA data would either force an SA release (contradicting `CLAUDE.md`'s Apache 2.0) or
require betting the release on the derivative-work question. We have enough data to proceed
without it, so the cheap and honest move is not to bet. Reversible later; a release is not.

## Question 2 — Sources with no declared licence at all

**Affected:** `fancyzhx/ag_news`, `stanfordnlp/sst` / `SetFit/sst5`, `allenai/openbookqa`,
`Rowan/hellaswag`, `allenai/winogrande`, `allenai/scitail`, `CogComp/trec`,
`SetFit/20_newsgroups`, `cardiffnlp/tweet_eval`, `sealuzh/app_reviews`.

Several of these are widely believed permissive (AG News is routinely described as free for
non-commercial research; several AllenAI sets are Apache-2.0 in their papers or repos) but the
Hub declares nothing, and "widely believed" is not a licence.

**Recommendation: keep refused, with one exception path.** For any source we actually want,
resolve the licence at its *primary* origin (the authors' repository or paper) rather than the
Hub mirror, record the URL and date in `docs/research/`, and add it to the allowlist by name in
a follow-up ADR. Do this only for sources we need — the audit should not become a research
project. **None currently blocks Phase 1.**

## Question 3 — Third-party re-uploads carrying a permissive declaration

**Affected, and this one matters:** `SetFit/amazon_reviews_multi_en` (declared `apache-2.0`) and
`jakartaresearch/google-play-review` (declared `cc-by-4.0`). These are **our only two
train-eligible `Score` sources**, and both are re-uploads rather than first-party releases. A
re-uploader can declare any licence; that declaration does not bind the rights-holder. The
original Amazon Reviews Multi corpus was withdrawn by Amazon, which makes an Apache-2.0
re-declaration of it questionable on its face.

`Score` is already the weakest primitive (0.579 accuracy zero-shot, ECE 0.240). So the primitive
with the least data also has the least defensible provenance.

**Recommendation — my preference, in order:**

1. **Do not rely on either re-upload.** Instead, **construct `Score` data from sources we already
   trust**: derive ordinal rubrics from `PolyAI/banking77`-style confidence, and generate
   synthetic ordinal tasks over permissively licensed states (structured JSON, tickets) with an
   open-weight teacher, per `CLAUDE.md`'s synthetic-data rule. This keeps provenance clean and is
   within the pipeline we are building anyway.
2. If a real ordinal corpus is wanted, find one with a **first-party** permissive licence and add
   it by name.
3. Use `jakartaresearch/google-play-review` (CC-BY-4.0) only if 1 and 2 fail, and record the
   provenance caveat in the model card verbatim.

**Not recommended:** `SetFit/amazon_reviews_multi_en`, despite being the most convenient option.

## Question 4 — Dataset licence versus underlying content

**Affected: everything, including sources we plan to use.** `go_emotions` is Apache-2.0 but the
text is Reddit comments; `google-play-review` is CC-BY-4.0 over user-written reviews; MASSIVE is
CC-BY-4.0 over crowd-written utterances.

Inheriting the dataset's declared terms is standard practice and is what every comparable open
release does. It is nonetheless an assumption.

**Recommendation: accept, and state it explicitly in the dataset card** rather than leaving it
implicit. No code change; a sentence that makes the assumption visible.

## Decision

*Awaiting the project owner.* Phase 1 Step 2 proceeds on the nine train-eligible sources
regardless of the outcome; nothing here blocks it. The one answer that would change near-term
work is **question 3**, since it determines where `Score` training data comes from.

## Consequences if the recommendations are accepted as written

- Training sources: banking77, clinc_oos, MASSIVE, go_emotions, MMLU, commonsense_qa, PubMedQA,
  plus synthetic ordinal data for `Score`.
- `Choice` is well supplied and heavily high-cardinality — **which makes the two-stage path
  (amendment A) load-bearing, not optional**.
- `Noul` is thinner than planned: go_emotions, PubMedQA, and Choice→Noul conversions.
- `Score` has **no natural corpus** and will be largely synthetic. That is a genuine weakness to
  state in the model card, and a reason to expect `Score` to remain the weakest primitive.
- ANLI, SciQ and the SA block remain available as **eval-only**, which is useful: they make good
  OOD evaluation sets precisely because we cannot train on them.
