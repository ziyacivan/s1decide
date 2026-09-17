# ADR 0005 — Ambiguous source licences: recommendations per source

- **Status:** **partially decided** (2026-09-17) — Q3 and the share-alike question are decided;
  Q1, Q2 and Q4 remain proposed and will be answered when they block work
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

### DECIDED (2026-09-17): share-alike sources are eval-only

Accepted as recommended. They become **OOD evaluation sets**, which is a genuinely good use for
them: we cannot train on them, so they stay clean.

**This is a risk-tolerance decision, not a legal finding.** Nobody here has determined that
model weights *are* a derivative work of their training data under CC-BY-SA; the question is
unsettled and we are declining to bet a release on either answer. It can be loosened later with
legal review, and loosening it would unlock SNLI, BoolQ, ARC, DBpedia and part of MultiNLI —
the single largest block of data the audit refused. Anyone revisiting this should read it as
"we chose not to find out yet", not as "we found out it was disallowed".

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

### DECIDED (2026-09-17): `Score` training data is synthetic

Neither re-upload is used. `Score` data is generated, under four binding rules. These are
requirements on `data/build/`, not guidance, and each gets a test.

**(a) Text provenance.** The *state* text may come only from train-eligible sources (the nine in
`docs/research/source-licences-2026-09-17.md`) or from our own programmatically constructed
structured states. No text from a refused or eval-only source ever appears in a `Score` row.

**(b) Labels from two independent open-weight teachers.** Both score a **defined, written
rubric**; no closed-weight models and no hosted APIs, per `CLAUDE.md`'s hard rules — which since
2026-09-17 permit any permissively licensed open-weight checkpoint run locally, whoever
published it. Keep only rows where the two teachers **agree exactly or are within ±1 level**;
drop the rest.

### DECIDED (2026-09-18): teacher 2 is `openai/gpt-oss-20b` at effort **medium**

Apache-2.0, 21B MoE with 3.6B active, run locally at MXFP4 — permitted under the amended
distillation rule, which bans closed-weight models and hosted APIs rather than publishers.

**Medium rather than low, and the reason is trace depth, not agreement.** At low effort its
reasoning averages 77 tokens; at medium, 220. The point of using teachers with reasoning on is
to distil *deliberate* judgements into a one-forward-pass student, and a 77-token trace is
barely deliberation — it is a single-token answer with a sentence in front of it, which ADR
0005 rule (b) exists to avoid. Medium costs 2.6 hours against 1.0 for the whole 6,000-row run,
which is nothing next to teacher 1's 27.6.

It is worth being explicit that medium **also** agrees with teacher 1 more often — 55.0% exact
against low's 47.0% — and that **this played no part in the choice**. Selecting the teacher that
agrees most would manufacture the consensus the agreement rate is supposed to measure. Had the
depth argument pointed at low, low would have been chosen on 47%.

**Runtime for teacher 1: bitsandbytes nf4, not llama.cpp.** A time-boxed experiment ran the same
100 rows through `llama-server` with Q4_K_M and continuous batching at `--parallel 16`. It was
**5.18x faster** — 0.312 rows/s against 0.060 — and truncated nothing. It also **disagreed with
the nf4 path on 22% of judgements**: 78.0% exact, 94.0% within one level. The pre-agreed rule
required both a 2.5x speed-up and 90% exact match, so the speed-up is declined. Two
quantizations of one model giving different answers on a fifth of an ordinal task is not a
runtime detail; adopting it would silently change what the teacher said. The client is kept at
`src/s1decide/engine/llamacpp_client.py` as the seed of the GGUF engine.

**Teacher 2 is selected on tractability, never on agreement.** The criteria are: does it load
cleanly on this machine, what throughput does it reach, and how often does it fail to produce a
parseable label. **Agreement with teacher 1 is a finding, not a selection criterion**, and the
distinction is the whole point of using two teachers. Picking the teacher that agrees most would
manufacture the consensus the agreement rate is supposed to measure, and would bias the corpus
toward exactly the rows where both models are confidently wrong together. The agreement rate is
therefore measured and published for whichever teacher is chosen, after it is chosen.

**The teachers run with reasoning ON, at low effort — not as single-token scorers.** This is
deliberate and is the point of the exercise: we are distilling *deliberate, System-2* judgements
into a System-1 student that will answer in one forward pass. A teacher that answers in a single
token is doing the student's job, badly, and would teach the student nothing it could not have
learned from a smaller model. Low effort rather than high because the rubric levels are coarse
and `xhigh` would cost hours for judgements that do not need it.

The dataset card records, generated by script and never typed:
- both teacher model IDs and revisions,
- the prompt hash for each teacher,
- **the reasoning setting used**,
- the rubric text,
- the **exact-agreement rate**, the ±1-agreement rate, and the **drop rate**.

Two teachers rather than one because a single teacher's systematic bias would be
indistinguishable from signal, and an ordinal task is exactly where such bias concentrates.

**(c) A programmatic ordinal control set.** A set whose labels are derived **by rule** from
structured state — no model in the loop — so ordinality is guaranteed by construction. It is the
sanity check for the ordinal loss and for calibration: if the ordinal loss is working, this set
should be close to solved, and if calibration is working, its reliability diagram should be
near-diagonal. A teacher-labelled set cannot play that role because its own noise is unknown.

**(d) The leakage guard applies.** No teacher-labelled state may share a state hash with any
evaluation set, Tier 1 or Tier 2 (Phase 1 plan, amendment B). Enforced by the same
`drop_leaked_rows` machinery, as a test.

**Consequence, stated plainly for the model card:** `Score` will be trained largely on synthetic
labels. That is a real weakness, and it is the primitive most likely to underperform. Rule (c)
exists so we can tell the difference between "the ordinal loss is broken" and "the synthetic
labels are noisy".

## Question 4 — Dataset licence versus underlying content

**Affected: everything, including sources we plan to use.** `go_emotions` is Apache-2.0 but the
text is Reddit comments; `google-play-review` is CC-BY-4.0 over user-written reviews; MASSIVE is
CC-BY-4.0 over crowd-written utterances.

Inheriting the dataset's declared terms is standard practice and is what every comparable open
release does. It is nonetheless an assumption.

**Recommendation: accept, and state it explicitly in the dataset card** rather than leaving it
implicit. No code change; a sentence that makes the assumption visible.

## Decision

- **Q3 (`Score` data): decided** — synthetic, under rules (a)–(d) above.
- **Share-alike: decided** — eval-only, as a risk-tolerance choice, usable as OOD sets.
- **Q1, Q2, Q4: still proposed.** None blocks Phase 1 Step 2. They will be answered when they
  block work, rather than pre-emptively.

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
