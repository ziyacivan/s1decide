# Dataset card

Sources used by this project, what they are licensed under, and what that permits.

**All counts live in [`dataset-build.md`](dataset-build.md)**, generated from
`data/processed/manifest.json` by `uv run task data`. Nothing numeric is typed here, so the
reasoning below and the figures there cannot drift apart. The licence audit behind the source
list is `docs/research/source-licences-2026-09-17.md`; the rules are ADR 0004 and ADR 0005.

## Currently used

### `pngwn/system-one-decisions`

| | |
|---|---|
| Used for | **Evaluation only, permanently** — ADR 0004. It never enters a train split. |
| Licence | **CC-BY-NC-4.0** |
| Splits | train 12,913 · val 1,452 · test 1,751 |
| Revision pinned | `fe081073641c58f336acafe090002e4435313c0a` |
| Schema | `task`, `question_type`, `ordered`, `state`, `question`, `options`, `answer_index` |
| Families | `go_emotions`, `mmlu`, `ag_news`, `banking77`, `yelp_score`, `tickets_queue`, `tickets_priority`, `tickets_language`, `tickets_type` |
| Primitives | `choice`, `score` (ordered), `noul` |

**The licence is the thing to notice. CC-BY-NC-4.0 forbids commercial use**, so weights derived
from it could not honestly be released under Apache 2.0. **Resolved by ADR 0004 (accepted
2026-09-17): this dataset is eval-only, permanently**, and Phase 1 rebuilds training data from
the underlying sources under their own licences. It stays a good yardstick precisely because we
never fit it.

Enforcement is a test, not this paragraph: every row in a train or validation split must carry
a licence from the ADR 0004 allowlist (Apache-2.0, MIT, BSD-2/3-Clause, CC0-1.0, CC-BY-3.0/4.0).
NC and ND sources may be tagged `eval` only; an unrecognised licence is refused rather than
assumed.

#### Normalisations applied at load

Both would be silent bugs if skipped; both are enforced in `eval/data.py` and tested.

* **Noul answer indices are flipped.** The source lists options as `('yes', 'no')` — index 0 is
  *yes*. `s1decide.primitives.Noul` fixes labels as `('no', 'yes')` so index 1 is always the
  true case and `Result.noul` means P(true). The index is remapped on load, and the loader
  refuses any noul row whose options are not exactly `('yes', 'no')`.
* **Option labels are stripped.** Some MMLU rows carry trailing spaces
  (`'No, because the killing was unintentional. '`). The primitives reject padded labels
  because they would render inconsistently. Stripping is applied at the boundary, and a row
  whose options *collide* once stripped is dropped and counted rather than silently merged.

#### Coverage limit (important when reading any result from this dataset)

Option counts in this dataset are exactly `{2, 4, 5, 52, 77}`. Two families exceed the 26
single-token labels available under ADR 0001:

| Family | Options | val rows | test rows |
|---|---|---|---|
| `banking77` | 77 | 100 | 150 |
| `tickets_queue` | 52 | 60 | 80 |

That is **160/1,452 of val (11.0%) and 230/1,751 of test (13.1%)**, and it is the *entire*
`17-77` option-count bucket. Until the two-stage high-cardinality path from ADR 0001 exists,
those rows are dropped — reported in every `metrics.json` under `coverage`, never hidden. Any
accuracy figure from this dataset is therefore an accuracy **on questions with at most 26
options**, and it is not comparable to a figure that includes banking77.

#### Splits

The three splits share **no states** (verified: zero overlap between train/val and train/test,
and between val and test). Temperature scaling is fitted on `val` and reported on `test`, and
the base-rate control learns its frequencies from `val` only.

## Planned, not yet used

The sources named in `AGENTS.md` for the `data-engineer` role — banking77, clinc150, MASSIVE
(keeping Turkish and other non-English subsets), MNLI/ANLI/FEVER/BoolQ, SST-5 and review
ratings, plus synthetic structured-state tasks. **Phase 1 opens with a source audit**: each
licence is resolved to an SPDX identifier and classified `train` / `eval-only` / `refused`
against the ADR 0004 allowlist *before any row is fetched*. Some of these will not survive the
audit, and finding that out before the GPU-hours is the point.

No teacher model has been used for synthetic labels yet; when one is, it will be an open-weight
model, and its licence and the prompt hash are recorded here.

## Teacher-labelled `Score` rows

4,804 rows in `train` under family `score_teacher`, produced by two open-weight models labelling
the same five-level rubric independently and keeping only the rows they agreed on. Neither is a
reference model and neither saw the other's answer.

| | |
|---|---|
| teacher 1 | `unsloth/Qwen3.8-27B-unsloth-bnb-4bit`, native effort low, 1,024-token cap, nf4 |
| teacher 2 | `openai/gpt-oss-20b`, effort medium, 1,024-token cap, MXFP4 |
| both licences | Apache-2.0, run locally, no hosted API touched (CLAUDE.md hard rule) |
| rows compared | 5,928 of 6,000 (72 lost to a teacher that never committed) |
| kept | 4,804 — 2,999 exact agreement, 1,805 one level apart |
| dropped | 1,124 two or more levels apart |
| agreement | 50.6% exact against 24.7% by chance; Cohen's κ 0.344, weighted κ 0.650 |
| evaluation batch (teacher run 2) | 959 more rows from 600 `val` + 600 `test` states, same teachers and rules: 468 land in `val`, 491 in `test`, none in `train` — [`results/teach-qwen-low-1024-eval-fold/fold.json`](../results/teach-qwen-low-1024-eval-fold/fold.json) |

**Reasoning effort, verified rather than asserted** — [`results/effort-audit-2026-09-23/effort.json`](../results/effort-audit-2026-09-23/effort.json).
Each teacher's chat template was rendered with its requested effort and with a contrasting one;
they must differ and the difference must name the requested value, or a run stops
(`data/build/teach_run.effort_check`, now run before every teacher job and stored in its meta).

| teacher | effort line the template adds | template default | fingerprint (SHA-256 of the effort lines) |
|---|---|---|---|
| Qwen3.8-27B | "Reasoning effort is set to low. Keep your thinking brief and focused…" | `xhigh` | `ce172c3527da`* |
| gpt-oss-20b | "Reasoning: medium" | `medium` — requesting it renders the same as not asking | `f3ba8dc6313e` |

\* full hashes in the JSON. The Qwen default matters: without the setting, teacher 1 would have
been told to reason at `xhigh`. The completed runs passed `low`, and the audit of the tokenizers
they loaded shows it applied.

**Open, not yet resolved:** this card records teacher 2 as MXFP4, while its runs' `meta.json` says
`nf4-bf16` — the default label of the setting, not a measurement. Which one the loaded weights were
has to be read from the model, and this line stays until it is.

**Exact agreement becomes a hard target. One level apart becomes a soft target split equally
across the two levels.** Adjacent levels are where a five-level rubric is genuinely ambiguous
rather than where it failed, and both teachers' levels stay on the row so the rule can be changed
without re-labelling. The split is equal because nothing measured justifies a tilt: on the
adjacent disagreements teacher 2 was the higher one 1,010 times and the lower 795, so there is no
consistently more reliable side.

The licence on each row is the licence of the **state**, inherited from the source it was drawn
from; the labels are ours. States come only from train-eligible sources (ADR 0005 rule a).

Two things a consumer should know. The kept set is skewed to the bottom of the scale — 41% of
rows are "none" and 10.6% "moderate" — and filtering to agreement is also filtering to *easy*,
which is a bias and not a quality guarantee. Neither teacher is ground truth; two models agreeing
is a lower bound on the error rate of the pair. Full measurement in
[`docs/research/teacher-agreement-2026-09-22.md`](research/teacher-agreement-2026-09-22.md).

## File format guarantees

Every split is UTF-8 JSONL with LF line endings, one JSON object per line, and these hold for
every row:

- **No raw line-breaking characters anywhere in the file.** `json.dumps(..., ensure_ascii=False)`
  writes U+0085 (NEL), U+2028 (LINE SEPARATOR), U+2029 (PARAGRAPH SEPARATOR) and the C0
  separators U+000B, U+000C, U+001C–U+001E out **unescaped**, and `str.splitlines()`, most
  JavaScript, and several JSONL readers treat all of them as line terminators. One inside a
  state silently splits a row in two. Four reached an early build from MMLU question text.
  They are now replaced with a space at ingestion, and a test asserts that every split parses
  **identically** whether a reader splits on `"\n"` alone or on Unicode line boundaries.
- **Whitespace is collapsed and trimmed** in `state`, `instructions` and every option.
- **Options within a question are distinct** after case-folding; rows that collide are dropped
  at build time rather than producing a question with two right answers.
- **`state` is non-empty** and `answer_idx` is always a valid index into `options`.
- **Option order is shuffled for `Choice` and fixed for `Noul` and `Score`.** A `Noul`'s labels
  are always `("no", "yes")` so index 1 always means true. A `Score`'s options are its ordinal
  scale in order, lowest first, because an expected level and an ordinal loss are both arithmetic
  over positions. Shuffling exists to stop a model learning a positional prior; on an ordinal
  scale the positional prior is the task.
- **`target_type` and `target`, on teacher-labelled `Score` rows only.** `target_type` is
  `"hard"` or `"soft"`; `target` is a distribution over the options, in the same order, summing
  to 1. On a hard row it puts all its mass on `answer_idx`. Consumers that ignore both fields and
  read `answer_idx` get a valid label on every row.

You can therefore read a split with any of these and get the same rows:

```python
[json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
[json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
```
