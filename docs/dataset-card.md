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
