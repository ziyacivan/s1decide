# Dataset card

Sources used by this project, what they are licensed under, and what that permits. Row counts
and per-split figures are produced by code (`eval/data.py`, recorded in each run's
`metrics.json` under `coverage`), never typed here by hand.

## Currently used

### `pngwn/system-one-decisions`

| | |
|---|---|
| Used for | **Evaluation only** (Phase 0 Step 5 zero-shot baseline and calibration) |
| Licence | **CC-BY-NC-4.0** |
| Splits | train 12,913 · val 1,452 · test 1,751 |
| Revision pinned | `fe081073641c58f336acafe090002e4435313c0a` |
| Schema | `task`, `question_type`, `ordered`, `state`, `question`, `options`, `answer_index` |
| Families | `go_emotions`, `mmlu`, `ag_news`, `banking77`, `yelp_score`, `tickets_queue`, `tickets_priority`, `tickets_language`, `tickets_type` |
| Primitives | `choice`, `score` (ordered), `noul` |

**The licence is the thing to notice. CC-BY-NC-4.0 forbids commercial use.** Measuring a model
against this data is fine and is all we do with it today. Training on it is a different matter:
weights derived from NC data cannot be released under Apache 2.0 with a straight face, which
directly contradicts the release plan in `CLAUDE.md`. Before Phase 1 uses any of this for
training, one of these has to be true:

1. we train only on sources whose licences permit redistribution of derived weights, and this
   dataset stays an eval set; or
2. we rebuild the equivalent data from the underlying sources under their own licences
   (`go_emotions`, `ag_news`, `banking77`, MMLU and Yelp each have their own terms, which are
   not all permissive either); or
3. the release is not Apache 2.0, which is a decision for the project owner and needs an ADR.

**Recommendation: option 1.** This dataset is a good, honest yardstick precisely because we did
not train on it, and keeping it that way also keeps it clean as a held-out benchmark.

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
ratings, plus synthetic structured-state tasks. **Each needs its licence checked against the
Apache-2.0 release before a single row is used for training**, per the note above. No teacher
model has been used for synthetic labels yet; when one is, it will be an open-weight model and
recorded here with its licence and the prompt hash.
