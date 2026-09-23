# ADR 0007 — Ordinal loss for `Score`

**Status:** accepted 2026-09-23
**Context:** locked decision 3 defines `Score` as an ordinal `Choice` trained with an
ordinal-aware loss and reporting an expected level plus the full distribution. This record says
which loss, and why not the other one.

## Decision

Train `Score` with **cross-entropy plus a distance penalty on the level distribution**:

```
L = CE(p, y) + λ · E_{k~p}[ |k − y| ]   ( + μ · unimodality penalty, optional )
```

- `CE(p, y)` is ordinary cross-entropy, and it accepts a **soft** `y`: 1,805 of the 4,804
  teacher-labelled rows carry a target split across two adjacent levels.
- `E[|k − y|]` is the expected absolute distance between the predicted level and the target,
  under the model's own distribution. It is what makes the loss ordinal: predicting level 1 when
  the answer is 4 costs three times what predicting level 3 does, where cross-entropy alone
  charges the same for both.
- The **unimodality penalty** is optional and off by default: it penalises local maxima at
  non-adjacent positions, i.e. a distribution that says "probably 1, possibly 4, definitely not
  2 or 3". On an ordinal scale that shape is usually a symptom rather than a belief.
- **λ ≈ 0.3** to start, selected on `val` by QWK and MAE **together with ECE** — a λ that
  improves the ordinal metrics by making the model overconfident is not an improvement, and
  looking at the three together is what catches it.

## Why not neighbour label smoothing

Rejected as the default. It moves a fixed slice of probability mass onto the neighbours of the
true level, which does make the loss ordinal-aware and is simpler to implement.

It is **not a proper scoring rule**. The optimum of a smoothed target is not the true
distribution: the model is trained to report the smoothing, so its confidence is wrong by
construction and wrong in a way that is baked into the weights rather than fitted afterwards.
This project's entire claim is measured, published calibration, with post-hoc temperature and
vector scaling applied on top of a model whose probabilities mean something. A training-time
distortion of exactly the quantity S2 exists to correct is the wrong foundation to build that on.

The distance penalty is not a proper scoring rule either, strictly speaking — any λ > 0 pulls the
optimum away from the true distribution. The difference is that λ is a dial we can take to zero
and measure against, and the pull is toward *the ordinal structure of the task* rather than
toward a fixed smear around the label. Keeping λ small and selecting it against ECE is how that
stays honest.

Neighbour label smoothing stays as **one A/B arm in S1 proper**, not in the smoke run. If it wins
on QWK, MAE *and* calibrated ECE, that is worth knowing and this ADR gets amended.

## Soft targets

Two kinds of row, because the evidence comes in two kinds:

| agreement | `target_type` | target |
|---|---|---|
| both teachers named the same level | `hard` | all mass on that level |
| one level apart | `soft` | 0.5 / 0.5 across the two |

The 0.5/0.5 split is equal because nothing measured justifies a tilt: on the adjacent
disagreements teacher 2 was the higher one 1,010 times and the lower 795
(`docs/research/teacher-agreement-2026-09-22.md`). `answer_idx` stays teacher 1's level, so a
consumer that ignores `target` still gets a valid label.

The loss takes a target distribution, not an index. A hard row is the one-hot case of the same
code path, so there is no second branch to keep in step.

## What this depends on

**Option order must be the ordinal scale, in order.** Both `E[|k − y|]` and the expected level
are arithmetic over positions. Until 2026-09-23 the build shuffled `Score` options along with
`Choice` options, which would have made this loss meaningless while running without error;
`ORDERED_QTYPES` now covers `noul` and `score`.

## Consequences

- `train/ordinal_loss.py` implements it, takes soft targets, and is tested against the properties
  rather than against remembered numbers.
- λ is a config value, swept on `val`, and the chosen value is recorded in the run's metadata.
- The model card reports QWK and MAE for `Score` beside accuracy, because accuracy on an ordinal
  scale throws away the thing that makes it ordinal.
- `Score` evaluation rows from a non-rule-based family are a prerequisite for any of these
  numbers; see the queued second teacher run in `docs/phase-1-plan.md`.
