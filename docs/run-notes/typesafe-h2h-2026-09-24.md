# TypeSafe public workflow evals, strict common subset — 2026-09-24

`eval/typesafe_headtohead.py`; report `results/typesafe-h2h-2026-09-24/headtohead.json`.

**Reference-label caveat, verbatim from https://evals.typesafe.ai:** "Instead of debating the
correctness of the harness and labels, we assume that the code is correct, and measure against the
current smartest large models. For this eval, the reference labels are generated via an average of
the responses of GPT-6 Astra and Claude Fable 5.1, both at high thinking, answering every question
in the harness. All other models are evaluated using the provider's default reasoning settings."
Every number here is agreement with that consensus, not with ground truth.

**Subset and data.** Kev's frozen copy (102 questions, 20 cases; 66 `Noul`, 36 `Choice`), pinned
by SHA-256 and read from outside the repo; TypeSafe states no terms for its data, so only derived
scores are committed. **Scoring cross-check:** our code scores TypeSafe's own published answers
in that file exactly as Kev's README quotes them (`cross_check.matches` in the report).

| row | how it ran |
|---|---|
| zero-shot 27B, S1 | nf4 on the 3090, batch 1, eval-time context 12,288 — all 102 answered. S1 trained at ≤ 2,048 tokens; most of these states are longer. |
| Kev-9B | its server, bf16, expandable segments; 73 answered, 13 refused by Kev as over its context (its own protocol), 16 **out of GPU memory on this card** — a hardware limit, listed separately in the meta |
| Laya ×2 | its package on CPU; all answered, **with the state truncated by Laya itself** to its configured context (it right-truncates) |
| Jev | TypeSafe's published answers in the file — quoted, not queried |

Three scorings in the report: `evaluated` (rows a model answered — Kev's headline convention),
`all_rows` (missing rows as agreement 0 / TVD 1) and `common_answered` (the 73 rows every model
answered: the strict like-for-like comparison).

## Reading it

- **Jev's published answers lead** on every scoring.
- **Zero-shot Qwen3.8-27B is ahead of S1 here**, on `Choice` especially; on `Noul` S1 is slightly
  ahead. This is the first set where fine-tuning did not help, and it is a real result, not noise
  to explain away. Candidate reasons, none tested yet: S1 never saw a state over 2,048 tokens; these
  `Choice` options carry long descriptions, which our training options never did; and the labels
  are closed-model consensus on multi-document workflows unlike any training family.
- Kev-9B trails both 27B rows on the common subset; Laya is far behind, with most of each state cut
  off by its own context.
