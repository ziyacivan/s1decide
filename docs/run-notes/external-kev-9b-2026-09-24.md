# Kev-9B — prepared, not run (2026-09-24)

Queued for the GPU after S1, pending the owner's go. Nothing has touched the GPU.

| | |
|---|---|
| code | `jaredpalmer/kev` at `c9c1f85`, cloned to `%LOCALAPPDATA%\s1decide-envs\kev` (outside the repo) |
| environment | Kev's own lock (`uv sync --extra serve`, Python 3.13, transformers 5.17.0), with **torch 2.8.0 swapped from the CPU-only PyPI wheel to `2.8.0+cu128`** — PyPI's Windows torch has no CUDA. Recorded because it is a change to Kev's published environment. |
| adapter | `jaredpalmer/kev-9b`, revision `2629c06` (rank-16 LoRA + pointer head, `head.pt`) |
| base | `Qwen/Qwen3.5-9B-Base`, revision `68c46c4`, 19 GB, cached |
| memory | ~17 GB to load per Kev's README; bf16, no 4-bit path documented |
| harness | Kev's server (`python -m kev.serve --run jaredpalmer/kev-9b`) behind `/v1/systemone`, scored by `eval/external_http.py` on the same slices as S1 and Laya, reported by `eval/external_laya.py report` |
| calibration | as served: one temperature fitted by Kev in distribution (`KEV_TEMPERATURE` left unset) |

Open before running: whether Kev-9B in bf16 fits beside nothing else on 24 GB with its own
activations at our longest rows, and the serve precision (`KEV_DTYPE`, default bf16) — both go in
the run meta.

## Run — 2026-09-24 07:37–07:54 UTC, after S1 finished (owner's go)

`results/external-kev-9b-2026-09-24/` — `metrics.json`, predictions and meta per split,
`kev_env.json` (Kev commit `c9c1f85`, adapter `2629c06`, base `68c46c4`, torch swap, bf16, Kev's
shipped temperature). Started automatically once S1 wrote `DONE` and its process exited, with the
card at 417 MiB; server on Kev's own venv python; ~20–21 GB in use while scoring; server stopped
and the card free afterwards. 8,329 val rows in 500 s, 8,350 test rows in 472 s.

**What it shows** (numbers in `metrics.json`): Kev-9B is ahead of both Laya checkpoints on
`choice`, `noul`, rule-labelled `Score` and teacher `Score`, with positive Brier skill on all four.
**Stage-1 rows are the exception, and badly so**: Brier skill far below the base rate, accuracy
well under the 99% "no" rate.

**Caveat on stage 1, as for Laya:** those rows are `noul` questions phrased "…?\nCandidate: X" —
our model's text, not a statement. A model that reads a `noul` as "is this plausible?" says yes to
most candidates, and at a 99:1 base rate that is catastrophic for Brier. It is what these rows
measure for every model, and the post must say it is a phrasing-sensitivity result rather than a
verdict on Kev's decisions. The aggregate selective-accuracy figure is dominated by these heavily
weighted rows; the per-primitive numbers are the ones to read.
