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
