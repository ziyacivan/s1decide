# s1decide

**Pre-alpha. Nothing here works yet, and there are deliberately no numbers on this page.**

`s1decide` is an open-source (Apache 2.0) decision model and inference library. It answers
typed questions — `Choice` (one option from a caller-supplied set), `Score` (a position on an
ordered rubric) and `Noul` (the probability a statement is true) — about a piece of state in a
single forward pass, with no autoregressive decoding anywhere in the path: each option is
mapped to a single token, the logits at one answer position are masked to the allowed tokens
and softmaxed, so an off-schema answer is impossible by construction. Many questions about the
same state are evaluated in one batched pass by broadcasting the prefilled KV cache, which
keeps latency close to flat in the number of questions. The point of the project is the part
comparable closed products do not publish: **measured, reproducible calibration** — ECE on
equal-mass bins reported alongside Brier, accuracy and a base-rate control, per option-count
bucket, in-distribution and out-of-distribution, re-fitted at the quantization actually
shipped. Every metric that ever appears in this repository is produced by a script in `eval/`
and committed as JSON under `results/`; none is typed by hand.

Start with [`CLAUDE.md`](CLAUDE.md) for the architecture decisions and hard rules,
[`AGENTS.md`](AGENTS.md) for the working roles, and [`docs/windows-setup.md`](docs/windows-setup.md)
for environment bring-up. All commands go through one entry point: `uv run task <name>`
(`uv run task --list`).
