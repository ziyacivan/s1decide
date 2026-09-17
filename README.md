# s1decide

**Pre-alpha. There is no trained model yet.** The data pipeline, the inference engine, the
evaluation harness and the benchmarks are built and tested; Stage-1 training has not run. Every
number on this page comes from a JSON file under `results/` and is checked against it by a test
— none is typed by hand, here or anywhere else in the repository.

`s1decide` answers typed questions about a piece of state in **one forward pass**:

| primitive | question | returns |
|---|---|---|
| `Choice` | one option from a caller-supplied set | probabilities over options, confidence |
| `Score` | a position on an ordered rubric | probabilities over levels, expected level, confidence |
| `Noul` | is this statement true | probability |

There is no autoregressive decoding anywhere in the path. Each option maps to a single token;
the logits at one answer position are masked to the allowed tokens and softmaxed, so an
off-schema answer is impossible by construction. Many questions about the same state are
answered in one batched pass by broadcasting the prefilled cache.

The point of the project is the part comparable closed products do not publish: **measured,
reproducible calibration.**

## What works today

| | status |
|---|---|
| Inference engine (`transformers`, 4-bit, cache broadcast) | working, benchmarked |
| Data pipeline — 15 sources, licence-gated, two-stage expansion | working |
| Evaluation harness — calibration, skill, risk-coverage, two negative controls | working |
| Latency benchmark against the definition of done | working, all four targets met |
| Zero-shot baseline on the 27B | measured and committed |
| **Stage-1 QLoRA training** | **not started** |
| Stage-2 calibration fitting | implemented, not yet fitted on a trained model |
| GGUF export, `/decide` server | not started |

## Measured so far

Base model, no fine-tuning, 4-bit on one RTX 3090. **This is the baseline S1 has to beat, not a
result.**

<!--metrics:zeroshot-->
| | value |
|---|---|
| Accuracy | 0.7645 |
| Brier skill vs base rate | +0.4602 |
| ECE (15 equal-mass bins) | 0.0449 |
| Accuracy at 80% coverage | 0.8314 |
<!--/metrics:zeroshot-->

Read the Brier skill score before the ECE. A base-rate control — a predictor that ignores the
state entirely — scores a *better* ECE than this model while being 33 accuracy points worse.
That is what ECE does, and it is why the headline calibration number here is skill against that
control, which a non-committal predictor cannot win.

Latency, same model and card, 1,617-token prefix:

<!--metrics:latency-->
| questions in one call | median | vs one call each | speedup |
|---|---|---|---|
| 1 | 1,515 ms | 1,550 ms | 1.02x |
| 4 | 1,761 ms | 6,207 ms | 3.53x |
| 16 | 2,614 ms | 24,899 ms | 9.52x |
| 64 | 5,847 ms | 99,092 ms | 16.95x |
<!--/metrics:latency-->

Latency is **not flat** in the number of questions and we do not claim it is — a 64-question
call costs about 3.9x a 1-question call at this state size. Flatness holds only when the shared
prefix is large relative to the per-question text. [ADR 0003](docs/adr/0003-latency-target.md)
derives the crossover and records the target that was retired for measuring the benchmark's
state size rather than the engine.

## Running it

Windows 11 with an RTX 3090 is the reference machine; everything runs unchanged on Linux.

```
uv sync
uv run task doctor     # GPU, CUDA, kernels, paths, encoding — read this first
uv run task data       # build the corpus into data/processed/ (deterministic)
uv run task test       # 679 CPU tests
uv run task test-gpu   # GPU tests, one file per process
uv run task eval       # score a model, write results/<run_id>/
uv run task bench      # latency against the definition of done
uv run task gpu-kill   # free the GPU after an interrupted run
```

`uv run task` with no arguments lists everything. Two commands are placeholders that fail
loudly rather than silently doing nothing: `smoke` and `train`.

Read [`docs/windows-setup.md`](docs/windows-setup.md) before the first GPU run. Two settings
there are not optional — the CUDA sysmem fallback policy, and git long paths.

## Documentation

| | |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | locked architecture decisions, hard rules, definition of done |
| [`AGENTS.md`](AGENTS.md) | the working roles and their hand-off checklists |
| [`docs/phase-1-plan.md`](docs/phase-1-plan.md) | what is being built now, step by step |
| [`docs/format-spec.md`](docs/format-spec.md) | the prompt format, versioned and generated |
| [`docs/model-card.md`](docs/model-card.md) | **draft** — every number a slot pointing at a `results/` path |
| [`docs/dataset-card.md`](docs/dataset-card.md) | sources, licences, file-format guarantees |
| [`docs/dataset-build.md`](docs/dataset-build.md) | generated: row counts, effective training mix, leakage |
| [`docs/windows-setup.md`](docs/windows-setup.md) | reproducing the environment on this machine |
| [`docs/adr/`](docs/adr/) | decisions, with the measurements behind them |
| [`docs/research/`](docs/research/) | dated notes; the factual basis for design choices |

The ADRs are the honest record. [0003](docs/adr/0003-latency-target.md) retires a latency
target that was mine and wrong. [0005](docs/adr/0005-ambiguous-source-licences.md) says which
data we declined to use and why. [0006](docs/adr/0006-release-scope.md) scopes v0.1 to a
pre-release and names what it will not measure.

## Licence

Apache-2.0. Training data is licence-gated by [ADR 0004](docs/adr/0004-training-data-licence-gate.md):
every row in a training split carries a licence from an allowlist, enforced by a test rather
than by a comment. Non-commercial and share-alike sources are evaluation-only.
