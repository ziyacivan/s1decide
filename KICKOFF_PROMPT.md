# Kickoff prompt for Claude Code

Copy everything below the line into Claude Code, run from the `s1decide/`
directory that contains `CLAUDE.md`, `AGENTS.md` and `docs/research/`.

---

You are starting **s1decide**, an open-source (Apache 2.0) decision model on
`Qwen/Qwen3.8-27B` that answers typed questions (Choice / Score / Noul) about a
piece of state in a single forward pass, with measured and published
calibration. It is meant to be the first ≥ 27B open alternative to TypeSafe's
Jev. Everything you need to know is in this repo:

1. Read `CLAUDE.md` fully. It contains locked architecture decisions, hard
   rules (no private data, no closed-API distillation, no push/publish/paid
   jobs without my explicit "go", no hand-typed metrics), the repo layout and
   the definition of done.
2. Read `AGENTS.md`. Work in the roles defined there and end every task with
   the hand-off block.
3. Read `docs/research/jev-landscape-2026-09-17.md`. It is the factual basis:
   what Jev is, its real accuracy numbers, the critiques, the existing ≤ 8B
   community replicas we build on, the papers (SALSA, RLCR), and the base
   model risks.

Do not re-derive any of that from memory; your training data predates this.

## Machine

**Windows 11 (64-bit), PowerShell, single RTX 3090 (24 GB VRAM). Unsloth
Studio is already installed natively** (its installer set up Long Paths, Git,
CMake, Visual Studio 2022 Build Tools and a CUDA toolkit). `uv` is available
or can be installed with `winget install astral-sh.uv`. There is no WSL2 in
scope unless I say so; no bash, no `make`, no FlashAttention, no vLLM on this
box. Assume nothing else. If you need a package or a model download, say so
and ask before downloading anything larger than 2 GB.

The project gets its own `uv` environment (Python 3.11). Do not touch the
Unsloth Studio venv under `%USERPROFILE%\.unsloth\`; you may *read* its
installed package versions (`pip list` inside it) to learn which torch/CUDA/
triton/bitsandbytes combination is known to work on this machine, and mirror
those pins in our `pyproject.toml`.

## Today's job: Phase 0 — foundation and de-risking

Work strictly in this order. After each step, stop and report with the
hand-off block; wait for my "continue" before the next step.

### Step 0 — Environment audit + plan (role: architect)
First, without installing anything, gather: `nvidia-smi`, `nvcc --version`,
Windows version, whether `uv` and `git` are on PATH, whether Long Paths is
enabled (`git config --get core.longpaths`, registry key
`LongPathsEnabled`), and the versions of torch / triton / bitsandbytes /
unsloth / transformers / peft / trl inside the Unsloth Studio venv. Write
them to `docs/windows-setup.md` (draft). Then write a plan for Phase 0
covering steps 1–6 below: files to create, tests to add, how each step is
verified, what could fail on Windows specifically. No code yet. Wait for my
confirmation.

### Step 1 — Repo scaffold (role: architect)
Create the layout from `CLAUDE.md` with `pyproject.toml` (package `s1decide`,
`uv`-managed, Python 3.11, ruff + pytest configured, torch index pinned to
the CUDA build matching the toolkit found in Step 0, optional extra `vllm`
marked Linux-only via environment markers), Apache-2.0 `LICENSE`,
`.gitattributes` (`* text=auto eol=lf`), `.python-version`,
`src/s1decide/tasks.py` exposing every task listed in `CLAUDE.md` through
`uv run task <name>` (tasks that aren't implementable yet must fail loudly
with "not implemented", not silently succeed) including a working `doctor`
that checks CUDA availability, torch↔toolkit version match, bnb 4-bit load,
triton import, unsloth import, llama-cpp-python import, free VRAM, HF cache
path sanity (no spaces, short), `PYTHONUTF8`. `README.md` (one paragraph,
clearly marked pre-alpha, no metrics), `docs/adr/0001-token-logit-approach.md`
and `docs/adr/0002-base-model.md` (status: proposed, to be filled in Step 3),
`CHANGELOG.md`. Run `uv sync` and `uv run task doctor`; fix until green.
Commit: `chore: scaffold repository`.

### Step 2 — Primitives and format spec (role: architect)
Implement `src/s1decide/primitives.py` (dataclasses `Choice`, `Score`,
`Noul`, `Question`, `Result` with validation: 2–255 options, unique option
labels, Score levels ordered), `src/s1decide/prompt.py` with
`FORMAT_VERSION = "0.1"` and a renderer that turns `(state, questions)` into
one prefix (the shared state block) and one suffix per question ending at
the answer position, and `src/s1decide/tokens.py` mapping options to
single-token labels. Write `docs/format-spec.md` with rendered examples.
Tests: validation errors, determinism of rendering, and — using the
`Qwen/Qwen3.8-27B` tokenizer only (no weights) — that every label token is a
single token with and without leading space, and that the chat template with
reasoning disabled emits no `<think>`. Commit.

### Step 3 — Base-model de-risking (role: training-engineer)
Without downloading the 27B weights, determine from installed library
versions and their source/docs whether `unsloth`, `transformers`, `peft`,
`trl`, `bitsandbytes` (Windows build) and `llama.cpp` support Qwen3.8-27B's
hybrid Gated-DeltaNet + attention architecture for: (a) 4-bit loading with
Unsloth `FastLanguageModel` on Windows, (b) LoRA on attention and MLP
projections, (c) the Triton kernels Unsloth needs for GDN layers compiling
under `triton-windows`, (d) GGUF conversion, (e) KV/state cache
expansion for batch broadcast (the hybrid cache has a recurrent state part —
find out how it is represented). Unsloth ships Qwen3.5 recipes (same
family): check their supported-models list and Studio's model catalogue for
3.8 first. Search the web where needed; add findings with dates to
`docs/research/`. Fill in `docs/adr/0002-base-model.md` with a
recommendation: proceed with Qwen3.8-27B, or fall back to Qwen3.6-27B. Name
the smallest model in the same architecture family (ideally ≤ 2 GB) to use
for smoke tests on this machine. Do not download anything yet. Report.

### Step 4 — Inference engine with KV broadcast (role: inference-engineer)
Implement `src/s1decide/engine/base.py` (protocol) and
`src/s1decide/engine/hf.py`: prefill the shared prefix once, expand the KV
cache to one row per question, run all question suffixes in a single forward
pass, read logits at the answer position, mask to allowed tokens, softmax.
Implement `src/s1decide/decide.py` on top, returning `Result` objects with
`probabilities`, `choice`/`score`/`noul`, and `confidence` (max prob for now;
calibrated later). Use `attn_implementation="sdpa"`; never import
`flash_attn`. Add `engine/mock.py` returning deterministic logits so the
whole API is testable on CPU in CI. Fuzz test: 10k random schemas → no
off-schema output. Add a test that an N-row broadcast equals N independent
single-question runs (this is where the hybrid recurrent-state cache can bite).
Commit. Only after my "go": download the small same-family model you
recommended in Step 3 at 4-bit and run the contract tests on the 3090; then,
with a second "go", the 27B at 4-bit.

### Step 5 — Zero-shot baseline + calibration harness (role: eval-scientist)
Implement `eval/metrics.py` (accuracy, ECE with 15 equal-mass bins, Brier,
NLL, AUROC of confidence vs correctness; base-rate control), `eval/run_eval.py`
writing `results/<run_id>/metrics.json`, and `eval/plots.py` for reliability
diagrams. Implement `src/s1decide/calibrate.py` (temperature scaling per
option-count bucket, save/load `calibration.json`). Data for now: only
`pngwn/system-one-decisions` (ask before downloading). Run the zero-shot
base model on its validation split through the engine from Step 4, fit
temperature on val, evaluate on test, commit results and plots. Report the
numbers by pointing to the JSON, not by retyping them.

### Step 6 — Latency benchmark (role: inference-engineer)
`eval/latency_bench.py`: median latency at 1 / 4 / 16 / 64 questions for a
fixed ~1,500-token state, 20 repeats, on the 3090, current engine. Plot and
commit. Report whether the 64-question call is < 2× the 1-question call; if
not, diagnose why.

## After Phase 0

Propose Phase 1 (data pipeline across all sources in `AGENTS.md`
`data-engineer` role, then S1 QLoRA smoke run) as a plan for my approval. Do
not start it unasked.

## Ground rules for this session

- Ask before: downloading > 2 GB, installing CUDA-level dependencies or
  anything via `winget`, touching the Unsloth Studio venv, anything that
  touches the network with my credentials, any `git push`.
- Everything you write must run identically from PowerShell on this machine
  and from bash on a Linux GPU box: Python only, `pathlib` only, `uv run task`
  as the single entry point. If a step needs a shell one-liner, write it as a
  Python function in `tasks.py` instead.
- If Unsloth Studio is running, tell me to close it before any GPU step
  (it holds VRAM).
- When a library or model fact is uncertain, search and record it in
  `docs/research/` with the date rather than guessing.
- Prefer boring, testable code over clever code. Every public function gets
  a test.
- Keep commits small with `area: imperative summary` messages.
- If you hit a wall, write the options and a recommendation as an ADR draft
  and stop.

Begin with Step 0.
