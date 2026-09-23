# CLAUDE.md — s1decide

> **Final name** (ADR 0006). `s1decide` = "System-One-style decisions". It is the
> PyPI package, the GitHub repository, and the Hugging Face organisation and model
> prefix. **No rename.** Never include "Jev", "TypeSafe" or "System One Model" in the
> project, package, model or repo name, the description, or any marketing copy —
> referring to them in docs and comparisons is expected, naming ourselves after them
> is not.

## What this project is

An open-source (Apache 2.0) decision model built on **Qwen3.8-27B** that
answers typed questions about a piece of state in **one forward pass**, with
**measured, published calibration**. Three primitives, same shape as the
TypeSafe Jev API so people can compare directly:

- `Choice` — one option from a caller-supplied set → probabilities over options + confidence
- `Score`  — position on an ordered rubric → probabilities over levels + expected level + confidence
- `Noul`   — probability that a statement is true

Deliverables: (1) BF16 LoRA adapter + merged BF16 weights + our own GGUFs on
Hugging Face, (2) `pip install s1decide` inference library with KV-broadcast
parallel questions, (3) `/decide` HTTP server, (4) reproducible train/eval
pipeline, (5) an eval report with calibration curves and a head-to-head against
TypeSafe's public workflow evals.

Read `docs/research/landscape-2026-09-24.md` (the open field, and this project's
position in it) and `docs/research/jev-landscape-2026-09-17.md` (Jev itself, as of 17 Sep)
before any design decision.
It holds every known fact about Jev, the community replicas, and the papers we
build on. Do not re-derive that context from memory; memory is stale.

## What this project is NOT

- Not a chat model, not a text generator. No decode loop anywhere in the
  inference path. If you find yourself calling `.generate()` with
  `max_new_tokens > 1`, stop.
- Not a general LLM benchmark project. We evaluate decisions and calibration only.
- Not an enterprise deployment. No private data enters this repo, ever
  (see Hard rules). A private company adapter will be trained later in a
  separate, non-public repo on top of what we release.

## Locked architecture decisions

Change these only with an ADR in `docs/adr/` and explicit human approval.

1. **Token-logit approach, not a classification head.** Each option is mapped
   to a single token (`A`…`Z`, digits, or `yes`/`no`). We read the logits at a
   placeholder position, mask to the allowed tokens, softmax. The model stays a
   standard causal LM → converts to GGUF, runs in llama.cpp/vLLM/MLX unchanged.
   (Rationale: community adoption. pngwn's classification-head route is valid
   but needs custom runtimes.)
2. **Parallel questions via KV-cache broadcast.** Prefill `state` once,
   broadcast the cache across one batch row per question, one forward pass for
   all questions. Latency must be ~flat in #questions. Reference:
   `rorshopping/jev-on-a-laptop`.
3. **Noul = 2-option Choice** (`yes`/`no`). **Score = ordinal Choice** trained
   with an ordinal-aware loss (neighbour label smoothing or CE + distance
   penalty) and reporting expected level + full distribution.
4. **High-cardinality Choice (> 26 options)**: two-stage — independent
   Noul-style scoring per option, then a final Choice over the top-k. Mirror
   the documented Jev limit (255) but verify the current docs first.
5. **Training backend is Unsloth Core** (`FastLanguageModel` + TRL), the same
   engine Unsloth Studio uses, driven by scripts in `train/` so runs are
   reproducible and versioned. Unsloth Studio (the web UI) is allowed for
   exploration, dataset inspection, quick sanity trainings and GGUF export —
   but every number we publish must come from a `train/` script + `eval/` run.
6. **Training stages**: (S1) QLoRA SFT with cross-entropy on the option token →
   (S2) post-hoc temperature scaling per option-count bucket, fitted at the
   *deployment quantization* → (S3, optional) GRPO with a proper scoring rule
   reward (Brier or log score) per RLCR. S3 runs only on rented Linux GPUs.
7. **Base**: `Qwen/Qwen3.8-27B`, reasoning disabled. Fallback: `Qwen/Qwen3.6-27B`
   if Unsloth/PEFT cannot handle the GDN hybrid layers. Verify support on day
   one and record the result in `docs/adr/0002-base-model.md`.
8. **Release artefact of record is the BF16 LoRA adapter.** Q4 is a
   downstream convenience, never the primary artefact.

## Hard rules

- **No private/enterprise data in this repo, in training data, in tests, in
  fixtures, in examples.** Public or synthetic only. If unsure, it's private.
- **No distillation from closed-weight models or hosted APIs.** Open-weight
  models under permissive licences, run locally, are permitted **regardless of
  publisher**; record model ID, licence and revision in the dataset card.
  (Amended 2026-09-17. The earlier wording named OpenAI, Anthropic, Google and
  TypeSafe — those were examples of *closed APIs*, not banned organisations. An
  Apache-2.0 checkpoint downloaded and run on our own GPU is not an API call,
  whoever released it.)
- **No fabricated numbers.** Every number in a README, model card or report
  must be produced by a script in `eval/` and committed as JSON under
  `results/`. Never type a metric by hand.
- **Never call `git push`, publish to Hugging Face, or start a paid cloud job
  without an explicit human "go" in the same session.** Local commits are fine.
- **Never spend GPU-hours on a config that has not passed the smoke test**
  (`uv run task smoke`, ≤ 5 min on the 3090).
- **Never use "Jev", "TypeSafe" or "System One Model" in names, package
  identifiers, model IDs or marketing copy.** Referring to them in docs and
  comparisons is fine and expected.
- Do not edit files under `data/raw/`. Derived data goes to `data/processed/`
  and is regenerated by `uv run task data`.
- Do not modify anything inside the Unsloth Studio installation
  (`%USERPROFILE%\.unsloth\`). The project has its own `uv` environment.

## Repo layout

```
s1decide/
  CLAUDE.md  AGENTS.md  README.md  LICENSE (Apache-2.0)  CHANGELOG.md
  pyproject.toml                 # package: s1decide; [project.scripts] task = "s1decide.tasks:main"
  .gitattributes                 # * text=auto eol=lf  (Windows: no CRLF drift)
  .python-version                # 3.11 (Studio uses 3.13; both allowed, 3.11 is ours)
  src/s1decide/
    tasks.py                     # cross-platform task runner (replaces Makefile), see Commands
    primitives.py                # Choice / Score / Noul dataclasses + validation
    prompt.py                    # canonical prompt/format spec (versioned: FORMAT_VERSION)
    tokens.py                    # option→token mapping, tokenizer sanity checks
    engine/
      base.py                    # Engine protocol: prefill, broadcast, score
      hf.py                      # transformers (+ optional unsloth fast path), BF16 / 4-bit — Windows OK
      llamacpp.py                # llama-cpp-python engine (GGUF) — Windows OK
      vllm.py                    # vLLM engine (prompt_logprobs) — Linux/WSL2 only, optional extra
      mock.py                    # deterministic engine for CPU tests / CI
    calibrate.py                 # temperature scaling, per-bucket, save/load
    decide.py                    # public API: decide(state, questions) -> Result
    server.py                    # FastAPI /decide, /healthz, /info
  data/
    raw/                         # downloaded sources, never edited
    processed/                   # unified jsonl, produced by `uv run task data`
    build/                       # scripts: fetch, normalise, augment, split
  train/
    sft_lora.py                  # Stage 1 — Unsloth Core QLoRA
    ordinal_loss.py
    grpo_calibration.py          # Stage 3 — Unsloth + TRL GRPO (Linux GPU only)
    configs/*.yaml               # each carries `hardware: rtx3090_windows | h100_linux`
  eval/
    metrics.py                   # acc, ECE (equal-mass bins), Brier, NLL, AUROC
    run_eval.py                  # produces results/<run>/*.json
    plots.py                     # reliability diagrams, latency curves
    baselines/                   # fair LLM baselines (constrained decode + logprobs)
    typesafe_headtohead.py       # strict common-subset vs evals.typesafe.ai
    latency_bench.py
  results/                       # committed JSON + PNG, one folder per run
  docs/
    research/                    # landscape notes
    adr/                         # architecture decision records
    format-spec.md               # the prompt format, versioned
    windows-setup.md             # env bring-up on this machine (written in Phase 0)
    model-card.md  dataset-card.md
  scripts/                       # convert_gguf.py, merge_lora.py, push_hf.py (gated)
  tests/                         # pytest; unit + a marked `gpu` set
```

## Environment (this machine)

- **Windows 11 (64-bit), single RTX 3090 (24 GB VRAM)**, NVIDIA driver
  current. **Unsloth Studio is already installed** natively (no WSL); its
  installer set up Long Paths, Git, CMake, Visual Studio 2022 Build Tools and a
  CUDA toolkit. Reuse those system tools; do not reinstall them.
- Shell is **PowerShell**. No bash, no `make`, no `sed`/`awk` assumptions.
  All automation lives in `src/s1decide/tasks.py` and is invoked with
  `uv run task <name>`; it must run unchanged on Linux (for rented GPUs).
- Python env: `uv` with Python **3.11** for the project (`uv sync`). Unsloth
  Studio's own venv (Python 3.13 under `%USERPROFILE%\.unsloth\studio\`) is
  separate — never import from it, never install into it.
- PyTorch: install the CUDA build that matches the toolkit the Studio
  installer chose (check `nvcc --version` and `nvidia-smi`). `uv sync` must
  pin the torch index (e.g. `--torch-backend=auto`), and `uv run task doctor`
  must assert `torch.cuda.is_available()` and that the torch CUDA version
  matches the toolkit.
- Windows-specific rules (each is enforced by `uv run task doctor` or a test):
  - Triton on Windows comes via the `triton-windows` wheel that Unsloth pulls
    in; if a kernel fails to compile, check Long Paths first (`git config
    core.longpaths true` as well).
  - **No FlashAttention-2 on Windows.** Use `attn_implementation="sdpa"`.
    Never add `flash-attn` to dependencies.
  - `bitsandbytes` must be a Windows-capable build; verify `bnb` 4-bit loads
    in `doctor`.
  - **vLLM is not available on Windows.** `engine/vllm.py` is an optional extra
    (`uv sync --extra vllm`) guarded by `sys.platform != "win32"`; tests skip it.
  - Set `PYTHONUTF8=1` and `HF_HUB_ENABLE_HF_TRANSFER=0`; keep the HF cache on
    a path without spaces, ideally short (`D:\hf` or similar), and enable
    Windows Developer Mode or set `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.
  - Every script that uses multiprocessing or `torch.compile` has a
    `if __name__ == "__main__":` guard; DataLoader `num_workers=0` by default.
  - Line endings: `.gitattributes` forces LF in the repo; PowerShell scripts, if
    any, are `.ps1` and stay minimal.
- 27B on the 3090: inference in Q4/Q5 GGUF or 4-bit bnb; training only as
  Unsloth QLoRA (4-bit base, gradient checkpointing `"unsloth"`, sequences
  ≤ 2k tokens, batch 1–2). Expect to be VRAM-bound; every training run logs
  `torch.cuda.max_memory_allocated()`. Close Unsloth Studio (frees VRAM) before
  training or benchmarking; `doctor` warns if another process holds > 1 GB.
- Big runs (final SFT, S3 GRPO, BF16 evals, vLLM baselines): rented **Linux**
  H100 80GB. Config files carry `hardware:`; the runner refuses to start an
  `h100_linux` config on this machine and vice versa.
- Reasoning/thinking mode of the base model must be **off** in every prompt
  template. Confirm with a unit test that the chat template emits no
  `<think>` block.

## Commands

All via `uv run task <name>` (PowerShell and Linux identical):

```
doctor      # GPU/CUDA/torch/bnb/triton/unsloth/llama-cpp checks, VRAM in use, path sanity
setup       # uv sync (+ optional extras), pre-commit hooks
data        # fetch + normalise + augment + split → data/processed/
smoke       # 200-sample end-to-end: data → tiny Unsloth LoRA (rank 8, one pass) → eval; ≤ 5 min on 3090
train       # --cfg train/configs/sft_3090.yaml
eval        # --run <run_id>
bench       # latency vs #questions, current engine
serve       # uvicorn s1decide.server:app
test        # pytest -m "not gpu"
test-gpu    # pytest -m gpu
gguf        # merge LoRA → BF16 → GGUF Q8/Q5_K_M/Q4_K_M into artifacts/ (llama.cpp convert + quantize)
push        # HEAD → origin/wip, wait for `ci` green on that commit (GitHub API), then fast-forward main
```

If a task in this list doesn't exist yet, creating it is part of the job.
Tasks that aren't implementable yet fail loudly with "not implemented".

## Conventions

- Python 3.11, type hints everywhere, `ruff` + `ruff format`, `pytest`.
  Use `pathlib` everywhere; never string-concatenate paths.
- Every public function in `src/s1decide/` has a docstring and a test.
- Prompt format is versioned (`FORMAT_VERSION = "0.x"`); bumping it requires
  updating `docs/format-spec.md` and re-running `uv run task smoke`.
- Datasets: one jsonl line per question:
  `{"id", "family", "state", "qtype": "choice|score|noul", "instructions", "options": [...], "answer_idx", "split", "source", "license"}`.
  Files are UTF-8 with LF, opened with `encoding="utf-8"` explicitly.
- Option order is shuffled at train time; the mapping token↔option is
  generated per example, never hard-coded.
- ECE uses **equal-mass** bins (15) and is always reported together with
  Brier and a base-rate negative control. Never report ECE alone.
- Results are written by code to `results/<run_id>/metrics.json` and plots to
  `results/<run_id>/*.png`. READMEs link to these; they do not restate them.
- Training scripts use Unsloth's `FastLanguageModel.from_pretrained(...,
  load_in_4bit=True)` + `get_peft_model`, save adapters with
  `save_pretrained` (BF16 safetensors), and GGUF export via our own
  `scripts/convert_gguf.py` so the artefact path is identical on Linux.
- **A kernel or attention backend change is a new result row, never a silent
  upgrade.** Installing or losing an accelerated kernel changes the numbers a run
  produces, so every run records its kernel configuration in `meta.kernels` and
  results produced under different configurations are reported as separate rows.
  `uv run task doctor` prints the active configuration; `docs/windows-setup.md`
  records which fast paths this machine can and cannot have.
- **Anti-pattern: a swallowed exception.** No broad `except` (`Exception`, `BaseException`,
  bare `except:`, `contextlib.suppress(Exception)`) that passes, skips or continues silently.
  Every exception that is caught and not re-raised is logged with its **type** (stderr, or a
  field in the run's JSON) and has a test that triggers it and asserts the log. Narrow handlers
  that fall back to a default follow the same rule. Two bugs on 2026-09-23 hid behind one: a
  `suppress(Exception)` around `model.to("meta")` hid that a 4-bit model refuses to move, and
  the 27B stayed resident through a second night's OOM.
- Commit messages: `area: imperative summary` (e.g. `engine: add KV broadcast for hf engine`).
  Small commits. Never amend a commit that has been pushed.
- Language of code, docs, commits, model card: English.

## Definition of done (v0.1 — **pre-release**, 3090-only; ADR 0006)

v0.1 is **S1 QLoRA trained on the 3090 plus S2 post-hoc temperature scaling**, published
as a pre-release. Two items are deferred to v0.2 because they need hardware we do not
own; they are listed at the end of this section rather than deleted, and the model card
states plainly what was not measured.


- [ ] `uv run task doctor` green on this machine; `docs/windows-setup.md` lets
      a stranger reproduce the environment.
- [ ] `decide()` returns type-safe results for all three primitives; fuzz test
      proves no off-schema output in 10k random schemas.
- [ ] Latency at 1 / 4 / 16 / 64 questions on the 3090 measured and plotted, and
      these targets met — all produced by `eval/latency_bench.py` into
      `results/<run_id>/latency.json` (see ADR 0003, which retired the old
      "64-question call < 2× the 1-question call" rule and explains why):
      - [ ] bundled-vs-N-separate-calls speedup ≥ 8× at 16 questions and ≥ 12× at 64;
      - [ ] marginal cost per question at 64, `(median(64) − median(1)) / 63`, ≤ 150 ms;
      - [ ] fitted cost model (`a·passes + b·suffix_tokens`, with R²) re-fitted and
            committed per release;
      - [ ] `format_overhead_controllable` ≤ 25% of suffix tokens — the overhead our
            format adds, excluding the base model's chat tail (ADR 0003 option B).
            The all-in `format_overhead` is reported beside it in every bench output
            and in `LATENCY.md`; the difference between the two is the chat tail, which
            the base model's template imposes and no format change of ours removes.
- [ ] Zero-shot baseline (no LoRA) + S1 LoRA + S2 calibration evaluated on
      held-out families and ≥ 2 fully OOD sets; ECE, Brier, acc, AUROC per
      option-count bucket; reliability diagrams committed.
- [ ] Quantization table **Q8 / Q5_K_M / Q4_K_M**, all on the 3090, each row reporting
      accuracy plus **ECE and BSS under both calibration methods** — `temperature` (per
      option-count bucket) and `vector` (temperature plus per-position bias, per exact option
      count). Both are fitted on `val` at that row's own quantization; `calibration.json`
      names the one deployed. Reporting one method per row would hide whether a quantization's
      cost is confidence (which temperature fixes) or a shifted positional preference (which
      it cannot).
      **BF16 is deferred to v0.2** — a BF16 27B forward pass does not fit in 24 GiB.
      The model card states that BF16 was not measured, in those words, next to the table.
- [ ] Fair LLM baselines (Qwen3.8-27B itself with constrained decode + logprobs,
      at least one other open model) on the same splits — **in 4-bit on the 3090**,
      matching the deployment quantization the table covers.
- [ ] Head-to-head on TypeSafe public workflow evals (strict common subset),
      with the reference-label caveat stated verbatim — **in 4-bit on the 3090**.
- [ ] Model card + dataset card list every source dataset and license, every
      teacher model, all hyperparameters, and the exact commit hash. The model card
      states that calibration is **post-hoc temperature scaling, not learned**.
- [ ] `uv run task smoke` green on a fresh clone, on Windows and on Linux.
- [ ] Published as a **pre-release**, under the name `s1decide` and no other (ADR 0006).

### Deferred to v0.2 (rented H100 — needs an explicit "go", costs money)

- [ ] **BF16 row** of the quantization table, closing the hole v0.1 ships with.
- [ ] **S3 GRPO calibration RL** with a proper scoring rule reward (already
      "Linux GPU only" in the locked decisions above; it was never a 3090 item).

v0.2 is a scope, not a commitment to rent anything.

## How to work in this repo

1. Start every session by reading this file, `AGENTS.md`, and
   `docs/research/jev-landscape-2026-09-17.md`. Then `git log --oneline -20`
   and `uv run task doctor`.
2. For any task > 30 min of work: write a short plan in the chat first
   (files to touch, tests to add, how to verify), wait for confirmation, then
   execute. For tasks that spend money or publish: always wait.
3. Prefer small, verifiable steps. Run `uv run task test` before every commit.
4. When a claim about Jev, a library or a model is needed, cite the line in
   `docs/research/…` or search the web and add the fact there with a date.
5. When blocked on a design choice, write an ADR draft in `docs/adr/` with
   options and a recommendation instead of guessing.
6. When something works in Unsloth Studio but not in our scripts, inspect
   the Studio-generated config/notebook for the exact arguments it used and
   port them — do not paper over the difference.
7. **Commits reach `main` only through the CI gate.** `main` is protected: the `ci` check is
   required and there is no admin bypass, so a direct push of an unchecked commit is rejected.
   Commit locally as before, then `uv run task push`, which:
   - refuses unless `HEAD` fast-forwards `origin/main` (rebase first otherwise);
   - pushes `HEAD` to `origin/wip` (forced — `wip` is a scratch lane, not history);
   - polls the GitHub check-runs API for `ci` on that exact commit (no `gh`; a token in
     `GITHUB_TOKEN`/`GH_TOKEN` is used if set, read from the user environment on Windows when
     the session predates it);
   - on green, pushes the same SHA to `main`; on red or timeout, prints the failing tests'
     annotations and leaves `main` alone.
   Docs-only commits go through the same gate — a CI run for docs is cheap. The hard rule on
   pushing still applies: `task push` needs the same explicit human "go" as `git push`.
