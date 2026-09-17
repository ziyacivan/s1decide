# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Phase 0 Step 0: `docs/windows-setup.md` — audit of the reference machine
  (Windows 11 + RTX 3090 + Unsloth Studio), including the package versions to mirror.
- Phase 0 Step 1: repository scaffold — `pyproject.toml` (uv, Python 3.11, torch pinned
  to the cu130 index, ruff + pytest), Apache-2.0 `LICENSE`, `.gitattributes` forcing LF,
  `.python-version`, and `src/s1decide/tasks.py`, the cross-platform task runner behind
  `uv run task <name>` with a working `doctor`.
- `docs/adr/0001-token-logit-approach.md` and `docs/adr/0002-base-model.md` (both proposed).
- Phase 0 Step 2: `src/s1decide/primitives.py` (`Choice`, `Score`, `Noul`, `Question`,
  `Result`, validated at construction), `src/s1decide/prompt.py` (`FORMAT_VERSION = "0.1"`,
  the prefix/suffix renderer, `ChatTemplate.from_tokenizer`), `src/s1decide/tokens.py`
  (option to single-token labelling and verification), and `docs/format-spec.md`,
  generated from the renderer by `scripts/render_format_spec.py` and checked for staleness
  by a test.

- Phase 0 Step 3: base-model de-risking. `docs/research/qwen38-stack-support-2026-09-17.md`
  audits transformers / peft / trl / bitsandbytes / unsloth / llama.cpp support for
  Qwen3.8-27B's GDN hybrid on the pinned stack; `docs/adr/0002-base-model.md` is now
  **accepted** (proceed with Qwen3.8-27B; smoke model `Qwen/Qwen3.5-0.8B`);
  `scripts/derisk_base_model.py` reproduces the GPU measurements without downloads.
- `pyproject.toml`: pin `torchao==0.17.0` (Studio's version; 0.18.0 wants torch >= 2.11).
- Phase 0 Step 4: inference engine. `engine/base.py` (protocol, `EngineOutput`,
  `validate_output`), `engine/mock.py` (deterministic, dependency-free), `engine/hf.py`
  (prefill once, `expand_cache` over both attention KV and gated-deltanet states, chunked
  suffix passes, SDPA only), `engine/qwen3_5_patch.py` (fixes transformers 5.5.0 ignoring
  cached GDN state on multi-token continuation), and `decide.py` (`decide()`, `Response`,
  `softmax`). Tests: 10k-schema fuzz, N-row broadcast == N independent runs on a tiny
  random Qwen3.5-class model on CPU, chunked == single pass, padding isolation.
- `tests/test_engine_gpu.py`: the same contract on real `Qwen/Qwen3.5-0.8B` weights on the
  RTX 3090 (fp32 exact to 2.3e-5, bf16 within two ulps, unpatched transformers off by 2.1),
  plus nf4 load, zero-shot state-sensitivity sanity checks and a timing report.
  `HFEngine.from_pretrained` now enforces the requested dtype (transformers 5.5 let the VLM
  `text_config.dtype` override it).
- `tests/test_engine_gpu_27b.py`: the contract on the target model,
  `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` (20.8 GiB), on the 3090. Loader now handles
  pre-quantized VLM checkpoints (text config + parent quant config), forces bf16 end to end
  (`force_quantized_model_dtype`; the checkpoint's unquantized tensors are fp16), sizes the
  broadcast batch from real free VRAM, and runs `lm_head` only at the answer positions.
  First 64-vs-1 question ratio at a 1.6k-token state: 3.81x — Step 6 work.

- Phase 0 Step 5: evaluation harness and calibration. `eval/metrics.py` (accuracy, ECE on 15
  equal-mass bins, MCE, multiclass and top-label Brier, NLL, rank-based AUROC, uniform and
  base-rate negative controls, per option-count / family / primitive breakdowns),
  `eval/data.py` (loads `pngwn/system-one-decisions`, remaps noul indices, reports coverage),
  `eval/run_eval.py` (`uv run task eval`, stores logits so `--rescore` recomputes everything
  without a GPU), `eval/plots.py` (reliability diagrams), and
  `src/s1decide/calibrate.py` (per-option-count-bucket temperature scaling, golden-section fit
  on the inverse temperature, `calibration.json` that refuses a quantization mismatch).
- `docs/dataset-card.md`: sources, licences and the coverage limit.
- Phase 0 Step 6: `eval/latency_bench.py` (`uv run task bench`) — median latency at 1/4/16/64
  questions over 20 repeats, the one-call-per-question comparison, and a least-squares
  decomposition of the suffix phase into per-pass and per-token cost. `eval/summary.py` gains a
  generated `LATENCY.md`. `docs/adr/0003-latency-target.md` records that the "64 questions
  < 2x one question" target is missed (4.31x) and why, with options and a recommendation.

- **Phase 0 complete** (tag `phase-0`). Decisions recorded at its close:
  - `docs/adr/0003-latency-target.md` **accepted** — the "64-question call < 2x the 1-question
    call" rule is retired. KV broadcast amortises the state, not the questions; latency is flat
    in question count only when `prefix_tokens >~ 61 x tokens_per_question`. Replaced in
    `CLAUDE.md` and `AGENTS.md` by four measurable targets produced by `eval/latency_bench.py`:
    bundling speedup (>= 8x at 16, >= 12x at 64), marginal cost per question at 64 (<= 150 ms),
    the fitted cost model committed per release, and format overhead (<= 25% of suffix tokens).
    Three are met; format overhead is the baseline for option B.
  - `docs/adr/0004-training-data-licence-gate.md` **accepted** — `pngwn/system-one-decisions` is
    eval-only permanently; train splits require a licence from an allowlist, enforced by a test.
  - `docs/research/latency-amortisation-2026-09-17.md` — why the `a/b` ratio is a property of
    the hardware rather than the model, and a falsifiable hypothesis for a competitor's
    flat-latency claim.
- `eval/latency_bench.py` now measures format overhead and scores the run against the four
  targets; `LATENCY.md` reports them.

- **Phase 1 Step 1** — licence gate and source audit. `data/build/licences.py` implements ADR
  0004's gate (allowlist, `classify_licence`, `assert_train_splits_are_licensed`, plus the
  Tier-2 `drop_leaked_rows` guard), enforced by `tests/test_licences.py` — 77 tests, no GPU, no
  network. `docs/research/source-licences-2026-09-17.md` records the audit;
  `docs/adr/0005-ambiguous-source-licences.md` collects the ambiguous cases with a
  recommendation per source. `docs/phase-1-plan.md` is the approved plan of record.
  Nine sources are train-eligible; MNLI, FEVER, BoolQ, SNLI, ARC, AG News, SST-5 and Yelp are
  not. `docs/research/latency-amortisation-2026-09-17.md` corrected: the predicted
  model-independence of `a/b` was **measured and refuted** on `Qwen/Qwen3.5-0.8B`.

### Notes

- vLLM is deliberately **not** a project extra: uv's universal lock cannot satisfy
  vLLM's exact `torch`/`transformers` pins alongside ours. It gets its own environment
  on the Linux box. See `pyproject.toml` and `docs/windows-setup.md` R7.
- Triton compiles on this machine **without** MSVC — `triton-windows` bundles `ptxas`
  and TCC. No Visual Studio Build Tools install is needed. See `docs/windows-setup.md` R1.

[Unreleased]: https://github.com/ziyacivan/s1decide
