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

### Notes

- vLLM is deliberately **not** a project extra: uv's universal lock cannot satisfy
  vLLM's exact `torch`/`transformers` pins alongside ours. It gets its own environment
  on the Linux box. See `pyproject.toml` and `docs/windows-setup.md` R7.
- Triton compiles on this machine **without** MSVC — `triton-windows` bundles `ptxas`
  and TCC. No Visual Studio Build Tools install is needed. See `docs/windows-setup.md` R1.

[Unreleased]: https://github.com/ziyacivan/s1decide
