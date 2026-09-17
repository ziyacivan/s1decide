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

### Notes

- vLLM is deliberately **not** a project extra: uv's universal lock cannot satisfy
  vLLM's exact `torch`/`transformers` pins alongside ours. It gets its own environment
  on the Linux box. See `pyproject.toml` and `docs/windows-setup.md` R7.
- Triton compiles on this machine **without** MSVC — `triton-windows` bundles `ptxas`
  and TCC. No Visual Studio Build Tools install is needed. See `docs/windows-setup.md` R1.

[Unreleased]: https://github.com/ziyacivan/s1decide
