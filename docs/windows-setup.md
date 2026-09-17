# Windows setup — s1decide (DRAFT)

> **Status: draft, written in Phase 0 Step 0 (2026-09-17).** This is an *audit*
> of the machine as found, not yet a reproduction recipe. It becomes a recipe in
> Step 1, when `uv run task doctor` exists and every claim here is machine-checked.
> Nothing was installed, changed or downloaded to produce this file.

## 1. Machine as found (2026-09-17)

| Item | Value | How it was read |
|---|---|---|
| OS | Microsoft Windows 11 Pro, 10.0.26200 (build 26200), 64-bit | `Get-CimInstance Win32_OperatingSystem` |
| GPU | 1x NVIDIA GeForce RTX 3090, 24576 MiB, WDDM, compute capability sm_86 | `nvidia-smi` |
| VRAM in use at audit time | 877 MiB (desktop / msedgewebview2 / explorer) | `nvidia-smi` |
| NVIDIA driver | 616.64 | `nvidia-smi` |
| CUDA UMD (driver-side) | 13.4 | `nvidia-smi` |
| Disks | single volume `C:` — 726 GB free / 204 GB used | `Get-PSDrive` |
| Shell | PowerShell (Windows PowerShell 5.1); Git Bash also present via Git for Windows | — |

## 2. Toolchain on PATH

| Tool | Present | Version / path |
|---|---|---|
| `git` | yes | 2.55.0.windows.3 — Git for Windows (mingw64) |
| `uv` | yes | 0.12.10 — `%LOCALAPPDATA%\Microsoft\WinGet\Packages\astral-sh.uv_...\uv.exe` (winget reports 0.12.15 available; **not** updated) |
| `python` | yes | 3.13.15 — `%LOCALAPPDATA%\Programs\Python\Python313\python.exe` (winget `Python.Python.3.13`) |
| `py` launcher | yes | `%LOCALAPPDATA%\Programs\Python\Launcher\py.exe` |
| `nvcc` | **no** | not on PATH; `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA` does not exist; `CUDA_PATH` unset |
| `cmake` | **no** | not on PATH; `C:\Program Files\CMake` does not exist |
| `cl.exe` / MSVC | **no** | not on PATH; no `Microsoft Visual Studio` directory under either Program Files; `vswhere.exe` absent |
| Windows SDK | **no** | `C:\Program Files (x86)\Windows Kits\10` does not exist |

### Python interpreters known to `uv`

`uv python list` finds only the system CPython 3.13.15 installed locally.
**Python 3.11 is not installed** — `uv sync` against `.python-version = 3.11`
will want to fetch `cpython-3.11.16-windows-x86_64-none` (~25-30 MB managed
download). This is the one download Step 1 needs.

## 3. Long Paths

Both layers are already enabled — no action needed.

| Layer | Value |
|---|---|
| `HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled` | `1` |
| `git config --get core.longpaths` (repo-local) | `true` |
| `git config --system --get core.longpaths` | unset (the repo-local value covers this repo) |

> The repo-local `core.longpaths true` was set during this session. A fresh clone
> on another machine must set it again, or set it globally. `doctor` checks this;
> it does not auto-fix it.

## 4. Unsloth Studio installation (read-only inspection)

Root: `%USERPROFILE%\.unsloth\` — **never modified, never installed into.**
Only directory listings and package metadata files were read; no interpreter
inside it was executed and no `pip` command was run against it.

```
%USERPROFILE%\.unsloth\
  llama.cpp\            prebuilt llama.cpp (binaries + full source tree)
  whisper.cpp\          prebuilt whisper.cpp
  node\                 bundled node/bun
  studio\
    unsloth_studio\     main venv, Python 3.13.15
    .venv_t5_510\       transformers-5.1.0 overlay venv (transformers + hub only)
    .venv_t5_530\       transformers-5.3.0 overlay venv
    .venv_t5_550\       transformers-5.5.0 overlay venv
    cache\ runs\ outputs\ logs\ compiled_cache\ TORCHINDUCTOR_CACHE_DIR\
```

`runs/` and `outputs/` are **empty** and `TORCHINDUCTOR_CACHE_DIR` holds 0 files:
no training has ever been executed in Studio on this machine. There is therefore
**no empirical proof yet that a Triton-backed training step compiles here** (see §7).

### 4.1 Package versions in the Studio venv (the known-good combination)

Read from `Lib\site-packages\*.dist-info` directory names — venv Python 3.13.15.

| Package | Version |
|---|---|
| `torch` | **2.10.0+cu130** |
| `torchvision` | 0.25.0+cu130 |
| `torchaudio` | 2.10.0+cu130 |
| `torchao` | 0.17.0 |
| `triton-windows` | **3.6.0.post26** |
| `bitsandbytes` | **0.50.2** |
| `unsloth` | **2026.9.2** |
| `unsloth_zoo` | **2026.9.1** |
| `transformers` | **5.5.0** |
| `peft` | **0.18.1** |
| `trl` | **0.23.1** |
| `accelerate` | 1.14.0 |
| `datasets` | 4.3.0 |
| `tokenizers` | 0.22.2 |
| `safetensors` | 0.8.0 |
| `numpy` | 2.5.2 |
| `xformers` | 0.0.34 |
| `cut_cross_entropy` | 25.1.1 |

Absent, as expected and as required by `CLAUDE.md`: `flash-attn`, `vllm`,
`llama-cpp-python`.

### 4.2 Where CUDA actually comes from

There are **no `nvidia-*` CUDA wheels** in the Studio venv. The Windows
`torch 2.10.0+cu130` wheel ships the CUDA runtime as DLLs inside the package:

```
site-packages\torch\lib\  cudart64_13.dll  cublas64_13.dll  cublasLt64_13.dll
                          cudnn64_9.dll    cudnn_*64_9.dll  ...
```

So the effective CUDA runtime is **CUDA 13.0 (cu130)**, supplied by torch, under
a **driver reporting CUDA 13.4** — forward-compatible, fine. There is no
standalone CUDA toolkit and therefore **no `nvcc`**.

**Consequence for `pyproject.toml`:** the torch index to pin is the **cu130**
build (`torch==2.10.0`, index `https://download.pytorch.org/whl/cu130`).
"Match the toolkit found in Step 0" resolves to *match the toolkit torch brings
with it*, because the box has no other one. `doctor` must therefore assert
`torch.version.cuda == "13.0"` rather than compare against an `nvcc` that does
not exist.

### 4.3 llama.cpp (prebuilt, not built here)

From `UNSLOTH_PREBUILT_INFO.json` + `BUILD_INFO.txt`:

| Field | Value |
|---|---|
| version | `b10798-mix-659e406` (upstream commit `d6b1d27`) |
| asset | `app-b10798-mix-659e406-windows-x64-cuda13-older.zip`, from `unslothai/llama.cpp` |
| backend | CUDA, **toolkit 13.3** (built on a GitHub runner, not here) |
| supported SMs | 75, 80, 86, 89 — **includes sm_86 = RTX 3090** |
| installed | 2026-09-06 |

Binaries present: `build\bin\Release\llama-server.exe`, `llama-quantize.exe`,
`llama-diffusion-gemma-visual-server.exe`. The **full llama.cpp source tree** is
there too, including `convert_hf_to_gguf.py` and `convert_lora_to_gguf.py` — which
is what `scripts/convert_gguf.py` will drive for the GGUF deliverable.

This install was **downloaded prebuilt**. It is *not* evidence that a C++/CUDA
compile works on this machine — it is evidence that nothing needed to be compiled.

## 5. Hugging Face cache

Default location `%USERPROFILE%\.cache\huggingface` — **68.9 GB already present**.
No `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `PYTHONUTF8` or `UV_CACHE_DIR`
environment variable is set.

Path sanity: `C:\Users\yusuf\.cache\huggingface` — no spaces, 33 chars, on the only
volume, 726 GB free. It satisfies the `CLAUDE.md` requirement as-is; moving it to
`D:\hf` is impossible (no D:) and unnecessary.

### Already-cached models (no download needed for these)

| Repo | Size | Files |
|---|---|---|
| `unsloth/Qwen3.8-27B-GGUF` | 34.6 GB | `Qwen3.8-27B-UD-Q4_K_M.gguf` (15.33 GB), `Qwen3.8-27B-UD-Q5_K_M.gguf` (18.41 GB), `mmproj-F16.gguf` (0.86 GB) |
| `orcarouter/Qwen3.8-27B-Uncensored-GGUF` | 16.5 GB | `Qwen3.8-27B-Uncensored-Q4_K_M.gguf` (15.66 GB), mmproj |
| `unsloth/Qwen3-Coder-Next-GGUF` | 17.6 GB | (unrelated to this project) |
| `unsloth/bge-small-en-v1.5` | 0.1 GB | (unrelated to this project) |

**No HF-format (safetensors) copy of Qwen3.8-27B is cached** — not the BF16 base,
not a bnb-4bit variant, and **not even the tokenizer**. Training and the `hf`
engine will need one; the Step 2 tokenizer tests need at minimum the tokenizer
files (a few MB).

## 6. Qwen3.8-27B support signals found locally (preliminary — Step 3 owns this)

Recorded here because they were observed during the audit. They are **not yet a
verified answer**; Step 3 verifies against upstream sources and writes
`docs/adr/0002-base-model.md`.

- `transformers 5.5.0` ships model packages `qwen3`, `qwen3_moe`, `qwen3_next`,
  `qwen3_5`, `qwen3_5_moe`, `qwen3_vl`, `qwen3_vl_moe`, `qwen3_omni_moe` — but
  **no `qwen3_8` and no `qwen3_6` package**. Qwen3.8-27B must therefore map onto
  an existing architecture class (most likely the `qwen3_5` / `qwen3_next`
  gated-deltanet hybrid) or require `trust_remote_code`. **Unverified.**
- `unsloth/models/mapper.py` contains an explicit entry
  `"unsloth/Qwen3.8-27B-unsloth-bnb-4bit" : ("unsloth/Qwen3.8-27B", "Qwen/Qwen3.8-27B", ...)`
  — Unsloth 2026.9.2 knows the model and publishes a pre-quantized 4-bit copy.
- `unsloth_zoo/temporary_patches/fla_vendor.py` vendors the
  `flash-linear-attention` (fla) Triton kernels and states they are used by
  "Qwen3.5 / Qwen3.6 / Qwen3-Next gated-deltanet models", with a
  **"several-times slower pure-PyTorch path"** when fla/Triton is unavailable.
  Escape hatch env var: `UNSLOTH_DISABLE_VENDORED_FLA=1`.
  So a missing Triton compiler degrades speed rather than blocking outright.
  **To be confirmed by actually running it.**

## 7. Risks found by this audit

**R1 — No MSVC, no Windows SDK, no CMake, no CUDA toolkit. (high)**
The kickoff brief and `CLAUDE.md` both state the Studio installer set these up.
On this machine it did **not**: it fetched prebuilt binaries instead.
`triton\windows_utils.py` (`find_msvc_env`, `find_msvc_vswhere`, `check_msvc`,
`find_msvc_winsdk`) searches `VCINSTALLDIR`, `vswhere.exe` and the registry for
`cl.exe` + `vcruntime.h` + `vcruntime.lib` — none of which exist here. Triton JIT
compiles kernels at runtime and needs that toolchain. Expected impact: Unsloth's
Triton kernels (and the vendored `fla` gated-deltanet kernels) fail to compile and
fall back to the pure-PyTorch path — slower training, possibly slower inference,
but per §6 not necessarily fatal. Also blocks any from-source build of
`llama-cpp-python`.
*This must be measured, not assumed* — `doctor` gets a real `triton.jit` compile
probe in Step 1, and the result decides whether we ask to install VS 2022 Build
Tools (a winget install, ~2-4 GB, requires explicit approval).

**R2 — `llama-cpp-python` may not install. (medium)**
No compiler means we depend on a prebuilt CUDA wheel matching Python 3.11 + cu13x,
which is not guaranteed to exist. Mitigation already on disk: Unsloth's prebuilt
`llama-server.exe` (CUDA 13.3, sm_86) — `engine/llamacpp.py` can drive it over its
HTTP API instead of linking the library. `doctor` reports `llama-cpp-python` as a
*warning*, not a hard failure.

**R3 — Python 3.11 is not installed. (low)** `uv` will fetch it (~30 MB).

**R4 — No HF-format Qwen3.8-27B, not even the tokenizer. (low/medium)**
Step 2's tokenizer tests need the tokenizer files (small, a few MB). The 4-bit
base for training is ~17 GB and needs explicit approval.

**R5 — 68.9 GB of GGUFs already cached; C: is the only volume. (low)**
726 GB free is comfortable, but 27B BF16 (~54 GB) + 4-bit (~17 GB) + GGUF exports
add up. Track it; do not move the cache.

**R6 — Studio is not running right now (0 MB held by it).** Good, but `doctor`
must still check, since Studio holds VRAM when open.

## 8. What this file still owes

- [ ] Every table above re-asserted by `uv run task doctor` (Step 1).
- [ ] A real `torch.cuda.is_available()` result from **our** env, not Studio's.
- [ ] A real Triton compile probe result (R1).
- [ ] Verdict on `llama-cpp-python` vs. prebuilt `llama-server.exe` (R2).
- [ ] A from-scratch reproduction recipe for a stranger.
