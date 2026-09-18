# Windows setup — s1decide (DRAFT)

> **Status: draft. Audited in Phase 0 Step 0, verified in Step 1 (2026-09-17).**
> Sections 1-6 are the machine as found. Section 7 has been updated with what
> `uv run task doctor` actually proved; the risks it closed are marked RESOLVED.
> This is still an audit, not yet a from-scratch reproduction recipe (§8).

## 0. Quick start (what exists today)

```powershell
git clone <repo> ; cd s1decide
git config core.longpaths true
uv sync                 # fetches CPython 3.11 + the pinned stack, creates .venv
uv run task doctor      # must end with "0 fail"
uv run task test
```

Last verified on this machine: **15 checks, 0 fail, 1 warn** (the warn is
`llama-cpp-python`, which is optional and warn-by-design — see R2).

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

## 3b. CUDA sysmem fallback — **required setting**

**NVIDIA Control Panel → Manage 3D settings → CUDA - Sysmem Fallback Policy →
"Prefer No Sysmem Fallback".** Set globally, or at minimum for the project's `python.exe`.

This is not a tuning preference; without it the card does not report being full.

By default the Windows NVIDIA driver satisfies a CUDA allocation that no longer fits in VRAM
out of host RAM over PCIe, instead of raising `OutOfMemoryError`. Nothing fails. The run simply
gets several times slower, and every number it produces is wrong in a way that looks like a
real result. Measured on this machine, 64-question calls at a 1,617-token prefix:

| peak VRAM | 64-question call | |
|---|---|---|
| 21.90 GiB | 5,806 ms | normal |
| 22.34 GiB | 15,990 ms | 2.8x slower, no error |
| 22.78 GiB | 37,289 ms | 6.4x slower, no error |

Prefill inflates along with everything else (1.42 s → 5.34 s → 11.88 s), which is the
signature: it is not a compute effect, it is memory traffic crossing the bus. The first time
this happened it was mistaken for a real regression.

`uv run task doctor` reports it as the `sysmem-fallback` check. The check **probes the
behaviour rather than reading the setting** — it asks for 2 GiB more than the card has free and
expects that to raise — because the setting itself lives in the driver's binary profile
database and cannot be read back reliably. A healthy machine reads:

```
[OK  ] sysmem-fallback   requested 24.79 GiB with 22.79 GiB free of 24.00 GiB: raised
                         OutOfMemoryError, as it should
```

The probe deliberately overshoots by 2 GiB. A smaller margin lands inside the variation of what
`mem_get_info` calls "free" (the driver holds back a reserve that changes), and at 512 MiB the
probe contradicted itself between consecutive runs.

**The engine does not rely on this setting.** `HFEngine` budgets rows per pass against a
predicted peak held under a per-hardware ceiling — 0.92 of total VRAM for `rtx3090_windows`, in
`src/s1decide/hardware.py` — so it stays below the cliff whether or not the fallback is on. The
setting is belt and braces: it turns a silent slowdown into a loud failure if the budget is ever
wrong.

## 3bb. Pause Windows Update — **required for the project duration**

**Settings → Windows Update → Pause updates**, and re-pause it before every multi-hour run.
Where the edition allows it, set *Active hours* to cover the whole day as well.

This is not a preference. Windows Update restarted this machine **13 hours into a 27-hour
teacher-labelling run**:

```
2026-09-18 04:58:03 UTC  id=1074  MoUsoCoreWorker.exe  ... "Operating System: Service pack (Planned)"
2026-09-18 04:59:04 UTC  id=1074  TrustedInstaller.exe ... "Operating System: Upgrade (Planned)"
2026-09-18 04:59:30 UTC  boot completed
```

The job had checkpointed 1,000 rows and reached 1,144, so **144 rows were lost** — cheap only
because the runner checkpoints every 200 rows and resumes without recomputing. It wrote no
`FAILED` file, because nothing caught anything: the process was killed with the operating
system. Pausing updates is what stops this; checkpointing is only what makes it survivable.

To check afterwards what happened and when:

```powershell
(Get-CimInstance Win32_OperatingSystem).LastBootUpTime
Get-WinEvent -FilterHashtable @{LogName='System'; Id=1074,6006,6008,41; StartTime=(Get-Date).AddDays(-2)}
```

`uv run task teach --status <run_id>` prints the boot time and flags `<-- REBOOTED MID-RUN`
when the machine booted after the job's last heartbeat, so this diagnosis takes one command
rather than a trip to the event log.

Standby and monitor timeout must also be off (`powercfg`) for unattended runs.

## 3c. Freeing the GPU after an interrupted run

`uv run task gpu-kill` — kills the whole process tree of any s1decide run still holding VRAM,
and lists (never kills) anything that is not ours. `--dry-run` to look first.

Needed because a `uv run task bench` is a shell, then a launcher, then the Python process that
actually holds ~20 GB. Killing the outermost one from a task manager or an editor leaves the
model resident; worse, a surviving loop shell will start the next iteration alongside it. Two
benchmarks competing for one card produced a 14-second single-question call — 10x the true
figure — and every measurement taken in that window was void.

`doctor`'s `gpu-processes` line lists holders by name and MiB. On Windows `nvidia-smi` reports
per-process memory as `N/A` under the WDDM driver model, so the number comes from the
`GPU Process Memory` performance counter set instead (see `src/s1decide/gpu.py`).

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

**R1 — No MSVC, no Windows SDK, no CMake, no CUDA toolkit.
— RESOLVED for Triton (Step 1); partially open for C++ builds.**

The kickoff brief and `CLAUDE.md` both state the Studio installer set these up. On
this machine it did **not**: it fetched prebuilt binaries instead. MSVC really is
absent, and Triton says so out loud:

```
triton/windows_utils.py:174: UserWarning: Failed to find MSVC.
triton/windows_utils.py:273: UserWarning: Failed to find Windows SDK.
```

`find_msvc_env()`, `find_msvc_vswhere()` and `find_msvc_winsdk()` all return empty.

**And Triton compiles and launches kernels anyway.** The `triton-compile` check in
`uv run task doctor` — a real `@triton.jit` elementwise kernel, launched, output
compared against `x + y` — passes. The reason is that `triton-windows`
3.6.0.post26 ships its own toolchain inside the wheel:

| Bundled | Path under `site-packages/triton/` | Role |
|---|---|---|
| `ptxas.exe` (24 MB) | `backends/nvidia/bin/ptxas.exe` | PTX to cubin — replaces the CUDA toolkit |
| `cuda.lib` | `backends/nvidia/lib/x64/cuda.lib` | link stub |
| **TCC** (Tiny C Compiler, 24 KB) | `runtime/tcc/tcc.exe` | compiles the small C launcher shims — replaces `cl.exe` |
| `libtriton.pyd` (101 MB) | `_C/libtriton.pyd` | prebuilt compiler core |

Evidence this was compiled here and not a stale artefact: `~/.triton/cache/` gained
`cuda_utils.cp311-win_amd64.pyd` and `__triton_launcher.cp311-win_amd64.pyd` with
mtimes **12:40:33 and 12:40:35 on 2026-09-17**, i.e. during the doctor run, on a box
with no C++ compiler. (A `cp313` copy dated 2026-09-06 predates us — that is Studio's
Python, not ours.)

**Conclusion: do not install VS 2022 Build Tools.** They are not needed for Triton,
and therefore not needed for Unsloth's vendored `fla` gated-deltanet kernels.

**Confirmed in Step 3 (2026-09-17):** the kernels that matter — Unsloth's vendored
`flash-linear-attention` gated-deltanet Triton kernels — compile and run through this
bundled toolchain. `scripts/derisk_base_model.py` showed 3/3 GDN modules on the fla path,
~9x faster than the pure-torch fallback, bf16-level agreement. Cold compile of the GDN
kernels took ~29 s; warm cache ~1.7 s.

**Still open:** TCC compiles C, not C++. Anything that builds a *C++* extension at
runtime — `torch.compile`'s inductor C++ backend, `torch.utils.cpp_extension` custom
ops, a from-source `llama-cpp-python` — remains blocked. Importing, patching and kernel
compilation in Unsloth needed none of these; whether a *training step* does is answered
by `task smoke`.

**R2 — `llama-cpp-python` may not install. (medium)**
No compiler means we depend on a prebuilt CUDA wheel matching Python 3.11 + cu13x,
which is not guaranteed to exist. Mitigation already on disk: Unsloth's prebuilt
`llama-server.exe` (CUDA 13.3, sm_86) — `engine/llamacpp.py` can drive it over its
HTTP API instead of linking the library. `doctor` reports `llama-cpp-python` as a
*warning*, not a hard failure.

**R3 — Python 3.11 is not installed. — RESOLVED.** `uv sync` fetched
`cpython-3.11.16-windows-x86_64-none` (24.0 MiB) and built `.venv`. `uv.lock`
resolved `torch==2.10.0+cu130` from the pinned index, exactly as intended.

**R7 — uv's universal lock cannot hold vLLM and our pinned stack at once. (new, Step 1)**
`uv lock` resolves every extra and platform simultaneously, so a `vllm` extra — even
one marked `sys_platform == 'linux'` — must still co-resolve with the rest. It cannot:
every vLLM release pins an exact `torch` and a `transformers` range that excludes
5.5.0 (`vllm 0.11.0` wants `torch==2.8.0`; `vllm >= 0.24` wants `transformers>=5.5.3`).
Making it fit means unpinning torch and transformers, which defeats the point of
pinning. **Decision:** vLLM is not a project extra. It is a *baseline runner* for the
eval, not part of this library's dependency closure, so on the Linux box it gets its
own environment:

```bash
uv venv .venv-vllm && uv pip install --python .venv-vllm vllm
```

`src/s1decide/engine/vllm.py` stays guarded by `sys.platform != "win32"` and its tests
skip when vLLM is absent. Recorded in `pyproject.toml` next to the other extras.

**R8 — Unsloth writes into the working directory. (new, Step 1, low)**
Importing `unsloth` (which `task doctor` does) creates `unsloth_compiled_cache/` in the
CWD and fills it with generated modules. It is git-ignored and excluded from ruff.
Also note `ruff format` reformats fenced Python blocks inside Markdown, which silently
edited `docs/research/`; `*.md` is now in ruff's `extend-exclude`.

**R4 — No HF-format Qwen3.8-27B, not even the tokenizer. (low/medium)**
Step 2's tokenizer tests need the tokenizer files (small, a few MB). The 4-bit
base for training is ~17 GB and needs explicit approval.

**R5 — 68.9 GB of GGUFs already cached; C: is the only volume. (low)**
726 GB free is comfortable, but 27B BF16 (~54 GB) + 4-bit (~17 GB) + GGUF exports
add up. Track it; do not move the cache.

**R6 — Studio is not running right now (0 MB held by it).** Good, but `doctor`
must still check, since Studio holds VRAM when open.

## 8. What this file still owes

- [x] Every table above re-asserted by `uv run task doctor` (Step 1) — 15 checks,
      0 fail, 1 warn.
- [x] A real `torch.cuda.is_available()` result from **our** env, not Studio's —
      True, `torch 2.10.0+cu130`, `torch.version.cuda == "13.0"`, RTX 3090 sm_86,
      22.8 / 24.0 GiB free.
- [x] A real Triton compile probe result (R1) — passes, via bundled TCC + ptxas.
- [x] bitsandbytes 0.50.2 nf4 forward on the GPU — passes.
- [ ] Verdict on `llama-cpp-python` vs. prebuilt `llama-server.exe` (R2) —
      still a warn; decided when `engine/llamacpp.py` is written.
- [ ] Whether the Unsloth training path needs a **C++** compiler (R1, still open).
- [ ] A from-scratch reproduction recipe for a stranger, verified on a clean machine.

## 9. Environment resolved by `uv sync` (2026-09-17)

Our project venv, `.venv`, Python **3.11.16**. Differences from Studio's stack are
deliberate and listed here so a later divergence is obvious:

| Package | Ours | Studio | Note |
|---|---|---|---|
| Python | 3.11.16 | 3.13.15 | `CLAUDE.md` pins 3.11 for this project |
| `torch` | 2.10.0+cu130 | 2.10.0+cu130 | same |
| `triton-windows` | 3.6.0.post26 | 3.6.0.post26 | same |
| `bitsandbytes` | 0.50.2 | 0.50.2 | same |
| `transformers` | 5.5.0 | 5.5.0 | same |
| `peft` / `trl` | 0.18.1 / 0.23.1 | 0.18.1 / 0.23.1 | same |
| `unsloth` / `unsloth-zoo` | 2026.9.2 / 2026.9.1 | 2026.9.2 / 2026.9.1 | same |
| `numpy` | 2.2.x | 2.5.2 | **forced**: the 2.5.2 wheel requires Python >= 3.12 |
| `torchao` | 0.18.0 | 0.17.0 | pulled transitively, not pinned by us |
| `xformers` | 0.0.35 | 0.0.34 | pulled transitively, not pinned by us |

The exact resolution is committed in `uv.lock`.
