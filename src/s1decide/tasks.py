"""Cross-platform task runner — the single entry point for every project command.

This replaces a Makefile. Everything here must behave identically under PowerShell
on Windows and bash on Linux, so: pure Python, ``pathlib`` only, no shell
one-liners, no ``make``, no bash.

Usage::

    uv run task <name> [args...]
    uv run task --list

Tasks that cannot be implemented yet raise :class:`NotImplementedError`; they never
silently succeed.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

__all__ = ["TASKS", "Task", "ensure_repo_on_path", "main", "run_task"]

# The CUDA runtime the pinned torch wheel carries. The reference machine has no
# standalone CUDA toolkit and no nvcc, so this — not nvcc — is the toolkit we match.
# See docs/windows-setup.md section 4.2.
EXPECTED_TORCH_CUDA = "13.0"

# `doctor` warns when another process is holding more than this much VRAM
# (typically a forgotten Unsloth Studio session).
VRAM_WARN_MIB = 1024

# `doctor` warns when less than this is free — 27B at 4-bit needs roughly 18 GB.
VRAM_FREE_WARN_GIB = 20.0

# `doctor` warns when the HF cache volume has less headroom than this.
DISK_FREE_WARN_GIB = 100.0


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Task:
    """One invocable project command.

    Attributes:
        name: The name used on the command line, e.g. ``test-gpu``.
        summary: One-line description shown by ``--list``.
        fn: Callable taking the remaining argv and returning a process exit code.
    """

    name: str
    summary: str
    fn: Callable[[list[str]], int]


TASKS: dict[str, Task] = {}


def register(
    name: str, summary: str
) -> Callable[[Callable[[list[str]], int]], Callable[[list[str]], int]]:
    """Register a task implementation under ``name``.

    Args:
        name: Command-line name of the task.
        summary: One-line description.

    Returns:
        A decorator that registers the wrapped function and returns it unchanged.
    """

    def decorator(fn: Callable[[list[str]], int]) -> Callable[[list[str]], int]:
        TASKS[name] = Task(name=name, summary=summary, fn=fn)
        return fn

    return decorator


def register_not_implemented(name: str, summary: str, reason: str) -> None:
    """Register a task that is planned but not implemented yet.

    The task fails loudly with :class:`NotImplementedError` rather than returning 0,
    so nothing downstream can mistake it for a successful no-op.

    Args:
        name: Command-line name of the task.
        summary: One-line description.
        reason: What still has to happen before this task can work.
    """

    def fn(_argv: list[str]) -> int:
        raise NotImplementedError(f"task {name!r} is not implemented yet — {reason}")

    TASKS[name] = Task(name=name, summary=f"{summary} [not implemented]", fn=fn)


def repo_root() -> Path:
    """Return the repository root (the directory holding ``pyproject.toml``).

    Falls back to the current working directory when the package is installed
    outside a checkout.

    Returns:
        Path to the repository root.
    """
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def ensure_repo_on_path() -> Path:
    """Put the repository root on ``sys.path`` so repo-level packages are importable.

    ``eval/``, ``train/`` and ``scripts/`` are part of the repository but not of the installed
    wheel, so the ``task`` console script cannot import them by default. pytest gets this from
    the ``pythonpath`` ini option; this is the same thing for the command line.

    Returns:
        The repository root.
    """
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _run(cmd: Sequence[str], cwd: Path | None = None) -> int:
    """Run a subprocess without a shell and stream its output.

    Args:
        cmd: Argument vector. Never a string, never ``shell=True``.
        cwd: Working directory; defaults to the repository root.

    Returns:
        The subprocess exit code.
    """
    printable = " ".join(str(part) for part in cmd)
    print(f"$ {printable}", flush=True)
    return subprocess.run(list(cmd), cwd=str(cwd or repo_root()), check=False).returncode


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #


class Status(StrEnum):
    """Outcome of a single :func:`doctor` check."""

    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"
    INFO = "INFO"


@dataclass(frozen=True)
class CheckResult:
    """Result of one environment check.

    Attributes:
        name: Short check identifier.
        status: Outcome.
        detail: Human-readable explanation, one line where possible.
    """

    name: str
    status: Status
    detail: str


def _first_line(exc: BaseException, limit: int = 220) -> str:
    """Render an exception as a single truncated line.

    Args:
        exc: The exception to render.
        limit: Maximum characters to keep.

    Returns:
        ``ExceptionType: first line of message``, truncated.
    """
    text = str(exc).strip().splitlines()
    head = text[0] if text else ""
    rendered = f"{type(exc).__name__}: {head}"
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


def check_platform() -> CheckResult:
    """Report OS, Python and interpreter location."""
    return CheckResult(
        "platform",
        Status.INFO,
        # platform.release() reports "10" on Windows 11, so include the build number.
        f"{platform.system()} {platform.version()} ({platform.machine()}) | "
        f"Python {platform.python_version()} | {sys.executable}",
    )


def check_gpu_processes() -> CheckResult:
    """Report VRAM already in use, before this process touches CUDA.

    Runs first so our own CUDA context does not pollute the reading. On Windows
    (WDDM) ``nvidia-smi`` reports per-process memory as ``N/A``, so we read the
    device total instead and list the process names.
    """
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return CheckResult("gpu-processes", Status.SKIP, "nvidia-smi not on PATH")
    try:
        used = (
            subprocess.run(
                [smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            .stdout.strip()
            .splitlines()
        )
        apps = subprocess.run(
            [smi, "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        return CheckResult("gpu-processes", Status.WARN, _first_line(exc))

    try:
        used_mib = max(int(line.strip()) for line in used if line.strip())
    except ValueError:
        return CheckResult(
            "gpu-processes", Status.WARN, f"could not parse nvidia-smi output: {used!r}"
        )

    # On a desktop Windows box this list is dominated by the compositor, the browser and
    # tray apps. Show the basenames of a handful and count the rest.
    names = sorted(
        {Path(line.split(",", 1)[-1].strip()).name for line in apps.splitlines() if line.strip()}
    )
    if not names:
        suffix = ""
    elif len(names) <= 4:
        suffix = f" | {', '.join(names)}"
    else:
        suffix = f" | {', '.join(names[:4])} and {len(names) - 4} more"
    if used_mib > VRAM_WARN_MIB:
        return CheckResult(
            "gpu-processes",
            Status.WARN,
            f"{used_mib} MiB already in use (> {VRAM_WARN_MIB} MiB). "
            f"Close Unsloth Studio and any other GPU job before training or benchmarking.{suffix}",
        )
    return CheckResult("gpu-processes", Status.OK, f"{used_mib} MiB in use{suffix}")


def check_torch() -> CheckResult:
    """Import torch and report its version."""
    try:
        import torch
    except Exception as exc:
        return CheckResult("torch", Status.FAIL, _first_line(exc))
    return CheckResult("torch", Status.OK, f"{torch.__version__}")


def check_cuda_available() -> CheckResult:
    """Assert that torch actually sees the GPU.

    This is the check that catches the documented Unsloth-on-Windows failure where
    a CPU-only torch wheel gets installed and training silently runs on CPU.
    """
    try:
        import torch
    except Exception as exc:
        return CheckResult("cuda-available", Status.FAIL, _first_line(exc))
    if not torch.cuda.is_available():
        return CheckResult(
            "cuda-available",
            Status.FAIL,
            f"torch.cuda.is_available() is False — torch {torch.__version__} looks like a CPU build",
        )
    return CheckResult("cuda-available", Status.OK, f"{torch.cuda.device_count()} device(s)")


def check_cuda_version() -> CheckResult:
    """Compare torch's bundled CUDA runtime against the pinned expectation."""
    try:
        import torch
    except Exception as exc:
        return CheckResult("cuda-version", Status.FAIL, _first_line(exc))
    got = torch.version.cuda
    if got is None:
        return CheckResult("cuda-version", Status.FAIL, "torch.version.cuda is None (CPU build)")
    if got != EXPECTED_TORCH_CUDA:
        return CheckResult(
            "cuda-version",
            Status.WARN,
            f"torch CUDA {got}, pinned expectation {EXPECTED_TORCH_CUDA} "
            "(fine on a different host; update EXPECTED_TORCH_CUDA if intended)",
        )
    return CheckResult("cuda-version", Status.OK, f"torch CUDA {got}")


def check_gpu() -> CheckResult:
    """Report GPU name, compute capability and free/total VRAM."""
    try:
        import torch

        if not torch.cuda.is_available():
            return CheckResult("gpu", Status.SKIP, "no CUDA device")
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        free_b, total_b = torch.cuda.mem_get_info(0)
    except Exception as exc:
        return CheckResult("gpu", Status.FAIL, _first_line(exc))

    free_gib = free_b / 1024**3
    total_gib = total_b / 1024**3
    detail = f"{name} | sm_{major}{minor} | {free_gib:.1f} / {total_gib:.1f} GiB free"
    if free_gib < VRAM_FREE_WARN_GIB:
        return CheckResult("gpu", Status.WARN, f"{detail} (< {VRAM_FREE_WARN_GIB:.0f} GiB free)")
    return CheckResult("gpu", Status.OK, detail)


def check_bitsandbytes() -> CheckResult:
    """Quantize and run a tiny 4-bit layer on the GPU.

    Proves the Windows bitsandbytes build actually works, not merely that it imports.
    """
    try:
        import bitsandbytes as bnb
        import torch

        if not torch.cuda.is_available():
            return CheckResult("bnb-4bit", Status.SKIP, "no CUDA device")
        layer = bnb.nn.Linear4bit(
            64, 64, bias=False, compute_dtype=torch.float16, quant_type="nf4"
        ).to("cuda")
        x = torch.randn(2, 64, device="cuda", dtype=torch.float16)
        y = layer(x)
        if not torch.isfinite(y).all():
            return CheckResult("bnb-4bit", Status.FAIL, "4-bit forward produced non-finite values")
    except Exception as exc:
        return CheckResult("bnb-4bit", Status.FAIL, _first_line(exc))
    return CheckResult("bnb-4bit", Status.OK, f"bitsandbytes {bnb.__version__} nf4 forward ok")


def check_triton_import() -> CheckResult:
    """Import Triton and report its version."""
    try:
        import triton
    except Exception as exc:
        return CheckResult("triton-import", Status.FAIL, _first_line(exc))
    return CheckResult("triton-import", Status.OK, f"triton {triton.__version__}")


def check_triton_compile() -> CheckResult:
    """Compile and launch a real Triton kernel.

    This is the decisive check for risk R1 in docs/windows-setup.md: Triton JITs
    kernels at runtime and on Windows needs MSVC + the Windows SDK to do it. An
    import success proves nothing; only a launched kernel does.
    """
    try:
        import torch
        import triton
        import triton.language as tl

        if not torch.cuda.is_available():
            return CheckResult("triton-compile", Status.SKIP, "no CUDA device")

        @triton.jit
        def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            tl.store(
                out_ptr + offs,
                tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask),
                mask=mask,
            )

        n = 1024
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        out = torch.empty_like(x)
        _add_kernel[(1,)](x, y, out, n, BLOCK=1024)
        torch.cuda.synchronize()
        if not torch.allclose(out, x + y, atol=1e-5):
            return CheckResult(
                "triton-compile", Status.FAIL, "kernel ran but produced wrong values"
            )
    except Exception as exc:
        return CheckResult(
            "triton-compile",
            Status.FAIL,
            f"{_first_line(exc)} | on Windows this usually means MSVC / Windows SDK is missing "
            "(see docs/windows-setup.md R1)",
        )
    return CheckResult("triton-compile", Status.OK, "jit kernel compiled and launched")


def check_unsloth() -> CheckResult:
    """Import Unsloth Core and report its version."""
    try:
        import unsloth
    except Exception as exc:
        return CheckResult("unsloth", Status.FAIL, _first_line(exc))
    return CheckResult(
        "unsloth", Status.OK, f"unsloth {getattr(unsloth, '__version__', 'unknown')}"
    )


def check_llama_cpp() -> CheckResult:
    """Import llama-cpp-python if present.

    Warn-only: on Windows without MSVC this package can only be installed from a
    prebuilt CUDA wheel, and the GGUF engine has a fallback that drives Unsloth's
    prebuilt ``llama-server.exe`` over HTTP instead.
    """
    try:
        import llama_cpp
    except Exception as exc:
        return CheckResult(
            "llama-cpp-python",
            Status.WARN,
            f"not available ({_first_line(exc)}) — GGUF engine unavailable; optional extra 'llamacpp'",
        )
    return CheckResult("llama-cpp-python", Status.OK, f"llama_cpp {llama_cpp.__version__}")


def check_no_flash_attn() -> CheckResult:
    """Assert flash-attn is absent.

    A hard rule in CLAUDE.md: there is no FlashAttention on Windows and we use SDPA
    everywhere, so the dependency must never creep in.
    """
    import importlib.util

    if importlib.util.find_spec("flash_attn") is not None:
        return CheckResult(
            "no-flash-attn",
            Status.FAIL,
            "flash_attn is installed — forbidden by CLAUDE.md; use attn_implementation='sdpa'",
        )
    return CheckResult("no-flash-attn", Status.OK, "absent, as required")


def hf_cache_dir() -> Path:
    """Resolve the Hugging Face hub cache directory.

    Returns:
        Path to the hub cache, from ``huggingface_hub`` when importable and from the
        documented default layout otherwise.
    """
    try:
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE)
    except Exception:
        env = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
        if env:
            base = Path(env)
            return base if base.name == "hub" else base / "hub"
        return Path.home() / ".cache" / "huggingface" / "hub"


def check_hf_cache() -> CheckResult:
    """Check the HF cache path for spaces, length and free disk space."""
    cache = hf_cache_dir()
    problems: list[str] = []
    if " " in str(cache):
        problems.append("path contains spaces (breaks some Windows tooling)")
    if len(str(cache)) > 80:
        problems.append(f"path is long ({len(str(cache))} chars)")

    free_gib: float | None = None
    probe = cache
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free_gib = shutil.disk_usage(probe).free / 1024**3
    except OSError as exc:
        problems.append(_first_line(exc))
    if free_gib is not None and free_gib < DISK_FREE_WARN_GIB:
        problems.append(f"only {free_gib:.0f} GiB free on that volume")

    space = f" | {free_gib:.0f} GiB free" if free_gib is not None else ""
    if problems:
        return CheckResult("hf-cache", Status.WARN, f"{cache}{space} — " + "; ".join(problems))
    return CheckResult("hf-cache", Status.OK, f"{cache}{space}")


def check_utf8() -> CheckResult:
    """Check that Python is running in UTF-8 mode."""
    utf8_env = os.environ.get("PYTHONUTF8")
    encoding = (sys.stdout.encoding or "").lower()
    if utf8_env == "1" or encoding.replace("-", "") == "utf8":
        return CheckResult("utf-8", Status.OK, f"PYTHONUTF8={utf8_env!r}, stdout={encoding!r}")
    return CheckResult(
        "utf-8",
        Status.WARN,
        f"PYTHONUTF8={utf8_env!r}, stdout={encoding!r} — set PYTHONUTF8=1 to avoid mojibake in data files",
    )


def check_long_paths() -> CheckResult:
    """Check that git long-path support is enabled (Windows only).

    Triton compilation and deep HF cache paths both exceed the legacy 260-character
    limit. Reports, never fixes.
    """
    if sys.platform != "win32":
        return CheckResult("git-longpaths", Status.SKIP, "not Windows")
    git = shutil.which("git")
    if git is None:
        return CheckResult("git-longpaths", Status.WARN, "git not on PATH")
    try:
        out = subprocess.run(
            [git, "config", "--get", "core.longpaths"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            cwd=str(repo_root()),
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        return CheckResult("git-longpaths", Status.WARN, _first_line(exc))
    if out.lower() != "true":
        return CheckResult(
            "git-longpaths",
            Status.WARN,
            "core.longpaths is not true — run: git config core.longpaths true",
        )
    return CheckResult("git-longpaths", Status.OK, "core.longpaths=true")


#: Checks run by `doctor`, in order. `gpu-processes` runs before anything imports
#: torch so that our own CUDA context is not counted as "someone else's VRAM".
DOCTOR_CHECKS: tuple[Callable[[], CheckResult], ...] = (
    check_platform,
    check_gpu_processes,
    check_torch,
    check_cuda_available,
    check_cuda_version,
    check_gpu,
    check_bitsandbytes,
    check_triton_import,
    check_triton_compile,
    check_unsloth,
    check_llama_cpp,
    check_no_flash_attn,
    check_hf_cache,
    check_utf8,
    check_long_paths,
)


def run_doctor() -> list[CheckResult]:
    """Run every environment check.

    Returns:
        One :class:`CheckResult` per check, in declaration order. Never raises:
        a check that blows up is reported as a FAIL.
    """
    results: list[CheckResult] = []
    for check in DOCTOR_CHECKS:
        try:
            results.append(check())
        except Exception as exc:
            results.append(CheckResult(check.__name__, Status.FAIL, _first_line(exc)))
    return results


@register("doctor", "Check GPU, CUDA, torch, bnb, triton, unsloth, paths and encoding")
def task_doctor(_argv: list[str]) -> int:
    """Print the environment report and exit non-zero if anything failed."""
    results = run_doctor()
    width = max(len(r.name) for r in results)
    for r in results:
        print(f"[{r.status.value:<4}] {r.name:<{width}}  {r.detail}")

    fails = [r for r in results if r.status is Status.FAIL]
    warns = [r for r in results if r.status is Status.WARN]
    print(
        f"\ndoctor: {len(results)} checks — "
        f"{len(fails)} fail, {len(warns)} warn, "
        f"{sum(1 for r in results if r.status is Status.OK)} ok"
    )
    if fails:
        print("failed: " + ", ".join(r.name for r in fails))
    return 1 if fails else 0


# --------------------------------------------------------------------------- #
# implemented tasks
# --------------------------------------------------------------------------- #


@register("setup", "uv sync the project environment")
def task_setup(argv: list[str]) -> int:
    """Sync the environment with ``uv``. Extra arguments are passed to ``uv sync``."""
    uv = shutil.which("uv")
    if uv is None:
        print("uv is not on PATH — install it with: winget install astral-sh.uv", file=sys.stderr)
        return 1
    return _run([uv, "sync", *argv])


@register("test", "Run the CPU test suite (pytest -m 'not gpu')")
def task_test(argv: list[str]) -> int:
    """Run pytest, excluding GPU-marked tests."""
    return _run([sys.executable, "-m", "pytest", "-m", "not gpu", *argv])


@register("test-gpu", "Run the GPU test suite (pytest -m gpu)")
def task_test_gpu(argv: list[str]) -> int:
    """Run pytest, only GPU-marked tests."""
    return _run([sys.executable, "-m", "pytest", "-m", "gpu", *argv])


@register("lint", "Run ruff check and ruff format --check")
def task_lint(argv: list[str]) -> int:
    """Lint and check formatting without modifying files."""
    rc = _run([sys.executable, "-m", "ruff", "check", ".", *argv])
    rc |= _run([sys.executable, "-m", "ruff", "format", "--check", "."])
    return rc


@register("fmt", "Apply ruff format and ruff check --fix")
def task_fmt(argv: list[str]) -> int:
    """Format the tree and apply safe lint fixes."""
    rc = _run([sys.executable, "-m", "ruff", "format", ".", *argv])
    rc |= _run([sys.executable, "-m", "ruff", "check", "--fix", "."])
    return rc


# --------------------------------------------------------------------------- #
# planned tasks — fail loudly until implemented
# --------------------------------------------------------------------------- #


@register("data", "Fetch, normalise, augment and split sources into data/processed/")
def task_data(argv: list[str]) -> int:
    """Build the dataset. ``--limit N`` caps rows per source for a smoke build."""
    import argparse
    import json

    ensure_repo_on_path()
    from data.build.pipeline import BuildConfig, build

    parser = argparse.ArgumentParser(prog="task data")
    parser.add_argument("--limit", type=int, default=None, help="cap rows per source")
    parser.add_argument("--seed", type=int, default=BuildConfig.seed)
    args = parser.parse_args(argv)

    manifest = build(BuildConfig(seed=args.seed, limit_per_source=args.limit))

    from data.build.report import write_build_report

    root = repo_root()
    report = write_build_report(
        root / "data" / "processed" / "manifest.json", root / "docs" / "dataset-build.md"
    )
    print(json.dumps({k: manifest[k] for k in ("rows", "splits", "by_family")}, indent=2))
    print(f"wrote {report.relative_to(root)}")
    print(f"two-stage parents expanded: {manifest['two_stage_parents_expanded']}")
    print(f"leakage: {manifest['leakage']}")
    return 0


register_not_implemented(
    "smoke",
    "200-sample end-to-end run: data -> tiny LoRA -> eval, <= 5 min on a 3090",
    "needs `data` and `train` first",
)
register_not_implemented(
    "train",
    "Stage-1 QLoRA SFT from a train/configs/*.yaml config",
    "Phase 1 training-engineer work; needs train/sft_lora.py",
)


@register("eval", "Run an evaluation and write results/<run_id>/metrics.json")
def task_eval(argv: list[str]) -> int:
    """Run `eval/run_eval.py`. Arguments are passed straight through."""
    ensure_repo_on_path()
    from eval.run_eval import main as run_eval_main

    return run_eval_main(argv)


@register("bench", "Latency vs number of questions on the current engine")
def task_bench(argv: list[str]) -> int:
    """Run `eval/latency_bench.py`. Arguments are passed straight through."""
    ensure_repo_on_path()
    from eval.latency_bench import main as bench_main

    return bench_main(argv)


register_not_implemented(
    "serve",
    "Run the /decide FastAPI server",
    "needs src/s1decide/server.py and a working engine",
)
register_not_implemented(
    "gguf",
    "Merge LoRA to BF16 and quantize to Q8 / Q5_K_M / Q4_K_M",
    "needs scripts/merge_lora.py and scripts/convert_gguf.py",
)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def _usage() -> str:
    """Render the task list for ``--help`` / ``--list``."""
    width = max(len(name) for name in TASKS)
    lines = ["usage: uv run task <name> [args...]", "", "tasks:"]
    lines += [f"  {name:<{width}}  {task.summary}" for name, task in TASKS.items()]
    return "\n".join(lines)


def run_task(name: str, argv: list[str] | None = None) -> int:
    """Run a single task by name.

    Args:
        name: Registered task name.
        argv: Arguments forwarded to the task.

    Returns:
        The task's exit code.

    Raises:
        KeyError: If no task is registered under ``name``.
        NotImplementedError: If the task is registered but not implemented yet.
    """
    if name not in TASKS:
        raise KeyError(name)
    return TASKS[name].fn(list(argv or []))


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for ``uv run task``.

    Args:
        argv: Argument vector excluding the program name; defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code. ``2`` for usage errors and for tasks that are not
        implemented yet; otherwise the task's own exit code.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "--list", "help", "list"}:
        print(_usage())
        return 0 if args else 2

    name, rest = args[0], args[1:]
    try:
        return run_task(name, rest)
    except KeyError:
        print(f"unknown task: {name}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2
    except NotImplementedError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
