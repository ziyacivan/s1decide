"""Which accelerated kernels are actually active.

Recorded in every run's metadata because a kernel change is a result change. `CLAUDE.md`'s
conventions make that explicit: **a kernel or attention backend change is a new result row,
never a silent upgrade.** A run whose metadata does not say which kernels it used cannot be
compared with one that does.

The awkward part is that "is the fast path on?" has no single answer on this machine. The
Qwen3.5 hybrid needs four functions, and transformers reports availability as `all()` of them —
so a build where the two expensive gated-deltanet kernels are present but the two cheap
depthwise-convolution ones are not logs *"the fast path is not available … falling back to
torch"* while half of it, and the expensive half, is running accelerated. That message is wrong
in the direction that matters, so this module reports each kernel separately rather than
repeating it.
"""

from __future__ import annotations

import importlib.util
from typing import Any

__all__ = [
    "FAST_PATH_KERNELS",
    "RECORDED_PACKAGES",
    "kernel_report",
    "kernel_summary",
    "package_versions",
]

#: The per-kernel attributes transformers resolves on the Qwen3.5 gated-deltanet layer. Each is
#: either an accelerated implementation or ``None``, in which case the layer uses its torch
#: fallback for that one function only.
FAST_PATH_KERNELS: tuple[str, ...] = (
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
    "causal_conv1d_fn",
    "causal_conv1d_update",
)


#: Distributions whose version every run records. `None` means not installed.
RECORDED_PACKAGES: tuple[str, ...] = (
    "torch",
    "transformers",
    "peft",
    "trl",
    "bitsandbytes",
    "unsloth",
    "unsloth_zoo",
    "triton",
    "triton-windows",
    "kernels",
    "flash-linear-attention",
    "accelerate",
)


def package_versions(names: tuple[str, ...] = RECORDED_PACKAGES) -> dict[str, str | None]:
    """Installed version of each distribution, ``None`` when it is not installed."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None  # recorded as absent; absence is the finding, not an error
    return out


def _installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def kernel_report() -> dict[str, Any]:
    """Everything a run should record about its kernel configuration.

    Returns:
        ``{"packages", "gated_deltanet", "fast_path_all", "attention", "notes"}``.
        ``gated_deltanet`` maps each kernel name to ``"accelerated"`` or ``"torch"``.
        ``fast_path_all`` is transformers' own all-or-nothing flag, kept so a reader can see
        why the log says what it says, not because it is the useful number.
    """
    packages = {
        name: _installed(name)
        for name in ("fla", "causal_conv1d", "triton", "triton_kernels", "kernels")
    }
    # Versions, not just presence: the teacher-2 runs recorded `kernels: true` and nothing
    # more, and which version they used can now only be inferred (dataset card).
    versions = package_versions()

    gated: dict[str, str] = {}
    fast_path_all: bool | None = None
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

        for name in FAST_PATH_KERNELS:
            gated[name] = "accelerated" if getattr(qwen, name, None) is not None else "torch"
        fast_path_all = bool(getattr(qwen, "is_fast_path_available", False))
    except ImportError:
        gated = dict.fromkeys(FAST_PATH_KERNELS, "unknown")

    accelerated = sum(1 for v in gated.values() if v == "accelerated")
    notes = []
    if accelerated and not fast_path_all:
        notes.append(
            f"{accelerated} of {len(gated)} gated-deltanet kernels are accelerated, but "
            "transformers reports is_fast_path_available=False because it is an all() over the "
            "four; the 'falling back to torch' warning in the log understates what is active"
        )
    if not packages.get("causal_conv1d"):
        notes.append(
            "causal-conv1d has no Windows wheel and fails to build; its two functions "
            "use the torch path (a 4-wide depthwise convolution, the cheap part)"
        )
    if not packages.get("kernels"):
        # Measured 2026-09-23 (results/teacher-quant-2026-09-23): with the `kernels` package,
        # gpt-oss-20b keeps its 24 expert blocks in MXFP4 (12.8 GiB after load); without it
        # transformers dequantizes them to bf16 (~42 GB), which cannot load on a 24 GB card.
        notes.append(
            "the `kernels` package is absent: MXFP4 checkpoints dequantize to bf16 and a 20B "
            "MoE will not fit on a 24 GB card; `kernels` is not in uv.lock"
        )

    return {
        "packages": packages,
        "versions": versions,
        "gated_deltanet": gated,
        "fast_path_all": fast_path_all,
        "attention": "sdpa",
        "notes": notes,
    }


def kernel_summary() -> str:
    """One line for the doctor report."""
    report = kernel_report()
    gated = report["gated_deltanet"]
    accelerated = sorted(k for k, v in gated.items() if v == "accelerated")
    torch_path = sorted(k for k, v in gated.items() if v == "torch")
    parts = [f"gdn {len(accelerated)}/{len(gated)} accelerated"]
    if accelerated:
        parts.append("fla: " + ", ".join(k.replace("_gated_delta_rule", "") for k in accelerated))
    if torch_path:
        parts.append(
            "torch: " + ", ".join(k.replace("causal_conv1d_", "conv:") for k in torch_path)
        )
    return " | ".join(parts)
