"""GPU process inventory and cleanup, plus the sysmem-fallback probe.

Two Windows-specific facts drive this module.

**Per-process VRAM is not visible through ``nvidia-smi`` on Windows.** Under the WDDM driver
model the display driver owns the allocations, so ``--query-compute-apps=used_memory`` returns
``N/A`` for every process. Windows does expose the number, through the ``GPU Process Memory``
performance counter set, keyed by an instance name that embeds the PID. That is what
:func:`gpu_processes` reads there; on Linux it reads ``nvidia-smi`` directly.

**Killing a run's shell does not kill the run.** A ``uv run task bench`` is a shell, then a
launcher, then the Python process that actually holds ~20 GB of VRAM. Killing the outermost one
leaves the model resident and the GPU unusable, and any measurement taken afterwards is
garbage. :func:`kill_process_tree` kills children first, parent last.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

__all__ = [
    "GpuProcess",
    "SysmemFallback",
    "gpu_processes",
    "kill_process_tree",
    "kill_s1decide_gpu_processes",
    "probe_sysmem_fallback",
]

#: Instance names look like ``pid_1964_luid_0x00000000_0x0000e1b5_phys_0``.
_PID_IN_INSTANCE = re.compile(r"pid_(\d+)_")

#: Processes below this are desktop noise (compositor, browser tabs, tray apps).
MIN_REPORTED_MIB = 64

#: Substrings that mark a GPU process as one of ours rather than the user's.
OUR_MARKERS: tuple[str, ...] = ("s1decide", "task.exe", "uv run task")


@dataclass(frozen=True)
class GpuProcess:
    """One process holding device memory.

    Attributes:
        pid: Process id.
        name: Executable name, or ``"?"`` when the process has already exited.
        mib: Dedicated GPU memory in MiB, or ``None`` when the platform will not report it.
        command: Full command line where readable, used to tell our runs from the user's.
    """

    pid: int
    name: str
    mib: int | None
    command: str = ""

    @property
    def is_ours(self) -> bool:
        """Whether this looks like an s1decide run rather than someone else's work."""
        haystack = f"{self.name} {self.command}".lower()
        return any(marker in haystack for marker in OUR_MARKERS)

    def describe(self) -> str:
        """One line for the doctor report."""
        size = "?" if self.mib is None else f"{self.mib} MiB"
        return f"{self.name} (pid {self.pid}) {size}{' [ours]' if self.is_ours else ''}"


def _command_line(pid: int) -> tuple[str, str]:
    """Return ``(name, command_line)`` for a pid, or ``("?", "")`` if it has gone."""
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is a declared dependency
        return "?", ""
    try:
        process = psutil.Process(pid)
        return process.name(), " ".join(process.cmdline())
    except Exception:
        # Exited between listing and inspection, or access denied for a system process.
        return "?", ""


def _windows_gpu_processes() -> list[GpuProcess]:
    """Read per-process dedicated GPU memory from the Windows performance counters."""
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        return []
    script = (
        "(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' -ErrorAction Stop)."
        'CounterSamples | ForEach-Object { "$($_.InstanceName)=$($_.CookedValue)" }'
    )
    try:
        completed = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return []

    # One counter instance per (process, adapter); a process on one GPU still reports once per
    # LUID, so sum rather than take the first.
    by_pid: dict[int, float] = {}
    for line in completed.stdout.splitlines():
        instance, _, value = line.strip().partition("=")
        match = _PID_IN_INSTANCE.search(instance)
        if not match or not value:
            continue
        try:
            by_pid[int(match.group(1))] = by_pid.get(int(match.group(1)), 0.0) + float(value)
        except ValueError:
            continue

    processes = []
    for pid, byte_count in by_pid.items():
        mib = int(byte_count / 2**20)
        if mib < MIN_REPORTED_MIB:
            continue
        name, command = _command_line(pid)
        processes.append(GpuProcess(pid=pid, name=name, mib=mib, command=command))
    return sorted(processes, key=lambda p: -(p.mib or 0))


def _nvidia_smi_gpu_processes() -> list[GpuProcess]:
    """Read per-process memory from ``nvidia-smi`` (accurate off Windows)."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return []
    try:
        completed = subprocess.run(
            [smi, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return []

    processes = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        mib = int(parts[1]) if parts[1].isdigit() else None
        name, command = _command_line(pid)
        processes.append(GpuProcess(pid=pid, name=name, mib=mib, command=command))
    return sorted(processes, key=lambda p: -(p.mib or 0))


def gpu_processes() -> list[GpuProcess]:
    """Every process holding device memory, largest first.

    Returns:
        The processes, with per-process MiB where the platform reports it. Empty when neither
        source is available — which is not the same as "nothing is using the GPU", and callers
        should say so rather than claiming the card is free.
    """
    if sys.platform == "win32":
        found = _windows_gpu_processes()
        if found:
            return found
    return _nvidia_smi_gpu_processes()


def kill_process_tree(pid: int, *, include_self: bool = True, timeout: float = 5.0) -> list[int]:
    """Kill a process and every descendant, children first.

    Children first because killing the parent orphans them: a ``uv run task bench`` shell can
    exit while the Python process holding 20 GB of VRAM keeps running, and the next
    measurement on that card is meaningless.

    Args:
        pid: Root of the tree.
        include_self: Whether to kill ``pid`` itself as well as its descendants.
        timeout: Seconds to wait for the processes to die.

    Returns:
        The pids actually killed.
    """
    import psutil

    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return []
    victims = root.children(recursive=True)
    if include_self:
        victims.append(root)

    killed = []
    for victim in victims:
        try:
            victim.kill()
            killed.append(victim.pid)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            continue
    psutil.wait_procs(victims, timeout=timeout)
    return killed


def kill_s1decide_gpu_processes(*, dry_run: bool = False) -> tuple[list[GpuProcess], list[int]]:
    """Kill the process tree of every s1decide run holding GPU memory.

    Only processes :attr:`GpuProcess.is_ours` recognises are touched — a stray browser or
    Unsloth Studio is reported, never killed.

    Args:
        dry_run: List what would be killed without killing it.

    Returns:
        ``(ours, killed_pids)``. ``ours`` is what matched; ``killed_pids`` is empty on a dry run.
    """
    import psutil

    ours = [p for p in gpu_processes() if p.is_ours and p.pid != os.getpid()]
    if dry_run:
        return ours, []

    killed: list[int] = []
    for process in ours:
        # Walk up to the outermost ancestor that is still recognisably one of our runs, so the
        # launcher shell goes too and cannot start the next iteration of a loop.
        root = process.pid
        try:
            current = psutil.Process(process.pid)
            while current.parent() is not None:
                parent = current.parent()
                haystack = f"{parent.name()} {' '.join(parent.cmdline())}".lower()
                if not any(marker in haystack for marker in OUR_MARKERS):
                    break
                current = parent
                root = parent.pid
        except Exception:
            pass
        killed.extend(kill_process_tree(root))
    return ours, sorted(set(killed))


@dataclass(frozen=True)
class SysmemFallback:
    """Result of probing whether the driver pages device memory to host RAM.

    Attributes:
        disabled: True when an over-large allocation raised instead of silently succeeding —
            the setting we want. ``None`` when the probe could not run.
        detail: What happened, for the doctor line.
    """

    disabled: bool | None
    detail: str


def probe_sysmem_fallback(headroom_mib: int = 2048) -> SysmemFallback:
    """Test directly whether an over-budget allocation fails loudly or pages silently.

    This probes the behaviour rather than the setting. The NVIDIA Control Panel's "CUDA -
    Sysmem Fallback Policy" lives in the binary driver profile database, not in a documented
    registry value, so reading it back is unreliable; what the project actually needs to know
    is whether an allocation past the card's capacity raises. It should: when it does not, a
    run that overcommits VRAM does not fail, it just gets several times slower with no error,
    which is how a 22.3 GiB peak turned a 5.8 s call into 16 s (ADR 0003).

    Args:
        headroom_mib: How far past free memory to reach. 2 GiB by default, not a few hundred
            MiB: ``mem_get_info``'s "free" excludes a driver reserve that varies, so a small
            margin lands inside the noise and the probe contradicts itself between runs (it
            did, at 512 MiB). 2 GiB is unambiguously past the card while still bounding how
            much gets paged if the fallback is on.

    Returns:
        A :class:`SysmemFallback`. ``disabled=True`` means the allocation raised, which is the
        healthy outcome.
    """
    try:
        import torch
    except ImportError:
        return SysmemFallback(None, "torch not installed")
    if not torch.cuda.is_available():
        return SysmemFallback(None, "CUDA not available")

    free, total = torch.cuda.mem_get_info()
    request = free + headroom_mib * 2**20
    # Stated in the result so the reading can be audited rather than trusted.
    context = (
        f"requested {request / 2**30:.2f} GiB with {free / 2**30:.2f} GiB free of "
        f"{total / 2**30:.2f} GiB"
    )
    buffer = None
    try:
        buffer = torch.empty(request, dtype=torch.uint8, device="cuda")
    except torch.cuda.OutOfMemoryError:
        return SysmemFallback(True, f"{context}: raised OutOfMemoryError, as it should")
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            return SysmemFallback(None, f"probe failed: {type(exc).__name__}: {exc}")
        return SysmemFallback(True, f"{context}: raised out of memory, as it should")
    else:
        return SysmemFallback(
            False,
            f"{context}: SUCCEEDED — the driver is paging device memory to host RAM, so an "
            "over-budget run will silently get slower instead of failing",
        )
    finally:
        del buffer
        torch.cuda.empty_cache()
