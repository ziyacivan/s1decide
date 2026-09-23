"""Per-hardware inference tuning, so a constant fitted on one card is not applied to another.

The rows-per-pass budget depends on facts about the machine, not about the model: how close to
the card's capacity a peak may go before something bad happens, and how much peak VRAM a
broadcast row costs beyond the cache bytes it holds. Both were measured on the 3090 under
Windows (ADR 0003). Hard-coding them would silently mis-tune the H100 runs, so they live here,
keyed by the same ``hardware:`` names the training configs use.

**Every profile states whether its numbers are measured or assumed.** An assumed profile is a
starting point for a measurement, not a result, and :attr:`HardwareProfile.measured` says which
it is so a report can never quote an unvalidated constant as a finding.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "PROFILES",
    "HardwareProfile",
    "detect_profile",
    "get_profile",
    "load_training_envelope",
]


@dataclass(frozen=True)
class HardwareProfile:
    """Inference tuning for one class of machine.

    Attributes:
        name: The ``hardware:`` key, matching ``train/configs/*.yaml``.
        peak_ceiling_fraction: The largest fraction of total VRAM a predicted peak may reach.
        peak_bytes_per_cache_byte: Peak VRAM a broadcast row costs as a multiple of its cache
            bytes, keyed by whether the expanded buffer is reused across passes.
        reuse_buffer: Default for :class:`~s1decide.engine.hf.HFEngine`'s buffer reuse.
        measured: Whether these numbers come from a benchmark on this hardware or are an
            informed starting point awaiting one.
        note: Why the numbers are what they are.
        training_envelope: Repository-relative path of the measured training envelope — peak
            VRAM and throughput per (seq cap, rank, batch) — written by
            ``eval/memory_envelope.py`` from the runs' own summaries. Empty when unmeasured.
    """

    name: str
    peak_ceiling_fraction: float
    peak_bytes_per_cache_byte: dict[bool, float] = field(default_factory=dict)
    reuse_buffer: bool = False
    measured: bool = False
    note: str = ""
    training_envelope: str = ""

    def to_json(self) -> dict[str, Any]:
        """Serialise into a run's ``meta``, so a result records what tuning produced it."""
        return {
            "name": self.name,
            "peak_ceiling_fraction": self.peak_ceiling_fraction,
            "peak_bytes_per_cache_byte": {
                str(k): v for k, v in self.peak_bytes_per_cache_byte.items()
            },
            "reuse_buffer": self.reuse_buffer,
            "measured": self.measured,
            "training_envelope": self.training_envelope,
        }


PROFILES: dict[str, HardwareProfile] = {
    "rtx3090_windows": HardwareProfile(
        name="rtx3090_windows",
        # 0.92 of 24 GiB is 22.08 GiB. Measured cliff: 21.90 GiB was fine, 22.34 GiB was 2.8x
        # slower and 22.78 GiB 6.4x slower, with no error — the Windows driver's sysmem
        # fallback paging device memory to host. `uv run task doctor` checks that the fallback
        # is turned off; the ceiling stays regardless, because the setting is a user preference
        # on a machine we do not fully control.
        peak_ceiling_fraction=0.92,
        peak_bytes_per_cache_byte={False: 1.4, True: 2.1},
        # Off: reuse saves ~1% but holds 2.1x its cache bytes instead of 1.4x, which costs three
        # rows per pass under the ceiling above. Three rows are worth ~15%.
        reuse_buffer=False,
        measured=True,
        note="ADR 0003 rows-per-pass sweep, Qwen3.8-27B nf4, 1,617-token prefix",
        training_envelope="results/memory-envelope-2026-09-23/envelope.json",
    ),
    "h100_linux": HardwareProfile(
        name="h100_linux",
        # Linux raises OOM rather than paging, so the ceiling exists to leave room for
        # allocator fragmentation, not to dodge a silent cliff. Assumed, not measured.
        peak_ceiling_fraction=0.95,
        peak_bytes_per_cache_byte={False: 1.4, True: 2.1},
        # Expected to win here and NOT YET MEASURED: 80 GiB means rows-per-pass is not VRAM
        # bound, so reuse's ~1% costs nothing in rows. Re-measure before trusting it (ADR 0003).
        reuse_buffer=True,
        measured=False,
        note="assumed from the 3090 measurements; re-run the rows-per-pass sweep on an H100",
    ),
    "unknown": HardwareProfile(
        name="unknown",
        # Deliberately the most conservative of the two: an unrecognised card may be a laptop
        # GPU sharing memory with the display, and being slightly slow beats thrashing.
        peak_ceiling_fraction=0.90,
        peak_bytes_per_cache_byte={False: 1.4, True: 2.1},
        reuse_buffer=False,
        measured=False,
        note="fallback for hardware with no profile; conservative on purpose",
    ),
}


def get_profile(name: str) -> HardwareProfile:
    """Look up a profile by ``hardware:`` name.

    Args:
        name: A key of :data:`PROFILES`.

    Returns:
        The profile.

    Raises:
        KeyError: If the name is not registered. Not silently defaulted: a config asking for
            hardware we have never tuned for should say so rather than run mis-tuned.
    """
    if name not in PROFILES:
        raise KeyError(f"unknown hardware profile {name!r}; known: {sorted(PROFILES)}")
    return PROFILES[name]


def detect_profile(device_name: str | None = None) -> HardwareProfile:
    """Pick the profile matching the current machine.

    Args:
        device_name: CUDA device name; read from torch when omitted.

    Returns:
        The matching profile, or the ``unknown`` fallback. Never raises — detection failing
        must not stop a run, only make it conservative.
    """
    if device_name is None:
        try:
            import torch

            device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        except Exception as exc:  # detection must not stop a run, but it must say why it failed
            print(
                f"hardware: device detection failed ({type(exc).__name__}: {exc}); "
                "using the conservative 'unknown' profile",
                file=sys.stderr,
            )
            device_name = ""

    lowered = (device_name or "").lower()
    if "3090" in lowered and sys.platform == "win32":
        return PROFILES["rtx3090_windows"]
    if "h100" in lowered and sys.platform.startswith("linux"):
        return PROFILES["h100_linux"]
    return PROFILES["unknown"]


def load_training_envelope(profile: HardwareProfile, root: Any) -> dict[str, Any] | None:
    """The measured training envelope for ``profile``, or ``None`` if it has none.

    Args:
        profile: A hardware profile.
        root: Repository root (a ``pathlib.Path``).

    Returns:
        The parsed ``envelope.json``.

    Raises:
        FileNotFoundError: If the profile names an envelope file that does not exist — a profile
            claiming a measurement it cannot produce is an error, not a missing value.
    """
    import json

    if not profile.training_envelope:
        return None
    path = root / profile.training_envelope
    return json.loads(path.read_text(encoding="utf-8"))
