"""Per-hardware inference tuning.

These constants were fitted on one card. The point of the module is that they cannot silently
travel to another one, so the tests are mostly about what must *not* happen.
"""

from __future__ import annotations

import pytest

from s1decide.hardware import PROFILES, HardwareProfile, detect_profile, get_profile


def test_every_profile_declares_both_reuse_costs() -> None:
    """A missing key would raise mid-run, inside a GPU call, hours into a job."""
    for name, profile in PROFILES.items():
        assert set(profile.peak_bytes_per_cache_byte) == {True, False}, name
        for cost in profile.peak_bytes_per_cache_byte.values():
            assert cost >= 1.0, f"{name}: a row cannot cost less peak than the cache it holds"


def test_every_profile_leaves_real_headroom() -> None:
    for name, profile in PROFILES.items():
        assert 0.5 < profile.peak_ceiling_fraction < 1.0, name


def test_reuse_costs_more_peak_than_not_reusing() -> None:
    """The measured asymmetry that decided the 3090 default — it must survive a careless edit."""
    for name, profile in PROFILES.items():
        assert profile.peak_bytes_per_cache_byte[True] > profile.peak_bytes_per_cache_byte[False], (
            f"{name}: keeping a buffer alive across passes cannot be cheaper than freeing it"
        )


def test_the_3090_profile_is_measured_and_the_h100_is_not() -> None:
    """An assumed constant must never be quotable as a finding (ADR 0003)."""
    assert PROFILES["rtx3090_windows"].measured is True
    assert PROFILES["h100_linux"].measured is False
    assert PROFILES["unknown"].measured is False


def test_the_3090_keeps_buffer_reuse_off_and_the_h100_expects_it_on() -> None:
    assert PROFILES["rtx3090_windows"].reuse_buffer is False
    assert PROFILES["h100_linux"].reuse_buffer is True


def test_the_unknown_profile_is_the_most_conservative() -> None:
    unknown = PROFILES["unknown"]
    for name, profile in PROFILES.items():
        if name != "unknown":
            assert unknown.peak_ceiling_fraction <= profile.peak_ceiling_fraction


def test_an_unregistered_name_is_refused_rather_than_defaulted() -> None:
    """Running mis-tuned is worse than not running."""
    with pytest.raises(KeyError, match="unknown hardware profile"):
        get_profile("a100_linux")


def test_get_profile_returns_the_named_profile() -> None:
    assert get_profile("rtx3090_windows") is PROFILES["rtx3090_windows"]


@pytest.mark.parametrize(
    "device_name",
    ["NVIDIA GeForce RTX 3090", "NVIDIA H100 80GB HBM3", "NVIDIA GeForce RTX 4070", ""],
)
def test_detection_always_returns_a_usable_profile(device_name: str) -> None:
    profile = detect_profile(device_name)
    assert isinstance(profile, HardwareProfile)
    assert profile.name in PROFILES


def test_detection_is_platform_aware() -> None:
    """The 3090 numbers describe the *Windows* driver's behaviour, not the card's."""
    import sys

    profile = detect_profile("NVIDIA GeForce RTX 3090")
    expected = "rtx3090_windows" if sys.platform == "win32" else "unknown"
    assert profile.name == expected


def test_profile_serialises_for_a_run_record() -> None:
    payload = PROFILES["rtx3090_windows"].to_json()
    assert payload["name"] == "rtx3090_windows"
    assert payload["measured"] is True
    assert set(payload["peak_bytes_per_cache_byte"]) == {"True", "False"}
