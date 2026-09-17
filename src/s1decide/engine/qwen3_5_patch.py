"""Fix multi-token cache continuation in transformers 5.5.0's ``Qwen3_5GatedDeltaNet``.

The bug (transformers 5.5.0, ``modeling_qwen3_5.py``, ``Qwen3_5GatedDeltaNet.forward``): with a
cache present, only the ``seq_len == 1`` decode step reads the cached state. Any longer
continuation takes the chunked path with ``initial_state=None`` and runs the causal conv over
the new tokens alone — **both the recurrent state and the conv window are ignored**, silently.
Nothing in a normal ``generate()`` call hits this (prefill has no state, decode is one token), which
is why it survives; our engine hits it on every call, because each question's suffix is a
multi-token continuation of the prefilled state.

Upstream ``main`` fixes it by feeding the cached conv window into the conv and passing
``initial_state=recurrent_state`` to the chunk kernel. This module does exactly that, per module
instance, leaving the untouched paths (no cache, decode step) on the library's own code. It is
applied by :class:`~s1decide.engine.hf.HFEngine` and verified by the broadcast-equals-independent
contract test — which is the test that caught it.
"""

from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn.functional as F

__all__ = ["is_patched", "patch_gated_deltanet"]

_ORIGINAL = "_s1decide_original_forward"


def _patched_forward(
    self: Any,
    hidden_states: torch.Tensor,
    cache_params: Any | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """``Qwen3_5GatedDeltaNet.forward`` with a correct multi-token continuation branch."""
    seq_len = hidden_states.shape[1]
    continuing = (
        cache_params is not None and cache_params.has_previous_state(self.layer_idx) and seq_len > 1
    )
    if not continuing:
        # No cache, first prefill, or a one-token decode step: the library code is right here.
        return getattr(self, _ORIGINAL)(hidden_states, cache_params, attention_mask)

    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape
    layer = cache_params.layers[self.layer_idx]
    kernel = self.conv_kernel_size

    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # (B, conv_dim, L)
    z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    # --- conv: prepend the cached window of the previous kernel-1 inputs ---------------------
    previous = layer.conv_states  # (B, conv_dim, kernel): the last `kernel` inputs seen
    if previous.shape[0] != batch_size:
        raise RuntimeError(
            f"conv state batch {previous.shape[0]} != input batch {batch_size} on layer "
            f"{self.layer_idx}; the cache was not broadcast to this batch size"
        )
    window = (
        torch.cat([previous[..., -(kernel - 1) :], mixed_qkv], dim=-1) if kernel > 1 else mixed_qkv
    )
    new_conv_state = torch.cat([previous, mixed_qkv], dim=-1)[..., -kernel:]

    if self.causal_conv1d_fn is not None:
        conv_out = self.causal_conv1d_fn(
            x=window,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
            seq_idx=None,
        )[..., -seq_len:]
    else:
        # conv1d pads kernel-1 on both sides; output index t covers window inputs [t-kernel+1, t].
        # The new tokens sit at window positions kernel-1 .. kernel-1+L-1.
        conv_out = F.silu(self.conv1d(window)[:, :, kernel - 1 : kernel - 1 + seq_len])
    layer.conv_states.copy_(new_conv_state)

    # --- gated delta rule, seeded with the cached recurrent state --------------------------
    mixed = conv_out.transpose(1, 2)
    query, key, value = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    groups = self.num_v_heads // self.num_k_heads
    if groups > 1:
        query = query.repeat_interleave(groups, dim=2)
        key = key.repeat_interleave(groups, dim=2)

    core, last_state = self.chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=layer.recurrent_states,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    cache_params.update_recurrent_state(last_state, self.layer_idx)

    core = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
    return self.out_proj(core.reshape(batch_size, seq_len, -1))


def is_patched(module: Any) -> bool:
    """Whether :func:`patch_gated_deltanet` has already been applied to this module."""
    return hasattr(module, _ORIGINAL)


def patch_gated_deltanet(model: Any) -> int:
    """Patch every ``Qwen3_5GatedDeltaNet`` module in ``model`` in place.

    Idempotent: modules that are already patched are skipped.

    Args:
        model: Any ``torch.nn.Module`` tree. Models without gated-deltanet layers are left
            untouched and return 0.

    Returns:
        Number of modules patched by this call.
    """
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    except ImportError:  # transformers without the qwen3_5 package — nothing to patch
        return 0

    patched = 0
    for module in model.modules():
        if isinstance(module, Qwen3_5GatedDeltaNet) and not is_patched(module):
            setattr(module, _ORIGINAL, module.forward)
            module.forward = types.MethodType(_patched_forward, module)
            patched += 1
    return patched
