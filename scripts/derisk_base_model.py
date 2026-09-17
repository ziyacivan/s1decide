"""Re-runnable evidence for ADR 0002: can the pinned stack handle Qwen3.8-27B's hybrid architecture?

Downloads nothing. Uses the cached ``Qwen/Qwen3.8-27B`` config.json (tokenizer-sized fetch
from Step 2) and a tiny random-weight model of the same architecture class. Answers, on the
GPU it runs on:

1. Does ``transformers`` load the real config, keeping Qwen3.8's extra fields?
2. What are the LoRA-able module names in a GDN layer and a full-attention layer?
3. How is the hybrid cache represented, and does ``batch_repeat_interleave`` work on it?
4. Do Unsloth's vendored ``fla`` Triton kernels compile and run here, and how do they compare
   with the pure-torch fallback (speed, numerics)?
5. Does a per-layer cache broadcast (attention KV *and* GDN conv/recurrent states) equal N
   independent runs, and does a KV-only broadcast fail loudly?

Run with::

    uv run python scripts/derisk_base_model.py

Findings as of 2026-09-17 are recorded in docs/research/qwen38-stack-support-2026-09-17.md.
"""

from __future__ import annotations

import copy
import os
import sys
import time
from pathlib import Path

# Keep Unsloth's generated modules out of the repository root.
os.environ.setdefault(
    "UNSLOTH_COMPILED_CACHE_DIR",
    str(Path(os.environ.get("TEMP", os.environ.get("TMPDIR", "/tmp"))) / "unsloth_derisk_cache"),
)

BASE_MODEL = "Qwen/Qwen3.8-27B"


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n=== {title} ===")


def main() -> int:
    """Run every probe and print a report. Returns a process exit code."""
    import torch

    if not torch.cuda.is_available():
        print("CUDA is required for the kernel and broadcast probes; aborting.")
        return 1

    # Unsloth must be imported before transformers builds any model: it injects the vendored
    # fla kernels into sys.modules and rebinds the kernel functions in modeling_qwen3_5.
    t0 = time.time()
    import unsloth

    print(f"unsloth {unsloth.__version__} imported in {time.time() - t0:.1f}s")
    import transformers
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForCausalLM,
        Qwen3_5GatedDeltaNet,
    )

    print(f"transformers {transformers.__version__} | torch {torch.__version__}")

    # --- 1. config ---------------------------------------------------------------------------
    section("1. real config under the pinned transformers")
    cfg = AutoConfig.from_pretrained(BASE_MODEL, local_files_only=True)
    tc = cfg.text_config
    print("config class:", type(cfg).__name__, "| architectures:", cfg.architectures)
    print(
        "text_config class:",
        type(tc).__name__,
        "| written by transformers",
        cfg.transformers_version,
    )
    for key in ("attn_output_gate", "output_gate_type", "mamba_ssm_dtype", "mtp_num_hidden_layers"):
        print(f"  text_config.{key} = {getattr(tc, key, '<MISSING>')}")
    counts = {t: tc.layer_types.count(t) for t in sorted(set(tc.layer_types))}
    print(
        "  layer_types:",
        counts,
        "| full_attention_interval pattern holds:",
        tc.layer_types[3] == "full_attention",
    )

    # --- tiny model of the same class ------------------------------------------------------
    small = type(tc)(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        max_position_embeddings=1024,
        attn_output_gate=True,
        output_gate_type="swish",
        mamba_ssm_dtype="float32",
    )
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(small).to("cuda", dtype=torch.bfloat16).eval()

    # --- 2. module names -------------------------------------------------------------------
    section("2. LoRA-able module names")
    for idx, kind in ((0, "linear_attention"), (3, "full_attention")):
        names = [
            f"{n} {tuple(mod.weight.shape)}"
            for n, mod in model.model.layers[idx].named_modules()
            if isinstance(mod, torch.nn.Linear | torch.nn.Conv1d)
        ]
        print(f"layer {idx} ({kind}):")
        for n in names:
            print("   ", n)

    # --- 3. cache representation -----------------------------------------------------------
    section("3. hybrid cache representation")
    with torch.no_grad():
        out = model(torch.randint(0, 512, (1, 12), device="cuda"), use_cache=True)
    cache = out.past_key_values
    print("cache class:", type(cache).__name__)
    for i, layer in enumerate(cache.layers):
        attrs = {
            a: tuple(getattr(layer, a).shape)
            for a in ("keys", "values", "conv_states", "recurrent_states")
            if torch.is_tensor(getattr(layer, a, None))
        }
        print(f"  layers[{i}] {type(layer).__name__}: {attrs}")
    try:
        copy.deepcopy(cache).batch_repeat_interleave(2)
        print("DynamicCache.batch_repeat_interleave: works")
    except AttributeError as exc:
        print(f"DynamicCache.batch_repeat_interleave: FAILS -> {exc}")

    # --- 4. fla kernels --------------------------------------------------------------------
    section("4. vendored fla Triton kernels")
    print("is_flash_linear_attention_available:", m.is_flash_linear_attention_available())
    print("is_causal_conv1d_available:", m.is_causal_conv1d_available())
    on_fla = sum(
        mod.chunk_gated_delta_rule is not m.torch_chunk_gated_delta_rule
        for mod in model.modules()
        if isinstance(mod, Qwen3_5GatedDeltaNet)
    )
    print(f"GDN modules bound to fla kernels: {on_fla}/3")

    slow = Qwen3_5ForCausalLM(small).to("cuda", dtype=torch.bfloat16).eval()
    slow.load_state_dict(model.state_dict())
    for mod in slow.modules():
        if isinstance(mod, Qwen3_5GatedDeltaNet):
            mod.chunk_gated_delta_rule = m.torch_chunk_gated_delta_rule
            mod.recurrent_gated_delta_rule = m.torch_recurrent_gated_delta_rule
            mod.causal_conv1d_fn = None
            mod.causal_conv1d_update = m.torch_causal_conv1d_update

    ids = torch.randint(0, 512, (2, 256), device="cuda")
    with torch.no_grad():
        t0 = time.time()
        a = model(ids).logits.float()
        torch.cuda.synchronize()
        t_first = time.time() - t0
        t0 = time.time()
        a = model(ids).logits.float()
        torch.cuda.synchronize()
        t_fla = time.time() - t0
        t0 = time.time()
        b = slow(ids).logits.float()
        torch.cuda.synchronize()
        t_torch = time.time() - t0
    print(
        f"first forward incl. compile: {t_first:.2f}s | fla steady: {t_fla * 1e3:.1f} ms | torch fallback: {t_torch * 1e3:.1f} ms"
    )
    print(
        f"fla vs torch: max|diff| {(a - b).abs().max().item():.3e}, mean|diff| {(a - b).abs().mean().item():.3e} (bf16)"
    )

    # --- 5. broadcast ----------------------------------------------------------------------
    section("5. prefix-cache broadcast vs independent runs")

    def expand(c, n: int, attrs: tuple[str, ...]):
        for layer in c.layers:
            for attr in attrs:
                t = getattr(layer, attr, None)
                if torch.is_tensor(t):
                    setattr(layer, attr, t.repeat_interleave(n, dim=0))
        return c

    n = 4
    prefix = torch.randint(0, 512, (1, 40), device="cuda")
    suffixes = torch.randint(0, 512, (n, 9), device="cuda")
    with torch.no_grad():
        pre = model(prefix, use_cache=True).past_key_values
        independent = torch.stack(
            [
                model(torch.cat([prefix, suffixes[i : i + 1]], dim=1)).logits[0, -1].float()
                for i in range(n)
            ]
        )
        full = expand(copy.deepcopy(pre), n, ("keys", "values", "conv_states", "recurrent_states"))
        bcast = model(suffixes, past_key_values=full, use_cache=True).logits[:, -1].float()
        d = (bcast - independent).abs().max().item()
        print(
            f"full broadcast (KV + conv + recurrent) vs independent: max|diff| {d:.3e} -> {'MATCH' if d < 5e-2 else 'MISMATCH'}"
        )
        try:
            naive = expand(copy.deepcopy(pre), n, ("keys", "values"))
            model(suffixes, past_key_values=naive, use_cache=True)
            print("KV-only broadcast: ran without error (would be silently wrong)")
        except RuntimeError as exc:
            print(f"KV-only broadcast: fails loudly -> {str(exc)[:110]}")

    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
