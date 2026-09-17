# ADR 0002 — Base model: proceed with Qwen3.8-27B

- **Status:** accepted (2026-09-17)
- **Date opened:** 2026-09-17 · **decided:** 2026-09-17, Phase 0 Step 3
- **Deciders:** project owner (human), training-engineer role
- **Evidence:** `docs/research/qwen38-stack-support-2026-09-17.md`, reproducible with
  `uv run python scripts/derisk_base_model.py`

## Context

`CLAUDE.md` locks the base model as `Qwen/Qwen3.8-27B` with reasoning disabled and names
`Qwen/Qwen3.6-27B` as the fallback "if Unsloth/PEFT cannot handle the GDN hybrid layers".
Qwen3.8-27B is Apache 2.0, dense 27B, 262,144 context, a hybrid of 48 Gated-DeltaNet layers
and 16 full-attention layers (3:1), with a vision tower and a multi-token-prediction head
(`docs/research/jev-landscape-2026-09-17.md` §6). The recorded risk was that the training
and serving stack we pin — Unsloth Studio's exact versions, on Windows — would not support the
hybrid layers, and that locked decision 2 (KV-cache broadcast) might not be implementable on a
cache that is mostly recurrent state rather than key/value.

The five questions were: (a) 4-bit load with Unsloth `FastLanguageModel` on Windows,
(b) LoRA on attention and MLP projections, (c) the GDN Triton kernels compiling under
`triton-windows`, (d) GGUF conversion, (e) cache expansion for batch broadcast and how the
recurrent state is represented.

## Decision

**Option A — proceed with `Qwen/Qwen3.8-27B`.** All five questions came back positive on the
pinned stack, with concrete conditions rather than blockers:

| # | Question | Answer | Evidence |
|---|---|---|---|
| a | 4-bit load on Windows | Yes. `Qwen3.8-27B` declares `model_type: qwen3_5`; transformers 5.5.0 ships that package, stores Qwen3.8's extra config fields, already hard-codes the attention output gate and swish GDN gate, ignores MTP weights, and offers `Qwen3_5ForCausalLM` that drops the vision tower. Unsloth maps the model to `unsloth/Qwen3.8-27B-unsloth-bnb-4bit`; bnb nf4 works on the 3090 (`task doctor`). | research §1, §2, §4, §5 |
| b | LoRA on attention + MLP | Yes. Names verified: `self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`; PEFT needs explicit `target_modules` (no `qwen3_5` default mapping). Matches Unsloth's published Qwen3.8 recipe. | research §3 |
| c | GDN Triton kernels on Windows | **Yes, measured.** Unsloth's vendored fla 0.5.1 compiles through the toolchain `triton-windows` bundles (TCC + ptxas, no MSVC); 3/3 GDN modules on the fast path; ~9x faster than the torch fallback; bf16-level agreement. | research §5, `docs/windows-setup.md` R1 |
| d | GGUF conversion | Yes. The installed llama.cpp converter registers both `Qwen3_5ForConditionalGeneration` and `Qwen3_5ForCausalLM`; the cached Unsloth GGUF header reads `general.architecture = qwen35`, 65 blocks incl. the MTP layer; `LLM_ARCH_QWEN35` is in the runtime. | research §6 |
| e | Cache broadcast | Yes, **but ours to implement.** The cache is a `DynamicCache` mixing `LinearAttentionLayer` (`conv_states`, `recurrent_states`) and `DynamicLayer` (`keys`, `values`). `batch_repeat_interleave` raises on the GDN layers. Expanding all four tensors per layer reproduces N independent runs to 3.9e-3 in bf16; expanding KV only fails at the first GDN conv update. | research §2, §8 |

The fallback to Qwen3.6-27B buys nothing: it is the same `qwen3_5` architecture class with the
same cache classes and the same kernels, so every constraint below would apply to it too.

## Conditions attached to the decision

These are not optional; each one has a named owner and a test.

1. **Write the cache broadcast ourselves** (`inference-engineer`, Step 4). One
   `expand_cache(cache, n)` that `repeat_interleave`s `keys`, `values`, `conv_states` and
   `recurrent_states` on every layer that has them. The contract test asserts equality against
   N independent full-sequence runs — not merely "no exception", because newer transformers
   versions carry a layer class that silently repeats only KV.
2. **No sequence packing in Stage-1 training** on transformers < 5.9 (`training-engineer`).
   In 5.2.0-5.8.1 the GDN conv/recurrent state leaks across packed samples silently. We are on
   5.5.0 to mirror Studio. `train/sft_lora.py` asserts `packing=False, padding_free=False` and
   logs the transformers version; lifting this requires bumping transformers and re-running
   `task smoke`.
3. **bf16 compute, never fp16, for the GDN layers** (`training-engineer`). Unsloth force-lists
   `qwen3_5` for float32 fallback because fp16 produces NaN grad norms; the prequantized
   checkpoint's config says `bnb_4bit_compute_dtype: float16`. Load with
   `dtype=torch.bfloat16`; `task smoke` logs the effective compute dtype.
4. **LoRA on attention + MLP only in S1** (`training-engineer`). GDN projections are LoRA-able
   but are left unquantized in the 4-bit checkpoint and are V-head-reordered by the GGUF
   converter. Adapting them is a separate, evaluated decision, not a default.
5. **Merge before GGUF** (`inference-engineer`, `scripts/convert_gguf.py`) — already locked
   decision 8. The converter's GDN V-head permutation then applies to merged weights and no
   adapter-side permutation is needed.
6. **Load text-only** — `text_only=True` with `FastLanguageModel`, or `FastModel` with
   `finetune_vision_layers=False`, or plain `Qwen3_5ForCausalLM`. All three drop the vision
   tower; the engine and trainer must agree on one and record it in the config.
7. **VRAM.** Unsloth's own figure for QLoRA on this model is 24 GB, i.e. the entire 3090.
   `max_seq_length <= 2048`, batch 1-2, `offload_embedding=True`, gradient checkpointing
   `"unsloth"`, Studio closed. If the smoke run does not fit, the fallback is *not* a different
   base model but the H100 config — the 3090 then does inference and calibration only.

## Smoke-test model

**`Qwen/Qwen3.5-0.8B`** — same `Qwen3_5ForConditionalGeneration` / `qwen3_5` class, same 3:1
`layer_types`, `attn_output_gate: true`, MTP head, same tokenizer vocabulary; one safetensors
shard of 1,746,942,600 bytes (1.63 GB) plus ~23 MB of tokenizer files. There is no small
Qwen3.8: the family is 27B, 2.4T-A95B (MoE), Flash-Next and Max only. The 0.8B differs from the
27B in having equal GDN key and value head counts (16/16 vs 16/48); the random-weight model in
`scripts/derisk_base_model.py` covers the grouped-V shape.

## Alternatives considered

- **B — `Qwen/Qwen3.6-27B`.** Rejected: same architecture class, same constraints, weaker
  model. It would only make sense if Qwen3.8 had shipped a new model package that 5.5.0 lacked;
  it did not.
- **C — a non-hybrid base (e.g. a pure-attention 27-32B).** Rejected for now: it would
  invalidate the project's positioning against the research note and remove the need for
  conditions 1-3, but the measured evidence says the hybrid is workable. Reconsider only if
  `task smoke` fails on VRAM or the Unsloth path turns out to need a C++ compiler on Windows.

## Consequences

- Locked decision 2 is renamed in spirit: it is a *hybrid-state* broadcast, not a KV broadcast.
  `CLAUDE.md` wording can stay; the engine protocol in Step 4 should say `expand_cache`.
- The transformers pin (5.5.0) now carries a known training bug we route around by config.
  Upgrading past 5.9 later is desirable and must be treated as a dependency change with a
  smoke re-run, not a casual bump.
- `pyproject.toml` pins `torchao==0.17.0` (Studio's version); the 0.18.0 that resolved
  transitively wants torch >= 2.11 and disables its C++ extensions on our 2.10.0.
- Still unmeasured, deliberately: the 27B itself at 4-bit (needs a ~17-18 GB download and a
  "go"), and a real training step (needs `task smoke`). Both are Phase 0 Step 4 / Phase 1 work
  and both have a clear fallback (H100 config) that does not change the base model.
