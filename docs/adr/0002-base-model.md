# ADR 0002 — Base model: Qwen3.8-27B or fall back to Qwen3.6-27B

- **Status:** proposed — **decision deferred to Phase 0 Step 3**
- **Date opened:** 2026-09-17
- **Deciders:** project owner (human), training-engineer role

> This file is a **stub with the questions written down**. It is filled in during Step 3
> (base-model de-risking) and only then moves to `accepted`. Nothing downstream may assume an
> answer before that.

## Context

`CLAUDE.md` locks the base model as `Qwen/Qwen3.8-27B` with reasoning disabled, and names
`Qwen/Qwen3.6-27B` as the fallback "if Unsloth/PEFT cannot handle the GDN hybrid layers".
From `docs/research/jev-landscape-2026-09-17.md` section 6: Qwen3.8-27B is Apache 2.0, dense
27B, 262,144 context, a **hybrid Mamba-Transformer (Gated DeltaNet + full attention)**, and its
Q4_K_M GGUF (~17 GB) fits a 24 GB RTX 3090. The recorded risk is that the GDN hybrid layers are
not supported by the training and serving stack we depend on.

The hybrid architecture is the crux. A Gated-DeltaNet layer carries a **recurrent state**, not a
key/value cache, so it touches three separate parts of this project: whether Unsloth can load
and LoRA it, whether its Triton kernels compile on this machine, and whether the KV-broadcast
design in locked decision 2 can broadcast a cache that is only half key/value.

## Evidence gathered so far (Step 0 audit, unverified)

From `docs/windows-setup.md` section 6 — observed on disk, not yet confirmed against upstream:

- `transformers 5.5.0` ships `qwen3`, `qwen3_moe`, `qwen3_next`, `qwen3_5`, `qwen3_5_moe`,
  `qwen3_vl`, `qwen3_vl_moe`, `qwen3_omni_moe` — and **no `qwen3_8` or `qwen3_6` package**.
  So Qwen3.8-27B either maps onto an existing architecture class or needs `trust_remote_code`.
- `unsloth 2026.9.2` `models/mapper.py` has an explicit entry
  `"unsloth/Qwen3.8-27B-unsloth-bnb-4bit" : ("unsloth/Qwen3.8-27B", "Qwen/Qwen3.8-27B", ...)`,
  i.e. Unsloth knows the model and publishes a pre-quantized 4-bit copy.
- `unsloth_zoo/temporary_patches/fla_vendor.py` vendors the `flash-linear-attention` Triton
  kernels "used by Qwen3.5 / Qwen3.6 / Qwen3-Next gated-deltanet models", with a documented
  **"several-times slower pure-PyTorch path"** when Triton or fla is unavailable
  (`UNSLOTH_DISABLE_VENDORED_FLA=1`).
- `unsloth/Qwen3.8-27B-GGUF` is **already in the local HF cache** (`UD-Q4_K_M` 15.3 GB,
  `UD-Q5_K_M` 18.4 GB), so a GGUF of this architecture demonstrably exists and llama.cpp
  demonstrably has a converter for it. No HF-format copy is cached, not even the tokenizer.
- The machine has **no MSVC, no Windows SDK, no CUDA toolkit** (`docs/windows-setup.md` R1),
  which puts Triton kernel compilation in doubt. `uv run task doctor` now answers this with a
  real `@triton.jit` compile probe.

## Evidence from `config.json` (Step 2, 2026-09-17 — read from the file, not inferred)

The tokenizer fetch in Step 2 also pulled `Qwen/Qwen3.8-27B/config.json`
(snapshot `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`). It answers question 1 directly and
reshapes question 5:

| Field | Value | Why it matters |
|---|---|---|
| `architectures` | `["Qwen3_5ForConditionalGeneration"]` | **Qwen3.8-27B declares itself as the `qwen3_5` architecture**, which `transformers 5.5.0` ships (`models/qwen3_5/`). No new model package is needed. |
| `model_type` | `qwen3_5` | same |
| `transformers_version` | `5.8.0.dev0` | The config was written by a **newer** transformers than our pinned 5.5.0. Unverified whether 5.5.0 reads every field. This is now the main open risk. |
| `num_hidden_layers` | 64 | |
| `layer_types` | 48x `linear_attention` + 16x `full_attention` | The hybrid, concretely: **3 GDN layers then 1 full-attention layer, repeated 16 times** (`full_attention_interval: 4`). |
| `linear_conv_kernel_dim` | 4 | The GDN state has a **short causal conv** component as well as a recurrent one — the broadcast in question 5 must copy both. |
| `linear_num_key_heads` / `linear_num_value_heads` | 16 / 48 | GDN state shape. |
| `num_attention_heads` / `num_key_value_heads` | 24 / 4 | GQA in the 16 full-attention layers. |
| `mamba_ssm_dtype` | `float32` | The recurrent state is kept in fp32 even at 4-bit. |
| `head_dim` / `hidden_size` | 256 / 5120 | |
| `vision_config` present, `language_model_only: false` | — | **It is a vision-language model**, `...ForConditionalGeneration`, not `...ForCausalLM`. Text-only loading, the LM head location and the GGUF path all need checking against that. |
| `mtp_num_hidden_layers` | 1 | Multi-token prediction head, as the research note said. |
| `vocab_size` | 248320 (tokenizer reports 248044 / `len` 248077) | Padded embedding; the mask must index the real logit width. |
| `max_position_embeddings` | 262144 | |

Consequences already visible: only 16 of 64 layers hold a conventional KV cache, so
"broadcast the KV cache" in locked decision 2 is really "broadcast 16 KV caches **and**
48 recurrent+conv states". Step 4's N-row-equals-N-runs test is therefore not a
formality — it is the check that the other 48 layers were copied rather than aliased.

## Questions Step 3 must answer

1. **4-bit load.** Does `FastLanguageModel.from_pretrained(..., load_in_4bit=True)` work for
   Qwen3.8-27B on Windows with the pinned stack? Which `transformers` architecture class does
   its `config.json` actually resolve to, and is `trust_remote_code` required?
2. **LoRA targets.** Which modules are LoRA-able on a GDN hybrid — the attention and MLP
   projections are the intent, but what are they named in the GDN blocks, and do the GDN
   layers need to be excluded explicitly?
3. **Triton / GDN kernels.** Do the vendored `fla` kernels compile here? If not, is the
   pure-PyTorch fallback fast enough to be usable for the smoke run, and by what factor is it
   slower? (Depends on the `triton-compile` doctor result.)
4. **GGUF.** Does the on-disk llama.cpp `convert_hf_to_gguf.py` handle this architecture, and
   does `convert_lora_to_gguf.py` handle an adapter over it? What `general.architecture` string
   does the cached GGUF declare?
5. **Cache broadcast.** How is the hybrid cache represented in `transformers 5.5.0` — what class,
   and how are the attention KV part and the recurrent-state part stored? Can both be expanded
   to N rows, and does an N-row broadcast give bit-comparable results to N independent runs?
   This determines whether locked decision 2 is implementable as written.

## Options

- **A — proceed with Qwen/Qwen3.8-27B.**
- **B — fall back to Qwen/Qwen3.6-27B** (same family, older, likelier to be a well-trodden path).
- **C — a different base entirely.** Would need its own ADR and a rewrite of the project's
  positioning; out of scope unless A and B both fail.

## Decision

*To be filled in Step 3.*

## Consequences

*To be filled in Step 3.*

## Smoke-test model

The smallest model in the same architecture family (ideally ≤ 2 GB) for contract tests on the
3090: *to be named in Step 3.*
