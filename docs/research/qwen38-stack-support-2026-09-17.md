# Qwen3.8-27B on the pinned stack — support audit (2026-09-17)

Phase 0 Step 3, role `training-engineer`. Answers the five questions in
`docs/adr/0002-base-model.md` for the exact versions in `uv.lock`:
**transformers 5.5.0, peft 0.18.1, trl 0.23.1, bitsandbytes 0.50.2, unsloth 2026.9.2,
unsloth-zoo 2026.9.1, triton-windows 3.6.0.post26, torch 2.10.0+cu130**, and the
llama.cpp build Unsloth Studio installed (`b10798-mix-659e406`).

Method: read the installed source, read the cached `Qwen/Qwen3.8-27B/config.json` and the
header of the cached `unsloth/Qwen3.8-27B-GGUF` file, and run a tiny random-weight model of
the same architecture class on the RTX 3090. **Nothing was downloaded** beyond the Step 2
tokenizer fetch and a few small web pages. Every GPU claim below is reproducible with
`uv run python scripts/derisk_base_model.py`. Web facts carry their URL and date.

## 0. Summary

| Question | Verdict | One-line evidence |
|---|---|---|
| (a) 4-bit load with Unsloth on Windows | **yes** (not yet run at 27B scale) | `Qwen/Qwen3.8-27B` is `model_type: qwen3_5`; transformers 5.5.0 ships it; Unsloth maps it to `unsloth/Qwen3.8-27B-unsloth-bnb-4bit`; bnb nf4 forward passes on this box |
| (b) LoRA on attention + MLP | **yes** | module names verified; Unsloth's published Qwen3.8 recipe targets exactly these |
| (c) GDN Triton kernels compile here | **yes, measured** | vendored fla 0.5.1 kernels compiled via bundled TCC + ptxas; 3/3 GDN modules on the fast path; 9x faster than the torch fallback on a tiny model |
| (d) GGUF conversion | **yes** | installed converter registers `Qwen3_5ForConditionalGeneration` and `Qwen3_5ForCausalLM`; cached GGUF header says `general.architecture = qwen35`; llama.cpp `LLM_ARCH_QWEN35` present |
| (e) hybrid cache broadcast | **yes, but we must write it** | `DynamicCache.batch_repeat_interleave` raises on GDN layers; a per-layer expand of KV + conv + recurrent states matches N independent runs |

**Recommendation: proceed with `Qwen/Qwen3.8-27B`.** No finding favours the Qwen3.6 fallback.
Smoke-test model: **`Qwen/Qwen3.5-0.8B`** (same class, 1.63 GB safetensors).

Three constraints came out of this that Steps 4 and S1 must honour: (1) **write our own cache
broadcast**, (2) **no sequence packing in training on transformers < 5.9**, (3) **bf16, never
fp16**, for the GDN layers.

## 1. What Qwen3.8-27B is, per its own config

`config.json`, snapshot `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`:

- `architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type: "qwen3_5"`. **Qwen3.8 is a
  Qwen3.5-architecture checkpoint.** There is no `qwen3_8` model package anywhere, and none is
  needed. Written by `transformers 5.8.0.dev0`.
- Vision-language model: `vision_config` present, `language_model_only: false`. The text tower
  is `Qwen3_5TextModel` under `model.language_model.*`.
- 64 layers, `layer_types` = 48 `linear_attention` + 16 `full_attention`, 3:1
  (`full_attention_interval: 4`). Model card: "16 x (3 x (Gated DeltaNet -> FFN) -> 1 x (Gated
  Attention -> FFN))".
- GDN: 16 key heads x 128, 48 value heads x 128, conv kernel 4, `mamba_ssm_dtype: float32`,
  `output_gate_type: swish`. Attention: 24 q heads, 4 kv heads, head_dim 256,
  `attn_output_gate: true`, partial rotary 0.25, interleaved mRoPE.
- `mtp_num_hidden_layers: 1` (multi-token prediction head), vocab 248,320, context 262,144.

The model card (fetched 2026-09-17) gives no minimum transformers version and does not mention
flash-linear-attention or causal-conv1d. It confirms `enable_thinking: False` disables
reasoning and that `reasoning_effort` takes only `xhigh` / `medium` / `low` — matching the
Step 2 measurement. It lists no other Qwen3.8 sizes.

## 2. transformers 5.5.0

**Config loads cleanly.** `AutoConfig.from_pretrained("Qwen/Qwen3.8-27B")` returns
`Qwen3_5Config` with `Qwen3_5TextConfig`. `Qwen3_5TextConfig` is decorated `@strict`, but
`PreTrainedConfig` wraps subclasses with `accept_kwargs`, so the fields 5.5.0 does not declare
— `attn_output_gate`, `output_gate_type`, `mamba_ssm_dtype`, `mtp_num_hidden_layers`,
`mtp_use_dedicated_embeddings` — are **stored as attributes, not rejected** (verified: all
five readable after load).

**The undeclared fields are already the hard-coded behaviour.** `Qwen3_5Attention.__init__`
makes `q_proj` twice the width and `forward` does `attn_output * torch.sigmoid(gate)`
unconditionally — that *is* `attn_output_gate: true`. `Qwen3_5RMSNormGated` applies
`F.silu(gate)`, which *is* `output_gate_type: swish`. `mamba_ssm_dtype` is not read; the GDN
path casts to float32 internally (`modeling_qwen3_5.py` lines 250, 322). MTP weights are
dropped on load: `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`.

**Text-only loading is first-class.** `Qwen3_5ForCausalLM` additionally ignores
`r"^model.visual.*"`, and `AutoModelForCausalLM` maps both `qwen3_5` ("VLM compatibility") and
`qwen3_5_text` to it (`modeling_auto.py` lines 727, 730). The VL -> text key remap lives in
`conversion_mapping.py` under `qwen3_5_text`.

**Kernels.** `modeling_qwen3_5.py` binds the GDN kernels **per module at `__init__`**
(`self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule`, line
407). Plain transformers finds neither `fla` nor `causal_conv1d` on this machine and logs "The
fast path is not available ... Falling back to torch implementation". See §5 for what Unsloth
changes.

**Cache.** A forward with `use_cache=True` returns a plain `DynamicCache` whose `layers` list
mixes two classes:

| Layer type | Cache class | Tensors (batch first) |
|---|---|---|
| `linear_attention` | `LinearAttentionLayer` | `conv_states (B, conv_dim, 4)`, `recurrent_states (B, H_v, d_k, d_v)` |
| `full_attention` | `DynamicLayer` | `keys (B, H_kv, T, d)`, `values (B, H_kv, T, d)` |

`LinearAttentionLayer` implements only `lazy_initialization`, `update_conv_state`,
`update_recurrent_state`. **It has no `batch_repeat_interleave`, `batch_select_indices`,
`crop` or `reorder_cache`.** `DynamicCache.batch_repeat_interleave` iterates every layer, so on
this model it raises `AttributeError: 'LinearAttentionLayer' object has no attribute
'batch_repeat_interleave'` (measured). Not fixed upstream as of transformers 5.17 either
([shamazharikh/qwen-rlcd PR #1](https://github.com/shamazharikh/qwen-rlcd/pull/1)). Related
open gaps: `PagedAttentionCache` and `cache_implementation="static"` both crash on
`linear_attention` ([#44530](https://github.com/huggingface/transformers/issues/44530),
[#46441](https://github.com/huggingface/transformers/issues/46441)). We only need
`DynamicCache`.

**Training bug in our version range.** In transformers **5.2.0-5.8.1**,
`Qwen3_5DecoderLayer.forward` calls `self.linear_attn(...)` without forwarding `**kwargs`, so
`cu_seq_lens` never reach the GDN and **the conv/recurrent state leaks across packed samples
silently** — "training loss may look normal but model quality degrades". Fixed in 5.9.0.
([modelscope/ms-swift #9618](https://github.com/modelscope/ms-swift/issues/9618), fetched
2026-09-17.) We are on 5.5.0 to mirror Studio, so **Stage-1 training must not use packing or
`padding_free`**, and `train/sft_lora.py` must assert that. Unsloth's `utils/packing.py`
mentions `qwen3_5`; whether it patches around this is to be verified when S1 is written, not
assumed.

## 3. Module names, PEFT 0.18.1, LoRA targets

From the tiny model (shapes are the tiny model's; names are the real ones):

```
linear_attention layer                    full_attention layer
  linear_attn.in_proj_qkv   Linear          self_attn.q_proj   Linear (2x width: q + gate)
  linear_attn.in_proj_z     Linear          self_attn.k_proj   Linear
  linear_attn.in_proj_b     Linear          self_attn.v_proj   Linear
  linear_attn.in_proj_a     Linear          self_attn.o_proj   Linear
  linear_attn.out_proj      Linear          mlp.gate_proj / up_proj / down_proj
  linear_attn.conv1d        Conv1d (depthwise)
  linear_attn.dt_bias, A_log, norm.weight   (parameters, not modules)
  mlp.gate_proj / up_proj / down_proj
```

- PEFT has **no `qwen3_5` entry** in `TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING`
  (only `qwen2`, `qwen3`), so `target_modules` must be explicit. PEFT does support LoRA on
  `nn.Conv1d` and `"all-linear"`.
- The standard target set `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` hits the 16
  attention layers' projections and **all 64 MLPs**; it never touches a GDN projection. That is
  what Unsloth's published Qwen3.8 recipe does (`finetune_attention_modules=True,
  finetune_mlp_modules=True`, r=16, alpha=16, dropout 0, `use_gradient_checkpointing="unsloth"`)
  — [unsloth.ai/docs/models/qwen3.8/train](https://unsloth.ai/docs/models/qwen3.8/train),
  fetched 2026-09-17.
- GDN projections are LoRA-able (`in_proj_qkv/z/b/a`, `out_proj` are `nn.Linear`) but carry
  two costs: (i) `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` deliberately leaves `in_proj_a`,
  `in_proj_b`, `in_proj_qkv` **unquantized** (`llm_int8_skip_modules`, see §4), so they are
  full-precision and heavier; (ii) the GGUF converter reorders GDN V heads (§6), which a LoRA
  adapter on those weights would have to mirror. **Decision for S1: attention + MLP only.**
  Revisit only if eval says the GDN layers need adapting.

## 4. bitsandbytes 0.50.2 (Windows) and the prequantized checkpoint

`task doctor` runs a real `Linear4bit` nf4 forward on the 3090: passes. bnb quantizes
`nn.Linear` generically and leaves `Conv1d` alone, so the GDN stack is not a special case.

`unsloth/Qwen3.8-27B-unsloth-bnb-4bit/config.json` (fetched 2026-09-17): `quant_method:
bitsandbytes`, `bnb_4bit_quant_type: nf4`, **`bnb_4bit_compute_dtype: float16`**,
`llm_int8_skip_modules: ["lm_head", "model.visual", ".*\.visual\..*", "in_proj_a", "in_proj_b",
"in_proj_qkv"]`, written by transformers 5.15.1.

Two things follow. First, the skip list means the GDN input projections stay in 16-bit — good
for numerics, and it is the reason the 4-bit file is larger than a naive estimate. Second,
**the compute dtype in that config is float16, and Unsloth itself lists `qwen3_5` in
`_FORCE_FLOAT32_FALLBACK` with the comment "Qwen3.5 GDN layers produce NaN grad norms in float16
training"** (`unsloth/models/loader.py` line 132). The RTX 3090 supports bf16; we load with
`dtype=torch.bfloat16` and let Unsloth's fallback do the rest. `task smoke` must log the
effective compute dtype.

## 5. Unsloth 2026.9.2 / unsloth-zoo 2026.9.1

- `models/mapper.py` maps `Qwen/Qwen3.8-27B` and `unsloth/Qwen3.8-27B` to
  `unsloth/Qwen3.8-27B-unsloth-bnb-4bit`. It is the **only** Qwen3.5/3.8-family entry in the
  mapper; small Qwen3.5 models are not pre-mapped (they still load, just without a prequantized
  twin). `FLA_MODEL_TYPE_PREFIXES = ("qwen3_next", "qwen3_5", "kimi_linear", "olmo_hybrid")`.
  Minimum transformers Unsloth demands for `qwen3_5`: 5.2.0.
- `text_only=True` on `FastLanguageModel/FastModel.from_pretrained` takes the VLM's
  `text_config`, remaps `model.language_model.*` keys, and loads `Qwen3_5ForCausalLM` — gated
  on `_is_family_text_decoder("qwen3_5", "qwen3_5_text")`, which is true. Unsloth's own recipe
  uses `FastModel.from_pretrained(..., load_in_4bit=True, offload_embedding=True)` with
  `finetune_vision_layers=False`. Either path avoids the vision tower.
- **fla injection, measured.** `import unsloth` registers `unsloth_zoo/_vendored/fla`
  (version 0.5.1) as top-level `fla` and rebinds the kernel functions in `modeling_qwen3_5`.
  After that: `is_flash_linear_attention_available() -> True`,
  `is_causal_conv1d_available() -> False` (conv uses torch), and a model built afterwards has
  **3/3 GDN modules bound to fla's `chunk_gated_delta_rule`**.
- **The kernels compile on this Windows box without MSVC.** First forward on the tiny model
  took ~29 s cold (Triton compile through the bundled TCC + `ptxas`, see
  `docs/windows-setup.md` R1); 1.7 s with a warm `~/.triton/cache`. Steady state on a
  2x256-token batch: **fla 8.0 ms vs torch fallback 71.7 ms** (~9x). Numerics vs the torch
  path: max |diff| 4.9e-3, mean 6.2e-4, on bf16 weights — bf16-level agreement.
- `unsloth/save.py` has `_is_qwen3_5_vlm` / `_qwen3_5_vlm_state_dict_for_save` for merged
  saves of these checkpoints, and `unsloth_zoo/llama_cpp.py` validates the converter against
  `_QWEN35_TENSOR_MAPPINGS`. `unsloth_zoo/gated_delta_vjp.py` is an **MLX** (Apple) memory-
  efficient GDN backward, irrelevant here.
- Windows: nothing in the loader is Windows-gated for this family. The unresolved Windows risk
  is whether any part of Unsloth's training path needs a **C++** compiler (TCC only does C);
  the import, patching and kernel compilation did not. `task smoke` settles it.
- Unsloth's published VRAM figure: **QLoRA on Qwen3.8-27B needs 24 GB** — i.e. the whole 3090.
  `max_seq_length=2048`, batch 1-2, `offload_embedding=True`, everything else closed.

## 6. llama.cpp (build `b10798-mix-659e406`, CUDA 13.3, sm_86) and GGUF

- The cached `Qwen3.8-27B-UD-Q4_K_M.gguf` (15.33 GiB, 866 tensors) declares
  `general.architecture = qwen35`, `qwen35.block_count = 65` (64 layers + 1 `nextn` MTP
  layer), `qwen35.ssm.conv_kernel = 4`, `ssm.state_size = 128`, `ssm.group_count = 16`,
  `ssm.time_step_rank = 48`, `ssm.inner_size = 6144`, `full_attention_interval = 4`, mixed
  quant types (Q4_K/Q5_K/Q6_K/Q8_0/IQ4_XS/IQ4_NL/IQ3_S, F32 for norms). Tensor families:
  `attn_q/k/v/output/gate/q_norm/k_norm`, `ssm_conv1d/ssm_a/ssm_alpha/ssm_beta/ssm_dt/ssm_norm/
  ssm_out`, `attn_qkv` (GDN in_proj), `ffn_*`, `nextn.*`.
- `src/llama-arch.*` and `llama-model.cpp` carry `LLM_ARCH_QWEN35`; the hybrid memory
  (recurrent + KV) is handled alongside `QWEN3NEXT`. So the installed `llama-server.exe` can
  serve the cached GGUF today.
- Converter: this build moved model classes from `convert_hf_to_gguf.py` (now a 317-line shim)
  into `conversion/`. `conversion/qwen.py` line 630:
  `@ModelBase.register("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM") class
  Qwen3_5TextModel(_Qwen35MRopeMixin, _LinearAttentionVReorderBase)`. `conversion/qwen3vl.py`
  registers the same class names for the `mmproj` (vision) output. **Both our checkpoint's
  architecture string and the text-only one are registered.**
- `_LinearAttentionVReorderBase` **reorders GDN V heads from grouped to tiled order** on
  `in_proj_qkv/z/a/b/out_proj` so ggml can use a cheap tiled broadcast (llama.cpp PR 19468).
  This is why merging the LoRA into BF16 *before* conversion (our plan, locked decision 8 /
  `scripts/convert_gguf.py`) is the clean path: the reorder is applied to merged weights. A
  `convert_lora_to_gguf.py` adapter on GDN projections would need the same permutation on its
  B matrices; an attention+MLP-only adapter does not (§3).
- `convert_lora_to_gguf.py` resolves the base class via
  `get_model_architecture(hparams, ModelType.TEXT)`, so it picks `Qwen3_5TextModel` for a VL
  base config.

## 7. trl 0.23.1

Nothing model-specific was found or is needed: `SFTTrainer` and `GRPOTrainer` operate on the
`PreTrainedModel` interface. The one interaction is §2's packing bug — SFT config must keep
`packing=False` and `padding_free=False` on this transformers version.

## 8. Measured broadcast result (question e)

On the tiny 3xGDN+1xattention model, bf16, N=4 suffixes of 9 tokens after a 40-token prefix:

| Broadcast strategy | Result |
|---|---|
| expand **keys, values, conv_states, recurrent_states** per layer (`repeat_interleave(N, dim=0)`) | max |diff| vs N independent full-sequence runs **3.9e-3** (bf16) -> match |
| expand keys + values only | `RuntimeError: output with shape [1, 128, 4] doesn't match the broadcast shape [4, 128, 4]` -> fails loudly at the first GDN conv update |
| `DynamicCache.batch_repeat_interleave(N)` | `AttributeError` on `LinearAttentionLayer` |

Independent corroboration: `shamazharikh/qwen-rlcd` PR #1 implements the same per-layer
`expand_cache` and reports 7e-7 (tiny fp32) and 4.77e-5 (real Qwen3.5-0.8B) max difference,
and notes that a `LinearAttentionAndFullAttentionLayer` variant in newer transformers inherits
`DynamicLayer`'s method and **silently repeats only keys/values** — i.e. on some versions the
naive path does *not* fail loudly. Our Step 4 test must therefore assert equality against
independent runs, not merely absence of an exception.

## 9. Smoke-test model

No small Qwen3.8 exists: the family is Qwen3.8-27B, Qwen3.8-2.4T-A95B (MoE), Qwen3.8-Flash-Next
and Qwen3.8-Max ([QwenLM/Qwen3.8](https://github.com/QwenLM/Qwen3.8),
[unsloth.ai/docs/models/qwen3.8](https://unsloth.ai/docs/models/qwen3.8), 2026-09-17).

The Qwen3.5 dense line has 0.8B, 2B, 4B, 9B, 27B
([artificialanalysis.ai](https://artificialanalysis.ai/articles/qwen3-5-small-models)).
**`Qwen/Qwen3.5-0.8B`** (`config.json`, `api/models` fetched 2026-09-17):
`Qwen3_5ForConditionalGeneration`, `model_type qwen3_5`, 24 layers in the same 3:1 pattern
(`full_attention_interval 4`), `attn_output_gate true`, MTP 1, hidden 1024, 8 q / 2 kv heads,
head_dim 256, GDN 16 key heads x 128 / 16 value heads x 128, conv 4, vocab 248,320, context
262,144, bf16, Apache-2.0. Single safetensors shard **1,746,942,600 bytes = 1.63 GB**, plus
~23 MB of tokenizer files. Under the 2 GB bar, same architecture class, same cache classes,
same GDN kernels, and it is the exact model `qwen-rlcd` validated its fork against. Note
`linear_num_value_heads == linear_num_key_heads` there (16/16) while the 27B has 48/16, so the
27B run additionally exercises the grouped-V path; the tiny random model in
`scripts/derisk_base_model.py` uses 4/2 to cover that shape on CPU-sized tensors.

## 10. Things this audit did not do

- Load the 27B at 4-bit. Needs the `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` download (~17-18 GB,
  approval required) and a closed desktop; it is the first thing Step 4's second "go" does.
- Run a training step. `task smoke` on Qwen3.5-0.8B is where the packing assertion, the bf16
  compute dtype and the "does Unsloth need C++ anywhere" question get answered.
- Verify the exact size of the bnb-4bit repo (the API call was not made; estimate from Q4 GGUF
  plus unquantized `in_proj_*`).

Sources: see inline links; local paths are under `.venv/Lib/site-packages/` and
`%USERPROFILE%/.unsloth/llama.cpp/`.
