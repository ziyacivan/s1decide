"""transformers engine: prefill the prefix once, broadcast the cache, one forward for all suffixes.

The base model is a hybrid: 48 gated-deltanet layers whose cache is a recurrent state plus a
short conv window, and 16 attention layers whose cache is key/value. So "KV broadcast" here
means broadcasting **both** kinds of state, and :func:`expand_cache` refuses any cache layer it
does not recognise rather than silently copying half of it — on transformers 5.5.0
``DynamicCache.batch_repeat_interleave`` raises on the deltanet layers, and newer versions
carry a layer class that repeats only keys/values (ADR 0002, condition 1).

Attention is always ``sdpa``. This module never imports ``flash_attn``.
"""

from __future__ import annotations

import contextlib
import copy
import time
from collections.abc import Sequence
from typing import Any

import torch

from s1decide.engine.base import EngineOutput
from s1decide.engine.qwen3_5_patch import patch_gated_deltanet
from s1decide.tokens import allowed_token_ids

__all__ = [
    "CACHE_BATCH_TENSORS",
    "HFEngine",
    "expand_cache",
    "force_quantized_model_dtype",
    "nf4_config",
]

#: Every per-layer tensor a transformers cache may hold along the batch dimension. Attention
#: layers (`DynamicLayer`) hold the first two; gated-deltanet layers (`LinearAttentionLayer`)
#: hold the last two. A layer holding none of them is unknown and is refused.
CACHE_BATCH_TENSORS: tuple[str, ...] = ("keys", "values", "conv_states", "recurrent_states")


def expand_cache(cache: Any, repeats: int) -> dict[str, list[str]]:
    """Broadcast a batch-1 prefix cache to ``repeats`` rows, in place, every layer and every state.

    Args:
        cache: A transformers ``DynamicCache`` (or anything with a ``.layers`` list) produced
            by a batch-1 prefill.
        repeats: Target batch size.

    Returns:
        Which tensors were expanded on each layer, keyed by layer class name — recorded in
        the engine's ``meta`` so a run can prove it broadcast the recurrent state too.

    Raises:
        ValueError: If ``repeats`` is below 1 or any tensor is not batch-1.
        NotImplementedError: If the cache has no ``layers`` or a layer holds none of
            :data:`CACHE_BATCH_TENSORS`. Refusing is the point: a cache we do not understand
            must not be half-broadcast.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise NotImplementedError(
            f"cannot broadcast a {type(cache).__name__}: it has no `.layers`; only "
            "transformers DynamicCache-style caches are supported"
        )

    expanded: dict[str, list[str]] = {}
    for index, layer in enumerate(layers):
        touched: list[str] = []
        for attr in CACHE_BATCH_TENSORS:
            tensor = getattr(layer, attr, None)
            if not torch.is_tensor(tensor):
                continue
            if tensor.shape[0] != 1:
                raise ValueError(
                    f"cache layer {index} ({type(layer).__name__}).{attr} has batch "
                    f"{tensor.shape[0]}, expected 1 — expand_cache only broadcasts a single prefill"
                )
            # .contiguous() materialises a real copy: the model updates these tensors in place
            # on the next forward, and an expanded view would alias every row onto one buffer.
            setattr(layer, attr, tensor.expand(repeats, *tensor.shape[1:]).contiguous())
            touched.append(attr)
        if not touched:
            raise NotImplementedError(
                f"cache layer {index} is a {type(layer).__name__} holding none of "
                f"{CACHE_BATCH_TENSORS}; refusing to broadcast a cache layer we do not understand"
            )
        expanded.setdefault(type(layer).__name__, touched)
    return expanded


def nf4_config(compute_dtype: torch.dtype = torch.bfloat16) -> Any:
    """The 4-bit quantization config used for local inference on the 3090.

    Args:
        compute_dtype: Matmul dtype. bf16, never fp16: the deltanet layers produce NaNs in
            fp16 (ADR 0002, condition 3).

    Returns:
        A ``transformers.BitsAndBytesConfig``.
    """
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def force_quantized_model_dtype(model: Any, dtype: torch.dtype) -> dict[str, int]:
    """Make a bitsandbytes 4-bit model compute in ``dtype`` end to end.

    A pre-quantized checkpoint fixes more than its compute dtype: transformers keeps every
    *unquantized* tensor in the dtype it was stored in, whatever ``dtype`` was requested.
    ``unsloth/Qwen3.8-27B-unsloth-bnb-4bit`` stores its embeddings, norms, conv and the
    gated-deltanet input projections in **fp16**, so without this the whole network runs in
    fp16 — the dtype ADR 0002 rules out for these layers — and then trips over its own bf16
    ``lm_head``. Casting fp16 -> bf16 drops three mantissa bits on exactly the tensors the
    quantizer left in high precision; that trade is recorded, not hidden.

    Args:
        model: A model loaded with a bitsandbytes 4-bit quantization config.
        dtype: The dtype to compute in (bf16 on this project).

    Returns:
        Counts of what was changed: ``linear4bit`` layers whose compute dtype was set,
        ``quant_states`` whose recorded dtype was aligned, and ``params_cast`` non-4-bit
        floating parameters converted. All zero if bitsandbytes is absent.
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        return {"linear4bit": 0, "quant_states": 0, "params_cast": 0}

    counts = {"linear4bit": 0, "quant_states": 0, "params_cast": 0}
    for module in model.modules():
        if isinstance(module, bnb.nn.Linear4bit):
            module.compute_dtype = dtype
            counts["linear4bit"] += 1
            state = getattr(module.weight, "quant_state", None)
            if state is not None and getattr(state, "dtype", None) != dtype:
                state.dtype = dtype
                counts["quant_states"] += 1
    for param in model.parameters():
        if (
            not isinstance(param, bnb.nn.Params4bit)
            and param.is_floating_point()
            and param.dtype != dtype
        ):
            param.data = param.data.to(dtype)
            counts["params_cast"] += 1
    quant_cfg = getattr(getattr(model, "config", None), "quantization_config", None)
    if quant_cfg is not None and hasattr(quant_cfg, "bnb_4bit_compute_dtype"):
        quant_cfg.bnb_4bit_compute_dtype = dtype
    return counts


class HFEngine:
    """Score many suffixes over one prefix with a transformers causal LM.

    Args:
        model: A loaded causal LM in eval mode (any device, any dtype, quantized or not).
        tokenizer: Its tokenizer. Only ``encode`` is used.
        max_rows_per_pass: Upper bound on suffixes per forward pass. The broadcast cache
            costs VRAM per row — on Qwen3.8-27B roughly 150 MB of fp32 recurrent state plus
            ~65 KB per prefix token of KV — so large question sets are processed in chunks
            over the same prefill. ``meta["passes"]`` records how many were needed.
    """

    name = "hf"

    #: Held back from the rows-per-pass budget on CUDA: bitsandbytes dequantises one weight
    #: matrix at a time (up to ~180 MB in bf16 on the 27B) and the pass needs activations.
    VRAM_RESERVE_BYTES = 768 * 2**20

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_rows_per_pass: int = 16,
        patch_gdn: bool = True,
    ) -> None:
        if max_rows_per_pass < 1:
            raise ValueError("max_rows_per_pass must be >= 1")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_rows_per_pass = max_rows_per_pass
        # The input embeddings live where the text tower lives. `next(model.parameters())` would
        # report the vision tower's device on a VLM class, and that tower may be parked on CPU.
        embeddings = (
            model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        )
        self.device = (
            embeddings.weight.device if embeddings is not None else next(model.parameters()).device
        )
        # transformers 5.5.0 ignores the cached GDN state on multi-token continuation — which is
        # every suffix we score. See engine/qwen3_5_patch.py. Off only for tests that demonstrate the bug.
        self.gdn_patched = patch_gated_deltanet(model) if patch_gdn else 0
        pad = getattr(tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(tokenizer, "eos_token_id", None)
        self.pad_token_id = int(pad) if pad is not None else 0

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        quantization_config: Any | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str | dict[str, Any] = "cuda",
        max_rows_per_pass: int = 16,
        prefer_fla: bool = True,
        **kwargs: Any,
    ) -> HFEngine:
        """Load a causal LM text-only with SDPA attention and wrap it.

        Args:
            model_name: Hub id or local path. For a vision-language checkpoint such as
                ``Qwen/Qwen3.8-27B`` this loads ``*ForCausalLM`` and drops the vision tower.
            quantization_config: e.g. :func:`nf4_config`. Leave ``None`` for a checkpoint
                that is already quantized (``*-bnb-4bit``) or for bf16.
            dtype: Weight/compute dtype for unquantized loads.
            device_map: Passed through to transformers.
            max_rows_per_pass: See the class docstring.
            prefer_fla: Import Unsloth first so its vendored flash-linear-attention Triton
                kernels back the gated-deltanet layers (measured ~9x faster than the torch
                path). The kernels are bound when the model is built, so this must happen
                before loading. Silently skipped if Unsloth is not installed.
            **kwargs: Passed through to ``from_pretrained``.

        Returns:
            A ready engine.
        """
        if prefer_fla:
            with contextlib.suppress(ImportError):
                import unsloth  # noqa: F401 - imported for its side effect: fla kernel injection

        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        config_kwargs = {
            k: v for k, v in kwargs.items() if k in {"local_files_only", "revision", "token"}
        }
        config = AutoConfig.from_pretrained(model_name, **config_kwargs)
        prequantized_vlm = (
            getattr(config, "quantization_config", None) is not None
            and getattr(config, "vision_config", None) is not None
        )
        common: dict[str, Any] = dict(
            dtype=dtype, device_map=device_map, attn_implementation="sdpa", **kwargs
        )
        # Passing `quantization_config=None` explicitly makes transformers 5.5 overwrite a
        # pre-quantized checkpoint's own config with None and then crash in the quantizer
        # lookup; only pass the key when we have something to say.
        if quantization_config is not None:
            common["quantization_config"] = quantization_config
        if prequantized_vlm:
            # `AutoModelForCausalLM` builds the text tower from `text_config`, which does not
            # carry the parent's `quantization_config`, so a pre-quantized VLM checkpoint such
            # as unsloth/*-bnb-4bit would load as if it were bf16 and fail on packed weights.
            # Hand it the text config with the quantization config copied over (the same
            # trick Unsloth's loader uses); the `model.language_model.*` -> `model.*` key remap
            # is transformers' own and covers the bitsandbytes aux tensors too.
            text_config = copy.deepcopy(config.get_text_config())
            text_config.quantization_config = config.quantization_config
            common["config"] = text_config
        model = AutoModelForCausalLM.from_pretrained(model_name, **common)
        if quantization_config is None and not getattr(model, "is_quantized", False):
            # transformers 5.5 builds the text tower from the VLM's `text_config`, whose own
            # `dtype` (bf16 for Qwen3.5/3.8) silently overrides the `dtype` we asked for.
            # Measured: dtype=float32 came back as bf16 parameters and bf16 logits.
            model = model.to(dtype)
            got = {p.dtype for p in model.parameters()}
            if got != {dtype}:
                raise RuntimeError(f"requested dtype {dtype} but model parameters are {got}")
        else:
            # A pre-quantized checkpoint carries its own dtypes — `unsloth/*-bnb-4bit` is fp16
            # throughout its unquantized tensors, and fp16 NaNs in the gated-deltanet layers
            # (ADR 0002, condition 3).
            force_quantized_model_dtype(model, dtype)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            **{k: v for k, v in kwargs.items() if k in {"local_files_only", "revision", "token"}},
        )
        return cls(model, tokenizer, max_rows_per_pass=max_rows_per_pass)

    def encode(self, text: str) -> list[int]:
        """Tokenize without special tokens — the chat template already supplies them."""
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    @torch.inference_mode()
    def score(
        self, prefix: str, suffixes: Sequence[str], labels: Sequence[Sequence[str]]
    ) -> EngineOutput:
        """Prefill the prefix once, then score every suffix's labels at its answer position."""
        if not suffixes:
            raise ValueError("score() needs at least one suffix")
        if len(suffixes) != len(labels):
            raise ValueError(f"{len(suffixes)} suffixes but {len(labels)} label sets")

        prefix_ids = self.encode(prefix)
        suffix_ids = [self.encode(s) for s in suffixes]
        if any(not ids for ids in suffix_ids):
            raise ValueError("every suffix must tokenize to at least one token")
        label_ids = [allowed_token_ids(self.tokenizer, row) for row in labels]

        started = time.perf_counter()
        # The prefill's logits are never read; keep one position instead of a full-vocab
        # tensor for every prefix token (~750 MB at 1,500 tokens on a 248k vocabulary).
        prefill = self.model(
            input_ids=torch.tensor([prefix_ids], device=self.device),
            use_cache=True,
            logits_to_keep=1,
        )
        prefix_cache = prefill.past_key_values
        if prefix_cache is None:
            raise RuntimeError("model returned no cache from the prefill; use_cache is required")
        del prefill
        self._sync()
        prefill_seconds = time.perf_counter() - started

        rows_per_pass, cache_bytes_per_row = self._rows_per_pass(prefix_cache)
        rows: list[tuple[float, ...]] = []
        passes = 0
        expanded: dict[str, list[str]] = {}
        started = time.perf_counter()
        for start in range(0, len(suffix_ids), rows_per_pass):
            chunk = suffix_ids[start : start + rows_per_pass]
            chunk_labels = label_ids[start : start + rows_per_pass]
            cache = copy.deepcopy(prefix_cache)
            expanded = expand_cache(cache, len(chunk))
            rows.extend(self._score_chunk(cache, len(prefix_ids), chunk, chunk_labels))
            del cache
            passes += 1
        self._sync()
        suffix_seconds = time.perf_counter() - started

        return EngineOutput(
            logits=tuple(rows),
            meta={
                "engine": self.name,
                "prefix_tokens": len(prefix_ids),
                "suffix_tokens": [len(ids) for ids in suffix_ids],
                "rows": len(rows),
                "passes": passes,
                "max_rows_per_pass": self.max_rows_per_pass,
                "rows_per_pass": rows_per_pass,
                "cache_bytes_per_row": cache_bytes_per_row,
                "cache_expanded": expanded,
                "gdn_patched": self.gdn_patched,
                "prefill_seconds": prefill_seconds,
                "suffix_seconds": suffix_seconds,
                "device": str(self.device),
            },
        )

    def _rows_per_pass(self, prefix_cache: Any) -> tuple[int, int]:
        """Decide how many suffix rows one pass may carry, from the cache's real size.

        Every row of the broadcast holds a full copy of the prefix cache — on the 27B roughly
        150 MB of fp32 recurrent state plus the attention KV — and the pass also needs room
        for activations and the suffix logits. On CUDA the count is capped so that the expanded
        caches take at most half of the currently free memory; elsewhere the configured
        maximum is used.

        Returns:
            ``(rows_per_pass, cache_bytes_per_row)``.
        """
        per_row = 0
        for layer in getattr(prefix_cache, "layers", []):
            for attr in CACHE_BATCH_TENSORS:
                tensor = getattr(layer, attr, None)
                if torch.is_tensor(tensor):
                    per_row += tensor.numel() * tensor.element_size()
        if self.device.type != "cuda" or per_row == 0:
            return self.max_rows_per_pass, per_row
        # Device-free memory plus what the caching allocator already holds but is not using;
        # keep a fixed reserve for bitsandbytes' dequantisation temporaries and the pass's
        # activations, then spend most of the rest on rows.
        free, _total = torch.cuda.mem_get_info(self.device)
        cached = torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
        available = free + max(0, cached) - self.VRAM_RESERVE_BYTES
        affordable = max(1, int(available * 0.8 // per_row))
        return min(self.max_rows_per_pass, affordable), per_row

    def _score_chunk(
        self,
        cache: Any,
        prefix_len: int,
        chunk: Sequence[Sequence[int]],
        chunk_labels: Sequence[Sequence[int]],
    ) -> list[tuple[float, ...]]:
        """Run one padded batch of suffixes over an already-broadcast cache."""
        rows = len(chunk)
        lengths = torch.tensor([len(ids) for ids in chunk], device=self.device)
        width = int(lengths.max())

        # Right padding: every real token precedes every pad token, so with causal attention
        # and causal recurrence the pads cannot influence the answer position we read.
        input_ids = torch.full(
            (rows, width), self.pad_token_id, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros(
            (rows, prefix_len + width), dtype=torch.long, device=self.device
        )
        attention_mask[:, :prefix_len] = 1
        for i, ids in enumerate(chunk):
            input_ids[i, : len(ids)] = torch.tensor(ids, device=self.device)
            attention_mask[i, prefix_len : prefix_len + len(ids)] = 1

        row_index = torch.arange(rows, device=self.device)
        base = getattr(self.model, "model", None)
        head = (
            self.model.get_output_embeddings()
            if hasattr(self.model, "get_output_embeddings")
            else None
        )
        if base is not None and head is not None:
            # Run the trunk, gather the one hidden state per row we need, then the head. The
            # full-vocabulary logits for every suffix position would cost rows x width x 248k
            # x 2 bytes (~550 MB at 16 rows on the 27B) for values that are never read.
            hidden = base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            ).last_hidden_state
            answer_logits = head(hidden[row_index, lengths - 1])
        else:
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
            answer_logits = out.logits[row_index, lengths - 1]
        return [
            tuple(answer_logits[i, list(ids)].float().tolist())
            for i, ids in enumerate(chunk_labels)
        ]

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
