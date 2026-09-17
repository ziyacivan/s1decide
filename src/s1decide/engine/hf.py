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

__all__ = ["CACHE_BATCH_TENSORS", "HFEngine", "expand_cache", "nf4_config"]

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
        self.device = next(model.parameters()).device
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

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map,
            attn_implementation="sdpa",
            quantization_config=quantization_config,
            **kwargs,
        )
        if quantization_config is None:
            # transformers 5.5 builds the text tower from the VLM's `text_config`, whose own
            # `dtype` (bf16 for Qwen3.5/3.8) silently overrides the `dtype` we asked for.
            # Measured: dtype=float32 came back as bf16 parameters and bf16 logits.
            model = model.to(dtype)
            got = {p.dtype for p in model.parameters()}
            if got != {dtype}:
                raise RuntimeError(f"requested dtype {dtype} but model parameters are {got}")
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
        prefill = self.model(
            input_ids=torch.tensor([prefix_ids], device=self.device), use_cache=True
        )
        prefix_cache = prefill.past_key_values
        if prefix_cache is None:
            raise RuntimeError("model returned no cache from the prefill; use_cache is required")
        self._sync()
        prefill_seconds = time.perf_counter() - started

        rows: list[tuple[float, ...]] = []
        passes = 0
        expanded: dict[str, list[str]] = {}
        started = time.perf_counter()
        for start in range(0, len(suffix_ids), self.max_rows_per_pass):
            chunk = suffix_ids[start : start + self.max_rows_per_pass]
            chunk_labels = label_ids[start : start + self.max_rows_per_pass]
            cache = copy.deepcopy(prefix_cache)
            expanded = expand_cache(cache, len(chunk))
            rows.extend(self._score_chunk(cache, len(prefix_ids), chunk, chunk_labels))
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
                "cache_expanded": expanded,
                "gdn_patched": self.gdn_patched,
                "prefill_seconds": prefill_seconds,
                "suffix_seconds": suffix_seconds,
                "device": str(self.device),
            },
        )

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

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
        answer_logits = out.logits[torch.arange(rows, device=self.device), lengths - 1]
        return [
            tuple(answer_logits[i, list(ids)].float().tolist())
            for i, ids in enumerate(chunk_labels)
        ]

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
