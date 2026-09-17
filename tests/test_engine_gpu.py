"""GPU contract tests for the hf engine on real Qwen3.5-0.8B weights.

Run with ``uv run task test-gpu``. Skips without CUDA or without the weights in the local HF
cache (``Qwen/Qwen3.5-0.8B``, ~1.63 GB). Same architecture class as the 27B target: 3:1
gated-deltanet / attention, vision tower dropped by the ``ForCausalLM`` load.
"""

from __future__ import annotations

import time

import pytest

from s1decide.decide import decide
from s1decide.primitives import Choice, Noul, Question, Score
from s1decide.prompt import render

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from s1decide.engine.hf import HFEngine, nf4_config  # noqa: E402
from s1decide.tokens import allowed_token_ids  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.weights]

SMOKE_MODEL = "Qwen/Qwen3.5-0.8B"


def _require_cuda_and_weights() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(SMOKE_MODEL, local_files_only=True, allow_patterns=["config.json"])
    except Exception as exc:
        pytest.skip(f"{SMOKE_MODEL} not in the local cache: {type(exc).__name__}")


def _load(dtype: torch.dtype, quantization_config=None, **kwargs) -> HFEngine:
    _require_cuda_and_weights()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    engine = HFEngine.from_pretrained(
        SMOKE_MODEL,
        dtype=dtype,
        quantization_config=quantization_config,
        device_map="cuda",
        local_files_only=True,
        **kwargs,
    )
    engine.load_seconds = time.perf_counter() - started  # type: ignore[attr-defined]
    engine.load_peak_gib = torch.cuda.max_memory_allocated() / 2**30  # type: ignore[attr-defined]
    return engine


@pytest.fixture(scope="module")
def engine_fp32() -> HFEngine:
    """fp32 weights on the pure-torch GDN path: the exactness reference (0.8B fits easily).

    `prefer_fla=False` keeps Unsloth out of this load. If another fixture has already imported
    Unsloth in this process, the fla kernels are bound anyway; `on_fla()` tells the tests which
    tolerance applies (fla kernels are ~5e-3 less exact in fp32 than the torch path).
    """
    return _load(torch.float32, prefer_fla=False)


@pytest.fixture(scope="module")
def engine_bf16() -> HFEngine:
    return _load(torch.bfloat16)


@pytest.fixture(scope="module")
def engine_nf4() -> HFEngine:
    """bnb nf4 with bf16 compute: the deployment quantization on the 3090."""
    return _load(torch.bfloat16, quantization_config=nf4_config(torch.bfloat16))


@pytest.fixture
def questions() -> list[Question]:
    return [
        Question("tone", Choice("What is the customer's tone?", ("calm", "frustrated", "angry"))),
        Question(
            "urgency", Score("How urgent is this ticket?", ("can wait", "this week", "today"))
        ),
        Question("billing", Noul("This ticket is about billing.")),
        Question("weather", Noul("This ticket is about the weather.")),
        Question(
            "lang", Choice("Which language is the ticket in?", ("English", "Turkish", "German"))
        ),
    ]


def on_fla(engine: HFEngine) -> bool:
    """Whether the engine's gated-deltanet modules are bound to the fla Triton kernels."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    gdn = [mod for mod in engine.model.modules() if isinstance(mod, m.Qwen3_5GatedDeltaNet)]
    return all(mod.chunk_gated_delta_rule is not m.torch_chunk_gated_delta_rule for mod in gdn)


def independent(engine: HFEngine, prefix: str, suffix: str, labels) -> torch.Tensor:
    ids = engine.encode(prefix) + engine.encode(suffix)
    with torch.inference_mode():
        logits = engine.model(input_ids=torch.tensor([ids], device=engine.device)).logits[0, -1]
    return logits[list(allowed_token_ids(engine.tokenizer, labels))].float().cpu()


# --- model loads as expected -------------------------------------------------------------


def test_loads_text_only_with_sdpa_and_patched_gdn(engine_fp32: HFEngine) -> None:
    model = engine_fp32.model
    assert {p.dtype for p in model.parameters()} == {torch.float32}, (
        "requested fp32 must be honoured"
    )
    assert type(model).__name__ == "Qwen3_5ForCausalLM"
    assert model.config._attn_implementation == "sdpa"
    assert not any("visual" in name for name, _ in model.named_modules())
    layer_types = model.config.layer_types
    assert layer_types.count("linear_attention") == 18 and layer_types.count("full_attention") == 6
    assert engine_fp32.gdn_patched == 18
    print(
        f"\n[load fp32] {engine_fp32.load_seconds:.1f}s, peak {engine_fp32.load_peak_gib:.2f} GiB"
    )


def test_gdn_layers_run_on_fla_kernels(engine_bf16: HFEngine) -> None:
    """The production load (bf16, prefer_fla=True) must end up on the vendored Triton kernels."""
    assert on_fla(engine_bf16), (
        "Unsloth's vendored fla kernels were not injected before model construction"
    )


# --- the contract: N-row broadcast == N independent runs, on real weights ----------------


def test_broadcast_equals_independent_runs_fp32(
    engine_fp32: HFEngine, ticket_state, questions
) -> None:
    rendered = render(ticket_state, questions)
    out = engine_fp32.score(rendered.prefix, rendered.suffixes, rendered.labels)
    # Measured on Qwen3.5-0.8B: torch GDN path 2.3e-5; fla kernels 5.5e-3 (fla is less exact in fp32).
    atol = 2e-2 if on_fla(engine_fp32) else 1e-3
    worst = 0.0
    for i, suffix in enumerate(rendered.suffixes):
        got = torch.tensor(out.logits[i])
        ref = independent(engine_fp32, rendered.prefix, suffix, rendered.labels[i])
        worst = max(worst, (got - ref).abs().max().item())
        assert torch.allclose(got, ref, atol=atol, rtol=0.0), (
            f"{rendered.names[i]}: {got.tolist()} vs {ref.tolist()}"
        )
        assert got.argmax() == ref.argmax()
    print(
        f"\n[broadcast fp32, fla={on_fla(engine_fp32)}] worst |diff| over {len(questions)} questions: {worst:.2e} (atol {atol}); passes={out.meta['passes']}, expanded={out.meta['cache_expanded']}"
    )


def test_broadcast_equals_independent_runs_bf16(
    engine_bf16: HFEngine, ticket_state, questions
) -> None:
    """bf16 is what runs in production; agreement is at bf16 noise level, argmax must match."""
    rendered = render(ticket_state, questions)
    out = engine_bf16.score(rendered.prefix, rendered.suffixes, rendered.labels)
    worst = 0.0
    for i, suffix in enumerate(rendered.suffixes):
        got = torch.tensor(out.logits[i])
        ref = independent(engine_bf16, rendered.prefix, suffix, rendered.labels[i])
        worst = max(worst, (got - ref).abs().max().item())
        # bf16 resolution at |logit| ~ 16-32 is 0.125; allow two ulps. Argmax may flip on near-ties.
        assert torch.allclose(got, ref, atol=0.3, rtol=0.0), (
            f"{rendered.names[i]}: {got.tolist()} vs {ref.tolist()}"
        )
    print(f"\n[broadcast bf16] worst |diff|: {worst:.2e}")


def test_chunked_passes_equal_single_pass(engine_fp32: HFEngine, ticket_state, questions) -> None:
    rendered = render(ticket_state, questions)
    single = engine_fp32.score(rendered.prefix, rendered.suffixes, rendered.labels)
    engine_fp32.max_rows_per_pass = 2
    try:
        chunked = engine_fp32.score(rendered.prefix, rendered.suffixes, rendered.labels)
    finally:
        engine_fp32.max_rows_per_pass = 16
    assert chunked.meta["passes"] == 3
    atol = 2e-2 if on_fla(engine_fp32) else 1e-3
    for a, b in zip(single.logits, chunked.logits):
        assert torch.allclose(torch.tensor(a), torch.tensor(b), atol=atol, rtol=0.0)


def test_unpatched_engine_would_fail_the_contract(ticket_state, questions) -> None:
    """The bug is real on real weights: without the patch the broadcast is wrong, not just noisy."""
    _require_cuda_and_weights()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        SMOKE_MODEL,
        dtype=torch.float32,
        device_map="cuda",
        attn_implementation="sdpa",
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(SMOKE_MODEL, local_files_only=True)
    engine = HFEngine(model, tokenizer, patch_gdn=False)
    assert engine.gdn_patched == 0
    rendered = render(ticket_state, questions)
    out = engine.score(rendered.prefix, rendered.suffixes, rendered.labels)
    worst = max(
        (torch.tensor(out.logits[i]) - independent(engine, rendered.prefix, s, rendered.labels[i]))
        .abs()
        .max()
        .item()
        for i, s in enumerate(rendered.suffixes)
    )
    print(f"\n[unpatched fp32] worst |diff|: {worst:.2e}")
    assert worst > 0.1, (
        "unpatched continuation agreed with the joint forward — transformers may have fixed the bug"
    )
    del model, engine
    torch.cuda.empty_cache()


# --- deployment quantization -------------------------------------------------------------


def test_nf4_engine_runs_and_agrees_on_the_easy_call(
    engine_nf4: HFEngine, engine_bf16: HFEngine, ticket_state, questions
) -> None:
    print(f"\n[load nf4] {engine_nf4.load_seconds:.1f}s, peak {engine_nf4.load_peak_gib:.2f} GiB")
    q4 = decide(ticket_state, questions, engine=engine_nf4)
    bf = decide(ticket_state, questions, engine=engine_bf16)
    assert q4.engine == "hf" and len(q4) == len(questions)
    # nf4 shifts probabilities; what must survive is the easy relative judgement.
    assert q4["billing"].noul > q4["weather"].noul
    assert bf["billing"].noul > bf["weather"].noul
    print(
        "[nf4 vs bf16] "
        + ", ".join(f"{r.name}: {r.confidence:.2f}/{bf[r.name].confidence:.2f}" for r in q4)
    )


# --- zero-shot sanity: the model actually reads the state ---------------------------------


def test_zero_shot_sanity_billing_ticket(engine_bf16: HFEngine, ticket_state, questions) -> None:
    response = decide(ticket_state, questions, engine=engine_bf16)
    for r in response:
        print(f"\n[zero-shot 0.8B] {r.name:8} -> {r.distribution}")
    assert response["billing"].noul > response["weather"].noul
    assert response["lang"].choice == "English"


def test_answer_depends_on_the_state(engine_bf16: HFEngine) -> None:
    q = [
        Question("lang", Choice("Which language is the text in?", ("English", "Turkish", "German")))
    ]
    en = decide("Please refund me, I was charged twice.", q, engine=engine_bf16)["lang"]
    tr = decide("Lütfen para iademi yapın, iki kez ücret alındı.", q, engine=engine_bf16)["lang"]
    print(f"\n[state sensitivity] en={en.distribution} tr={tr.distribution}")
    assert en.choice == "English"
    assert tr.probabilities[1] > en.probabilities[1]


def test_timing_and_memory_report(engine_bf16: HFEngine, ticket_state, questions) -> None:
    """Informational: prefill vs suffix time and peak VRAM for a 5-question call."""
    rendered = render(ticket_state, questions)
    engine_bf16.score(rendered.prefix, rendered.suffixes, rendered.labels)  # warm-up
    torch.cuda.reset_peak_memory_stats()
    out = engine_bf16.score(rendered.prefix, rendered.suffixes, rendered.labels)
    print(
        f"\n[timing bf16] prefix {out.meta['prefix_tokens']} tok: prefill {out.meta['prefill_seconds'] * 1e3:.1f} ms, "
        f"{len(questions)} suffixes {out.meta['suffix_seconds'] * 1e3:.1f} ms, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
    )
    assert out.meta["prefill_seconds"] > 0
