"""GPU contract tests for the hf engine on the target model, Qwen3.8-27B at 4-bit.

Run alone — ``uv run task test-gpu tests/test_engine_gpu_27b.py`` — so the VRAM numbers are
those of one 27B copy and nothing else. Skips without CUDA or without
``unsloth/Qwen3.8-27B-unsloth-bnb-4bit`` in the local HF cache (~17 GB).
"""

from __future__ import annotations

import time

import pytest

from s1decide.decide import decide
from s1decide.primitives import Choice, Noul, Question, Score
from s1decide.prompt import render

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from s1decide.engine.hf import HFEngine  # noqa: E402
from s1decide.tokens import allowed_token_ids  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.weights, pytest.mark.slow]

MODEL_4BIT = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit"

GIB = 2**30


def _require() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(MODEL_4BIT, local_files_only=True, allow_patterns=["config.json"])
    except Exception as exc:
        pytest.skip(f"{MODEL_4BIT} not in the local cache: {type(exc).__name__}")


@pytest.fixture(scope="module")
def engine() -> HFEngine:
    """The one 27B copy. bf16 compute over the pre-quantized nf4 weights, fla kernels on."""
    _require()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    eng = HFEngine.from_pretrained(
        MODEL_4BIT, dtype=torch.bfloat16, device_map="cuda", local_files_only=True
    )
    eng.load_seconds = time.perf_counter() - started  # type: ignore[attr-defined]
    eng.resident_gib = torch.cuda.memory_allocated() / GIB  # type: ignore[attr-defined]
    print(
        f"\n[27B load] {eng.load_seconds:.0f}s, resident {eng.resident_gib:.2f} GiB, peak {torch.cuda.max_memory_allocated() / GIB:.2f} GiB"
    )
    return eng


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


def long_state(engine: HFEngine, target_tokens: int = 1500) -> str:
    """A realistic-looking state of about ``target_tokens`` tokens (fixed text, deterministic)."""
    paragraph = (
        "Ticket #{i}: The customer reports that their subscription was charged twice on the same day. "
        "They have attached two bank statements and ask for a refund of the duplicate charge. Previous "
        "contact on this account: none in the last twelve months. Plan: Pro, monthly, auto-renew enabled. "
    )
    text, i = "", 0
    while len(engine.encode(text)) < target_tokens:
        text += paragraph.format(i=i)
        i += 1
    return text


def independent(engine: HFEngine, prefix: str, suffix: str, labels) -> torch.Tensor:
    ids = engine.encode(prefix) + engine.encode(suffix)
    with torch.inference_mode():
        logits = engine.model(input_ids=torch.tensor([ids], device=engine.device)).logits[0, -1]
    return logits[list(allowed_token_ids(engine.tokenizer, labels))].float().cpu()


# --- load ---------------------------------------------------------------------------------


def test_loads_text_only_4bit_with_bf16_compute(engine: HFEngine) -> None:
    import bitsandbytes as bnb
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    model = engine.model
    # Pre-quantized VLM checkpoints load as the full class (quant config lives on the parent
    # config); the vision tower is parked on the CPU so it costs no VRAM.
    assert type(model).__name__ in ("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration")
    assert model.config._attn_implementation == "sdpa"
    visual_on_gpu = [
        n for n, p in model.named_parameters() if "visual" in n and p.device.type == "cuda"
    ]
    assert not visual_on_gpu, visual_on_gpu[:3]
    text_config = getattr(model.config, "text_config", model.config)
    assert text_config.layer_types.count("linear_attention") == 48
    assert text_config.layer_types.count("full_attention") == 16
    assert engine.gdn_patched == 48

    linear4bit = [mod for mod in model.modules() if isinstance(mod, bnb.nn.Linear4bit)]
    assert linear4bit, "expected bitsandbytes 4-bit layers"
    # The checkpoint stores its unquantized tensors in fp16; the engine must have cast them.
    plain_dtypes = {
        p.dtype
        for p in model.parameters()
        if not isinstance(p, bnb.nn.Params4bit) and p.is_floating_point()
    }
    assert plain_dtypes == {torch.bfloat16}, plain_dtypes
    assert {str(mod.weight.quant_state.dtype) for mod in linear4bit} == {"torch.bfloat16"}
    assert {mod.compute_dtype for mod in linear4bit} == {torch.bfloat16}, (
        "compute dtype must be bf16, not the repo's fp16"
    )
    assert model.config.quantization_config.bnb_4bit_compute_dtype in (torch.bfloat16, "bfloat16")

    gdn = [mod for mod in model.modules() if isinstance(mod, m.Qwen3_5GatedDeltaNet)]
    assert all(mod.chunk_gated_delta_rule is not m.torch_chunk_gated_delta_rule for mod in gdn), (
        "GDN not on fla kernels"
    )
    # The repo skips quantizing the GDN input projections; they must be bf16 Linear, not 4-bit.
    skipped = {
        type(gdn[0].in_proj_qkv).__name__,
        type(gdn[0].in_proj_a).__name__,
        type(gdn[0].in_proj_b).__name__,
    }
    assert skipped == {"Linear"}, skipped
    assert engine.resident_gib < 22.0, (
        f"27B resident {engine.resident_gib:.2f} GiB leaves no room for the cache"
    )


# --- the contract on the target model ----------------------------------------------------


def test_broadcast_equals_independent_runs(engine: HFEngine, ticket_state, questions) -> None:
    rendered = render(ticket_state, questions)
    out = engine.score(rendered.prefix, rendered.suffixes, rendered.labels)
    expected_passes = -(-len(questions) // out.meta["rows_per_pass"])
    assert out.meta["passes"] == expected_passes
    print(
        f"\n[27B rows] rows_per_pass={out.meta['rows_per_pass']} "
        f"(cache {out.meta['cache_bytes_per_row'] / 2**20:.0f} MiB/row), passes={out.meta['passes']}"
    )
    worst = 0.0
    for i, suffix in enumerate(rendered.suffixes):
        got = torch.tensor(out.logits[i])
        ref = independent(engine, rendered.prefix, suffix, rendered.labels[i])
        worst = max(worst, (got - ref).abs().max().item())
        # Measured 2026-09-17: the *same* sequence run at batch 1 vs duplicated to batch 4, with
        # no cache involved, already differs by up to 0.5 on this model (bf16 kernels change with
        # batch shape across 64 layers). The broadcast is held to that noise floor plus margin,
        # and to the probability-level agreement that actually matters downstream.
        assert torch.allclose(got, ref, atol=0.75, rtol=0.0), (
            f"{rendered.names[i]}: {got.tolist()} vs {ref.tolist()}"
        )
        prob_gap = (got.softmax(0) - ref.softmax(0)).abs().max().item()
        assert prob_gap < 0.05, f"{rendered.names[i]}: probability gap {prob_gap:.3f}"
    print(
        f"\n[27B broadcast] worst |diff| over {len(questions)} questions: {worst:.2e}; expanded={out.meta['cache_expanded']}"
    )


def test_chunked_passes_equal_single_pass(engine: HFEngine, ticket_state, questions) -> None:
    rendered = render(ticket_state, questions)
    single = engine.score(rendered.prefix, rendered.suffixes, rendered.labels)
    engine.max_rows_per_pass = 2
    try:
        chunked = engine.score(rendered.prefix, rendered.suffixes, rendered.labels)
    finally:
        engine.max_rows_per_pass = 16
    assert chunked.meta["rows_per_pass"] <= 2
    assert chunked.meta["passes"] >= single.meta["passes"]
    for a, b in zip(single.logits, chunked.logits):
        a_t, b_t = torch.tensor(a), torch.tensor(b)
        assert torch.allclose(a_t, b_t, atol=0.75, rtol=0.0)  # batch-shape kernel noise, see above
        assert (a_t.softmax(0) - b_t.softmax(0)).abs().max().item() < 0.05


# --- VRAM and first timings on a ~1,500-token state ----------------------------------------


def test_vram_and_timing_on_a_1500_token_state(engine: HFEngine) -> None:
    """The number Step 6 needs before benchmarking: does 16 rows over a 1.5k state fit, and at what cost?"""
    state = long_state(engine, 1500)
    base = [
        Question(
            f"q{i}", Noul(f"Statement {i}: the customer asks for a refund of a duplicate charge.")
        )
        for i in range(64)
    ]
    report = []
    for n in (1, 4, 16, 64):
        rendered = render(state, base[:n])
        engine.score(
            rendered.prefix, rendered.suffixes, rendered.labels
        )  # warm-up (Triton compiles, allocator)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        out = engine.score(rendered.prefix, rendered.suffixes, rendered.labels)
        torch.cuda.synchronize()
        total = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated() / GIB
        report.append(
            (
                n,
                total,
                out.meta["prefill_seconds"],
                out.meta["suffix_seconds"],
                out.meta["passes"],
                peak,
            )
        )
        assert peak < 23.5, f"{n} questions peaked at {peak:.2f} GiB"
    print(
        f"\n[27B nf4 bf16, prefix {out.meta['prefix_tokens']} tok, max_rows_per_pass={engine.max_rows_per_pass}]"
    )
    for n, total, prefill, suffix, passes, peak in report:
        print(
            f"   n={n:3d}: total {total * 1e3:7.1f} ms | prefill {prefill * 1e3:7.1f} ms | suffixes {suffix * 1e3:7.1f} ms in {passes} pass(es) | peak {peak:.2f} GiB"
        )
    one, sixty_four = report[0][1], report[-1][1]
    print(
        f"   64-question call / 1-question call = {sixty_four / one:.2f}x (single measurement; Step 6 does 20 repeats)"
    )


# --- zero-shot sanity: does the 27B read the state? ---------------------------------------


def test_zero_shot_sanity(engine: HFEngine, ticket_state, questions) -> None:
    response = decide(ticket_state, questions, engine=engine)
    for r in response:
        print(
            f"\n[27B zero-shot] {r.name:8} -> "
            + ", ".join(f"{k}: {v:.3f}" for k, v in r.distribution.items())
        )
    assert response["billing"].noul > response["weather"].noul
    assert response["lang"].choice == "English"
    assert response["tone"].choice != "calm"


def test_answer_depends_on_the_state(engine: HFEngine) -> None:
    q = [
        Question("lang", Choice("Which language is the text in?", ("English", "Turkish", "German")))
    ]
    en = decide("Please refund me, I was charged twice.", q, engine=engine)["lang"]
    tr = decide("Lütfen para iademi yapın, iki kez ücret alındı.", q, engine=engine)["lang"]
    de = decide(
        "Bitte erstatten Sie mir den Betrag, ich wurde zweimal belastet.", q, engine=engine
    )["lang"]
    print(
        f"\n[27B lang] en={en.choice} ({en.confidence:.2f}) tr={tr.choice} ({tr.confidence:.2f}) de={de.choice} ({de.confidence:.2f})"
    )
    assert (en.choice, tr.choice, de.choice) == ("English", "Turkish", "German")


def test_buffer_reuse_does_not_move_a_single_logit(
    engine: HFEngine, ticket_state, questions
) -> None:
    """ADR 0003 option C is an allocation change, not a numerics change — on real weights.

    Same batch shapes on both sides, so unlike ``test_chunked_passes_equal_single_pass`` there is
    no kernel-shape noise to allow for: the two paths differ only in where the cache tensors
    live. If this holds, no metric can move, which is what the condition on option C asks.
    """
    rendered = render(ticket_state, questions)
    engine.reuse_buffer = True
    reused = engine.score(rendered.prefix, rendered.suffixes, rendered.labels)
    engine.reuse_buffer = False
    try:
        copied = engine.score(rendered.prefix, rendered.suffixes, rendered.labels)
    finally:
        engine.reuse_buffer = True
    assert reused.meta["rows_per_pass"] == copied.meta["rows_per_pass"]
    worst = max(
        (torch.tensor(a) - torch.tensor(b)).abs().max().item()
        for a, b in zip(reused.logits, copied.logits)
    )
    print(f"\n[option C] worst |logit| difference reuse-on vs reuse-off: {worst:.3e}")
    assert worst == 0.0, f"buffer reuse changed the logits by {worst:.3e}"
