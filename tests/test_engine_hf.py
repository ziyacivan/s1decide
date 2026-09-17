"""Contract tests for the transformers engine, on CPU, with a tiny random Qwen3.5-class model.

No weights are downloaded. The model is the real ``Qwen3_5ForCausalLM`` class with the real
Qwen3.8 tokenizer (from the local cache) and random weights, shrunk to 4 layers — 3
gated-deltanet + 1 attention, grouped V heads like the 27B. That is enough to exercise the
hybrid cache, which is where locked decision 2 can break. These tests skip if the tokenizer
is not cached; ``gpu``-marked tests against real weights are added once those are downloaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from s1decide.decide import decide
from s1decide.primitives import Choice, Noul, Question, Score
from s1decide.prompt import render

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from s1decide.engine.hf import CACHE_BATCH_TENSORS, HFEngine, expand_cache  # noqa: E402

pytestmark = pytest.mark.tokenizer


@pytest.fixture(scope="module")
def tiny_model(tokenizer):
    """A 4-layer hybrid model with the real vocabulary, fp32, on CPU, seeded."""
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    text_cfg = AutoConfig.from_pretrained("Qwen/Qwen3.8-27B", local_files_only=True).text_config
    small = type(text_cfg)(
        vocab_size=text_cfg.vocab_size,
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
        max_position_embeddings=4096,
        attn_output_gate=True,
        output_gate_type="swish",
        mamba_ssm_dtype="float32",
    )
    small._attn_implementation = "sdpa"
    torch.manual_seed(0)
    # flash-linear-attention's kernels are CUDA-only, and transformers wires them in at
    # construction — including the gated RMS norm, which is a module and cannot be swapped
    # afterwards. Installing fla broke every test in this file with "Pointer argument cannot be
    # accessed from Triton (cpu tensor?)" until the model was built as if it were absent.
    from s1decide.engine.qwen3_5_patch import without_fla_kernels

    with without_fla_kernels():
        return Qwen3_5ForCausalLM(small).eval()


@pytest.fixture(scope="module")
def engine(tiny_model, tokenizer) -> HFEngine:
    return HFEngine(tiny_model, tokenizer, max_rows_per_pass=16)


@pytest.fixture
def questions() -> list[Question]:
    return [
        Question("tone", Choice("What is the customer's tone?", ("calm", "frustrated", "angry"))),
        Question(
            "urgency", Score("How urgent is this ticket?", ("can wait", "this week", "today"))
        ),
        Question("billing", Noul("This ticket is about billing.")),
        Question("refund", Noul("The customer asks for a refund.")),
        Question("lang", Choice("Which language is the ticket in?", ("English", "Turkish"))),
    ]


def independent_logits(model, tokenizer, prefix: str, suffix: str, labels) -> torch.Tensor:
    """The reference: one un-broadcast forward over the full single-question prompt."""
    from s1decide.tokens import allowed_token_ids

    ids = tokenizer.encode(prefix, add_special_tokens=False) + tokenizer.encode(
        suffix, add_special_tokens=False
    )
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([ids])).logits[0, -1]
    return logits[list(allowed_token_ids(tokenizer, labels))].float()


# --- the test that matters: N-row broadcast == N independent runs -------------


def test_broadcast_equals_independent_runs(
    engine, tiny_model, tokenizer, ticket_state, questions
) -> None:
    rendered = render(ticket_state, questions)
    out = engine.score(rendered.prefix, rendered.suffixes, rendered.labels)
    assert out.meta["passes"] == 1, "five questions must fit one pass at max_rows_per_pass=16"
    for i, suffix in enumerate(rendered.suffixes):
        reference = independent_logits(
            tiny_model, tokenizer, rendered.prefix, suffix, rendered.labels[i]
        )
        got = torch.tensor(out.logits[i])
        assert torch.allclose(got, reference, atol=1e-4, rtol=1e-4), (
            f"question {rendered.names[i]}: broadcast {got.tolist()} vs independent {reference.tolist()}"
        )


def test_broadcast_expanded_both_kinds_of_cache_state(engine, ticket_state, questions) -> None:
    rendered = render(ticket_state, questions)
    expanded = engine.score(rendered.prefix, rendered.suffixes, rendered.labels).meta[
        "cache_expanded"
    ]
    assert set(expanded["LinearAttentionLayer"]) == {"conv_states", "recurrent_states"}
    assert set(expanded["DynamicLayer"]) == {"keys", "values"}


def test_chunked_passes_equal_a_single_pass(tiny_model, tokenizer, ticket_state, questions) -> None:
    rendered = render(ticket_state, questions)
    single = HFEngine(tiny_model, tokenizer, max_rows_per_pass=16).score(
        rendered.prefix, rendered.suffixes, rendered.labels
    )
    chunked = HFEngine(tiny_model, tokenizer, max_rows_per_pass=2).score(
        rendered.prefix, rendered.suffixes, rendered.labels
    )
    assert chunked.meta["passes"] == 3
    assert torch.allclose(
        torch.tensor(single.logits[0]), torch.tensor(chunked.logits[0]), atol=1e-5
    )
    for a, b in zip(single.logits, chunked.logits):
        assert torch.allclose(torch.tensor(a), torch.tensor(b), atol=1e-5)


def test_padding_does_not_leak_into_shorter_suffixes(
    engine, tiny_model, tokenizer, ticket_state
) -> None:
    """Right-padded short suffixes must score exactly as they do alone."""
    short = Question("n", Noul("Short."))
    long = Question(
        "c",
        Choice(
            "A much longer question with many more words in it?",
            tuple(f"opt{i}" for i in range(12)),
        ),
    )
    rendered = render(ticket_state, [short, long])
    out = engine.score(rendered.prefix, rendered.suffixes, rendered.labels)
    reference = independent_logits(
        tiny_model, tokenizer, rendered.prefix, rendered.suffixes[0], rendered.labels[0]
    )
    assert torch.allclose(torch.tensor(out.logits[0]), reference, atol=1e-4)


def test_decide_runs_end_to_end_on_the_hf_engine(engine, ticket_state, questions) -> None:
    response = decide(ticket_state, questions, engine=engine)
    assert response.engine == "hf"
    assert response["tone"].choice in ("calm", "frustrated", "angry")
    assert 0.0 <= response["billing"].noul <= 1.0
    assert response.meta["prefix_tokens"] > 0
    assert len(response.meta["suffix_tokens"]) == len(questions)


def test_score_output_meta_is_complete(engine, ticket_state, questions) -> None:
    rendered = render(ticket_state, questions)
    meta = engine.score(rendered.prefix, rendered.suffixes, rendered.labels).meta
    for key in (
        "prefix_tokens",
        "suffix_tokens",
        "rows",
        "passes",
        "cache_expanded",
        "prefill_seconds",
        "suffix_seconds",
    ):
        assert key in meta


def test_score_rejects_empty_and_mismatched_requests(engine) -> None:
    with pytest.raises(ValueError, match="at least one"):
        engine.score("p", [], [])
    with pytest.raises(ValueError, match="label sets"):
        engine.score("p", ["a", "b"], [("A", "B")])


def test_max_rows_per_pass_must_be_positive(tiny_model, tokenizer) -> None:
    with pytest.raises(ValueError, match="max_rows_per_pass"):
        HFEngine(tiny_model, tokenizer, max_rows_per_pass=0)


# --- expand_cache -------------------------------------------------------------


def prefill_cache(model, tokens: int = 7):
    with torch.inference_mode():
        return model(input_ids=torch.randint(0, 1000, (1, tokens)), use_cache=True).past_key_values


def test_expand_cache_broadcasts_every_layer(tiny_model) -> None:
    cache = prefill_cache(tiny_model)
    expanded = expand_cache(cache, 3)
    assert set(expanded) == {"LinearAttentionLayer", "DynamicLayer"}
    for layer in cache.layers:
        for attr in CACHE_BATCH_TENSORS:
            tensor = getattr(layer, attr, None)
            if torch.is_tensor(tensor):
                assert tensor.shape[0] == 3
                assert tensor.is_contiguous()


def test_expand_cache_makes_real_copies_not_views(tiny_model) -> None:
    """Rows must not alias: the model updates cache tensors in place on the next forward."""
    cache = prefill_cache(tiny_model)
    expand_cache(cache, 2)
    for layer in cache.layers:
        for attr in CACHE_BATCH_TENSORS:
            tensor = getattr(layer, attr, None)
            if torch.is_tensor(tensor):
                assert tensor.stride(0) != 0


def test_expand_cache_with_one_repeat_is_a_no_op_in_shape(tiny_model) -> None:
    cache = prefill_cache(tiny_model)
    before = [tuple(getattr(cache.layers[0], a).shape) for a in ("conv_states", "recurrent_states")]
    expand_cache(cache, 1)
    after = [tuple(getattr(cache.layers[0], a).shape) for a in ("conv_states", "recurrent_states")]
    assert before == after


def test_expand_cache_refuses_a_cache_that_is_not_batch_one(tiny_model) -> None:
    cache = prefill_cache(tiny_model)
    expand_cache(cache, 2)
    with pytest.raises(ValueError, match="expected 1"):
        expand_cache(cache, 2)


def test_expand_cache_refuses_unknown_layers() -> None:
    class Opaque:
        pass

    class FakeCache:
        layers: tuple[object, ...] = (Opaque(),)

    with pytest.raises(NotImplementedError, match="do not understand"):
        expand_cache(FakeCache(), 2)


def test_expand_cache_refuses_caches_without_layers() -> None:
    with pytest.raises(NotImplementedError, match=r"no `\.layers`"):
        expand_cache(object(), 2)


def test_expand_cache_rejects_zero_repeats(tiny_model) -> None:
    with pytest.raises(ValueError, match=">= 1"):
        expand_cache(prefill_cache(tiny_model), 0)


def test_kv_only_broadcast_is_not_silently_accepted(tiny_model) -> None:
    """The failure mode ADR 0002 warns about: expanding keys/values but not the GDN states.

    On transformers 5.5.0 the model itself rejects the shape mismatch. If a future version
    stops doing so, this test starts failing, and expand_cache's equality test above is the
    remaining line of defence.
    """
    cache = prefill_cache(tiny_model)
    for layer in cache.layers:
        for attr in ("keys", "values"):
            tensor = getattr(layer, attr, None)
            if torch.is_tensor(tensor):
                setattr(layer, attr, tensor.expand(2, *tensor.shape[1:]).contiguous())
    with pytest.raises(RuntimeError), torch.inference_mode():
        tiny_model(input_ids=torch.randint(0, 1000, (2, 3)), past_key_values=cache, use_cache=True)


# --- hard rules ---------------------------------------------------------------


def test_engine_sources_never_mention_flash_attn() -> None:
    engine_dir = Path(__file__).resolve().parents[1] / "src" / "s1decide" / "engine"
    for path in engine_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import flash_attn" not in text and "flash_attention_2" not in text, path.name


def test_hf_engine_uses_sdpa(tiny_model) -> None:
    assert tiny_model.config._attn_implementation == "sdpa"


# --- the transformers 5.5.0 continuation bug and our patch -----------------------


def fresh_tiny_model(tokenizer):
    """An unpatched copy of the tiny model (the module fixture is patched by the engine)."""
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    text_cfg = AutoConfig.from_pretrained("Qwen/Qwen3.8-27B", local_files_only=True).text_config
    small = type(text_cfg)(
        vocab_size=1000,
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
        max_position_embeddings=4096,
    )
    torch.manual_seed(0)
    # CUDA-only fla kernels on a CPU model — see the fixture above.
    from s1decide.engine.qwen3_5_patch import without_fla_kernels

    with without_fla_kernels():
        return Qwen3_5ForCausalLM(small).eval()


def cached_vs_joint(model, prefix_len: int, suffix_len: int) -> float:
    """max |logit diff| between a cached continuation and one joint forward, last position."""
    import copy

    torch.manual_seed(1)
    prefix = torch.randint(0, 1000, (1, prefix_len))
    suffix = torch.randint(0, 1000, (1, suffix_len))
    with torch.inference_mode():
        joint = model(torch.cat([prefix, suffix], dim=1)).logits[0, -1]
        cache = copy.deepcopy(model(prefix, use_cache=True).past_key_values)
        cached = model(suffix, past_key_values=cache, use_cache=True).logits[0, -1]
    return (joint - cached).abs().max().item()


def test_unpatched_transformers_ignores_the_gdn_state_on_multi_token_continuation(
    tokenizer,
) -> None:
    """Documents the bug we patch around. If this starts failing, transformers fixed it upstream."""
    import transformers

    from s1decide.engine.qwen3_5_patch import is_patched

    model = fresh_tiny_model(tokenizer)
    assert not any(is_patched(m) for m in model.modules())
    diff = cached_vs_joint(model, prefix_len=100, suffix_len=5)
    assert diff > 1e-5, (
        f"transformers {transformers.__version__} now continues GDN state correctly (diff {diff:.2e}); "
        "re-evaluate whether engine/qwen3_5_patch.py is still needed"
    )


@pytest.mark.parametrize("suffix_len", [2, 3, 5, 20, 64, 65])
def test_patched_continuation_equals_joint_forward(tokenizer, suffix_len: int) -> None:
    """Covers suffixes shorter than the conv kernel (4), at the chunk size (64) and past it."""
    from s1decide.engine.qwen3_5_patch import patch_gated_deltanet

    model = fresh_tiny_model(tokenizer)
    assert patch_gated_deltanet(model) == 3
    assert cached_vs_joint(model, prefix_len=100, suffix_len=suffix_len) < 1e-5


@pytest.mark.parametrize("prefix_len", [1, 3, 4, 63, 64, 65, 130])
def test_patched_continuation_is_robust_to_prefix_length(tokenizer, prefix_len: int) -> None:
    from s1decide.engine.qwen3_5_patch import patch_gated_deltanet

    model = fresh_tiny_model(tokenizer)
    patch_gated_deltanet(model)
    assert cached_vs_joint(model, prefix_len=prefix_len, suffix_len=5) < 1e-5


def test_patch_leaves_the_one_token_decode_path_alone(tokenizer) -> None:
    from s1decide.engine.qwen3_5_patch import patch_gated_deltanet

    model = fresh_tiny_model(tokenizer)
    patch_gated_deltanet(model)
    assert cached_vs_joint(model, prefix_len=50, suffix_len=1) < 1e-5


def test_patch_is_idempotent_and_counts(tokenizer) -> None:
    from s1decide.engine.qwen3_5_patch import is_patched, patch_gated_deltanet

    model = fresh_tiny_model(tokenizer)
    assert patch_gated_deltanet(model) == 3
    assert patch_gated_deltanet(model) == 0
    assert sum(is_patched(m) for m in model.modules()) == 3


def test_patch_ignores_models_without_gdn_layers() -> None:
    from s1decide.engine.qwen3_5_patch import patch_gated_deltanet

    assert patch_gated_deltanet(torch.nn.Linear(2, 2)) == 0


def test_engine_reports_the_patch_and_can_disable_it(tokenizer) -> None:
    model = fresh_tiny_model(tokenizer)
    off = HFEngine(model, tokenizer, patch_gdn=False)
    assert off.gdn_patched == 0
    on = HFEngine(model, tokenizer)
    assert on.gdn_patched == 3
    assert HFEngine(model, tokenizer).gdn_patched == 0  # already patched


# --- BroadcastCache: buffer reuse across passes (ADR 0003 option C) ----------


def test_broadcast_cache_expands_every_layer(tiny_model) -> None:
    from s1decide.engine.hf import BroadcastCache

    buffer = BroadcastCache(prefill_cache(tiny_model), 3)
    assert set(buffer.expanded) == {"LinearAttentionLayer", "DynamicLayer"}
    assert buffer.rows == 3
    assert buffer.bytes_held > 0
    for layer in buffer.cache.layers:
        for attr in CACHE_BATCH_TENSORS:
            tensor = getattr(layer, attr, None)
            if torch.is_tensor(tensor):
                assert tensor.shape[0] == 3


def test_reset_restores_the_prefix_state_after_the_model_mutates_it(tiny_model) -> None:
    """The model updates GDN states in place and rebinds attention keys; reset must undo both."""
    from s1decide.engine.hf import BroadcastCache

    prefix = torch.randint(0, 1000, (1, 9))
    with torch.inference_mode():
        prefix_cache = tiny_model(input_ids=prefix, use_cache=True).past_key_values
    buffer = BroadcastCache(prefix_cache, 2)

    before = {
        (i, a): getattr(layer, a).clone()
        for i, layer in enumerate(buffer.cache.layers)
        for a in CACHE_BATCH_TENSORS
        if torch.is_tensor(getattr(layer, a, None))
    }
    with torch.inference_mode():
        tiny_model(
            input_ids=torch.randint(0, 1000, (2, 4)), past_key_values=buffer.reset(), use_cache=True
        )

    # After a pass the cache has moved on: GDN states changed, attention keys grew.
    grew = any(
        getattr(buffer.cache.layers[i], a).shape != before[(i, a)].shape
        or not torch.equal(getattr(buffer.cache.layers[i], a), before[(i, a)])
        for (i, a) in before
    )
    assert grew, "the model did not mutate the cache; this test would prove nothing"

    buffer.reset()
    for (i, a), original in before.items():
        restored = getattr(buffer.cache.layers[i], a)
        assert restored.shape == original.shape, (i, a)
        assert torch.equal(restored, original), (i, a)


def test_scoring_the_same_chunk_twice_gives_identical_logits(
    engine, ticket_state, questions
) -> None:
    """The real risk of buffer reuse: pass 2 inheriting pass 1's state."""
    rendered = render(ticket_state, questions)
    first = engine.score(rendered.prefix, rendered.suffixes, rendered.labels)
    second = engine.score(rendered.prefix, rendered.suffixes, rendered.labels)
    for a, b in zip(first.logits, second.logits):
        assert torch.allclose(torch.tensor(a), torch.tensor(b), atol=1e-6)


def test_many_short_passes_still_match_one_wide_pass(
    tiny_model, tokenizer, ticket_state, questions
) -> None:
    """Every row count must agree, including the padded final chunk."""
    rendered = render(ticket_state, questions)
    reference = HFEngine(tiny_model, tokenizer, max_rows_per_pass=16).score(
        rendered.prefix, rendered.suffixes, rendered.labels
    )
    for rows_per_pass in (1, 2, 3, 4, 5):
        chunked = HFEngine(tiny_model, tokenizer, max_rows_per_pass=rows_per_pass).score(
            rendered.prefix, rendered.suffixes, rendered.labels
        )
        assert len(chunked.logits) == len(questions)
        for a, b in zip(reference.logits, chunked.logits):
            assert torch.allclose(torch.tensor(a), torch.tensor(b), atol=1e-4), rows_per_pass


def test_broadcast_cache_rejects_a_bad_row_count(tiny_model) -> None:
    from s1decide.engine.hf import BroadcastCache

    with pytest.raises(ValueError, match="rows must be >= 1"):
        BroadcastCache(prefill_cache(tiny_model), 0)


def test_broadcast_cache_refuses_an_unknown_cache() -> None:
    from s1decide.engine.hf import BroadcastCache

    with pytest.raises(NotImplementedError, match=r"no `\.layers`"):
        BroadcastCache(object(), 2)


def test_the_caller_prefix_cache_is_not_mutated_by_scoring(
    tiny_model, tokenizer, ticket_state, questions
) -> None:
    """BroadcastCache keeps its own pristine copy; the prefill it was built from stays clean."""
    from s1decide.engine.hf import BroadcastCache

    with torch.inference_mode():
        prefix_cache = tiny_model(
            input_ids=torch.randint(0, 1000, (1, 11)), use_cache=True
        ).past_key_values
    snapshot = {
        (i, a): getattr(layer, a).clone()
        for i, layer in enumerate(prefix_cache.layers)
        for a in CACHE_BATCH_TENSORS
        if torch.is_tensor(getattr(layer, a, None))
    }
    buffer = BroadcastCache(prefix_cache, 2)
    with torch.inference_mode():
        tiny_model(
            input_ids=torch.randint(0, 1000, (2, 3)), past_key_values=buffer.reset(), use_cache=True
        )
    for (i, a), original in snapshot.items():
        assert torch.equal(getattr(prefix_cache.layers[i], a), original), (i, a)


def test_reuse_buffer_flag_reaches_the_engine(tiny_model, tokenizer) -> None:
    """A/B flags that silently default are worse than no flag: this one is load-bearing."""
    assert HFEngine(tiny_model, tokenizer).reuse_buffer is False, (
        "reuse is off by default: it costs more rows per pass than it saves time (ADR 0003)"
    )
    assert HFEngine(tiny_model, tokenizer, reuse_buffer=True).reuse_buffer is True


def test_both_cache_paths_give_the_same_logits(
    tiny_model, tokenizer, ticket_state, questions
) -> None:
    """Buffer reuse is an optimisation; it must not change a single number."""
    rendered = render(ticket_state, questions)
    reused = HFEngine(tiny_model, tokenizer, max_rows_per_pass=2, reuse_buffer=True).score(
        rendered.prefix, rendered.suffixes, rendered.labels
    )
    copied = HFEngine(tiny_model, tokenizer, max_rows_per_pass=2, reuse_buffer=False).score(
        rendered.prefix, rendered.suffixes, rendered.labels
    )
    assert reused.meta["reuse_buffer"] is True
    assert copied.meta["reuse_buffer"] is False
    for a, b in zip(reused.logits, copied.logits):
        assert torch.allclose(torch.tensor(a), torch.tensor(b), atol=1e-5)


def test_pinned_rows_per_pass_overrides_the_budget(
    tiny_model, tokenizer, ticket_state, questions
) -> None:
    """Pinning is how "does N rows fit?" gets answered by measurement rather than by estimate."""
    rendered = render(ticket_state, questions)
    out = HFEngine(tiny_model, tokenizer, max_rows_per_pass=16, rows_per_pass=2).score(
        rendered.prefix, rendered.suffixes, rendered.labels
    )
    assert out.meta["rows_per_pass"] == 2
    assert out.meta["passes"] == 3
    reference = HFEngine(tiny_model, tokenizer, max_rows_per_pass=16).score(
        rendered.prefix, rendered.suffixes, rendered.labels
    )
    for a, b in zip(out.logits, reference.logits):
        assert torch.allclose(torch.tensor(a), torch.tensor(b), atol=1e-5)


def test_pinned_rows_per_pass_must_be_positive(tiny_model, tokenizer) -> None:
    with pytest.raises(ValueError, match="rows_per_pass"):
        HFEngine(tiny_model, tokenizer, rows_per_pass=0)
