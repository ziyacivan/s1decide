"""The adapter a run trains must be the adapter its config and ADR 0002 describe.

The second 27B attempt fell through to a path that adapted q/k/v/o in 16 of 64 layers and no MLP,
and reported success. These tests pin the layout check that now refuses that before step one,
and the one-pass step rule that makes a smoke run's coverage table true by construction.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from train.sft_lora import (
    ATTENTION_TARGETS,
    LORA_TARGETS,
    MLP_TARGETS,
    TrainConfig,
    check_lora_layout,
    expected_lora_layout,
    plan_steps,
)


def hybrid_config(layers: int = 8, every: int = 4) -> SimpleNamespace:
    """A Qwen3.5/3.8-shaped text config: full attention every `every`-th layer, GDN otherwise."""
    kinds = [
        "full_attention" if (i + 1) % every == 0 else "linear_attention" for i in range(layers)
    ]
    return SimpleNamespace(num_hidden_layers=layers, layer_types=kinds)


def peft_names(layout: set[tuple[int, str]]) -> list[str]:
    block = {p: "self_attn" for p in ATTENTION_TARGETS} | {p: "mlp" for p in MLP_TARGETS}
    return [f"base_model.model.model.layers.{i}.{block[p]}.{p}" for i, p in sorted(layout)]


def test_the_targets_are_adr_0002s_attention_and_mlp_and_nothing_else() -> None:
    assert set(LORA_TARGETS) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }


def test_mlp_on_every_layer_attention_on_every_attention_layer() -> None:
    layout = expected_lora_layout(hybrid_config(8, 4))
    assert {i for i, p in layout if p in MLP_TARGETS} == set(range(8))
    assert {i for i, p in layout if p in ATTENTION_TARGETS} == {3, 7}
    assert len(layout) == 8 * 3 + 2 * 4


def test_a_complete_adapter_passes_as_module_names_and_as_file_keys() -> None:
    config = hybrid_config()
    layout = expected_lora_layout(config)
    report = check_lora_layout(peft_names(layout), config)
    assert report == {"expected": 32, "adapted": 32, "attention_layers": 2, "mlp_layers": 8}
    keys = [f"{n}.lora_{ab}.weight" for n in peft_names(layout) for ab in "AB"]
    assert check_lora_layout(keys, config)["adapted"] == 32


def test_the_second_27b_attempts_adapter_is_refused() -> None:
    """q/k/v/o on the attention layers, no MLP: what the fallback trained on 2026-09-23."""
    config = hybrid_config(64, 4)
    attention_only = {(i, p) for i in range(3, 64, 4) for p in ATTENTION_TARGETS}
    with pytest.raises(RuntimeError, match=r"64/256 expected modules.*gate_proj.: 64"):
        check_lora_layout(peft_names(attention_only), config)


@pytest.mark.parametrize(
    "stray",
    [
        "base_model.model.model.layers.0.linear_attn.in_proj_qkv",
        "base_model.model.mtp.layers.0.mlp.gate_proj",
        "base_model.model.model.visual.blocks.0.mlp.gate_proj",
    ],
)
def test_anything_outside_the_layout_is_refused(stray: str) -> None:
    """GDN projections are a separate decision; MTP and vision modules are not the text model."""
    config = hybrid_config()
    names = [*peft_names(expected_lora_layout(config)), stray]
    with pytest.raises(RuntimeError, match="unexpected"):
        check_lora_layout(names, config)


def test_the_real_27b_config_expects_256_modules() -> None:
    transformers = pytest.importorskip("transformers")
    try:
        config = transformers.AutoConfig.from_pretrained(
            "unsloth/Qwen3.8-27B-unsloth-bnb-4bit", local_files_only=True
        )
    except Exception:  # pragma: no cover - the model may not be cached
        pytest.skip("27B config not cached locally")
    layout = expected_lora_layout(config)
    assert len(layout) == 16 * 4 + 64 * 3


@pytest.mark.parametrize(
    ("rows", "batch", "accum", "steps"),
    [(200, 1, 1, 200), (200, 2, 1, 100), (201, 2, 1, 101), (200, 4, 2, 25), (7, 4, 1, 2)],
)
def test_one_pass_is_ceil_of_rows_over_the_effective_batch(rows, batch, accum, steps) -> None:
    assert plan_steps(rows, TrainConfig(batch_size=batch, grad_accum=accum)) == steps


def test_an_explicit_step_count_is_kept() -> None:
    assert plan_steps(200, TrainConfig(max_steps=50)) == 50


def test_one_pass_is_the_default() -> None:
    assert TrainConfig().max_steps is None


def test_the_reread_floor_matches_loss_parts() -> None:
    """eval/smoke_floor.py recomputes floors by hand; they must equal the trainer's own."""
    torch = pytest.importorskip("torch")
    from eval.smoke_floor import row_floor
    from train.ordinal_loss import loss_parts

    for target in ([0.0, 0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.5, 0.5, 0, 0, 0]):
        parts = loss_parts(torch.zeros(1, 5), torch.tensor([target]), lambda_distance=0.3)
        assert row_floor(target, "score", 0.3) == pytest.approx(float(parts["floor"]))
    assert row_floor([0.0, 1.0], "noul", 0.3) == pytest.approx(0.0)


def test_a_config_key_the_trainer_does_not_implement_is_refused(tmp_path) -> None:
    from train.sft_lora import train

    with pytest.raises(RuntimeError, match="sample_budget"):
        train(TrainConfig(extra={"sample_budget": 30000}), root=tmp_path)


def test_longest_selection_takes_the_longest_rows_first() -> None:
    from train.sft_lora import longest_examples

    examples = [{"input_ids": [0] * n, "id": n} for n in (5, 50, 20, 50, 1)]
    picked = longest_examples(examples, 3)
    assert [len(e["input_ids"]) for e in picked] == [50, 50, 20]


def test_an_unknown_selection_is_refused(tmp_path) -> None:
    from train.sft_lora import train

    with pytest.raises(ValueError, match="random-ish"):
        train(TrainConfig(selection="random-ish"), root=tmp_path)


def test_lr_warms_up_linearly_then_decays_by_cosine_to_zero() -> None:
    from train.sft_lora import lr_multiplier

    total, warmup = 100, 3
    assert [round(lr_multiplier(s, total, warmup, "cosine"), 3) for s in range(3)] == [
        0.333,
        0.667,
        1.0,
    ]
    assert lr_multiplier(3, total, warmup, "cosine") == pytest.approx(1.0)
    middle = warmup + (total - warmup) // 2
    assert lr_multiplier(middle, total, warmup, "cosine") == pytest.approx(0.5, abs=0.02)
    assert lr_multiplier(total, total, warmup, "cosine") == pytest.approx(0.0, abs=1e-9)
    assert lr_multiplier(50, total, 0, "constant") == 1.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"sampling": "roughly"}, "unknown sampling"),
        ({"sampling": "family_balanced"}, "sample_budget"),
        ({"eval_mode": "full"}, "case-control"),
        ({"lr_schedule": "linear-ish"}, "lr_schedule"),
    ],
)
def test_a_config_the_trainer_cannot_honour_is_refused_before_any_io(tmp_path, overrides, message):
    from train.sft_lora import train

    with pytest.raises(ValueError, match=message):
        train(TrainConfig(**overrides), root=tmp_path)


def test_a_checkpoint_carries_optimiser_schedule_and_position(tmp_path) -> None:
    """A resume without these restarts Adam's moments and the LR schedule: a different run."""
    torch = pytest.importorskip("torch")
    from train.sft_lora import lr_multiplier, save_training_state

    param = torch.nn.Parameter(torch.ones(3))
    optimiser = torch.optim.AdamW([param], lr=2e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda s: lr_multiplier(s, 100, 3, "cosine")
    )
    for _ in range(5):
        param.sum().backward()
        optimiser.step()
        scheduler.step()
        optimiser.zero_grad()

    path = save_training_state(tmp_path / "rows-000040", optimiser, scheduler, 40, TrainConfig())
    state = torch.load(path, weights_only=False)
    assert state["rows_done"] == 40
    assert state["optimizer_steps_done"] == 5
    assert "exp_avg" in state["optimizer"]["state"][0]
    assert state["config"]["learning_rate"] == TrainConfig().learning_rate

    fresh = torch.optim.AdamW([torch.nn.Parameter(torch.ones(3))], lr=2e-5)
    fresh.load_state_dict(state["optimizer"])  # loads back into a new optimiser


def test_a_resume_may_change_the_eval_batch_and_nothing_else() -> None:
    from dataclasses import asdict

    from train.sft_lora import check_resume_config

    saved = asdict(TrainConfig(learning_rate=2e-5, eval_batch_size=8))
    check_resume_config(saved, TrainConfig(learning_rate=2e-5, eval_batch_size=4, resume_from="x"))
    with pytest.raises(ValueError, match="learning_rate"):
        check_resume_config(saved, TrainConfig(learning_rate=1e-4, eval_batch_size=8))


def test_a_resume_restores_adapter_optimiser_and_schedule(tmp_path) -> None:
    """Round trip on CPU with a tiny PEFT model: what a checkpoint saves is what a resume loads."""
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    from train.sft_lora import lr_multiplier, resume_training_state, save_training_state

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.q_proj(x)

    def build(seed: int):
        torch.manual_seed(seed)
        model = peft.get_peft_model(Tiny(), peft.LoraConfig(r=2, target_modules=["q_proj"]))
        params = [p for p in model.parameters() if p.requires_grad]
        optimiser = torch.optim.AdamW(params, lr=2e-5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimiser, lambda s: lr_multiplier(s, 100, 3, "cosine")
        )
        return model, optimiser, scheduler

    config = TrainConfig(learning_rate=2e-5)
    model, optimiser, scheduler = build(0)
    for _ in range(4):
        model(torch.ones(1, 4)).sum().backward()
        optimiser.step()
        scheduler.step()
        optimiser.zero_grad()
    checkpoint = tmp_path / "rows-000032"
    model.save_pretrained(str(checkpoint))
    save_training_state(checkpoint, optimiser, scheduler, 32, config)

    fresh, fresh_opt, fresh_sched = build(1)  # different init: the resume must overwrite it
    step, rows = resume_training_state(checkpoint, fresh, fresh_opt, fresh_sched, config)
    assert (step, rows) == (4, 32)
    trained = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    restored = {k: v for k, v in fresh.state_dict().items() if "lora_" in k}
    assert trained.keys() == restored.keys()
    assert all(torch.equal(trained[k], restored[k]) for k in trained)
    assert fresh_sched.last_epoch == 4
    assert fresh_sched.get_last_lr() == pytest.approx(scheduler.get_last_lr())
    first = next(iter(fresh_opt.state.values()))
    assert int(first["step"]) == 4
