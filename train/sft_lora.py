"""Stage-1 QLoRA SFT on the option token — the training half of the masked-logit path.

The model we publish answers by having its logits read at one position and masked to the allowed
options. Training has to optimise **that**, not next-token likelihood over a rendered answer: a
model trained to emit the string "B" and a model whose logit at the answer position ranks `B`
highest are different objectives, and only the second is what `decide()` reads.

So the loss is cross-entropy over the allowed option tokens at the answer position, and for
`Score` it is ADR 0007's ordinal loss over the same positions, accepting the soft targets the
teacher fold produced.

The prompts come from `s1decide.prompt.render` — the same function the engine calls. That is not
tidiness: a training prompt that differs from an inference prompt by one token trains the model
for a question it will never be asked, and the difference is invisible unless something asserts
it. `tests/test_train_render.py` asserts it byte for byte.

Two stages by design, per ADR 0002 and the smoke rule in CLAUDE.md: a small same-family model
proves the pipeline in minutes, and only then does a 27B get GPU hours.
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ATTENTION_TARGETS",
    "LORA_TARGETS",
    "MLP_TARGETS",
    "SMALL_MODEL",
    "TrainConfig",
    "adapted_module_names",
    "adapter_file_keys",
    "build_examples",
    "check_lora_layout",
    "check_resume_config",
    "collate",
    "coverage",
    "expected_lora_layout",
    "format_coverage",
    "is_prequantized",
    "load_config",
    "longest_examples",
    "lr_multiplier",
    "main",
    "plan_steps",
    "resume_training_state",
    "save_training_state",
    "select_rows",
    "train",
    "write_report",
]

#: The pipeline-proving model: same family and tokenizer conventions as the 27B, small enough
#: that a wrong shape or a mis-rendered prompt shows up in minutes instead of hours.
SMALL_MODEL = "Qwen/Qwen3.5-0.8B"


@dataclass
class TrainConfig:
    """One training run, as a config file describes it.

    Attributes:
        hardware: ``rtx3090_windows`` or ``h100_linux``. The runner refuses a config built for
            the other machine rather than discovering the difference at hour three.
        model: Hub id or local path.
        run_id: Output directory under ``results/``.
        max_steps: Optimiser steps. ``None`` (the default, and what the smoke configs use)
            means exactly one pass: ``ceil(rows / (batch_size * grad_accum))`` steps, every
            selected row seen once, so the coverage table describes what was trained on by
            construction. A number wraps around the selection as often as it needs to.
        rank: LoRA rank.
        lora_alpha: LoRA alpha.
        learning_rate: Peak learning rate.
        batch_size: Sequences per device step.
        grad_accum: Gradient accumulation steps.
        max_seq_len: Tokens per example, prompt included; longer examples are skipped, not
            truncated, because truncating a prompt moves the answer position.
        limit: Cap on training examples, for the smoke run.
        lambda_distance: ADR 0007's distance weight for `Score` rows.
        mu_unimodality: ADR 0007's shape penalty weight.
        load_in_4bit: QLoRA. False for the small pipeline model, which fits in bf16.
        seed: Everything sampled is sampled from this.
        selection: ``stratified`` (default) draws round-robin across primitives for code
            coverage; ``longest`` takes the ``limit`` longest rows of the corpus as rendered —
            the worst case for activation memory, used to measure the envelope.
        sampling: ``stratified`` (smoke) or ``family_balanced`` (S1): draw ``sample_budget``
            rows with replacement under ``sampling_weights.json``, `Score` uniform by row.
        sample_budget: Rows drawn for a ``family_balanced`` run.
        warmup_fraction: Share of optimiser steps spent in linear warmup.
        lr_schedule: ``constant`` or ``cosine`` after warmup.
        checkpoint_every_rows: Save the adapter every this many rows.
        eval_every_rows: Evaluate the fixed `val` slice every this many rows, and at row 0.
        eval_mode: Only ``case-control`` is implemented for periodic evaluation.
        eval_stage1_questions: Stage-1 questions in the slice.
        eval_negatives: Negatives kept per stage-1 question, weighted back.
        eval_batch_size: Rows per forward pass during evaluation.
        resume_from: A checkpoint directory (``results/<run>/checkpoints/rows-NNNNNN``) to continue
            from: adapter, optimiser, schedule and position. Every other field must match the
            checkpoint's config except those in :data:`RESUME_MAY_CHANGE`.
    """

    hardware: str = "rtx3090_windows"
    model: str = SMALL_MODEL
    run_id: str = "smoke"
    max_steps: int | None = None
    rank: int = 8
    lora_alpha: int = 16
    learning_rate: float = 2e-4
    batch_size: int = 1
    grad_accum: int = 1
    max_seq_len: int = 1024
    limit: int | None = 200
    lambda_distance: float = 0.3
    mu_unimodality: float = 0.0
    load_in_4bit: bool = False
    seed: int = 20260923
    selection: str = "stratified"
    sampling: str = "stratified"
    sample_budget: int | None = None
    warmup_fraction: float = 0.0
    lr_schedule: str = "constant"
    checkpoint_every_rows: int | None = None
    eval_every_rows: int | None = None
    eval_mode: str = "case-control"
    eval_stage1_questions: int = 300
    eval_negatives: int = 16
    eval_batch_size: int = 8
    resume_from: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> TrainConfig:
    """Read a YAML config and refuse one built for another machine.

    Args:
        path: Config file.

    Returns:
        The config.

    Raises:
        RuntimeError: If the config's ``hardware`` is not this machine's. A 27B bf16 run started
            by accident on a 24 GiB card fails eventually; failing here costs a second.
    """
    import yaml

    from s1decide.hardware import detect_profile

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = {f for f in TrainConfig.__dataclass_fields__ if f != "extra"}
    config = TrainConfig(
        **{k: v for k, v in raw.items() if k in known},
        extra={k: v for k, v in raw.items() if k not in known},
    )
    here = detect_profile().name
    if config.hardware != here:
        raise RuntimeError(
            f"config is for {config.hardware!r} and this machine is {here!r}; "
            "run it where it belongs or change the config deliberately"
        )
    return config


def select_rows(rows: Sequence[dict[str, Any]], config: TrainConfig) -> list[dict[str, Any]]:
    """Take a stratified sample across primitives, soft-target rows included.

    The first version of this took the head of the file, and the file is ordered by family. The
    smoke run therefore trained on 200 banking77 rows, 159 of which had the same answer and none
    of which was a `Score` — so the ordinal loss, the soft targets and every primitive but two
    went entirely unexercised while the loss fell to 4e-7 and looked like success.

    A smoke run's job is to fail when the pipeline is wrong. It cannot do that for code paths it
    never enters, so the sample is drawn round-robin across `qtype` and, within `score`, across
    hard and soft targets.

    Args:
        rows: The whole training split.
        config: ``limit`` and ``seed``.

    Returns:
        Rows, shuffled, covering every primitive present in the corpus.
    """
    if not config.limit:
        return list(rows)

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = row["qtype"]
        if key == "score":
            key = f"score/{row.get('target_type', 'hard')}"
        elif row.get("stage") == 1:
            key = "noul/stage1"
        groups.setdefault(key, []).append(row)

    rng = random.Random(config.seed)
    for group in groups.values():
        rng.shuffle(group)

    # Four times the limit so that length filtering and the option-count ceiling have something
    # to discard without emptying a group.
    wanted = config.limit * 4
    picked: list[dict[str, Any]] = []
    order = sorted(groups)
    cursors = dict.fromkeys(order, 0)
    while len(picked) < wanted and any(cursors[k] < len(groups[k]) for k in order):
        for key in order:
            if cursors[key] < len(groups[key]) and len(picked) < wanted:
                picked.append(groups[key][cursors[key]])
                cursors[key] += 1
    rng.shuffle(picked)
    return picked


def build_examples(
    rows: Sequence[dict[str, Any]],
    tokenizer: Any,
    config: TrainConfig,
    stats: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Render every row the way inference will, and attach its option tokens and target.

    Args:
        rows: Corpus rows.
        tokenizer: The model's tokenizer.
        config: Sequence cap and sampling seed.
        stats: If given, ``stats["dropped_over_cap"]`` counts rows longer than the cap. They are
            dropped, never truncated, and a drop is reported, never silent.

    Returns:
        Examples with ``input_ids``, ``label_token_ids``, ``target`` and ``qtype``.
    """
    from s1decide.primitives import Choice, Noul, Question, Score
    from s1decide.prompt import render
    from s1decide.tokens import allowed_token_ids

    examples: list[dict[str, Any]] = []
    for row in rows:
        options = list(row["options"])
        if row["qtype"] == "noul":
            spec: Any = Noul(instructions=row["instructions"])
        elif row["qtype"] == "score":
            spec = Score(instructions=row["instructions"], levels=tuple(options))
        else:
            spec = Choice(instructions=row["instructions"], options=tuple(options))
        rendered = render(row["state"], [Question(name="q", spec=spec)])
        prompt = rendered.prefix + rendered.suffixes[0]
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(input_ids) > config.max_seq_len:
            # Dropped, never truncated: cutting a prompt moves the answer position, and the
            # model would be trained to answer at a place it is never read at.
            if stats is not None:
                stats["dropped_over_cap"] = stats.get("dropped_over_cap", 0) + 1
            continue
        labels = list(rendered.labels[0])
        target = row.get("target")
        if not isinstance(target, list) or len(target) != len(labels):
            target = [0.0] * len(labels)
            target[int(row["answer_idx"])] = 1.0
        examples.append(
            {
                "input_ids": input_ids,
                "label_token_ids": list(allowed_token_ids(tokenizer, labels)),
                "target": [float(v) for v in target],
                "qtype": row["qtype"],
                "family": row.get("family", "?"),
                "stage1": row.get("stage") == 1,
                "soft": row["qtype"] == "score" and row.get("target_type") == "soft",
                "n_options": len(labels),
                "answer_idx": int(row["answer_idx"]),
                "eval_weight": float(row.get("eval_weight", 1.0)),
                "row_index": row.get("row_index"),
            }
        )
    random.Random(config.seed).shuffle(examples)
    return examples


def longest_examples(examples: Sequence[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """The ``limit`` longest examples, longest first; ties keep their original order.

    Args:
        examples: Rendered examples (``input_ids`` set).
        limit: How many to keep; ``None`` keeps all, sorted.

    Returns:
        The selection, sorted by descending token count.
    """
    ranked = sorted(examples, key=lambda ex: -len(ex["input_ids"]))
    return ranked[:limit] if limit else ranked


def coverage(examples: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Count what a run actually trained on: primitive, family, target type, option count.

    A falling loss says nothing about which code paths produced it. The first smoke run fell to
    4e-7 on 200 banking77 rows and no `Score` row at all; this table is what makes that visible
    in the report instead of only in hindsight.

    Args:
        examples: Output of :func:`build_examples`, after any ``limit``.

    Returns:
        ``by_qtype`` (``noul`` split into genuine and ``noul/stage1``), ``by_family``,
        ``by_target`` (hard/soft) and ``by_option_count``, each sorted by key.
    """
    tables: dict[str, dict[str, int]] = {
        "by_qtype": {},
        "by_family": {},
        "by_target": {},
        "by_option_count": {},
    }
    for ex in examples:
        keys = {
            "by_qtype": "noul/stage1" if ex.get("stage1") else ex["qtype"],
            "by_family": ex.get("family", "?"),
            "by_target": "soft" if ex.get("soft") else "hard",
            "by_option_count": str(ex.get("n_options", len(ex["label_token_ids"]))),
        }
        for table, key in keys.items():
            tables[table][key] = tables[table].get(key, 0) + 1
    tables["by_option_count"] = dict(
        sorted(tables["by_option_count"].items(), key=lambda kv: int(kv[0]))
    )
    return {
        name: dict(sorted(t.items())) if name != "by_option_count" else t
        for name, t in tables.items()
    }


def format_coverage(cov: dict[str, dict[str, int]]) -> str:
    """Render :func:`coverage` as Markdown tables, one per dimension, for run notes."""
    lines: list[str] = []
    for name, table in cov.items():
        total = sum(table.values()) or 1
        lines += [f"| {name.removeprefix('by_')} | rows | share |", "|---|---|---|"]
        lines += [f"| {k} | {v} | {v / total:.1%} |" for k, v in table.items()]
        lines.append("")
    return "\n".join(lines)


def collate(batch: Sequence[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    """Left-pad a batch so every answer position is the last column.

    Left padding rather than right: the answer is read at the final position, and with left
    padding that is the same index for every row regardless of length. Right padding would need
    a per-row gather and one off-by-one away from reading a pad token's logits.
    """
    import torch

    width = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, item in enumerate(batch):
        ids = item["input_ids"]
        input_ids[i, width - len(ids) :] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, width - len(ids) :] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_token_ids": [item["label_token_ids"] for item in batch],
        "target": [item["target"] for item in batch],
        "qtype": [item["qtype"] for item in batch],
    }


PART_KEYS: tuple[str, ...] = (
    "total",
    "kl",
    "cross_entropy",
    "distance",
    "distance_excess",
    "floor",
)


def _batch_loss(
    model: Any, batch: dict[str, Any], config: TrainConfig
) -> tuple[Any, list[dict[str, float]]]:
    """Masked-option loss at the answer position, ordinal where the row is a `Score`.

    Returns the batch mean for the backward pass and, per row, the loss broken into the parts
    :func:`train.ordinal_loss.loss_parts` defines. The parts are what the reports read: a soft
    target's loss has a floor of ``ln 2`` plus the distance term's own floor, so the raw total
    over `Score` rows moves with the hard/soft mix, while ``kl`` is zero at the optimum on
    every row.
    """
    import torch

    from train.ordinal_loss import loss_parts

    device = next(model.parameters()).device
    out = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
    )
    answer_logits = out.logits[:, -1, :].float()

    total = answer_logits.new_zeros(())
    per_row: list[dict[str, float]] = []
    for i, (ids, target, qtype) in enumerate(
        zip(batch["label_token_ids"], batch["target"], batch["qtype"], strict=True)
    ):
        index = torch.tensor(ids, device=device)
        row = answer_logits[i, index].unsqueeze(0)
        distribution = torch.tensor([target], device=device, dtype=row.dtype)
        is_score = qtype == "score"
        parts = loss_parts(
            row,
            distribution,
            lambda_distance=config.lambda_distance if is_score else 0.0,
            mu_unimodality=config.mu_unimodality if is_score else 0.0,
        )
        total = total + parts["total"].sum()
        per_row.append({key: float(parts[key].detach().sum()) for key in PART_KEYS})
    return total / len(batch["qtype"]), per_row


def plan_steps(n_examples: int, config: TrainConfig) -> int:
    """Optimiser steps for a run: ``max_steps`` if set, else exactly one pass.

    Args:
        n_examples: Rows selected for the run.
        config: ``max_steps``, ``batch_size``, ``grad_accum``.

    Returns:
        ``config.max_steps``, or ``ceil(n_examples / (batch_size * grad_accum))``.
    """
    if config.max_steps is not None:
        return config.max_steps
    # A family-balanced run's examples are already its draw sequence, so one pass over them is
    # the sample budget (less any draw dropped over the cap).
    return math.ceil(n_examples / (config.batch_size * config.grad_accum))


def _window_means(rows: Sequence[dict[str, float]], key: str) -> dict[str, float]:
    values = [row[key] for row in rows]
    return {
        "mean_first_10": sum(values[:10]) / len(values[:10]),
        "mean_last_10": sum(values[-10:]) / len(values[-10:]),
    }


def lr_multiplier(step: int, total_steps: int, warmup_steps: int, schedule: str) -> float:
    """Learning-rate multiplier at optimiser ``step`` (0-based): linear warmup, then the schedule.

    Args:
        step: Optimiser step about to be taken.
        total_steps: Steps in the run.
        warmup_steps: Linear warmup from ~0 to 1 over this many steps.
        schedule: ``constant`` or ``cosine`` (to zero at the last step).

    Returns:
        The factor the base learning rate is multiplied by.
    """
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "constant":
        return 1.0
    if schedule == "cosine":
        span = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / span)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"unknown lr_schedule {schedule!r}")


def save_training_state(
    path: Path, optimiser: Any, scheduler: Any, rows_done: int, config: TrainConfig
) -> Path:
    """Write ``trainer_state.pt`` beside a checkpoint's adapter: what a faithful resume needs.

    Args:
        path: The checkpoint directory (created if missing).
        optimiser: The optimiser; its moments are part of the run's state.
        scheduler: The learning-rate scheduler; its step count is the schedule position.
        rows_done: Rows trained so far — the position in the seeded draw sequence.
        config: The run's config, stored so a resume can refuse a changed one.

    Returns:
        The file written.
    """
    import torch

    path.mkdir(parents=True, exist_ok=True)
    target = path / "trainer_state.pt"
    torch.save(
        {
            "optimizer": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
            "optimizer_steps_done": scheduler.last_epoch,
            "rows_done": rows_done,
            "config": asdict(config),
        },
        target,
    )
    return target


#: Config fields a resume may change. Everything else would make the continuation a different run;
#: `eval_batch_size` is the owner's approved remedy for an evaluation slowdown (2026-09-24).
RESUME_MAY_CHANGE: frozenset[str] = frozenset({"eval_batch_size", "resume_from", "extra"})


def check_resume_config(saved: dict[str, Any], current: TrainConfig) -> None:
    """Refuse a resume whose config differs from the checkpoint's beyond :data:`RESUME_MAY_CHANGE`.

    Raises:
        ValueError: Naming every field that differs.
    """
    now = asdict(current)
    differs = sorted(
        key
        for key in set(saved) | set(now)
        if key not in RESUME_MAY_CHANGE and saved.get(key) != now.get(key)
    )
    if differs:
        detail = ", ".join(f"{k}: {saved.get(k)!r} -> {now.get(k)!r}" for k in differs)
        raise ValueError(f"cannot resume: config changed since the checkpoint ({detail})")


def resume_training_state(
    checkpoint: Path, model: Any, optimiser: Any, scheduler: Any, config: TrainConfig
) -> tuple[int, int]:
    """Load a checkpoint's adapter weights, optimiser and schedule into a freshly built run.

    Args:
        checkpoint: A ``rows-NNNNNN`` directory written by the trainer.
        model: The PEFT model, already built with the same adapter layout.
        optimiser: The freshly built optimiser.
        scheduler: The freshly built scheduler.
        config: The resuming run's config.

    Returns:
        ``(optimiser steps done, rows done)``.

    Raises:
        ValueError: If the config differs beyond what a resume may change.
    """
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    state = torch.load(checkpoint / "trainer_state.pt", weights_only=False)
    check_resume_config(state["config"], config)
    result = set_peft_model_state_dict(
        model, load_file(str(checkpoint / "adapter_model.safetensors"))
    )
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if unexpected:
        raise ValueError(f"checkpoint has adapter weights the model does not: {unexpected[:5]}")
    optimiser.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return int(state["optimizer_steps_done"]), int(state["rows_done"])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()
    ]


def _select_examples(
    config: TrainConfig,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    root: Path,
    stats: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The training sequence, in the order it is trained on, and a report of how it was drawn."""
    if config.sampling == "family_balanced":
        from train.sampling import row_weights, sample_indices, score_level_marginal

        weights_file = json.loads(
            (root / "data/processed/sampling_weights.json").read_text(encoding="utf-8")
        )
        indexed = [{**row, "row_index": i} for i, row in enumerate(rows)]
        weights = row_weights(indexed, weights_file["weights"])
        draw = sample_indices(weights, int(config.sample_budget), seed=config.seed)
        unique = sorted(set(draw))
        rendered = build_examples([indexed[i] for i in unique], tokenizer, config, stats)
        by_index = {ex["row_index"]: ex for ex in rendered}
        sequence = [by_index[i] for i in draw if i in by_index]
        drawn_rows = [rows[i] for i in draw]
        report = {
            "sampling": "family_balanced",
            "uniform_groups": ["score"],
            "budget": config.sample_budget,
            "drawn": len(draw),
            "unique_rows": len(unique),
            "dropped_over_cap_draws": len(draw) - len(sequence),
            "score_level_marginal_training": score_level_marginal(rows),
            "score_level_marginal_drawn": score_level_marginal(drawn_rows),
            "weights_file_seed": weights_file.get("seed"),
        }
        return sequence, report
    if config.selection == "longest":
        return longest_examples(build_examples(rows, tokenizer, config, stats), config.limit), {
            "selection": "longest"
        }
    examples = build_examples(select_rows(rows, config), tokenizer, config, stats)
    if config.limit:
        examples = examples[: config.limit]
    return examples, {"selection": "stratified"}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
    )


def train(config: TrainConfig, root: Path | None = None) -> dict[str, Any]:
    """Run one training job and write its summary.

    Two modes. A **smoke** run (``sampling: stratified`` or ``selection: longest``) makes one
    pass over a small selection. An **S1** run (``sampling: family_balanced``) draws
    ``sample_budget`` rows with replacement under the family weights, `Score` uniform by row,
    and — when configured — saves a checkpoint and evaluates a fixed `val` slice every
    ``eval_every_rows`` rows, starting with the untrained model at row 0.

    Returns:
        Summary: steps, loss curve, VRAM peak, tokens per second, and what it ran on.
    """
    import torch

    from s1decide.kernels import kernel_report
    from s1decide.tasks import repo_root

    if config.extra:
        # load_config keeps unknown keys in `extra`; training on a config whose sample budget,
        # sampling or eval schedule is silently ignored would be a different run from the one
        # the file describes.
        raise RuntimeError(
            f"config keys not implemented by this trainer: {sorted(config.extra)}; "
            "refusing to run a config it would partly ignore"
        )
    if config.selection not in {"stratified", "longest"}:
        raise ValueError(f"unknown selection {config.selection!r}")
    if config.sampling not in {"stratified", "family_balanced"}:
        raise ValueError(f"unknown sampling {config.sampling!r}")
    if config.sampling == "family_balanced" and not config.sample_budget:
        raise ValueError("family_balanced sampling needs a sample_budget")
    if config.eval_mode != "case-control":
        raise ValueError(
            f"only case-control periodic evaluation is implemented, not {config.eval_mode!r}"
        )
    lr_multiplier(0, 1, 0, config.lr_schedule)  # an unknown schedule fails here, not at step 1
    root = root or repo_root()
    out_dir = root / "results" / config.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model, local_files_only=True)
    rows = _read_jsonl(root / "data/processed/train.jsonl")
    render_stats: dict[str, int] = {"dropped_over_cap": 0}
    examples, selection_report = _select_examples(config, rows, tokenizer, root, render_stats)
    if not examples:
        raise RuntimeError("no training examples survived rendering; check max_seq_len")

    # The evaluation slice and its base-rate controls are built and checked before any weight
    # is loaded: a row with no control would otherwise raise at the first checkpoint, hours in.
    eval_examples: list[dict[str, Any]] = []
    eval_report: dict[str, Any] = {}
    rates: dict[str, list[float]] = {}
    if config.eval_every_rows:
        from train.periodic_eval import base_rates, build_eval_rows, eval_group

        eval_rows, eval_report = build_eval_rows(
            _read_jsonl(root / "data/processed/val.jsonl"),
            stage1_questions=config.eval_stage1_questions,
            negatives=config.eval_negatives,
            seed=config.seed,
        )
        eval_stats: dict[str, int] = {"dropped_over_cap": 0}
        eval_examples = build_examples(eval_rows, tokenizer, config, eval_stats)
        eval_report["rows_dropped_over_cap"] = eval_stats["dropped_over_cap"]
        rates = base_rates(rows)
        missing = sorted(
            {f"{eval_group(ex)}/{len(ex['target'])}" for ex in eval_examples} - set(rates)
        )
        if missing:
            raise RuntimeError(f"evaluation rows with no base-rate control: {missing}")
        _write_json(out_dir / "eval_slice.json", {**eval_report, "base_rates": rates})

    model, loader = _load_model(config)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config.learning_rate
    )
    one_pass = config.max_steps is None
    steps = plan_steps(len(examples), config)
    warmup_steps = round(config.warmup_fraction * steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: lr_multiplier(step, steps, warmup_steps, config.lr_schedule)
    )
    start_step, rows_offset, resumed = 0, 0, None
    if config.resume_from:
        start_step, rows_offset = resume_training_state(
            Path(config.resume_from), model, optimiser, scheduler, config
        )
        resumed = {"from": config.resume_from, "step": start_step, "rows": rows_offset}
        print(f"  resumed from {config.resume_from} at step {start_step}, row {rows_offset}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    checkpoints: list[dict[str, Any]] = []
    stop_reasons: list[str] = []
    if resumed and (out_dir / "evals.jsonl").is_file():
        # The stop rule reads the whole history ("two consecutive checkpoints"), not only the
        # evaluations this process ran.
        checkpoints = [
            entry for entry in _read_jsonl(out_dir / "evals.jsonl") if entry["rows"] <= rows_offset
        ]

    def evaluate(rows_done: int) -> None:
        from train.periodic_eval import eval_metrics, predict_logits

        began = time.perf_counter()

        def beat(done: int, total: int) -> None:
            (out_dir / "heartbeat").write_text(
                time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8", newline="\n"
            )
            print(f"    eval @ {rows_done} rows: {done}/{total}", flush=True)

        logits = predict_logits(
            model, eval_examples, pad_token_id, config.eval_batch_size, on_progress=beat
        )
        entry: dict[str, Any] = {
            "rows": rows_done,
            "metrics": eval_metrics(eval_examples, logits, rates),
            "seconds": round(time.perf_counter() - began, 1),
        }
        checkpoints.append(entry)
        from train.stop_rule import stop_decision

        entry["stop_rule"] = stop_decision(checkpoints)
        if entry["stop_rule"]["stop"]:
            stop_reasons.extend(entry["stop_rule"]["reasons"])
        with (out_dir / "evals.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")
        brief = {g: round(m["kl"], 4) for g, m in entry["metrics"].items()}
        print(f"  eval @ {rows_done} rows ({entry['seconds']} s): KL {brief}", flush=True)
        for line in entry["stop_rule"]["warnings"]:
            print(f"  warning: {line}", flush=True)
        for line in entry["stop_rule"]["reasons"]:
            print(f"  STOP: {line}", flush=True)

    def save_checkpoint(rows_done: int) -> None:
        """Adapter plus everything a faithful resume needs: optimiser, schedule, position.

        The draw sequence is a deterministic function of the seed, so ``rows_done`` is the
        position in it; without the optimiser and scheduler state a resume would restart Adam's
        moments and the learning-rate schedule, which is a different run.
        """
        path = out_dir / "checkpoints" / f"rows-{rows_done:06d}"
        model.save_pretrained(str(path))
        save_training_state(path, optimiser, scheduler, rows_done, config)

    progress_path = out_dir / "progress.jsonl"
    if config.eval_every_rows and not resumed:
        evaluate(0)

    losses: list[float] = []
    seen: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    tokens = 0
    started = time.perf_counter()
    model.train()
    # On a resume the draw sequence is the same seeded sequence; its first `rows_offset` rows were
    # trained by the earlier process, so this one starts after them.
    cursor = rows_offset
    rows_done = rows_offset
    for step in range(start_step, steps):
        optimiser.zero_grad(set_to_none=True)
        for _ in range(config.grad_accum):
            chunk = examples[cursor : cursor + config.batch_size]
            if not chunk:
                if one_pass:
                    break  # the last step of a pass can be short; it never wraps
                cursor = 0
                chunk = examples[: config.batch_size]
            cursor += config.batch_size
            batch = collate(chunk, pad_token_id)
            tokens += int(batch["attention_mask"].sum())
            loss, per_row = _batch_loss(model, batch, config)
            (loss / config.grad_accum).backward()
            losses.append(float(loss.detach()))
            for item, parts in zip(chunk, per_row, strict=True):
                seen.append(item)
                loss_rows.append(
                    {
                        "qtype": "noul/stage1" if item.get("stage1") else item["qtype"],
                        "soft": bool(item.get("soft")),
                        **{k: round(v, 5) for k, v in parts.items()},
                    }
                )
        optimiser.step()
        scheduler.step()
        before, rows_done = rows_done, rows_offset + len(seen)
        if step % 10 == 0:
            print(f"  step {step:4d}  loss {losses[-1]:.4f}", flush=True)
        if step % 25 == 0 or step == steps - 1:
            elapsed_now = time.perf_counter() - started
            with progress_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    json.dumps(
                        {
                            "step": step + 1,
                            "steps": steps,
                            "rows": rows_done,
                            "loss_mean_last_25": sum(losses[-25:]) / len(losses[-25:]),
                            "lr": scheduler.get_last_lr()[0],
                            "elapsed_seconds": round(elapsed_now, 1),
                            "rows_per_second": round(rows_done / max(elapsed_now, 1e-9), 4),
                            "vram_peak_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3)
                            if torch.cuda.is_available()
                            else None,
                        }
                    )
                    + "\n"
                )
            (out_dir / "heartbeat").write_text(
                time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8", newline="\n"
            )
        for every, hook in (
            (config.checkpoint_every_rows, save_checkpoint),
            (config.eval_every_rows, evaluate),
        ):
            if every and rows_done // every > before // every:
                hook(rows_done)
        if stop_reasons:
            # The rule is only evaluated at a checkpoint row, where the checkpoint (saved before
            # the evaluation) already holds this exact state.
            (out_dir / "STOPPED").write_text(
                f"stopped at {rows_done} rows\n" + "\n".join(stop_reasons) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            print(f"  stopping at {rows_done} rows: {stop_reasons}", flush=True)
            break
    if one_pass and rows_offset + len(seen) != len(examples) and not stop_reasons:
        raise RuntimeError(f"one pass saw {len(seen)} rows of {len(examples)} selected")

    elapsed = time.perf_counter() - started
    peak = float(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0.0
    summary = {
        "run_id": config.run_id,
        "config": asdict(config),
        "examples": len(examples),
        "selection": selection_report,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "one_pass": one_pass,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_mean_first_10": sum(losses[:10]) / max(1, len(losses[:10])),
        "loss_mean_last_10": sum(losses[-10:]) / max(1, len(losses[-10:])),
        "elapsed_seconds": round(elapsed, 2),
        "tokens_per_second": round(tokens / elapsed, 1) if elapsed else None,
        "loss_curve": [round(v, 5) for v in losses],
        # Per primitive, KL(target || pred) is the headline: it equals cross-entropy on a hard
        # row and has floor 0 on a soft one, so a window's hard/soft mix cannot move it. The raw
        # total and its floor are kept beside it so the difference stays visible.
        "loss_by_qtype": {
            qtype: {
                "rows": len(rows_q),
                "soft_rows": sum(1 for r in rows_q if r["soft"]),
                "kl": _window_means(rows_q, "kl"),
                "distance_excess": _window_means(rows_q, "distance_excess"),
                "total": _window_means(rows_q, "total"),
                "floor": _window_means(rows_q, "floor"),
            }
            for qtype in sorted({r["qtype"].split("/")[0] for r in loss_rows})
            for rows_q in [[r for r in loss_rows if r["qtype"].split("/")[0] == qtype]]
        },
        # What the optimiser saw, repeats included. In one-pass mode this equals the selection.
        "rows_seen": len(seen),
        "rows_dropped_over_cap": render_stats["dropped_over_cap"],
        # Activation memory follows the longest row actually trained on, not the cap.
        "max_row_tokens": max(len(item["input_ids"]) for item in seen),
        "coverage": coverage(seen),
        "coverage_selected": coverage(examples),
        "loss_rows": loss_rows,
        "eval_slice": eval_report or None,
        "checkpoint_evals": [c["rows"] for c in checkpoints],
        "stopped": stop_reasons or None,
        "resumed": resumed,
        "loader": loader,
        "vram_peak_bytes": peak,
        "vram_peak_gib": round(peak / 1024**3, 3),
        "kernels": kernel_report(),
        "logit_precision": "fp32",
    }
    # Saved before the summary is written, so whether it worked is recorded in the summary rather
    # than only in the filesystem — the first run reported no adapter while one sat on disk.
    adapter = out_dir / "adapter"
    try:
        model.save_pretrained(str(adapter))
        summary["adapter"] = str(adapter)
    except Exception as exc:  # pragma: no cover - depends on the peft wrapper
        summary["adapter_error"] = repr(exc)
    else:
        # The file is the artefact, so the file is what gets checked, not the live model.
        from transformers import AutoConfig

        try:
            summary["adapter_layout"] = check_lora_layout(
                adapter_file_keys(adapter),
                AutoConfig.from_pretrained(config.model, local_files_only=True),
            )
        except RuntimeError as exc:
            summary["adapter_layout_error"] = str(exc)

    _write_json(out_dir / "train_summary.json", summary)
    write_report(out_dir, summary)
    return summary


def write_report(out_dir: Path, summary: dict[str, Any]) -> Path:
    """Write ``report.md`` (coverage tables beside the loss) and ``loss.png`` for a run.

    Every smoke report carries the coverage table next to the loss curve, because a falling loss
    on one family is not a pass. Generated from ``train_summary.json`` so the report cannot
    disagree with the run.

    Args:
        out_dir: The run's ``results/<run_id>`` directory.
        summary: The run's summary, as :func:`train` wrote it.

    Returns:
        Path of the written ``report.md``.
    """
    plotted = _plot_loss(out_dir / "loss.png", summary.get("loss_rows") or [])
    lines = [
        f"# {summary['run_id']} — coverage and loss",
        "",
        "Generated by `train/sft_lora.py:write_report` from `train_summary.json`.",
        "",
        f"Rows seen by the optimiser: {summary.get('rows_seen', '?')} "
        f"(of {summary.get('examples', '?')} selected), {summary.get('steps')} steps.",
        "",
        "![loss per row, by qtype](loss.png)"
        if plotted
        else "_loss.png not drawn (no matplotlib)_",
        "",
        "## Loss by qtype (per row, rows seen)",
        "",
        "KL(target ‖ pred) is the column to read: it is cross-entropy on a hard row and zero at",
        "the optimum on a soft one. The raw total carries a soft row's floor (ln 2 plus",
        "λ·0.5 for a 50/50 split) and so moves with the hard/soft mix of the window.",
        "",
        "| qtype | rows | soft | KL first 10 | KL last 10 | total first 10 | total last 10 "
        "| floor first 10 | floor last 10 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for qtype, stats in (summary.get("loss_by_qtype") or {}).items():
        lines.append(
            f"| {qtype} | {stats['rows']} | {stats['soft_rows']} "
            + " ".join(
                f"| {stats[key][window]:.4f}"
                for key in ("kl", "total", "floor")
                for window in ("mean_first_10", "mean_last_10")
            )
            + " |"
        )
    lines += ["", "## Coverage (rows seen)", "", format_coverage(summary.get("coverage") or {})]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")
    return path


def _plot_loss(path: Path, loss_rows: Sequence[dict[str, Any]]) -> bool:
    """Two panels: KL per row by qtype, and `Score`'s CE and distance components apart.

    Returns False if nothing was drawn (no matplotlib, or no rows).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    if not loss_rows:
        return False
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)
    for qtype in sorted({str(r["qtype"]) for r in loss_rows}):
        points = [(i, r["kl"]) for i, r in enumerate(loss_rows) if r["qtype"] == qtype]
        top.scatter([p[0] for p in points], [p[1] for p in points], s=12, label=qtype)
    top.set_ylabel("KL(target ‖ pred)")
    top.legend(frameon=False, fontsize=8)

    score = [(i, r) for i, r in enumerate(loss_rows) if r["qtype"] == "score"]
    if score:
        xs = [i for i, _ in score]
        bottom.scatter(xs, [r["cross_entropy"] for _, r in score], s=12, label="cross-entropy")
        bottom.scatter(
            xs,
            [r["total"] - r["cross_entropy"] for _, r in score],
            s=12,
            marker="x",
            label="λ·distance",
        )
        bottom.scatter(xs, [r["floor"] for _, r in score], s=8, marker="_", label="row floor")
        bottom.legend(frameon=False, fontsize=8)
    bottom.set_ylabel("Score components")
    bottom.set_xlabel("row seen")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


#: ADR 0002, condition 4: LoRA on attention and MLP projections. The gated-deltanet projections
#: (``linear_attn.*``) are LoRA-able but deliberately excluded — adapting them is its own
#: evaluated decision, not a default.
ATTENTION_TARGETS: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_TARGETS: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")
LORA_TARGETS: tuple[str, ...] = ATTENTION_TARGETS + MLP_TARGETS

_LORA_NAME = re.compile(
    r"layers\.(?P<layer>\d+)\.(?P<block>self_attn|mlp)\.(?P<proj>[a-z_]+)"
    r"(?:\.lora_[AB](?:\.[A-Za-z_]+)*(?:\.weight)?)?$"
)


def expected_lora_layout(hf_config: Any) -> set[tuple[int, str]]:
    """Every ``(layer, projection)`` ADR 0002 says a Stage-1 adapter must cover.

    MLP projections in every decoder layer; attention projections in every layer that has
    softmax attention. On the Qwen3.5/3.8 hybrid that is 16 of 64 layers — the other 48 are
    gated-deltanet and have no ``self_attn`` to adapt — so "attention + MLP on all layers" means
    192 MLP modules and 64 attention modules, 256 in all.

    Args:
        hf_config: A ``transformers`` config; a vision-language config's text config is used.

    Returns:
        The set of ``(layer_index, projection_name)`` pairs.
    """
    text = hf_config.get_text_config() if hasattr(hf_config, "get_text_config") else hf_config
    layers = int(text.num_hidden_layers)
    types = list(getattr(text, "layer_types", None) or ["full_attention"] * layers)
    attention = [i for i, kind in enumerate(types) if kind == "full_attention"]
    return {(i, p) for i in attention for p in ATTENTION_TARGETS} | {
        (i, p) for i in range(layers) for p in MLP_TARGETS
    }


def check_lora_layout(adapted: Iterable[str], hf_config: Any) -> dict[str, Any]:
    """Compare the modules an adapter actually covers with ADR 0002's layout, and refuse a gap.

    The first 27B attempt trained on a fallback that adapted ``q/k/v/o`` in 16 of 64 layers and
    no MLP at all, and nothing said so until the adapter file was opened afterwards. This check
    runs before the first step and again on the saved file.

    Args:
        adapted: Module or tensor names carrying LoRA weights — ``named_modules`` names of
            layers with ``lora_A``, or safetensors keys such as
            ``base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight``.
        hf_config: The base model's config.

    Returns:
        ``expected``, ``adapted``, ``attention_layers`` and ``mlp_layers`` counts.

    Raises:
        RuntimeError: If any expected module is not adapted, or anything outside the layout is
            (a ``linear_attn`` projection, the vision tower) — naming what is wrong.
    """
    expected = expected_lora_layout(hf_config)
    got: set[tuple[int, str]] = set()
    unexpected: set[str] = set()
    for name in adapted:
        match = _LORA_NAME.search(name)
        parts = set(name.split("."))
        foreign = bool(parts & {"mtp", "visual", "vision_tower"})
        if match and match["proj"] in LORA_TARGETS and not foreign:
            got.add((int(match["layer"]), match["proj"]))
        else:
            unexpected.add(re.sub(r"\.lora_[AB].*$", "", name))
    missing = expected - got
    unexpected |= {f"layers.{i}.{p}" for i, p in got - expected}
    if missing or unexpected:
        missing_by_proj: dict[str, int] = {}
        for _, proj in missing:
            missing_by_proj[proj] = missing_by_proj.get(proj, 0) + 1
        raise RuntimeError(
            "LoRA layout does not match ADR 0002 (attention + MLP): "
            f"{len(got & expected)}/{len(expected)} expected modules adapted; "
            f"missing by projection {dict(sorted(missing_by_proj.items()))}; "
            f"unexpected {sorted(unexpected)[:8]}{' ...' if len(unexpected) > 8 else ''}. "
            "Refusing to train a different adapter from the one the config describes."
        )
    return {
        "expected": len(expected),
        "adapted": len(got),
        "attention_layers": len({i for i, p in got if p in ATTENTION_TARGETS}),
        "mlp_layers": len({i for i, p in got if p in MLP_TARGETS}),
    }


def adapted_module_names(model: Any) -> list[str]:
    """Names of every module in ``model`` that carries LoRA weights."""
    return [name for name, module in model.named_modules() if hasattr(module, "lora_A")]


def adapter_file_keys(adapter_dir: Path) -> list[str]:
    """Tensor names in a saved adapter's ``adapter_model.safetensors``."""
    from safetensors import safe_open

    with safe_open(str(adapter_dir / "adapter_model.safetensors"), "pt") as handle:
        return list(handle.keys())


def is_prequantized(hf_config: Any) -> bool:
    """Whether a checkpoint ships its own quantization config and must not be quantized again.

    Args:
        hf_config: A ``transformers`` config, as ``AutoConfig.from_pretrained`` returns it.

    Returns:
        True for e.g. ``unsloth/*-bnb-4bit``, whose config carries ``quantization_config``.
    """
    return getattr(hf_config, "quantization_config", None) is not None


def _load_model(config: TrainConfig) -> tuple[Any, dict[str, Any]]:
    """Load through Unsloth where it can, and fall back to plain peft where it cannot.

    CLAUDE.md pins Unsloth Core as the training backend, and that is what a real run uses. The
    fallback exists for the small pipeline-proving model, which Unsloth may not recognise.

    Both paths apply the same ADR 0002 layout (:data:`LORA_TARGETS`) and both are checked
    against it with :func:`check_lora_layout` before a step is taken: a path that cannot build
    that adapter raises rather than training a smaller one.

    Quantization is asked for differently on each path, and deliberately:

    - **Unsloth** gets ``load_in_4bit=config.load_in_4bit``. Unsloth resolves the repo name from
      that flag: told ``False`` it maps ``unsloth/*-bnb-4bit`` to the bf16 repo, which is what
      sent the second 27B attempt to an uncached checkpoint. Told ``True`` on a pre-quantized
      checkpoint, it keeps the repo and uses the checkpoint's own quantization.
    - **transformers + peft** never adds a quantization config to a checkpoint that ships one.
      Doing so produced ``A inner dim (5120) does not match weight (1)`` on the first attempt.

    Returns ``(model, loader)``: ``loader`` records the path, whether the checkpoint was
    pre-quantized, why Unsloth was skipped if it was, and the adapter layout.
    """
    import copy

    # Unsloth first, so its kernels are bound before transformers builds anything. A failure
    # here is recorded, not hidden: it is also why the Unsloth path below would then fail.
    unsloth_import_error = None
    try:
        import unsloth  # noqa: F401
    except Exception as exc:  # Unsloth absent, or no accelerator to patch
        unsloth_import_error = f"{type(exc).__name__}: {exc}"[:300]
        print(f"  unsloth import failed ({unsloth_import_error})", file=sys.stderr)
    import torch
    from transformers import AutoConfig

    from s1decide.engine.hf import force_quantized_model_dtype

    hf_config = AutoConfig.from_pretrained(config.model, local_files_only=True)
    prequantized = is_prequantized(hf_config)
    loader: dict[str, Any] = {
        "prequantized": prequantized,
        "targets": list(LORA_TARGETS),
        "unsloth_import_error": unsloth_import_error,
    }

    try:
        from unsloth import FastLanguageModel

        model, _ = FastLanguageModel.from_pretrained(
            model_name=config.model,
            max_seq_length=config.max_seq_len,
            load_in_4bit=config.load_in_4bit,
            dtype=torch.bfloat16,
            local_files_only=True,
            text_only=True,
            offload_embedding=config.load_in_4bit,
        )
        if prequantized:
            loader["dtype_fix"] = force_quantized_model_dtype(model, torch.bfloat16)
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.rank,
            lora_alpha=config.lora_alpha,
            target_modules=list(LORA_TARGETS),
            use_gradient_checkpointing="unsloth",
            random_state=config.seed,
        )
        loader["path"] = "unsloth"
    except Exception as exc:
        loader["unsloth_error"] = f"{type(exc).__name__}: {exc}"[:500]
        print(f"  unsloth path unavailable ({loader['unsloth_error']}); using transformers + peft")
        # A half-built 27B from the failed attempt would otherwise still hold the card, and
        # the fallback would then OOM for a reason that has nothing to do with training.
        import gc

        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        kwargs: dict[str, Any] = {
            "dtype": torch.bfloat16,
            "local_files_only": True,
            "attn_implementation": "sdpa",
        }
        if config.load_in_4bit and not prequantized:
            from s1decide.engine.hf import nf4_config

            kwargs["quantization_config"] = nf4_config()
        if config.load_in_4bit:
            kwargs["device_map"] = {"": 0}
        if prequantized and getattr(hf_config, "vision_config", None) is not None:
            # The text tower is built from `text_config`, which does not carry the parent's
            # quantization config; same fix as engine/hf.py.
            text_config = copy.deepcopy(hf_config.get_text_config())
            text_config.quantization_config = hf_config.quantization_config
            kwargs["config"] = text_config
        model = AutoModelForCausalLM.from_pretrained(config.model, **kwargs)
        if prequantized:
            loader["dtype_fix"] = force_quantized_model_dtype(model, torch.bfloat16)
        if not config.load_in_4bit and torch.cuda.is_available():
            model = model.to("cuda")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model = get_peft_model(
            model,
            LoraConfig(
                r=config.rank,
                lora_alpha=config.lora_alpha,
                target_modules=list(LORA_TARGETS),
                task_type="CAUSAL_LM",
            ),
        )
        loader["path"] = "transformers+peft"

    loader["quantized"] = bool(getattr(model, "is_quantized", False)) or config.load_in_4bit
    loader["layout"] = check_lora_layout(adapted_module_names(model), hf_config)
    return model, loader


def main(argv: Sequence[str] | None = None) -> int:
    """Command line for ``uv run task train``.

    ``--detach`` starts the same command in its own process (`s1decide.jobs.spawn_detached`),
    logging to ``results/<run_id>/log.txt``. A run ends by writing ``DONE`` or ``FAILED`` (with
    the traceback) beside its summary, so a monitor can tell finished from dead.
    """
    import argparse
    import traceback

    from s1decide.tasks import repo_root

    parser = argparse.ArgumentParser(prog="task train")
    parser.add_argument("--cfg", required=True, help="path to a train/configs/*.yaml")
    parser.add_argument("--detach", action="store_true", help="run in a detached process")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.cfg)
    out_dir = repo_root() / "results" / config.run_id
    if args.detach:
        from s1decide.jobs import spawn_detached

        out_dir.mkdir(parents=True, exist_ok=True)
        pid = spawn_detached([sys.executable, "-m", "train.sft_lora", "--cfg", args.cfg], out_dir)
        print(f"detached training {config.run_id} (pid {pid}); log: {out_dir / 'log.txt'}")
        return 0

    print(f"training {config.run_id}: {config.model} on {config.hardware}", flush=True)
    try:
        summary = train(config)
    except BaseException:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "FAILED").write_text(traceback.format_exc(), encoding="utf-8", newline="\n")
        raise
    bulky = {"config", "loss_rows", "loss_curve"}
    print(json.dumps({k: v for k, v in summary.items() if k not in bulky}, indent=2, default=str))
    if "adapter_layout_error" in summary or "adapter_error" in summary:
        print(summary.get("adapter_layout_error") or summary.get("adapter_error"))
        (out_dir / "FAILED").write_text("adapter check failed\n", encoding="utf-8", newline="\n")
        return 1
    if summary.get("stopped"):
        # STOPPED was written by the loop with its reasons; not DONE, and not a crash either.
        return 3
    if summary["loss_mean_last_10"] is not None and math.isfinite(summary["loss_mean_last_10"]):
        (out_dir / "DONE").write_text(
            time.strftime("%Y-%m-%dT%H:%M:%S\n"), encoding="utf-8", newline="\n"
        )
        return 0
    (out_dir / "FAILED").write_text("non-finite loss\n", encoding="utf-8", newline="\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
