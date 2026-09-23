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
    "collate",
    "coverage",
    "expected_lora_layout",
    "format_coverage",
    "is_prequantized",
    "load_config",
    "main",
    "plan_steps",
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
    rows: Sequence[dict[str, Any]], tokenizer: Any, config: TrainConfig
) -> list[dict[str, Any]]:
    """Render every row the way inference will, and attach its option tokens and target.

    Args:
        rows: Corpus rows.
        tokenizer: The model's tokenizer.
        config: Sequence cap and sampling seed.

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
            # Skipped, never truncated: cutting a prompt moves the answer position, and the
            # model would be trained to answer at a place it is never read at.
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
            }
        )
    random.Random(config.seed).shuffle(examples)
    return examples


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
    return math.ceil(n_examples / (config.batch_size * config.grad_accum))


def _window_means(rows: Sequence[dict[str, float]], key: str) -> dict[str, float]:
    values = [row[key] for row in rows]
    return {
        "mean_first_10": sum(values[:10]) / len(values[:10]),
        "mean_last_10": sum(values[-10:]) / len(values[-10:]),
    }


def train(config: TrainConfig, root: Path | None = None) -> dict[str, Any]:
    """Run one training job and write its summary.

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
    root = root or repo_root()
    out_dir = root / "results" / config.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model, local_files_only=True)
    rows = [
        json.loads(line)
        for line in (root / "data/processed/train.jsonl").read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    rows = select_rows(rows, config)
    examples = build_examples(rows, tokenizer, config)
    if config.limit:
        examples = examples[: config.limit]
    if not examples:
        raise RuntimeError("no training examples survived rendering; check max_seq_len")

    model, loader = _load_model(config)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config.learning_rate
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    losses: list[float] = []
    seen: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    tokens = 0
    steps = plan_steps(len(examples), config)
    one_pass = config.max_steps is None
    started = time.perf_counter()
    model.train()
    cursor = 0
    for step in range(steps):
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
        if step % 10 == 0:
            print(f"  step {step:4d}  loss {losses[-1]:.4f}", flush=True)
    if one_pass and len(seen) != len(examples):
        raise RuntimeError(f"one pass saw {len(seen)} rows of {len(examples)} selected")

    elapsed = time.perf_counter() - started
    peak = float(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0.0
    summary = {
        "run_id": config.run_id,
        "config": asdict(config),
        "examples": len(examples),
        "steps": steps,
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
        "coverage": coverage(seen),
        "coverage_selected": coverage(examples),
        "loss_rows": loss_rows,
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

    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
    )
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
    """Command line for ``uv run task train``."""
    import argparse

    parser = argparse.ArgumentParser(prog="task train")
    parser.add_argument("--cfg", required=True, help="path to a train/configs/*.yaml")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.cfg)
    print(f"training {config.run_id}: {config.model} on {config.hardware}", flush=True)
    summary = train(config)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2, default=str))
    if "adapter_layout_error" in summary or "adapter_error" in summary:
        print(summary.get("adapter_layout_error") or summary.get("adapter_error"))
        return 1
    if summary["loss_mean_last_10"] is not None and math.isfinite(summary["loss_mean_last_10"]):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
