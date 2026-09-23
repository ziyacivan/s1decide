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
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "SMALL_MODEL",
    "TrainConfig",
    "build_examples",
    "collate",
    "coverage",
    "format_coverage",
    "is_prequantized",
    "load_config",
    "main",
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
        max_steps: Optimiser steps. The smoke run is deliberately tiny.
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
    max_steps: int = 50
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


def _batch_loss(model: Any, batch: dict[str, Any], config: TrainConfig) -> tuple[Any, list[float]]:
    """Masked-option loss at the answer position, ordinal where the row is a `Score`.

    Returns the batch mean for the backward pass and each row's own loss, so the summary can
    show the curve per primitive rather than one number that a single family can carry.
    """
    import torch

    from train.ordinal_loss import ordinal_loss

    device = next(model.parameters()).device
    out = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
    )
    answer_logits = out.logits[:, -1, :].float()

    total = answer_logits.new_zeros(())
    per_row: list[float] = []
    for i, (ids, target, qtype) in enumerate(
        zip(batch["label_token_ids"], batch["target"], batch["qtype"], strict=True)
    ):
        index = torch.tensor(ids, device=device)
        row = answer_logits[i, index].unsqueeze(0)
        distribution = torch.tensor([target], device=device, dtype=row.dtype)
        if qtype == "score":
            loss = ordinal_loss(
                row,
                distribution,
                lambda_distance=config.lambda_distance,
                mu_unimodality=config.mu_unimodality,
            )
        else:
            loss = ordinal_loss(row, distribution, lambda_distance=0.0)
        per_row.append(float(loss.detach()))
        total = total + loss
    return total / len(batch["qtype"]), per_row


def train(config: TrainConfig, root: Path | None = None) -> dict[str, Any]:
    """Run one training job and write its summary.

    Returns:
        Summary: steps, loss curve, VRAM peak, tokens per second, and what it ran on.
    """
    import torch

    from s1decide.kernels import kernel_report
    from s1decide.tasks import repo_root

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
    seen_losses: dict[str, list[float]] = {}
    seen: list[dict[str, Any]] = []
    loss_rows: list[list[Any]] = []
    tokens = 0
    started = time.perf_counter()
    model.train()
    cursor = 0
    for step in range(config.max_steps):
        optimiser.zero_grad(set_to_none=True)
        for _ in range(config.grad_accum):
            chunk = examples[cursor : cursor + config.batch_size]
            if not chunk:
                cursor = 0
                chunk = examples[: config.batch_size]
            cursor += config.batch_size
            batch = collate(chunk, pad_token_id)
            tokens += int(batch["attention_mask"].sum())
            loss, per_row = _batch_loss(model, batch, config)
            (loss / config.grad_accum).backward()
            losses.append(float(loss.detach()))
            for item, value in zip(chunk, per_row, strict=True):
                seen_losses.setdefault(item["qtype"], []).append(value)
                seen.append(item)
                loss_rows.append(["noul/stage1" if item.get("stage1") else item["qtype"], value])
        optimiser.step()
        if step % 10 == 0:
            print(f"  step {step:4d}  loss {losses[-1]:.4f}", flush=True)

    elapsed = time.perf_counter() - started
    peak = float(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0.0
    summary = {
        "run_id": config.run_id,
        "config": asdict(config),
        "examples": len(examples),
        "steps": config.max_steps,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_mean_first_10": sum(losses[:10]) / max(1, len(losses[:10])),
        "loss_mean_last_10": sum(losses[-10:]) / max(1, len(losses[-10:])),
        "elapsed_seconds": round(elapsed, 2),
        "tokens_per_second": round(tokens / elapsed, 1) if elapsed else None,
        "loss_curve": [round(v, 5) for v in losses],
        "loss_by_qtype": {
            qtype: {
                "rows": len(values),
                "mean_first_10": sum(values[:10]) / len(values[:10]),
                "mean_last_10": sum(values[-10:]) / len(values[-10:]),
            }
            for qtype, values in sorted(seen_losses.items())
        },
        # What the optimiser saw, repeats included. 50 steps at batch 1 over 200 selected rows
        # sees a quarter of them, and the table has to describe that quarter, not the 200.
        "rows_seen": len(seen),
        "coverage": coverage(seen),
        "coverage_selected": coverage(examples),
        "loss_rows": [[q, round(v, 5)] for q, v in loss_rows],
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
        "## Loss by qtype (per-row, rows seen)",
        "",
        "| qtype | rows | mean first 10 | mean last 10 |",
        "|---|---|---|---|",
    ]
    for qtype, stats in (summary.get("loss_by_qtype") or {}).items():
        lines.append(
            f"| {qtype} | {stats['rows']} | {stats['mean_first_10']:.4f} "
            f"| {stats['mean_last_10']:.4f} |"
        )
    lines += ["", "## Coverage (rows seen)", "", format_coverage(summary.get("coverage") or {})]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")
    return path


def _plot_loss(path: Path, loss_rows: Sequence[Sequence[Any]]) -> bool:
    """Scatter per-row loss against its position, one colour per qtype. False if not drawn."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    if not loss_rows:
        return False
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for qtype in sorted({str(q) for q, _ in loss_rows}):
        points = [(i, v) for i, (q, v) in enumerate(loss_rows) if q == qtype]
        ax.scatter([p[0] for p in points], [p[1] for p in points], s=12, label=qtype)
    ax.set_xlabel("row seen")
    ax.set_ylabel("loss")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


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
    fallback exists for the small pipeline-proving model, which Unsloth may not recognise: the
    point of that run is to prove the *data and loss* path, and refusing to run it because a
    0.8B is not in a support matrix would remove the cheapest check in the project.

    Returns ``(model, loader)``: ``loader`` records which path ran, whether the checkpoint was
    pre-quantized, and why Unsloth was skipped if it was — the first 27B attempt left no trace of
    which path had failed.

    A checkpoint that ships its own ``quantization_config`` (``unsloth/*-bnb-4bit``) is loaded
    with **no** quantization request on top: asking for 4-bit again is what produced
    ``A inner dim (5120) does not match weight (1)`` on the first 27B attempt, the same failure
    ``engine/hf.py`` and ``data/build/teach_run.py`` already guard against.
    """
    import copy

    import torch
    from transformers import AutoConfig

    from s1decide.engine.hf import force_quantized_model_dtype

    hf_config = AutoConfig.from_pretrained(config.model, local_files_only=True)
    prequantized = is_prequantized(hf_config)
    quantize = config.load_in_4bit and not prequantized
    loader: dict[str, Any] = {"prequantized": prequantized, "quantize_on_load": quantize}

    try:
        from unsloth import FastLanguageModel

        model, _ = FastLanguageModel.from_pretrained(
            model_name=config.model,
            max_seq_length=config.max_seq_len,
            load_in_4bit=quantize,
            dtype=torch.bfloat16,
            local_files_only=True,
        )
        if prequantized:
            loader["dtype_fix"] = force_quantized_model_dtype(model, torch.bfloat16)
        loader["path"] = "unsloth"
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.rank,
            lora_alpha=config.lora_alpha,
            use_gradient_checkpointing="unsloth",
            random_state=config.seed,
        )
        return model, loader
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
        if quantize:
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
        loader["path"] = "transformers+peft"
        return get_peft_model(
            model,
            LoraConfig(
                r=config.rank,
                lora_alpha=config.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type="CAUSAL_LM",
            ),
        ), loader


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
    if summary["loss_mean_last_10"] is not None and math.isfinite(summary["loss_mean_last_10"]):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
