"""The teacher-labelling job itself: load a teacher, generate with reasoning, checkpoint.

Runs in the foreground of whatever process starts it. `uv run task teach --detach` spawns that
process detached so the run survives the session that launched it; `--status` inspects it from
outside. The bookkeeping — resume, heartbeat, progress, failure — lives in `s1decide.jobs`; this
module only knows how to talk to a teacher.

This is the one place in the project that runs a decode loop, and that is fine: `CLAUDE.md`'s
no-decode-loop rule is about `decide()`, the inference path we publish. Generating training data
is not inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from data.build.teach import (
    RUBRIC,
    TeacherSetting,
    build_prompt,
    parse_level,
    prompt_hash,
    trace_stats,
    write_summary,
)

from s1decide.jobs import JobPaths, check_resume_meta, job_status, run_job, spawn_detached
from s1decide.kernels import kernel_report

__all__ = ["TEACHERS", "load_teacher", "main", "teacher_rows"]

#: The candidates, as approved 2026-09-17. Teacher 1 is fixed; teacher 2 is chosen from the
#: pilot on tractability — loads, throughput, truncation — never on agreement (ADR 0005).
TEACHERS: dict[str, TeacherSetting] = {
    "qwen-low": TeacherSetting(
        model="unsloth/Qwen3.8-27B-unsloth-bnb-4bit",
        label="qwen-low",
        effort="low",
        max_new_tokens=384,
    ),
    "gptoss-low": TeacherSetting(
        model="openai/gpt-oss-20b",
        label="gptoss-low",
        effort="low",
        max_new_tokens=384,
    ),
    "gptoss-medium": TeacherSetting(
        model="openai/gpt-oss-20b",
        label="gptoss-medium",
        effort="medium",
        max_new_tokens=384,
    ),
    "qwen-low-1024": TeacherSetting(
        model="unsloth/Qwen3.8-27B-unsloth-bnb-4bit",
        label="qwen-low-1024",
        effort="low",
        max_new_tokens=1024,
    ),
    "gptoss-low-1024": TeacherSetting(
        model="openai/gpt-oss-20b",
        label="gptoss-low-1024",
        effort="low",
        max_new_tokens=1024,
    ),
    "gptoss-medium-1024": TeacherSetting(
        model="openai/gpt-oss-20b",
        label="gptoss-medium-1024",
        effort="medium",
        max_new_tokens=1024,
    ),
    "magistral-1024": TeacherSetting(
        model="unsloth/Magistral-Small-2509-unsloth-bnb-4bit",
        label="magistral-1024",
        effort=None,
        max_new_tokens=1024,
    ),
    "magistral-384": TeacherSetting(
        model="unsloth/Magistral-Small-2509-unsloth-bnb-4bit",
        label="magistral-384",
        effort=None,
        max_new_tokens=384,
    ),
}


def load_teacher(setting: TeacherSetting) -> tuple[Any, Any]:
    """Load a teacher for generation, text-only and 4-bit where it is not already quantized.

    Args:
        setting: Which teacher and at what reasoning setting.

    Returns:
        ``(model, tokenizer)``.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from s1decide.engine.hf import force_quantized_model_dtype, nf4_config

    tokenizer = AutoTokenizer.from_pretrained(setting.model, local_files_only=True)
    kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "device_map": {"": 0},
        "local_files_only": True,
        "attn_implementation": "sdpa",
    }
    # A checkpoint that ships pre-quantized (bnb-4bit, MXFP4) carries its own config; adding
    # ours on top is what broke the 27B load three times in Phase 0.
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(setting.model, local_files_only=True)
    if getattr(config, "quantization_config", None) is None:
        kwargs["quantization_config"] = nf4_config()
    elif getattr(config, "vision_config", None) is not None:
        # A pre-quantized vision-language checkpoint: `AutoModelForCausalLM` builds the text
        # tower from `text_config`, which does not carry the parent's `quantization_config`, so
        # the weights arrive quantized and the model does not expect them to be. Hand it the
        # text config with the quantization config copied across — the same fix `engine/hf.py`
        # carries, and the reason Qwen3.8-27B took three attempts to load in Phase 0.
        import copy

        text_config = copy.deepcopy(config.get_text_config())
        text_config.quantization_config = config.quantization_config
        kwargs["config"] = text_config

    def _load(**extra: Any) -> Any:
        """Try the causal-LM head, then the image-text head for vision-language checkpoints.

        Magistral and Qwen3.8 are both VLM classes whose text tower is what we want. There is no
        text-only AutoModel for them, so the full class is loaded and only text is ever passed.
        """
        merged = {**kwargs, **extra}
        try:
            return AutoModelForCausalLM.from_pretrained(setting.model, **merged)
        except ValueError as exc:
            if "Unrecognized configuration class" not in str(exc):
                raise
            from transformers import AutoModelForImageTextToText

            print(f"  {setting.model}: vision-language class, loading the full model", flush=True)
            return AutoModelForImageTextToText.from_pretrained(setting.model, **merged)

    try:
        model = _load().eval()
    except ValueError as exc:
        if "scaled_dot_product_attention" not in str(exc):
            raise
        # gpt-oss has no SDPA path in transformers 5.5. Falling back is fine *here* and would
        # not be in `engine/hf.py`: CLAUDE.md pins sdpa for the inference path we publish and
        # benchmark, not for a teacher that generates training data once.
        kwargs["attn_implementation"] = "eager"
        print(f"  {setting.model}: no sdpa kernel, falling back to eager", flush=True)
        model = _load().eval()

    force_quantized_model_dtype(model, torch.bfloat16)

    # `device_map="cuda"` does not always place an MXFP4 MoE's expert weights: gpt-oss loaded
    # with its experts left on CPU and failed mid-generate with "mat2 is on cpu". Check rather
    # than hope — a half-placed model fails deep inside a kernel, hours in, with a message that
    # says nothing about loading.
    stragglers = [
        name
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
        if tensor.device.type != "cuda"
    ]
    if stragglers:
        print(f"  {setting.model}: {len(stragglers)} tensors left on CPU, moving", flush=True)
        model = model.to("cuda")
        still = [
            name
            for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
            if tensor.device.type != "cuda"
        ]
        if still:
            raise RuntimeError(
                f"{setting.model}: {len(still)} tensors are still not on the GPU after .to(), "
                f"first few {still[:5]} — generation would fail inside a kernel instead"
            )
    return model, tokenizer


def _render(tokenizer: Any, setting: TeacherSetting, prompt: str) -> str:
    """Apply the chat template, asking for reasoning where the model has a knob for it."""
    messages = [{"role": "user", "content": prompt}]
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if setting.effort:
        # gpt-oss takes reasoning_effort through its template; models without it ignore the
        # kwarg, and a TypeError means the template does not accept it at all.
        try:
            return tokenizer.apply_chat_template(
                messages, reasoning_effort=setting.effort, **kwargs
            )
        except TypeError:
            pass
    return tokenizer.apply_chat_template(messages, **kwargs)


def teacher_rows(
    model: Any,
    tokenizer: Any,
    setting: TeacherSetting,
    batch: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Label one batch, returning a row per item.

    Args:
        model: Loaded teacher.
        tokenizer: Its tokenizer.
        setting: Reasoning setting and generation cap.
        batch: Items with ``id``, ``state`` and ``question``.

    Returns:
        One row per item: the parsed level (``None`` when the model never committed), the
        generated token count, whether it hit the cap, and the prompt hash.
    """
    import torch

    prompts = [build_prompt(item["state"], item["question"]) for item in batch]
    rendered = [_render(tokenizer, setting, p) for p in prompts]

    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    encoded = tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to(
        model.device
    )

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=setting.max_new_tokens,
            do_sample=False,
            pad_token_id=pad,
        )

    prompt_length = encoded["input_ids"].shape[1]
    rows: list[dict[str, Any]] = []
    for item, prompt, sequence in zip(batch, prompts, generated):
        new_tokens = sequence[prompt_length:]
        # Trailing pad is not generation; counting it would inflate every trace-length number.
        kept = [int(t) for t in new_tokens.tolist() if int(t) != pad]
        text = tokenizer.decode(kept, skip_special_tokens=True)
        level = parse_level(text)
        rows.append(
            {
                "id": item["id"],
                "teacher": setting.label,
                "model": setting.model,
                "effort": setting.effort,
                "level": level,
                "trace_tokens": len(kept),
                "truncated": len(kept) >= setting.max_new_tokens and level is None,
                "hit_cap": len(kept) >= setting.max_new_tokens,
                "prompt_hash": prompt_hash(prompt),
                "text": text[-600:],
            }
        )
    return rows


def load_items(path: Path, limit: int | None) -> list[dict[str, Any]]:
    """Read the states to label from a JSONL file.

    Args:
        path: A JSONL file with ``id``, ``state`` and ``question`` per line.
        limit: Cap for a pilot.

    Returns:
        The items.
    """
    items = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        items.append({"id": str(row["id"]), "state": row["state"], "question": row["question"]})
    return items[:limit] if limit else items


def items_digest(items: Sequence[dict[str, Any]]) -> str:
    """A fingerprint of exactly what a run was asked to label.

    Hashes the id, state and question of every item, in order. It exists because the input file
    is not tracked by git and the ids embed a position (``teach-00042-9f3c1ab2``): regenerating
    ``teach_items.jsonl`` after an upstream change shifts every id from the first altered row
    onward, and a resume would then re-label thousands of rows and write a corpus containing
    two numbering schemes. Recording this in ``meta.json`` turns that from a silent corruption
    into a refused resume.

    Args:
        items: The items, in the order they will be processed.

    Returns:
        A hex sha256.
    """
    digest = hashlib.sha256()
    for item in items:
        digest.update(
            json.dumps(
                [item["id"], item["state"], item["question"]], ensure_ascii=False, sort_keys=True
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def run(
    setting: TeacherSetting,
    items: Sequence[dict[str, Any]],
    directory: Path,
    batch_size: int,
    force: bool = False,
) -> dict[str, Any]:
    """Label every item with one teacher, checkpointing as it goes.

    Args:
        setting: Which teacher, at what reasoning setting and generation cap.
        items: The states and questions to label.
        directory: Job directory; an existing one is resumed.
        batch_size: Items per generate call.
        force: Resume even when the configuration has changed since the rows on disk.

    Returns:
        The run summary, also written to ``summary.json``.

    Raises:
        MetaMismatch: When resuming under a changed configuration without ``force``.
    """
    meta = {
        "setting": setting.to_json(),
        "rubric": list(RUBRIC),
        "batch_size": batch_size,
        "items_digest": items_digest(items),
        "items_count": len(items),
        "kernels": kernel_report(),
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    # Before the model load, not after: a refused resume should cost a second, not five minutes.
    check_resume_meta(directory, meta, force=force)
    model, tokenizer = load_teacher(setting)

    def work(batch: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = teacher_rows(model, tokenizer, setting, batch)
        rows[-1]["progress"] = {
            "teacher": setting.label,
            "truncated": sum(1 for r in rows if r["truncated"]),
            "uncommitted": sum(1 for r in rows if r["level"] is None),
        }
        return rows

    summary = run_job(
        directory,
        list(items),
        work,
        key=lambda item: item["id"],
        batch_size=batch_size,
        meta=meta,
        force=force,
    )

    rows = [
        json.loads(line)
        for line in JobPaths(directory).rows.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    committed = [r for r in rows if r["level"] is not None]
    payload = {
        **summary,
        "setting": setting.to_json(),
        "committed": len(committed),
        "uncommitted": len(rows) - len(committed),
        "truncation_rate": sum(1 for r in rows if r["truncated"]) / max(1, len(rows)),
        "hit_cap_rate": sum(1 for r in rows if r["hit_cap"]) / max(1, len(rows)),
        "trace_tokens": trace_stats([r["trace_tokens"] for r in rows]),
        "tokens_per_second": (
            sum(r["trace_tokens"] for r in rows) / summary["elapsed_seconds"]
            if summary["elapsed_seconds"]
            else None
        ),
    }
    write_summary(directory, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Command line for ``uv run task teach``."""
    from s1decide.tasks import repo_root

    parser = argparse.ArgumentParser(prog="task teach")
    parser.add_argument("--teacher", choices=sorted(TEACHERS), help="which teacher and setting")
    parser.add_argument("--items", default="data/processed/teach_items.jsonl")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--limit", type=int, default=None, help="cap the item count (pilot)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--detach", action="store_true", help="run as a detached process")
    parser.add_argument("--status", default=None, metavar="RUN_ID", help="report on a run")
    parser.add_argument(
        "--force",
        action="store_true",
        help="resume even if the configuration changed since the rows on disk",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = repo_root()
    if args.status:
        print(job_status(root / "results" / args.status).render())
        return 0

    if not args.teacher:
        parser.error("--teacher is required unless --status is given")
    setting = TEACHERS[args.teacher]
    run_id = args.run_id or f"teach-{setting.label}"
    directory = root / "results" / run_id

    if args.detach:
        argv_child = [
            sys.executable,
            "-m",
            "data.build.teach_run",
            "--teacher",
            args.teacher,
            "--items",
            args.items,
            "--run-id",
            run_id,
            "--batch-size",
            str(args.batch_size),
        ]
        if args.limit:
            argv_child += ["--limit", str(args.limit)]
        if args.force:
            argv_child.append("--force")
        pid = spawn_detached(argv_child, directory)
        print(f"detached run {run_id} (pid {pid})")
        print(f"  status: uv run task teach --status {run_id}")
        return 0

    items = load_items(
        Path(args.items) if Path(args.items).is_absolute() else root / args.items, args.limit
    )
    payload = run(setting, items, directory, args.batch_size, force=args.force)
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
