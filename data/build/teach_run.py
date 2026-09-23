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

from s1decide.jobs import (
    JobPaths,
    check_resume_meta,
    job_status,
    read_done_ids,
    run_job,
    spawn_detached,
)
from s1decide.kernels import kernel_report

__all__ = [
    "TEACHERS",
    "EffortNotApplied",
    "effort_check",
    "load_teacher",
    "main",
    "pending_items",
    "rebuild_summary",
    "release_teacher",
    "require_quantization_support",
    "teacher_rows",
]

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
        # Measured, not assumed: results/teacher-quant-2026-09-23/gptoss-medium-1024.json
        quantization="mxfp4-experts+bf16",
        label="gptoss-low",
        effort="low",
        max_new_tokens=384,
    ),
    "gptoss-medium": TeacherSetting(
        model="openai/gpt-oss-20b",
        # Measured, not assumed: results/teacher-quant-2026-09-23/gptoss-medium-1024.json
        quantization="mxfp4-experts+bf16",
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
        # Measured, not assumed: results/teacher-quant-2026-09-23/gptoss-medium-1024.json
        quantization="mxfp4-experts+bf16",
        label="gptoss-low-1024",
        effort="low",
        max_new_tokens=1024,
    ),
    "gptoss-medium-1024": TeacherSetting(
        model="openai/gpt-oss-20b",
        # Measured, not assumed: results/teacher-quant-2026-09-23/gptoss-medium-1024.json
        quantization="mxfp4-experts+bf16",
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


def require_quantization_support(config: Any) -> None:
    """Refuse a checkpoint whose quantization this environment would silently undo.

    An MXFP4 checkpoint (gpt-oss) stays MXFP4 only when the `kernels` package is installed;
    without it transformers dequantizes to bf16 — for gpt-oss-20b ~42 GB, which then fails on a
    24 GB card with an OOM that says nothing about the cause. The completed teacher-2 runs had
    `kernels`; a teacher run without it would not be the same teacher.

    Raises:
        RuntimeError: For an MXFP4 checkpoint when `kernels` is not importable.
    """
    import importlib.util

    quant = getattr(config, "quantization_config", None)
    method = (
        quant.get("quant_method")
        if isinstance(quant, dict)
        else getattr(quant, "quant_method", None)
    )
    method = str(getattr(method, "value", method)) if method is not None else None
    if method == "mxfp4" and importlib.util.find_spec("kernels") is None:
        raise RuntimeError(
            "MXFP4 checkpoint but the `kernels` package is not installed: it would be "
            "dequantized to bf16 and would not be the teacher the dataset card describes. "
            "Install the pinned version with `uv sync --extra teachers`."
        )


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
    require_quantization_support(config)
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
    """Apply the chat template, passing the reasoning effort where the setting has one.

    No fallback: a template that rejects ``reasoning_effort`` used to be retried without it,
    silently, while the run's meta went on recording the effort. :func:`effort_check` proves the
    template reads the knob before a run starts; here a ``TypeError`` simply propagates.
    """
    messages = [{"role": "user", "content": prompt}]
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if setting.effort:
        kwargs["reasoning_effort"] = setting.effort
    return tokenizer.apply_chat_template(messages, **kwargs)


class EffortNotApplied(RuntimeError):
    """The chat template does not demonstrably apply the requested reasoning effort."""


def _changed_lines(a: str, b: str) -> list[str]:
    """Lines of ``b`` that are not in ``a`` — what the second render added."""
    import difflib

    return [
        line[1:]
        for line in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0)
        if line.startswith("+") and not line.startswith("+++")
    ]


def effort_check(tokenizer: Any, setting: TeacherSetting, prompt: str) -> dict[str, Any] | None:
    """Prove the chat template applies ``setting.effort``, and fingerprint how.

    Two renders are compared. **Requested vs a contrasting effort** (``low`` against ``high``)
    must differ, and the requested value must appear in what changed: that is the proof the
    template reads the knob at all. **Requested vs no effort** is recorded too, but an empty diff
    there is not a failure — gpt-oss documents ``reasoning_effort`` as "defaults to medium", so
    asking for medium renders identically to asking for nothing while being fully in effect.
    Qwen3.8 defaults to ``xhigh``, which is why the difference matters.

    Args:
        tokenizer: The teacher's tokenizer.
        setting: The teacher setting; nothing to check when it has no effort.
        prompt: A user prompt to render.

    Returns:
        ``requested``, ``contrast``, ``matches_template_default``, the SHA-256 of the lines the
        requested effort adds over the contrast and over the default, and of the full render.
        ``None`` when the setting has no effort.

    Raises:
        EffortNotApplied: If the template rejects the kwarg (``TypeError``), renders the same for
            two different efforts, or does not show the requested value in what changed.
    """
    import hashlib

    if not setting.effort:
        return None
    contrast = "high" if setting.effort != "high" else "low"
    messages = [{"role": "user", "content": prompt}]
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    try:
        requested = tokenizer.apply_chat_template(
            messages, reasoning_effort=setting.effort, **kwargs
        )
        contrasted = tokenizer.apply_chat_template(messages, reasoning_effort=contrast, **kwargs)
    except TypeError as exc:
        raise EffortNotApplied(
            f"{setting.model}: chat template rejects reasoning_effort ({type(exc).__name__}: "
            f"{exc}); effort={setting.effort} would not be applied"
        ) from exc
    default = tokenizer.apply_chat_template(messages, **kwargs)

    over_contrast = _changed_lines(contrasted, requested)
    if not over_contrast:
        raise EffortNotApplied(
            f"{setting.model}: effort={setting.effort} and effort={contrast} render identically; "
            "the template ignores the knob"
        )
    if not any(setting.effort in line for line in over_contrast):
        raise EffortNotApplied(
            f"{setting.model}: the render changed but does not mention effort={setting.effort}: "
            f"{over_contrast[:2]}"
        )
    over_default = _changed_lines(default, requested)

    def digest(lines: list[str] | str) -> str:
        text = "\n".join(lines) if isinstance(lines, list) else lines
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    return {
        "requested": setting.effort,
        "contrast": contrast,
        "matches_template_default": not over_default,
        "effort_lines": over_contrast,
        "diff_vs_contrast_sha256": digest(over_contrast),
        "diff_vs_default_sha256": digest(over_default) if over_default else None,
        "render_sha256": digest(requested),
    }


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


def release_teacher(*held: Any) -> int:
    """Give a teacher's weights back to the driver before the next one is loaded.

    The overnight chain runs both legs in one process. Leg 1 finished 6,000 rows and leg 2 died
    loading its model with "0 bytes is free ... 19.45 GiB is allocated by PyTorch, and 3.76 GiB
    is reserved by PyTorch but unallocated" — leg 1's 27B was still resident. Two separate
    reasons, and dropping either one is not enough:

    * the `work` closure keeps a reference to the model, so it stays alive after `run()`'s locals
      would otherwise have died, and a `del` of the caller's names frees nothing;
    * PyTorch's caching allocator does not return freed blocks to the driver on its own, so even
      once the references are gone the next `cudaMalloc` still fails.

    A failing leg costs a whole night here, and it fails *after* the previous leg has finished,
    which is the most expensive moment for it to happen.

    Args:
        held: Every object that might hold device memory — the model, the tokenizer, and the
            closure that captured them.

    Returns:
        Bytes still reserved afterwards, or 0 when there is no CUDA device. Returned rather than
        logged so a test can assert on it.
    """
    import gc

    for item in held:
        moved = False
        if hasattr(item, "to"):
            try:
                item.to("meta")
                moved = True
            except Exception as exc:  # a 4-bit model refuses; fall through to dropping storage
                print(
                    f"  release: {type(item).__name__}.to('meta') refused "
                    f"({type(exc).__name__}: {exc}); dropping parameter storage instead",
                    file=sys.stderr,
                )
        if not moved and hasattr(item, "parameters"):
            # A bitsandbytes 4-bit model refuses `.to()`, and the suppress above hid that: on
            # 2026-09-23 the 27B stayed resident and leg 2 died exactly as it had on 09-22.
            # Replacing each tensor's storage frees it even while references to the module live.
            _drop_storage(item)
    held = ()
    gc.collect()
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a declared dependency
        return 0
    if not torch.cuda.is_available():
        return 0
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return int(torch.cuda.memory_reserved())


def _drop_storage(module: Any) -> int:
    """Point every parameter and buffer of ``module`` at an empty CPU tensor, freeing its memory.

    Returns:
        Tensors that could not be emptied; each is also reported on stderr with its error type.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a declared dependency
        return 0
    empty = torch.empty(0)
    failed = 0
    for tensor in [*module.parameters(), *module.buffers()]:
        try:
            tensor.data = empty
        except Exception as exc:  # reported, and counted so a caller can see memory was kept
            failed += 1
            print(
                f"  release: could not empty a {tuple(tensor.shape)} tensor "
                f"({type(exc).__name__}: {exc})",
                file=sys.stderr,
            )
    return failed


def pending_items(directory: Path, items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Items not yet in the job's checkpoint — the ones a resume would still label.

    Args:
        directory: The job directory (may not exist yet).
        items: Everything the job was asked to label.

    Returns:
        The items whose id is not among the finished rows.
    """
    done = read_done_ids(JobPaths(directory)) if directory.is_dir() else set()
    return [item for item in items if item["id"] not in done]


def rebuild_summary(directory: Path, setting: TeacherSetting) -> dict[str, Any]:
    """Recompute a finished job's summary from its rows and its progress log.

    The progress log is appended once per batch and records elapsed time since that
    invocation started, so its last line carries the timing of the invocation that finished
    the job. Used when ``summary.json`` is missing or was overwritten.

    Args:
        directory: A job directory whose rows are complete.
        setting: The teacher that produced them.

    Returns:
        The summary, also written to ``summary.json``.
    """
    paths = JobPaths(directory)
    rows = [
        json.loads(line)
        for line in paths.rows.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    last = [
        json.loads(line)
        for line in paths.progress.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ][-1]
    elapsed = float(last["elapsed_seconds"])
    this_run = int(last["rows_this_run"])
    committed = [r for r in rows if r["level"] is not None]
    payload = {
        "run_id": directory.name,
        "finished": last["at"],
        "rows_total": int(last["total_rows"]),
        "rows_this_run": this_run,
        "rows_resumed": int(last["rows_done"]) - this_run,
        "elapsed_seconds": round(elapsed, 2),
        "setting": setting.to_json(),
        "committed": len(committed),
        "uncommitted": len(rows) - len(committed),
        "truncation_rate": sum(1 for r in rows if r["truncated"]) / max(1, len(rows)),
        "hit_cap_rate": sum(1 for r in rows if r["hit_cap"]) / max(1, len(rows)),
        "trace_tokens": trace_stats([r["trace_tokens"] for r in rows]),
        # Tokens generated in the finishing invocation over its elapsed time: rows resumed from
        # an earlier invocation are excluded, since their time is not in `elapsed`.
        "tokens_per_second": (
            sum(r["trace_tokens"] for r in rows[len(rows) - this_run :]) / elapsed
            if elapsed
            else None
        ),
        "rebuilt_from": "rows.jsonl + progress.jsonl",
    }
    write_summary(directory, payload)
    return payload


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
    if not pending_items(directory, items):
        # Nothing to label: load nothing — not a 27B, not even a tokenizer — and touch neither
        # meta, DONE nor the summary. The first relaunch of the eval chain did both, and rewrote
        # leg 1's summary with a zero elapsed time.
        existing = directory / "summary.json"
        if existing.is_file():
            return json.loads(existing.read_text(encoding="utf-8"))
        return rebuild_summary(directory, setting)
    if setting.effort:
        # Before anything is generated, prove the template applies the effort the meta records.
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(setting.model, local_files_only=True)
        meta["effort_check"] = effort_check(
            tokenizer, setting, build_prompt(items[0]["state"], items[0]["question"])
        )
    # Before the model load, not after: a refused resume should cost a second, not five minutes.
    check_resume_meta(directory, meta, force=force)

    # The model lives in this dict and nowhere else, so clearing it drops the last reference
    # before the caching allocator is emptied; `work` reads it through the dict rather than
    # capturing it.
    loaded: dict[str, Any] = {}
    loaded["model"], loaded["tokenizer"] = load_teacher(setting)

    def work(batch: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = teacher_rows(loaded["model"], loaded["tokenizer"], setting, batch)
        rows[-1]["progress"] = {
            "teacher": setting.label,
            "truncated": sum(1 for r in rows if r["truncated"]),
            "uncommitted": sum(1 for r in rows if r["level"] is None),
        }
        return rows

    try:
        summary = run_job(
            directory,
            list(items),
            work,
            key=lambda item: item["id"],
            batch_size=batch_size,
            meta=meta,
            force=force,
        )
    finally:
        # Popped straight into the call: the argument tuple is then the only reference left.
        release_teacher(*[loaded.pop(key) for key in list(loaded)])

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
