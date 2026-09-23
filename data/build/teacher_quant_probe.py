"""Measure how a teacher's weights were actually held on the GPU — from the loaded model.

A run's ``setting.quantization`` is a label (the setting's default), not a measurement, and the
dataset card once disagreed with it. This loads the teacher exactly as a teacher run does
(`data.build.teach_run.load_teacher`) and records what came back: VRAM allocated after the load,
the class of every linear layer that holds weights, the checkpoint's own quantization config and
the dtypes of the parameters.

    uv run python -m data.build.teacher_quant_probe gptoss-medium-1024 \\
        --out results/teacher-quant-2026-09-23
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from typing import Any

__all__ = ["classify_quantization", "main"]


def classify_quantization(linear_classes: dict[str, int], checkpoint_quant: str | None) -> str:
    """Name the representation the linear weights were held in, from what was loaded.

    Args:
        linear_classes: Class name to count, over modules holding the big weight matrices.
        checkpoint_quant: The checkpoint's ``quantization_config.quant_method``, if any.

    Returns:
        ``bnb-nf4`` when bitsandbytes 4-bit layers are present, ``mxfp4`` when MXFP4 layers are,
        ``bf16/unquantized`` when only plain linear layers remain, else ``unverified``.
    """
    names = " ".join(linear_classes).lower()
    if "linear4bit" in names:
        return "bnb-nf4"
    if "mxfp4" in names:
        return "mxfp4"
    if linear_classes and all(n in {"Linear"} for n in linear_classes):
        return (
            "bf16/unquantized"
            if checkpoint_quant is None
            else f"dequantized from {checkpoint_quant}"
        )
    return "unverified"


def main(argv: Sequence[str] | None = None) -> int:
    """Load one teacher and write ``<label>.json`` with what its weights were."""
    import torch
    from data.build.teach_run import TEACHERS, load_teacher, release_teacher

    from s1decide.tasks import repo_root

    parser = argparse.ArgumentParser(prog="data.build.teacher_quant_probe")
    parser.add_argument("teacher", choices=sorted(TEACHERS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    setting = TEACHERS[args.teacher]
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    model, tokenizer = load_teacher(setting)
    allocated = torch.cuda.memory_allocated() - before

    classes: Counter[str] = Counter()
    for _, module in model.named_modules():
        name = type(module).__name__
        lowered = name.lower()
        holds_linear_weight = getattr(module, "weight", None) is not None and "linear" in lowered
        # MoE expert blocks hold their weights in custom buffers, not in `.weight`.
        is_expert_block = "mxfp4" in lowered or "experts" in lowered
        if holds_linear_weight or is_expert_block:
            classes[name] += 1
    quant_cfg = getattr(model.config, "quantization_config", None)
    method = None
    if quant_cfg is not None:
        method = (
            quant_cfg.get("quant_method")
            if isinstance(quant_cfg, dict)
            else getattr(quant_cfg, "quant_method", None)
        )
        method = str(getattr(method, "value", method)) if method is not None else None
    dtypes = Counter(str(p.dtype) for p in model.parameters())
    payload: dict[str, Any] = {
        "teacher": args.teacher,
        "model": setting.model,
        "label_in_setting": setting.quantization,
        "allocated_after_load_gib": round(allocated / 1024**3, 3),
        "linear_classes": dict(classes),
        "checkpoint_quant_method": method,
        "parameter_dtypes": dict(dtypes),
        "measured_quantization": classify_quantization(dict(classes), method),
    }
    release_teacher(model, tokenizer)

    out = repo_root() / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.teacher}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
