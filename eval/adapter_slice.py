"""Our own adapter on the comparison slices — raw, and with S2's temperature fitted on `val`.

The rows in the comparison table must be scored on the same slices by the same code. External
models go through `eval/external_laya.py` / `eval/external_http.py`; this module produces our
adapter's rows in the same format, so `python -m eval.external_laya report` scores all of them
identically.

``score`` (GPU): builds the model with the trainer's own loader (Unsloth 4-bit, the ADR 0002
adapter layout), loads the adapter weights, renders every slice row with the trainer's
`build_examples` and reads the masked logits with the batched `predict_logits` S1 evaluated
with. Its `val` output must reproduce S1's final checkpoint evaluation exactly — the same model,
slice and code path — and ``score`` checks that before anything else is written.

``calibrate`` (CPU): fits S2 on the `val` logits per (primitive, option-count bucket) with
`s1decide.calibrate.fit_calibration_cells` (ADR 0008) — temperature and vector scaling both
fitted in every cell, the method chosen per cell on held-out val folds — applies it to `test`,
and writes a second run directory. ``--method`` forces one method everywhere, so both can be
reported beside the chosen one.

    uv run python -m eval.adapter_slice score --adapter results/s1-3090/adapter \\
        --config train/configs/sft_3090_s1.yaml --out results/s1-3090-slices
    uv run --no-sync python -m eval.adapter_slice calibrate --raw results/s1-3090-slices \\
        --out results/s1-3090-slices-s2
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["main"]

LAYA_SLICES = Path("results/external-laya-2026-09-24")


def _softmax(logits: Sequence[float]) -> list[float]:
    top = max(logits)
    exps = [math.exp(v - top) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]


def _write_predictions(
    out: Path,
    split: str,
    rows: Sequence[dict[str, Any]],
    logits: Sequence[Sequence[float]],
    meta: dict[str, Any],
) -> None:
    with (out / f"predictions-{split}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row, values in zip(rows, logits, strict=True):
            handle.write(
                json.dumps(
                    {"id": row["id"], "logits": list(values), "probabilities": _softmax(values)}
                )
                + "\n"
            )
    (out / f"predictions-{split}.meta.json").write_text(
        json.dumps({**meta, "split": split, "rows": len(rows)}, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _score(
    adapter: Path | None,
    config_path: Path,
    out: Path,
    splits: Sequence[str],
    s1_run: Path,
    eval_batch_size: int = 4,
    slices: Path = LAYA_SLICES,
) -> None:
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from train.periodic_eval import base_rates, eval_metrics, predict_logits
    from train.sft_lora import _load_model, build_examples, load_config
    from transformers import AutoTokenizer

    from s1decide.kernels import kernel_report
    from s1decide.tasks import repo_root

    root = repo_root()
    config = load_config(config_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model, local_files_only=True)
    model, loader = _load_model(config)
    # With no adapter the freshly initialised LoRA is a no-op (its B matrices are zero): the
    # zero-shot model, through exactly the same loader and read path as the trained one.
    if adapter is not None:
        state = load_file(str(adapter / "adapter_model.safetensors"))
        result = set_peft_model_state_dict(model, state)
        if getattr(result, "unexpected_keys", None):
            raise ValueError(
                f"adapter weights the model does not have: {result.unexpected_keys[:5]}"
            )
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    rates = base_rates(_read(root / "data/processed/train.jsonl"))
    out.mkdir(parents=True, exist_ok=True)

    meta = {
        "model": config.model,
        "adapter": str(adapter) if adapter is not None else None,
        "config": str(config_path),
        "loader": loader,
        "quantization": "nf4-bf16",
        "kernels": kernel_report(),
        "calibration": "none (raw logits)",
        "eval_batch_size": eval_batch_size,
    }
    for split in splits:
        for suffix in (".jsonl", ".json"):
            shutil.copy(root / slices / f"slice-{split}{suffix}", out / f"slice-{split}{suffix}")
        rows = _read(out / f"slice-{split}.jsonl")
        stats: dict[str, int] = {"dropped_over_cap": 0}
        examples = build_examples(rows, tokenizer, config, stats)
        if stats["dropped_over_cap"] or len(examples) != len(rows):
            raise ValueError(
                f"{split}: {stats['dropped_over_cap']} rows over the cap; slices must be scored whole"
            )
        # Scored in build_examples' seeded order — the order S1's own evaluation used — because
        # the recurrent layers see left padding, so batch composition is part of the computation.
        # Only the outputs are put back in slice order.
        shuffled_logits = predict_logits(model, examples, pad, eval_batch_size)
        by_id = {ex["id"]: (ex, lg) for ex, lg in zip(examples, shuffled_logits, strict=True)}
        ordered = [by_id[row["id"]][0] for row in rows]
        logits = [by_id[row["id"]][1] for row in rows]
        if split == "val" and adapter is not None:
            # The reproduction check: S1's own final evaluation of the same slice.
            final = _read(s1_run / "evals.jsonl")[-1]["metrics"]
            ours = eval_metrics(ordered, logits, rates)
            for group, m in final.items():
                if abs(ours[group]["kl"] - m["kl"]) > 1e-3:
                    raise RuntimeError(
                        f"val does not reproduce S1's final evaluation: {group} KL "
                        f"{ours[group]['kl']:.5f} vs {m['kl']:.5f}"
                    )
            meta["reproduces_s1_final_eval"] = True
        _write_predictions(out, split, rows, logits, meta)
        print(f"wrote {split}: {len(rows)} rows", flush=True)
    meta["vram_peak_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    (out / "score_meta.json").write_text(
        json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
    )


def calibration_group(row: dict[str, Any]) -> str:
    """The S2 cell a slice row is calibrated in (ADR 0008): its primitive, stage 1 apart."""
    return "stage1" if row.get("stage") == 1 else str(row["qtype"])


def _calibrate(
    raw: Path, out: Path, method: str | None = None, exclude_families: Sequence[str] = ()
) -> None:
    from eval.metrics import Prediction
    from s1decide.calibrate import fit_calibration_cells

    val_rows = {r["id"]: r for r in _read(raw / "slice-val.jsonl")}
    val = [
        Prediction(
            id=p["id"],
            family=val_rows[p["id"]]["family"],
            qtype=calibration_group(val_rows[p["id"]]),
            logits=tuple(p["logits"]),
            answer_idx=int(val_rows[p["id"]]["answer_idx"]),
            weight=float(val_rows[p["id"]].get("eval_weight", 1.0)),
        )
        for p in _read(raw / "predictions-val.jsonl")
    ]
    cells = fit_calibration_cells(
        val,
        quantization="nf4-bf16",
        group_of=lambda p: p.qtype,
        exclude_families=exclude_families,
    )
    out.mkdir(parents=True, exist_ok=True)
    cells.save(out / "calibration.json")
    deployed = method or "chosen per (primitive, bucket) on val, out of sample"
    base_meta = json.loads((raw / "predictions-val.meta.json").read_text(encoding="utf-8"))
    for split in ("val", "test"):
        for suffix in (".jsonl", ".json"):
            shutil.copy(raw / f"slice-{split}{suffix}", out / f"slice-{split}{suffix}")
        rows = _read(raw / f"slice-{split}.jsonl")
        predictions = {p["id"]: p["logits"] for p in _read(raw / f"predictions-{split}.jsonl")}
        calibrated = [
            list(
                cells.get("nf4-bf16", calibration_group(r)).apply(
                    predictions[r["id"]], method=method
                )
            )
            for r in rows
        ]
        # `apply` returns probabilities; store their logs so downstream code sees logits.
        logits = [[math.log(max(p, 1e-12)) for p in probs] for probs in calibrated]
        meta = {
            **base_meta,
            "calibration": f"S2 per cell (ADR 0008), method: {deployed}; fitted on the val slice "
            "(calibration.json)",
            "calibration_excluded_families": sorted(exclude_families),
            "calibration_fit_on_this_split": split == "val",
        }
        _write_predictions(out, split, rows, logits, meta)
    print(f"wrote {out}")


def main(argv: Sequence[str] | None = None) -> int:
    """``score`` (GPU) or ``calibrate`` (CPU)."""
    parser = argparse.ArgumentParser(prog="eval.adapter_slice")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--adapter", default=None, help="omit to score the zero-shot model")
    s.add_argument("--eval-batch-size", type=int, default=4, help="4 by default since 2026-09-24")
    s.add_argument("--config", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--splits", nargs="+", default=["val", "test"])
    s.add_argument("--s1-run", default="results/s1-3090")
    s.add_argument(
        "--slices",
        default=str(LAYA_SLICES),
        help="directory holding slice-<split>.jsonl (the fair-ground slice: --splits eval)",
    )
    c = sub.add_parser("calibrate")
    c.add_argument("--raw", required=True)
    c.add_argument("--out", required=True)
    c.add_argument(
        "--method",
        choices=["temperature", "vector"],
        default=None,
        help="force one method in every cell; omit to deploy the per-cell choice (ADR 0008)",
    )
    c.add_argument("--exclude-families", nargs="*", default=[])
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.cmd == "score":
        _score(
            Path(args.adapter) if args.adapter else None,
            Path(args.config),
            Path(args.out),
            args.splits,
            Path(args.s1_run),
            args.eval_batch_size,
            Path(args.slices),
        )
    else:
        _calibrate(Path(args.raw), Path(args.out), args.method, args.exclude_families)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
