"""The training row and the inference call must be the same question.

A prompt that differs by one token trains the model for something it will never be asked, and
nothing at training time notices: the loss goes down, the eval goes through a different renderer,
and the gap only shows as a model that is mysteriously worse than its training curve. The
stage-1 case is pinned in `test_data_build.py`; this covers `Score`, including the soft-target
rows, and the option tokens the loss actually indexes.
"""

from __future__ import annotations

import json

import pytest
from train.sft_lora import TrainConfig, build_examples, collate

from s1decide.tasks import repo_root


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            "unsloth/Qwen3.8-27B-unsloth-bnb-4bit", local_files_only=True
        )
    except Exception:  # pragma: no cover - the model may not be cached
        pytest.skip("tokenizer not cached locally")


def score_row(target: list[float] | None = None) -> dict:
    row = {
        "qtype": "score",
        "state": "The customer was charged twice and has written three times about it.",
        "instructions": "How strongly does this express frustration?",
        "options": ["none", "slight", "moderate", "strong", "decisive"],
        "answer_idx": 3,
    }
    if target is not None:
        row["target"] = target
        row["target_type"] = "soft" if sum(1 for v in target if v) > 1 else "hard"
    return row


def test_a_score_training_row_renders_byte_identically_to_inference(tokenizer) -> None:
    from s1decide.primitives import Question, Score
    from s1decide.prompt import render

    row = score_row()
    example = build_examples([row], tokenizer, TrainConfig())[0]

    rendered = render(
        row["state"],
        [
            Question(
                name="q", spec=Score(instructions=row["instructions"], levels=tuple(row["options"]))
            )
        ],
    )
    expected = rendered.prefix + rendered.suffixes[0]
    assert tokenizer.decode(example["input_ids"]) == expected


def test_the_option_tokens_the_loss_indexes_are_the_ones_inference_masks_to(tokenizer) -> None:
    """The loss gathers `label_token_ids` out of the answer-position logits, and the engine masks
    to `allowed_token_ids`. If those ever differ, training optimises tokens nobody reads."""
    from s1decide.primitives import Question, Score
    from s1decide.prompt import render
    from s1decide.tokens import allowed_token_ids

    row = score_row()
    example = build_examples([row], tokenizer, TrainConfig())[0]
    rendered = render(
        row["state"],
        [
            Question(
                name="q", spec=Score(instructions=row["instructions"], levels=tuple(row["options"]))
            )
        ],
    )
    assert example["label_token_ids"] == list(allowed_token_ids(tokenizer, rendered.labels[0]))


def test_a_soft_target_survives_into_the_example(tokenizer) -> None:
    soft = [0.0, 0.0, 0.5, 0.5, 0.0]
    example = build_examples([score_row(soft)], tokenizer, TrainConfig())[0]
    assert example["target"] == soft
    assert len(example["target"]) == len(example["label_token_ids"])


def test_a_row_without_a_target_becomes_one_hot_on_its_answer(tokenizer) -> None:
    """Every other family has no `target` field; they must not silently train on zeros."""
    example = build_examples([score_row()], tokenizer, TrainConfig())[0]
    assert example["target"] == [0.0, 0.0, 0.0, 1.0, 0.0]


def test_an_over_long_prompt_is_skipped_not_truncated(tokenizer) -> None:
    """Truncating moves the answer position, which is the one position the model is read at."""
    row = score_row()
    row["state"] = "word " * 4000
    assert build_examples([row], tokenizer, TrainConfig(max_seq_len=1024)) == []


def test_the_real_corpus_renders(tokenizer) -> None:
    """Against committed rows rather than a fixture, so a schema drift shows up here."""
    path = repo_root() / "data" / "processed" / "train.jsonl"
    if not path.is_file():
        pytest.skip("no built corpus")
    rows = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["family"] == "score_teacher":
            rows.append(row)
        if len(rows) == 40:
            break
    examples = build_examples(rows, tokenizer, TrainConfig())
    assert len(examples) == len(rows)
    for example in examples:
        assert sum(example["target"]) == pytest.approx(1.0)
        assert len(example["target"]) == len(example["label_token_ids"]) == 5


# --- batching ---------------------------------------------------------------------


def test_padding_is_on_the_left_so_the_answer_is_always_the_last_column() -> None:
    """The loss reads `logits[:, -1, :]`. With right padding that is a pad token for every row
    but the longest, and the model would be trained on the wrong position."""
    batch = collate(
        [
            {"input_ids": [1, 2, 3], "label_token_ids": [9], "target": [1.0], "qtype": "noul"},
            {"input_ids": [4, 5], "label_token_ids": [9], "target": [1.0], "qtype": "noul"},
        ],
        pad_token_id=0,
    )
    assert batch["input_ids"][0].tolist() == [1, 2, 3]
    assert batch["input_ids"][1].tolist() == [0, 4, 5]
    assert batch["attention_mask"][1].tolist() == [0, 1, 1]
    assert batch["input_ids"][0][-1] == 3 and batch["input_ids"][1][-1] == 5


# --- what the smoke run actually exercises -------------------------------------------


def corpus_rows() -> list[dict]:
    path = repo_root() / "data" / "processed" / "train.jsonl"
    if not path.is_file():
        pytest.skip("no built corpus")
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()
    ]


def test_the_sample_covers_every_primitive() -> None:
    """A smoke run's job is to fail when the pipeline is wrong, and it cannot do that for a code
    path it never enters.

    The first version took the head of the file, which is ordered by family: 200 banking77 rows,
    159 with the same answer, zero `Score`. The ordinal loss and the soft targets — the two newest
    and least-exercised things in the repo — were never touched while the loss fell to 4e-7.
    """
    from train.sft_lora import select_rows

    picked = select_rows(corpus_rows(), TrainConfig(limit=200))
    qtypes = {row["qtype"] for row in picked[:800]}
    assert {"choice", "score", "noul"} <= qtypes, f"missing primitives: {qtypes}"


def test_the_sample_includes_soft_target_rows() -> None:
    from train.sft_lora import select_rows

    picked = select_rows(corpus_rows(), TrainConfig(limit=200))
    assert sum(1 for row in picked if row.get("target_type") == "soft") > 0
    assert sum(1 for row in picked if row.get("target_type") == "hard") > 0


def test_the_sample_is_not_one_family() -> None:
    from train.sft_lora import select_rows

    picked = select_rows(corpus_rows(), TrainConfig(limit=200))
    assert len({row["family"] for row in picked}) > 3


def test_the_sample_is_deterministic() -> None:
    from train.sft_lora import select_rows

    rows = corpus_rows()
    a = [r["id"] for r in select_rows(rows, TrainConfig(limit=100))]
    b = [r["id"] for r in select_rows(rows, TrainConfig(limit=100))]
    assert a == b


def test_no_limit_means_the_whole_corpus() -> None:
    from train.sft_lora import select_rows

    rows = corpus_rows()
    assert len(select_rows(rows, TrainConfig(limit=None))) == len(rows)


def test_coverage_counts_every_dimension_a_smoke_report_needs() -> None:
    from train.sft_lora import coverage, format_coverage

    examples = [
        {"qtype": "choice", "family": "banking77", "n_options": 8, "label_token_ids": [0] * 8},
        {"qtype": "noul", "family": "go_emotions", "n_options": 2, "label_token_ids": [0, 1]},
        {
            "qtype": "noul",
            "family": "banking77",
            "stage1": True,
            "n_options": 2,
            "label_token_ids": [0, 1],
        },
        {
            "qtype": "score",
            "family": "teacher",
            "soft": True,
            "n_options": 5,
            "label_token_ids": [0] * 5,
        },
    ]
    cov = coverage(examples)
    assert cov["by_qtype"] == {"choice": 1, "noul": 1, "noul/stage1": 1, "score": 1}
    assert cov["by_family"] == {"banking77": 2, "go_emotions": 1, "teacher": 1}
    assert cov["by_target"] == {"hard": 3, "soft": 1}
    # Numeric order, not string order: "10" must not sort before "2".
    assert list(cov["by_option_count"]) == ["2", "5", "8"]
    table = format_coverage(cov)
    assert "| noul/stage1 | 1 | 25.0% |" in table
    assert "| soft | 1 | 25.0% |" in table


def test_a_prequantized_checkpoint_is_not_quantized_again() -> None:
    """The first 27B attempt died on `inner dim (5120) does not match weight (1)` for this."""
    from types import SimpleNamespace

    from train.sft_lora import is_prequantized

    assert is_prequantized(SimpleNamespace(quantization_config={"quant_method": "bitsandbytes"}))
    assert not is_prequantized(SimpleNamespace(quantization_config=None))
    assert not is_prequantized(SimpleNamespace())


def test_every_run_writes_a_report_with_coverage_beside_the_loss(tmp_path) -> None:
    from train.sft_lora import coverage, write_report

    seen = [
        {
            "qtype": "score",
            "family": "score_teacher",
            "soft": True,
            "n_options": 5,
            "label_token_ids": [0] * 5,
        },
        {"qtype": "noul", "family": "go_emotions", "n_options": 2, "label_token_ids": [0, 1]},
    ]
    summary = {
        "run_id": "unit",
        "examples": 2,
        "rows_seen": 2,
        "steps": 2,
        "loss_rows": [
            {
                "qtype": "score",
                "soft": True,
                "total": 1.2,
                "kl": 0.3,
                "cross_entropy": 1.0,
                "distance": 0.66,
                "distance_excess": 0.16,
                "floor": 0.843,
            },
            {
                "qtype": "noul",
                "soft": False,
                "total": 0.4,
                "kl": 0.4,
                "cross_entropy": 0.4,
                "distance": 0.0,
                "distance_excess": 0.0,
                "floor": 0.0,
            },
        ],
        "loss_by_qtype": {
            q: {
                "rows": 1,
                "soft_rows": int(q == "score"),
                **{key: {"mean_first_10": v, "mean_last_10": v} for key, v in values.items()},
            }
            for q, values in {
                "noul": {"kl": 0.4, "distance_excess": 0.0, "total": 0.4, "floor": 0.0},
                "score": {"kl": 0.3, "distance_excess": 0.16, "total": 1.2, "floor": 0.843},
            }.items()
        },
        "coverage": coverage(seen),
    }
    report = write_report(tmp_path, summary).read_text(encoding="utf-8")
    assert "## Coverage (rows seen)" in report
    assert "| score_teacher | 1 | 50.0% |" in report
    assert "| soft | 1 | 50.0% |" in report
    # KL first, then the raw total and its floor, so the two can be read against each other.
    assert "| score | 1 | 1 | 0.3000 | 0.3000 | 1.2000 | 1.2000 | 0.8430 | 0.8430 |" in report
    assert "KL(target" in report
    if "loss.png" in report and "not drawn" not in report:
        assert (tmp_path / "loss.png").is_file()


def test_a_row_over_the_cap_is_dropped_and_counted(tokenizer) -> None:
    stats: dict[str, int] = {}
    long_row = {**score_row(), "state": "word " * 3000}
    assert build_examples([long_row], tokenizer, TrainConfig(max_seq_len=1024), stats) == []
    assert stats == {"dropped_over_cap": 1}
