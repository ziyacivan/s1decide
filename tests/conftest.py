"""Shared fixtures.

The tokenizer fixture deliberately loads from the local Hugging Face cache only. Tests
that need it skip when it is absent, so a fresh clone can run the whole CPU suite with
no network and no weights.
"""

from __future__ import annotations

from typing import Any

import pytest

from s1decide.primitives import Choice, Noul, Question, Score

BASE_MODEL = "Qwen/Qwen3.8-27B"


@pytest.fixture(scope="session")
def tokenizer() -> Any:
    """The base-model tokenizer, from the local cache.

    Skips the test when transformers is missing or the tokenizer has not been
    downloaded. Fetch it with roughly 22 MB and no weights::

        huggingface_hub.snapshot_download(
            "Qwen/Qwen3.8-27B",
            allow_patterns=["tokenizer*", "vocab.json", "merges.txt", "chat_template.jinja"],
        )
    """
    transformers = pytest.importorskip("transformers", reason="transformers not installed")
    try:
        return transformers.AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    except Exception as exc:
        pytest.skip(f"{BASE_MODEL} tokenizer not in the local cache: {type(exc).__name__}")


@pytest.fixture
def tone() -> Question:
    """A three-option Choice."""
    return Question(
        name="tone",
        spec=Choice(
            instructions="What is the customer's tone?",
            options=("calm", "frustrated", "angry"),
        ),
    )


@pytest.fixture
def urgency() -> Question:
    """A three-level Score."""
    return Question(
        name="urgency",
        spec=Score(
            instructions="How urgent is this ticket?",
            levels=("can wait", "this week", "today"),
        ),
    )


@pytest.fixture
def billing() -> Question:
    """A Noul."""
    return Question(name="billing", spec=Noul(instructions="This ticket is about billing."))


@pytest.fixture
def ticket_state() -> str:
    """A short free-text state."""
    return "I was charged twice for my subscription this month. Please fix this ASAP."
