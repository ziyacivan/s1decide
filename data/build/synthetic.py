"""The programmatic ordinal control set (ADR 0005, rule c).

Labels here are derived **by rule** from a structured state, with no model in the loop. That is
the whole point: it is the one Score set whose ground truth is known exactly, so it can tell
apart "the ordinal loss is broken" from "the synthetic labels are noisy". A teacher-labelled set
cannot play that role, because its own error rate is unknown.

It is a *control*, not a substitute for real ordinal data. It is easy by construction, and a
model that does well here and badly on teacher-labelled Score has learned the rule, not the task
— which is exactly the diagnosis we want to be able to make.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from typing import Any

__all__ = ["ORDINAL_RUBRICS", "generate_ordinal_control"]

#: Each rubric is a state generator plus a rule mapping the generated state to a level index.
#: Levels are ordered lowest to highest, matching `s1decide.primitives.Score`.
ORDINAL_RUBRICS: dict[str, dict[str, Any]] = {
    "ticket_backlog": {
        "instructions": "How severe is this support backlog?",
        "levels": ("negligible", "minor", "moderate", "serious", "critical"),
        "thresholds": (5, 25, 100, 400),
        "field": "open_tickets",
    },
    "response_delay": {
        "instructions": "How overdue is the first response on this ticket?",
        "levels": ("on time", "slightly late", "late", "very late"),
        "thresholds": (1, 8, 48),
        "field": "hours_since_opened",
    },
    "error_rate": {
        "instructions": "How healthy is this service?",
        "levels": ("healthy", "degraded", "failing"),
        "thresholds": (1, 10),
        "field": "error_percent",
    },
}


def _level_for(value: float, thresholds: tuple[int, ...]) -> int:
    """Index of the first band ``value`` falls into; the rule the whole set rests on."""
    level = 0
    for threshold in thresholds:
        if value >= threshold:
            level += 1
    return level


def generate_ordinal_control(count: int = 900, seed: int = 20260917) -> Iterator[dict[str, Any]]:
    """Generate rule-labelled ordinal questions over structured states.

    The state is JSON with several fields, only one of which the rubric reads. The distractor
    fields matter: without them the task is a lookup, and a model could score well by reading
    the only number present.

    Args:
        count: How many questions to generate.
        seed: Seed; the same seed gives byte-identical output.

    Yields:
        Normalised question dicts, ready for the pipeline.
    """
    rng = random.Random(seed)
    names = sorted(ORDINAL_RUBRICS)
    for i in range(count):
        key = names[i % len(names)]
        rubric = ORDINAL_RUBRICS[key]
        thresholds = rubric["thresholds"]
        # Spread values across the bands rather than uniformly, so every level is populated.
        band = rng.randrange(len(thresholds) + 1)
        low = 0 if band == 0 else thresholds[band - 1]
        high = thresholds[band] if band < len(thresholds) else thresholds[-1] * 4
        value = rng.randint(int(low), max(int(low), int(high) - 1))

        state = {
            rubric["field"]: value,
            "team": rng.choice(["billing", "platform", "identity", "payments"]),
            "region": rng.choice(["eu-west", "us-east", "ap-south"]),
            "on_call": rng.choice([True, False]),
            "unrelated_count": rng.randint(0, 500),
        }
        yield {
            "state": json.dumps(state, sort_keys=True, indent=2),
            "instructions": rubric["instructions"],
            "options": tuple(rubric["levels"]),
            "answer_idx": _level_for(value, thresholds),
            "rubric": key,
        }
