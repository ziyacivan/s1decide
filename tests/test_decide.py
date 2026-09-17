"""The public API, and the fuzz test that proves type safety by construction."""

from __future__ import annotations

import random
import string

import pytest

from s1decide.decide import Response, decide, softmax
from s1decide.engine.base import EngineError, EngineOutput
from s1decide.engine.mock import MockEngine
from s1decide.primitives import Choice, Noul, Question, Result, Score
from s1decide.prompt import FORMAT_VERSION
from s1decide.tokens import MAX_SINGLE_TOKEN_OPTIONS

# --- softmax -----------------------------------------------------------------


def test_softmax_sums_to_one_and_preserves_order() -> None:
    probs = softmax([1.0, 2.0, 3.0])
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] < probs[1] < probs[2]


def test_softmax_is_shift_invariant_and_stable_for_huge_logits() -> None:
    assert softmax([1000.0, 1001.0]) == pytest.approx(softmax([0.0, 1.0]))
    assert all(p == p for p in softmax([1e308, -1e308]))  # no NaN


def test_softmax_temperature_flattens_and_sharpens() -> None:
    hot = softmax([0.0, 2.0], temperature=10.0)
    cold = softmax([0.0, 2.0], temperature=0.1)
    assert max(hot) < max(softmax([0.0, 2.0])) < max(cold)


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_softmax_rejects_non_positive_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        softmax([0.0, 1.0], temperature=temperature)


def test_softmax_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        softmax([])


# --- decide ------------------------------------------------------------------


def test_decide_returns_one_validated_result_per_question(
    ticket_state, tone, urgency, billing
) -> None:
    response = decide(ticket_state, [tone, urgency, billing], engine=MockEngine())
    assert isinstance(response, Response)
    assert len(response) == 3
    assert [r.name for r in response] == ["tone", "urgency", "billing"]
    assert response.engine == "mock"
    assert response.format_version == FORMAT_VERSION
    assert response.temperature == 1.0
    for result in response:
        assert isinstance(result, Result)
        assert sum(result.probabilities) == pytest.approx(1.0)


def test_decide_exposes_typed_views(ticket_state, tone, urgency, billing) -> None:
    response = decide(ticket_state, [tone, urgency, billing], engine=MockEngine())
    assert response.choices["tone"].choice in tone.labels
    assert response.scores["urgency"].score in urgency.labels
    assert 0.0 <= response.nouls["billing"].noul <= 1.0
    assert 0.0 <= response.scores["urgency"].expected_level <= 2.0
    assert response["tone"] is response.results[0]
    with pytest.raises(KeyError):
        response["missing"]


def test_decide_accepts_a_name_to_spec_mapping(ticket_state) -> None:
    response = decide(
        ticket_state,
        {
            "tone": Choice("What is the tone?", ("calm", "angry")),
            "billing": Noul("This is about billing."),
        },
        engine=MockEngine(),
    )
    assert [r.name for r in response] == ["tone", "billing"]


def test_decide_is_deterministic(ticket_state, tone, urgency) -> None:
    a = decide(ticket_state, [tone, urgency], engine=MockEngine(seed=1))
    b = decide(ticket_state, [tone, urgency], engine=MockEngine(seed=1))
    assert a.results == b.results


def test_decide_results_are_independent_of_sibling_questions(ticket_state, tone, urgency) -> None:
    """Answering in isolation: adding a question must not change another question's answer."""
    alone = decide(ticket_state, [tone], engine=MockEngine())["tone"]
    together = decide(ticket_state, [tone, urgency], engine=MockEngine())["tone"]
    assert alone == together


def test_decide_temperature_changes_confidence_not_choice(ticket_state, tone) -> None:
    sharp = decide(ticket_state, [tone], engine=MockEngine(), temperature=0.2)["tone"]
    flat = decide(ticket_state, [tone], engine=MockEngine(), temperature=5.0)["tone"]
    assert sharp.choice == flat.choice
    assert sharp.confidence > flat.confidence


def test_decide_rejects_non_positive_temperature(ticket_state, tone) -> None:
    with pytest.raises(ValueError, match="temperature"):
        decide(ticket_state, [tone], engine=MockEngine(), temperature=0.0)


def test_decide_rejects_duplicate_names(ticket_state, tone) -> None:
    with pytest.raises(ValueError, match="unique"):
        decide(ticket_state, [tone, tone], engine=MockEngine())


def test_decide_rejects_no_questions(ticket_state) -> None:
    with pytest.raises(ValueError, match="at least one"):
        decide(ticket_state, [], engine=MockEngine())


class _WrongShapeEngine:
    name = "broken"

    def score(self, prefix, suffixes, labels):
        return EngineOutput(logits=tuple((0.0,) for _ in suffixes))


class _NaNEngine:
    name = "nan"

    def score(self, prefix, suffixes, labels):
        return EngineOutput(logits=tuple(tuple(float("nan") for _ in row) for row in labels))


def test_decide_refuses_an_engine_that_answers_the_wrong_shape(ticket_state, tone) -> None:
    with pytest.raises(EngineError, match="logits for"):
        decide(ticket_state, [tone], engine=_WrongShapeEngine())


def test_decide_refuses_non_finite_logits(ticket_state, tone) -> None:
    with pytest.raises(EngineError, match="non-finite"):
        decide(ticket_state, [tone], engine=_NaNEngine())


# --- fuzz: 10k random schemas, zero off-schema outputs -----------------------

ALPHABET = string.ascii_letters + string.digits + "çğıöşüéèàñ-_ /"


def _label(rng: random.Random) -> str:
    return "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 12))).strip() or "x"


def _unique_labels(rng: random.Random, count: int) -> tuple[str, ...]:
    seen: dict[str, str] = {}
    while len(seen) < count:
        label = _label(rng)
        if label and label == label.strip() and label.casefold() not in seen:
            seen[label.casefold()] = label
    return tuple(seen.values())


def random_question(rng: random.Random, index: int) -> Question:
    """One random Choice, Score or Noul with a random, valid option set."""
    kind = rng.choice(("choice", "score", "noul"))
    instructions = f"Q{index}: " + " ".join(_label(rng) for _ in range(rng.randint(1, 8)))
    if kind == "choice":
        spec = Choice(instructions, _unique_labels(rng, rng.randint(2, MAX_SINGLE_TOKEN_OPTIONS)))
    elif kind == "score":
        spec = Score(instructions, _unique_labels(rng, rng.randint(2, 9)))
    else:
        spec = Noul(instructions)
    return Question(name=f"q{index}_{_label(rng).replace(' ', '_')}", spec=spec)


def random_state(rng: random.Random) -> str | dict[str, object]:
    if rng.random() < 0.5:
        return " ".join(_label(rng) for _ in range(rng.randint(1, 40)))
    return {
        f"k{i}": rng.choice([_label(rng), rng.randint(0, 10**6), rng.random() < 0.5])
        for i in range(rng.randint(1, 6))
    }


def check_on_schema(question: Question, result: Result) -> None:
    """Every guarantee the API makes, asserted for one question."""
    assert result.name == question.name
    assert result.qtype == question.qtype
    assert result.options == question.labels
    assert len(result.probabilities) == len(question.labels)
    assert all(0.0 <= p <= 1.0 for p in result.probabilities)
    assert abs(sum(result.probabilities) - 1.0) < 1e-6
    assert 0.0 < result.confidence <= 1.0
    if question.qtype == "choice":
        assert result.choice in question.labels
    elif question.qtype == "score":
        assert result.score in question.labels
        assert 0.0 <= result.expected_level <= len(question.labels) - 1
    else:
        assert 0.0 <= result.noul <= 1.0
        assert result.options == ("no", "yes")


@pytest.mark.slow
def test_fuzz_ten_thousand_schemas_never_leave_the_schema() -> None:
    rng = random.Random(20260917)
    engine = MockEngine(seed=7)
    schemas = 0
    questions_seen = 0
    while schemas < 10_000:
        questions = [random_question(rng, i) for i in range(rng.randint(1, 6))]
        response = decide(random_state(rng), questions, engine=engine)
        assert len(response) == len(questions)
        for question, result in zip(questions, response):
            check_on_schema(question, result)
        schemas += 1
        questions_seen += len(questions)
    assert schemas == 10_000
    assert questions_seen > 10_000
