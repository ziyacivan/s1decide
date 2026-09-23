"""Reading masked option probabilities out of llama-server.

This is the half of the GGUF engine that `decide()` needs and the quantization table is measured
with. It has to reproduce `engine/hf.py`'s rule exactly — distribution at the answer position,
keep the allowed option tokens, renormalise — because the whole point of having it is to compare
the two runtimes on the same question.

The tests that matter here are the ones covering the ways it can be wrong *and look right*: a
point-mass distribution from the post-sampler path, an option missing from the top N, and a
tokenizer that disagrees with the reference.
"""

from __future__ import annotations

import pytest

from s1decide.engine.llamacpp_client import (
    DEFAULT_N_PROBS,
    LlamaServerClient,
    OptionOutsideTopN,
    TokenizerMismatch,
)


class FakeServer(LlamaServerClient):
    """A client whose HTTP layer is a canned response, so the parsing can be tested on CPU."""

    def __init__(self, response: dict, tokens: dict[str, list[int]] | None = None) -> None:
        super().__init__()
        self.response = response
        self.tokens = tokens or {}
        self.posted: list[tuple[str, dict]] = []

    def _post(self, path: str, payload: dict) -> dict:
        self.posted.append((path, payload))
        if path == "/tokenize":
            return {"tokens": self.tokens.get(payload["content"], [1])}
        return self.response


def probs(entries: list[dict]) -> dict:
    return {"completion_probabilities": [{"content": "A", "probs": entries}]}


# --- the request itself ----------------------------------------------------------


def test_scoring_asks_for_the_pre_sampler_distribution() -> None:
    """The single most dangerous default in this file.

    With `post_sampling_probs` true the server reports the distribution *after* sampling, which
    at temperature 0 is a point mass on the argmax. Every question would come back at confidence
    1.0, every calibration number would be garbage, and nothing would look broken.
    """
    server = FakeServer(probs([{"tok_str": "A", "prob": 0.7}, {"tok_str": "B", "prob": 0.3}]))
    server.score_options("prompt", ["A", "B"])
    _, payload = server.posted[0]
    assert payload["post_sampling_probs"] is False
    assert payload["temperature"] == 0.0
    assert payload["n_predict"] == 1, "scoring must not generate an answer, only read one"
    assert payload["n_probs"] == DEFAULT_N_PROBS


def test_scoring_renormalises_over_the_options_only() -> None:
    """The mask half of the rule: probability mass outside the option set is not ours."""
    server = FakeServer(
        probs(
            [
                {"tok_str": "the", "prob": 0.5},
                {"tok_str": "A", "prob": 0.3},
                {"tok_str": "B", "prob": 0.1},
            ]
        )
    )
    a, b = server.score_options("prompt", ["A", "B"])
    assert a == pytest.approx(0.75) and b == pytest.approx(0.25)
    assert a + b == pytest.approx(1.0)


def test_logprobs_are_accepted_as_well_as_probs() -> None:
    """llama.cpp reports one or the other depending on version and endpoint."""
    import math

    server = FakeServer(
        probs(
            [
                {"tok_str": "A", "logprob": math.log(0.8)},
                {"tok_str": "B", "logprob": math.log(0.2)},
            ]
        )
    )
    a, b = server.score_options("prompt", ["A", "B"])
    assert a == pytest.approx(0.8) and b == pytest.approx(0.2)


# --- the ways it goes silently wrong ---------------------------------------------


def test_an_option_outside_the_top_n_raises_instead_of_renormalising() -> None:
    """`B` absent is not `B` improbable.

    Renormalising over what happened to be present would report a confident, plausible, invented
    distribution — the worst kind of wrong for a calibration measurement.
    """
    server = FakeServer(probs([{"tok_str": "A", "prob": 0.9}, {"tok_str": "the", "prob": 0.1}]))
    with pytest.raises(OptionOutsideTopN, match="B"):
        server.score_options("prompt", ["A", "B"])


def test_an_unrecognised_response_shape_raises_rather_than_scoring_nothing() -> None:
    """An empty probability list is indistinguishable downstream from "no option was in top N"."""
    with pytest.raises(RuntimeError, match="n_probs"):
        FakeServer({"content": "A"}).score_options("prompt", ["A", "B"])


def test_all_zero_probabilities_do_not_divide_by_zero() -> None:
    server = FakeServer(probs([{"tok_str": "A", "prob": 0.0}, {"tok_str": "B", "prob": 0.0}]))
    with pytest.raises(OptionOutsideTopN):
        server.score_options("prompt", ["A", "B"])


def test_matching_on_token_id_beats_matching_on_text() -> None:
    """A leading space makes a different token with the same visible text; ids are unambiguous."""
    server = FakeServer(
        probs(
            [
                {"id": 32, "tok_str": " A", "prob": 0.6},
                {"id": 64, "tok_str": "A", "prob": 0.3},
                {"id": 65, "tok_str": "B", "prob": 0.1},
            ]
        )
    )
    a, _ = server.score_options("prompt", ["A", "B"], token_ids={"A": 64, "B": 65})
    assert a == pytest.approx(0.75), "matched the space-prefixed token instead of the real one"


# --- tokenizer agreement ---------------------------------------------------------


def test_a_label_that_is_not_one_token_on_the_server_is_fatal() -> None:
    server = FakeServer({}, tokens={"A": [12, 34]})
    with pytest.raises(TokenizerMismatch, match="2 tokens"):
        server.check_label_tokens(["A"])


def test_a_label_the_two_tokenizers_disagree_about_is_fatal() -> None:
    """The failure this guard exists for: a tokenizer difference reported as a quantization one.

    If `A` is token 64 in transformers and token 99 on the server, the two runtimes are being
    asked different questions and every disagreement downstream is an artifact.
    """
    server = FakeServer({}, tokens={"A": [99], "B": [65]})
    with pytest.raises(TokenizerMismatch, match="disagree"):
        server.check_label_tokens(["A", "B"], reference={"A": 64, "B": 65})


def test_agreeing_tokenizers_return_the_mapping() -> None:
    server = FakeServer({}, tokens={"A": [64], "B": [65]})
    assert server.check_label_tokens(["A", "B"], reference={"A": 64, "B": 65}) == {"A": 64, "B": 65}


# --- batching ---------------------------------------------------------------------


def test_mismatched_prompt_and_label_counts_raise() -> None:
    """Zipping to the shorter one would drop questions from a comparison and change its answer."""
    server = FakeServer(probs([{"tok_str": "A", "prob": 0.5}, {"tok_str": "B", "prob": 0.5}]))
    with pytest.raises(ValueError, match="2 prompts but 1 label"):
        server.score_many(["one", "two"], [["A", "B"]])


def test_scoring_needs_at_least_two_options() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        FakeServer({}).score_options("prompt", ["A"])


def test_score_many_preserves_input_order() -> None:
    server = FakeServer(probs([{"tok_str": "A", "prob": 0.6}, {"tok_str": "B", "prob": 0.4}]))
    out = server.score_many(["one", "two", "three"], [["A", "B"]] * 3)
    assert len(out) == 3
    assert all(row[0] == pytest.approx(0.6) for row in out)


def test_scoring_many_is_sequential_by_default() -> None:
    """Measured, not cautious: concurrency changes the answers.

    Scoring one question ten times on a live server, sequentially the result is bit-identical
    (spread 0.00000); concurrently at --parallel 4 it spreads 0.02662 and its mean shifts 0.0219,
    because continuous batching makes a result depend on which sequences shared its batch. That
    is larger than the effects this path exists to measure.
    """
    calls: list[str] = []

    class Recording(FakeServer):
        def score_options(self, prompt, labels, **kwargs):
            import threading

            calls.append(threading.current_thread().name)
            return super().score_options(prompt, labels, **kwargs)

    server = Recording(probs([{"tok_str": "A", "prob": 0.6}, {"tok_str": "B", "prob": 0.4}]))
    server.score_many(["one", "two", "three"], [["A", "B"]] * 3)
    assert len(set(calls)) == 1, "default scoring ran on more than one thread"
