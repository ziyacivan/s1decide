"""A minimal llama.cpp server client — the seed of the GGUF engine.

`llama-server` gives us two things the transformers path does not: GGUF Q4_K_M weights, and
**continuous batching**, which keeps the GPU fed across many concurrent requests instead of
waiting for the slowest sequence in a static batch. For generation-heavy work — the teacher run
— that is the whole ballgame.

It now carries both halves. The *generation* half drove the teacher-speed experiment. The
*scoring* half reads masked option probabilities out of ``/completion``'s ``n_probs`` without
generating anything beyond one token, which is what `decide()` needs and what the GGUF rows of
the quantization table are measured with.

The scoring half exists to be compared against `engine/hf.py`, so it has to reproduce that path's
rule exactly: take the distribution at the answer position, keep only the allowed option tokens,
renormalise over them. Two details decide whether it does.

``post_sampling_probs`` must be false. With it true the server reports the distribution *after*
the sampler has run, and at temperature 0 that is a point mass on the argmax — every question
would come back with confidence 1.0 and the calibration numbers would be meaningless while
looking plausible.

``n_probs`` returns only the top N tokens. An option outside that window is not a small
probability, it is *absent*, and renormalising over the ones that happened to be present would
silently invent a distribution. That raises :class:`OptionOutsideTopN` instead.

Nothing here imports torch. The server owns the GPU; this process only talks to it.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DEFAULT_N_PROBS",
    "LlamaServerClient",
    "OptionOutsideTopN",
    "ServerBusy",
    "TokenizerMismatch",
]

#: How many top tokens to ask ``/completion`` for when scoring options.
#:
#: Generous on purpose. The cost is a slightly larger JSON response, once per question; the cost
#: of it being too small is :class:`OptionOutsideTopN` and a re-run.
#:
#: 40 is not arbitrary. A `Noul` renders as ``Answer (yes/no):`` and the model puts ``yes`` and
#: ``no`` in the top five — comfortable. But a throwaway prompt ending in ``Answer:`` put the
#: option tokens at ranks 12 and 14, behind ``\n\n``, ``Does`` and ``Is``. How far down an option
#: sits depends on how hard the prompt commits the model to answering, so the window has to
#: survive a prompt that commits less well than the current format does.
DEFAULT_N_PROBS = 40


class OptionOutsideTopN(RuntimeError):
    """An option's token was not among the top ``n_probs`` the server returned.

    Not recoverable by renormalising over the options that *were* present: that would report a
    distribution the model did not produce, and it would look entirely reasonable. Raise, widen
    ``n_probs``, and run again.
    """


class TokenizerMismatch(RuntimeError):
    """The server and the reference tokenizer disagree about an option label.

    Fatal for a quantization comparison specifically. If ``A`` is one token id here and a
    different one there, the two runtimes are being asked different questions, and the
    disagreement that comes out is a tokenizer artifact wearing the costume of a quantization
    effect — which is the exact thing the comparison exists to measure.
    """


class ServerBusy(RuntimeError):
    """The server refused a request because every slot is occupied.

    Worth its own type: with ``--parallel N`` the server rejects the N+1th request rather than
    queueing it, and retrying is correct where failing is not.
    """


@dataclass
class LlamaServerClient:
    """Talk to a running ``llama-server``.

    Attributes:
        base_url: Where the server is listening.
        timeout: Per-request timeout in seconds. Generous: a 1,024-token completion under
            contention from fifteen siblings is not fast.
        parallel: How many requests to keep in flight, matching the server's ``--parallel``.
    """

    base_url: str = "http://127.0.0.1:8080"
    timeout: float = 600.0
    parallel: int = 16

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code in (503, 429):
                raise ServerBusy(f"{exc.code} from {path}") from exc
            raise

    def health(self) -> bool:
        """Whether the server is up and has finished loading."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=10) as response:
                return json.loads(response.read()).get("status") == "ok"
        except OSError:
            return False

    def wait_until_ready(self, seconds: float = 600.0) -> bool:
        """Block until the model is loaded, or give up.

        Loading a 16 GB GGUF takes tens of seconds; polling beats guessing at a sleep.
        """
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.health():
                return True
            time.sleep(2.0)
        return False

    def complete(self, prompt: str, *, max_tokens: int, temperature: float = 0.0) -> dict[str, Any]:
        """One completion, greedy by default.

        Args:
            prompt: Already chat-templated text.
            max_tokens: Generation cap.
            temperature: 0.0 for greedy, matching the transformers path's ``do_sample=False``
                so the two runtimes are compared on the same decoding rule.

        Returns:
            ``{"text", "tokens", "stopped_by_limit"}``.
        """
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "cache_prompt": True,
            "stream": False,
        }
        result = self._post("/completion", payload)
        timings = result.get("timings", {})
        return {
            "text": result.get("content", ""),
            "tokens": int(timings.get("predicted_n", 0)),
            "stopped_by_limit": bool(result.get("stopped_limit", False)),
        }

    def complete_many(
        self, prompts: list[str], *, max_tokens: int, temperature: float = 0.0
    ) -> list[dict[str, Any]]:
        """Run prompts concurrently so continuous batching has something to batch.

        Sequential requests would leave the server with one sequence at a time, which is the
        configuration continuous batching exists to avoid — and would measure the wrong thing.

        Args:
            prompts: Chat-templated prompts.
            max_tokens: Generation cap per prompt.
            temperature: Sampling temperature.

        Returns:
            One result per prompt, in the input order.
        """
        with ThreadPoolExecutor(max_workers=self.parallel) as pool:
            futures = [
                pool.submit(self.complete, p, max_tokens=max_tokens, temperature=temperature)
                for p in prompts
            ]
            return [f.result() for f in futures]

    def tokenize(self, text: str) -> list[int]:
        """Token ids the *server's* tokenizer gives for a string.

        Args:
            text: The text to tokenize.

        Returns:
            Token ids, without BOS.
        """
        return [int(t) for t in self._post("/tokenize", {"content": text}).get("tokens", [])]

    def check_label_tokens(
        self, labels: Sequence[str], reference: Mapping[str, int] | None = None
    ) -> dict[str, int]:
        """Confirm each option label is one token here, and the same one the reference used.

        Call this once before a scoring run, not per question. It is the cheap guard against the
        most expensive way a quantization comparison can go wrong: measuring a tokenizer
        difference and reporting it as a quantization effect.

        Args:
            labels: Option labels, e.g. ``("A", "B")``.
            reference: Optional label to token id mapping from the transformers tokenizer, as
                `s1decide.tokens.allowed_token_ids` produces.

        Returns:
            Label to token id, according to the server.

        Raises:
            TokenizerMismatch: If a label is not exactly one token, or disagrees with
                ``reference``.
        """
        ids: dict[str, int] = {}
        for label in labels:
            tokens = self.tokenize(label)
            if len(tokens) != 1:
                raise TokenizerMismatch(
                    f"the server tokenizes {label!r} as {len(tokens)} tokens ({tokens}); "
                    "option labels must be single tokens on both sides"
                )
            ids[label] = tokens[0]
        if reference:
            differing = {
                label: (ids[label], reference[label])
                for label in labels
                if label in reference and ids[label] != reference[label]
            }
            if differing:
                raise TokenizerMismatch(
                    "server and reference tokenizers disagree, "
                    f"label to (server, reference) = {differing}"
                )
        return ids

    @staticmethod
    def _probability_entries(result: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Pull the per-token probability list out of a completion response.

        llama.cpp has moved this field more than once, so the shape is probed rather than
        assumed. An unrecognised response raises instead of returning an empty list, because
        downstream an empty list is indistinguishable from "no option was in the top N".
        """
        for key in ("completion_probabilities", "probs"):
            entries = result.get(key)
            if isinstance(entries, list) and entries:
                first = entries[0]
                if isinstance(first, dict):
                    for inner in ("probs", "top_logprobs", "top_probs"):
                        if isinstance(first.get(inner), list):
                            return list(first[inner])
                    if "prob" in first or "logprob" in first:
                        return list(entries)
        raise RuntimeError(
            "no token probabilities in the completion response; "
            f"keys were {sorted(result)} - is n_probs set and post_sampling_probs false?"
        )

    def score_options(
        self,
        prompt: str,
        labels: Sequence[str],
        *,
        token_ids: Mapping[str, int] | None = None,
        n_probs: int = DEFAULT_N_PROBS,
    ) -> tuple[float, ...]:
        """Probabilities over the option labels at the answer position.

        The same rule as `engine/hf.py`: read the distribution once, keep the allowed option
        tokens, renormalise over them. One token is generated because the server has no way to
        report a distribution without stepping, and it is discarded.

        Args:
            prompt: Already chat-templated text, ending where the answer goes.
            labels: Option labels in index order.
            token_ids: Label to token id from :meth:`check_label_tokens`. Matching on ids is
                exact; without them the match falls back to token strings, which is looser
                because a leading space makes a different token with the same visible text.
            n_probs: How many top tokens to request.

        Returns:
            One probability per label, in the order given, summing to 1.

        Raises:
            OptionOutsideTopN: If any label is missing from the returned top tokens.
            ValueError: If fewer than two labels are given.
        """
        if len(labels) < 2:
            raise ValueError(f"scoring needs at least 2 options, got {len(labels)}")
        result = self._post(
            "/completion",
            {
                "prompt": prompt,
                "n_predict": 1,
                "temperature": 0.0,
                "n_probs": n_probs,
                # False, or the server reports the post-sampler distribution, which at
                # temperature 0 is a point mass and would make every answer look certain.
                "post_sampling_probs": False,
                "cache_prompt": True,
                "stream": False,
            },
        )
        found: dict[str, float] = {}
        for entry in self._probability_entries(result):
            probability = entry.get("prob")
            if probability is None and entry.get("logprob") is not None:
                probability = math.exp(float(entry["logprob"]))
            if probability is None:
                continue
            for label in labels:
                if label in found:
                    continue
                if token_ids is not None and "id" in entry:
                    if int(entry["id"]) == token_ids[label]:
                        found[label] = float(probability)
                elif str(entry.get("tok_str", entry.get("token", ""))).strip() == label:
                    found[label] = float(probability)
        missing = [label for label in labels if label not in found]
        if missing:
            raise OptionOutsideTopN(
                f"{missing} not in the top {n_probs} tokens; raise n_probs and re-run. "
                "Renormalising over the options that were present would report a distribution "
                "the model did not produce."
            )
        total = sum(found[label] for label in labels)
        if total <= 0.0:
            raise OptionOutsideTopN(f"every option had zero probability: {found}")
        return tuple(found[label] / total for label in labels)

    def score_many(
        self,
        prompts: Sequence[str],
        labels: Sequence[Sequence[str]],
        *,
        token_ids: Mapping[str, int] | None = None,
        n_probs: int = DEFAULT_N_PROBS,
        concurrent: bool = False,
    ) -> list[tuple[float, ...]]:
        """Score many questions in input order, sequentially by default.

        **Sequential is the default here and concurrent is the default for generation, and the
        difference is measured rather than cautious.** Scoring the same question ten times on
        this server: sequentially the answer is bit-identical every time, spread 0.00000;
        concurrently at ``--parallel 4`` the same question spreads 0.02662 and its mean moves by
        0.0219. Continuous batching makes a result depend on which other sequences happened to
        share its batch.

        Two and a half points of probability is noise larger than the effects this scoring path
        exists to measure — the quantization comparison is looking for whether 4-bit moves an
        answer at all. Run it concurrently and the finding would be llama.cpp's scheduler.

        Pass ``concurrent=True`` only for throughput work where the numbers are not the product.

        Args:
            prompts: One chat-templated prompt per question.
            labels: One label set per prompt.
            token_ids: As :meth:`score_options`.
            n_probs: As :meth:`score_options`.
            concurrent: Trade reproducibility for speed.

        Returns:
            One probability tuple per prompt.

        Raises:
            ValueError: If the two sequences are different lengths. Zipping to the shorter one
                would quietly drop questions from a comparison and change its result.
        """
        if len(prompts) != len(labels):
            raise ValueError(f"{len(prompts)} prompts but {len(labels)} label sets")
        if not concurrent:
            return [
                self.score_options(prompt, row, token_ids=token_ids, n_probs=n_probs)
                for prompt, row in zip(prompts, labels, strict=True)
            ]
        with ThreadPoolExecutor(max_workers=self.parallel) as pool:
            futures = [
                pool.submit(self.score_options, prompt, row, token_ids=token_ids, n_probs=n_probs)
                for prompt, row in zip(prompts, labels, strict=True)
            ]
            return [f.result() for f in futures]
