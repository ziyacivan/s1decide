"""A minimal llama.cpp server client — the seed of the GGUF engine.

`llama-server` gives us two things the transformers path does not: GGUF Q4_K_M weights, and
**continuous batching**, which keeps the GPU fed across many concurrent requests instead of
waiting for the slowest sequence in a static batch. For generation-heavy work — the teacher run
— that is the whole ballgame.

This is deliberately the *generation* client only. `engine/llamacpp.py` will need the
`/completion` endpoint's ``n_probs`` to read masked option logits for `decide()`, which is a
different problem: the project's inference path has no decode loop, and this one is all decode.
Kept here so the HTTP plumbing, the health check and the concurrency handling are already
written and tested when that lands.

Nothing here imports torch. The server owns the GPU; this process only talks to it.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

__all__ = ["LlamaServerClient", "ServerBusy"]


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
