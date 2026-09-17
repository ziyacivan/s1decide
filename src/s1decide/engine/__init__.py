"""Inference engines. All engines implement the same protocol and pass the same contract tests.

`hf` needs torch and is imported lazily; `mock` has no dependencies.
"""

from s1decide.engine.base import Engine, EngineError, EngineOutput
from s1decide.engine.mock import MockEngine

__all__ = ["Engine", "EngineError", "EngineOutput", "MockEngine"]
