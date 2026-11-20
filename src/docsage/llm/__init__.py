"""LLM providers."""

from __future__ import annotations

from typing import Any

from docsage.llm.base import LLM, Message
from docsage.llm.fake import ScriptedLLM

__all__ = ["LLM", "Message", "ScriptedLLM", "load_llm"]


def load_llm(name: str, **kwargs: Any) -> LLM:
    """Build an LLM by name: ``openai``, ``anthropic`` or ``ollama``."""
    if name == "openai":
        from docsage.llm.openai import OpenAILLM

        return OpenAILLM(**kwargs)
    if name == "anthropic":
        from docsage.llm.anthropic import AnthropicLLM

        return AnthropicLLM(**kwargs)
    if name == "ollama":
        from docsage.llm.ollama import OllamaLLM

        return OllamaLLM(**kwargs)
    raise ValueError(f"unknown llm {name!r}; choose 'openai', 'anthropic' or 'ollama'")
