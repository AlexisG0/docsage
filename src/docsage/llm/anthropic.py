"""Anthropic backend. Install with ``pip install docsage[anthropic]``."""

from __future__ import annotations

import os
from typing import Any

from docsage.llm.base import Message


class AnthropicLLM:
    def __init__(self, model: str = "claude-haiku-5-5", api_key: str | None = None) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic is not installed; run `pip install docsage[anthropic]`"
            ) from exc
        self._client: Any = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._model = model

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        kwargs: dict[str, Any] = {}
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(
            model=self._model,
            messages=turns,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
