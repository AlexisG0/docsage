"""OpenAI backend. Install with ``pip install docsage[openai]``."""

from __future__ import annotations

import os
from typing import Any

from docsage.llm.base import Message


class OpenAILLM:
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai is not installed; run `pip install docsage[openai]`") from exc
        self._client: Any = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._model = model

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return str(resp.choices[0].message.content or "")
