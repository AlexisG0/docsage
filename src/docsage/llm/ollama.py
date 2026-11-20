"""Ollama backend over its local HTTP API; needs no extra dependency."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from docsage.llm.base import Message


class OllamaLLM:
    def __init__(
        self, model: str = "llama3.1", host: str | None = None, timeout: float = 120.0
    ) -> None:
        self._model = model
        self._host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self._timeout = timeout

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self._host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            data: dict[str, Any] = json.load(resp)
        return data

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        data = self._post(
            {
                "model": self._model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }
        )
        return str(data["message"]["content"])
