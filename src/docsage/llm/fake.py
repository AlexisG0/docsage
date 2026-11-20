"""Scripted LLM for tests: replays canned replies and records every call."""

from __future__ import annotations

from docsage.llm.base import Message


class ScriptedLLM:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        self.calls.append(list(messages))
        if not self._replies:
            raise RuntimeError("ScriptedLLM has no replies left")
        return self._replies.pop(0)
