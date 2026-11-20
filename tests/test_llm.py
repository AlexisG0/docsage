import io
import json
import sys
from typing import Any

import pytest

from docsage.llm import LLM, Message, ScriptedLLM, load_llm
from docsage.llm.ollama import OllamaLLM


def test_scripted_llm_replays_and_records() -> None:
    llm = ScriptedLLM(["one", "two"])
    assert isinstance(llm, LLM)
    msgs = [Message("user", "hi")]
    assert llm.complete(msgs) == "one"
    assert llm.complete(msgs) == "two"
    assert llm.calls == [msgs, msgs]


def test_scripted_llm_runs_out() -> None:
    with pytest.raises(RuntimeError):
        ScriptedLLM([]).complete([Message("user", "hi")])


def test_load_llm_unknown() -> None:
    with pytest.raises(ValueError):
        load_llm("nope")


def test_ollama_builds_request_and_parses_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float) -> io.BytesIO:
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data)
        return io.BytesIO(json.dumps({"message": {"content": "pong"}}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    llm = OllamaLLM(model="m", host="http://h:1/")
    out = llm.complete([Message("system", "s"), Message("user", "ping")], max_tokens=5)
    assert out == "pong"
    assert seen["url"] == "http://h:1/api/chat"
    assert seen["body"]["stream"] is False
    assert seen["body"]["options"]["num_predict"] == 5
    assert seen["body"]["messages"][0] == {"role": "system", "content": "s"}


def test_ollama_host_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://box:9")
    assert OllamaLLM()._host == "http://box:9"


@pytest.mark.parametrize("name", ["openai", "anthropic"])
def test_sdk_backends_explain_missing_extra(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError, match=f"docsage\\[{name}\\]"):
        load_llm(name)
