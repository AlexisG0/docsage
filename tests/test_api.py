from pathlib import Path

from fastapi.testclient import TestClient

from docsage.api import create_app
from docsage.embeddings import HashingEmbedder
from docsage.llm import ScriptedLLM
from docsage.workspace import Workspace

DOCS = {
    "documents": [
        {"path": "guide/install.md", "content": "# Install\n\nRun pip install docsage.\n"},
        {"path": "guide/config.md", "content": "# Config\n\nSet DOCSAGE_HOME for the index.\n"},
    ]
}


def _client(tmp_path: Path, replies: list[str] | None = None) -> tuple[TestClient, ScriptedLLM]:
    fake = ScriptedLLM(replies or ["Run pip install docsage [1]."])
    ws = Workspace.open(tmp_path / "idx", HashingEmbedder())
    return TestClient(create_app(ws, fake)), fake


def test_ingest_skips_unchanged_documents(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    first = client.post("/ingest", json=DOCS).json()
    second = client.post("/ingest", json=DOCS).json()
    assert first["unchanged"] == 0
    assert second["unchanged"] == 2 and second["chunks"] == 0
    assert second["total_chunks"] == first["total_chunks"]


def test_health_reports_empty_index(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["documents"] == 0


def test_ingest_then_query_returns_cited_sources(tmp_path: Path) -> None:
    client, fake = _client(tmp_path)
    r = client.post("/ingest", json=DOCS)
    assert r.status_code == 200, r.text
    assert r.json()["total_documents"] == 2

    r = client.post("/query", json={"question": "how to install docsage?", "k": 2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == "Run pip install docsage [1]."
    assert body["sources"][0]["number"] == 1
    assert body["sources"][0]["path"] in {"guide/install.md", "guide/config.md"}
    assert len(fake.calls) == 1


def test_reingest_replaces_and_persists(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    client.post("/ingest", json=DOCS)
    r = client.post("/ingest", json=DOCS)
    assert r.json()["total_chunks"] == 2
    reopened = Workspace.open(tmp_path / "idx", HashingEmbedder())
    assert reopened.stats() == {"documents": 2, "chunks": 2}


def test_ingest_rejects_unsupported_type_and_chunker(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    bad = {"documents": [{"path": "x.exe", "content": "nope"}]}
    assert client.post("/ingest", json=bad).status_code == 422
    bad_chunker = {**DOCS, "chunker": "magic"}
    assert client.post("/ingest", json=bad_chunker).status_code == 422


def test_query_validates_input(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    assert client.post("/query", json={"question": ""}).status_code == 422
    assert client.post("/query", json={"question": "x", "k": 0}).status_code == 422


def test_query_on_empty_index_does_not_call_llm(tmp_path: Path) -> None:
    client, fake = _client(tmp_path)
    body = client.post("/query", json={"question": "anything"}).json()
    assert body["sources"] == []
    assert fake.calls == []


def test_openapi_documents_examples(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    schema = client.get("/openapi.json").json()
    assert set(schema["paths"]) == {"/health", "/ingest", "/query"}
    assert schema["components"]["schemas"]["IngestRequest"]["examples"]
