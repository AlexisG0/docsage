"""FastAPI service: ``/ingest``, ``/query`` and ``/health`` on top of a :class:`Workspace`.

Documents are uploaded as text in the request body, so the server never reads arbitrary paths
from its own disk. Run it with ``uvicorn docsage.api:create_app --factory``.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from docsage import __version__
from docsage.answer import Source, answer_question
from docsage.chunking import (
    Chunker,
    FixedSizeChunker,
    RecursiveChunker,
    SentenceWindowChunker,
    chunk_documents,
)
from docsage.embeddings import Embedder, load_embedder
from docsage.llm import LLM, load_llm
from docsage.loaders import load_file
from docsage.workspace import Workspace, content_hash

_TEXT_SUFFIXES = {".md", ".markdown", ".html", ".htm"}
_CHUNKERS: dict[str, Callable[[], Chunker]] = {
    "recursive": RecursiveChunker,
    "fixed": FixedSizeChunker,
    "sentence": SentenceWindowChunker,
}


class DocumentIn(BaseModel):
    path: str = Field(
        min_length=1,
        description="Logical path of the document. Its extension (.md or .html) picks the loader.",
    )
    content: str = Field(description="Full text of the document.")


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1)
    chunker: str = Field(default="recursive", description="recursive, fixed or sentence.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "documents": [
                        {
                            "path": "guide/install.md",
                            "content": "# Install\n\nRun pip install docsage to install it.\n",
                        }
                    ],
                    "chunker": "recursive",
                }
            ]
        }
    }


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    total_documents: int
    total_chunks: int
    unchanged: int = 0


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=50, description="Passages to retrieve.")

    model_config = {
        "json_schema_extra": {"examples": [{"question": "How do I install docsage?", "k": 5}]}
    }


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "Run pip install docsage [1].",
                    "sources": [
                        {
                            "number": 1,
                            "path": "guide/install.md",
                            "title": "Install",
                            "page": None,
                            "snippet": "Run pip install docsage to install it.",
                        }
                    ],
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    status: str
    version: str
    documents: int
    chunks: int


def create_app(
    workspace: Workspace | None = None,
    llm: LLM | Callable[[], LLM] | None = None,
) -> FastAPI:
    """Build the app. Without arguments it is configured from the environment.

    ``DOCSAGE_HOME`` (default ``.docsage``), ``DOCSAGE_EMBEDDER`` (default ``hashing``) and
    ``DOCSAGE_LLM`` (default ``ollama``) with an optional ``DOCSAGE_LLM_MODEL``.
    """
    if workspace is None:
        embedder: Embedder = load_embedder(os.environ.get("DOCSAGE_EMBEDDER", "hashing"))
        workspace = Workspace.open(os.environ.get("DOCSAGE_HOME", ".docsage"), embedder)
    ws = workspace

    def get_llm() -> LLM:
        if llm is None:
            model = os.environ.get("DOCSAGE_LLM_MODEL")
            name = os.environ.get("DOCSAGE_LLM", "ollama")
            return load_llm(name, **({"model": model} if model else {}))
        return llm() if callable(llm) else llm

    app = FastAPI(
        title="docsage",
        version=__version__,
        description="Question answering over technical docs with hybrid retrieval and citations.",
    )

    @app.get("/health", response_model=HealthResponse, summary="Liveness and index size")
    def health() -> HealthResponse:
        stats = ws.stats()
        return HealthResponse(
            status="ok",
            version=__version__,
            documents=stats["documents"],
            chunks=stats["chunks"],
        )

    @app.post("/ingest", response_model=IngestResponse, summary="Index documents")
    def ingest(req: IngestRequest) -> IngestResponse:
        if req.chunker not in _CHUNKERS:
            raise HTTPException(422, f"unknown chunker {req.chunker!r}; use {list(_CHUNKERS)}")
        for d in req.documents:
            if PurePosixPath(d.path).suffix.lower() not in _TEXT_SUFFIXES:
                raise HTTPException(422, f"unsupported file type for {d.path!r}; use .md or .html")
        strategy = _CHUNKERS[req.chunker]()
        total = unchanged = 0
        with tempfile.TemporaryDirectory() as tmp:
            for i, d in enumerate(req.documents):
                digest = content_hash(d.content, req.chunker)
                if ws.is_current(d.path, digest):
                    unchanged += 1
                    continue
                src = Path(tmp) / f"{i}{PurePosixPath(d.path).suffix.lower()}"
                src.write_text(d.content, encoding="utf-8")
                docs = [doc.model_copy(update={"path": d.path}) for doc in load_file(src)]
                if docs and docs[0].title == src.stem:
                    docs[0] = docs[0].model_copy(update={"title": PurePosixPath(d.path).stem})
                chunks = chunk_documents(docs, strategy)
                ws.replace(d.path, chunks, digest)
                total += len(chunks)
        ws.save()
        stats = ws.stats()
        return IngestResponse(
            documents=len(req.documents),
            chunks=total,
            total_documents=stats["documents"],
            total_chunks=stats["chunks"],
            unchanged=unchanged,
        )

    @app.post("/query", response_model=QueryResponse, summary="Ask a question")
    def query(req: QueryRequest) -> QueryResponse:
        hits = ws.retriever().search(req.question, k=req.k)
        result = answer_question(get_llm(), req.question, hits)
        return QueryResponse(answer=result.text, sources=result.sources)

    return app
