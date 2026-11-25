"""Command line entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from docsage import __version__
from docsage.answer import answer_question
from docsage.chunking import (
    Chunker,
    FixedSizeChunker,
    RecursiveChunker,
    SentenceWindowChunker,
    chunk_documents,
)
from docsage.embeddings import load_embedder
from docsage.eval import evaluate, load_dataset
from docsage.llm import LLM, load_llm
from docsage.loaders import load_file
from docsage.workspace import Workspace

app = typer.Typer(help="Ask questions about your technical docs.", no_args_is_help=True)

_SUFFIXES = {".md", ".markdown", ".html", ".htm", ".pdf"}

Home = Annotated[
    Path,
    typer.Option(envvar="DOCSAGE_HOME", help="Directory that holds the index."),
]
EmbedderOpt = Annotated[str, typer.Option("--embedder", help="hashing or sentence-transformers.")]


@app.callback()
def main() -> None:
    """Keep subcommands addressable by name even while there is only one."""


def _chunker(name: str) -> Chunker:
    chunkers: dict[str, Chunker] = {
        "recursive": RecursiveChunker(),
        "fixed": FixedSizeChunker(),
        "sentence": SentenceWindowChunker(),
    }
    if name not in chunkers:
        raise typer.BadParameter(f"choose one of: {', '.join(chunkers)}", param_hint="--chunker")
    return chunkers[name]


def _make_llm(name: str, model: str | None) -> LLM:
    return load_llm(name, **({"model": model} if model else {}))


def _open(home: Path, embedder: str) -> Workspace:
    try:
        return Workspace.open(home, load_embedder(embedder))
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def ingest(
    paths: Annotated[list[Path], typer.Argument(help="Files or directories to index.")],
    home: Home = Path(".docsage"),
    chunker: Annotated[str, typer.Option(help="recursive, fixed or sentence.")] = "recursive",
    embedder: EmbedderOpt = "hashing",
) -> None:
    """Load, chunk and index documents. Re-ingesting a file replaces its old chunks."""
    files: list[Path] = []
    for p in paths:
        if not p.exists():
            typer.echo(f"error: {p} does not exist", err=True)
            raise typer.Exit(1)
        if p.is_dir():
            files.extend(sorted(f for f in p.rglob("*") if f.suffix.lower() in _SUFFIXES))
        else:
            files.append(p)
    ws = _open(home, embedder)
    strategy = _chunker(chunker)
    total = 0
    for f in files:
        try:
            docs = load_file(f)
        except ValueError as exc:
            typer.echo(f"skipping {f}: {exc}", err=True)
            continue
        chunks = chunk_documents(docs, strategy)
        ws.replace(docs[0].path, chunks)
        total += len(chunks)
    ws.save()
    stats = ws.stats()
    typer.echo(
        f"indexed {total} chunks from {len(files)} files "
        f"({stats['documents']} documents, {stats['chunks']} chunks in the index)"
    )


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Your question.")],
    home: Home = Path(".docsage"),
    k: Annotated[int, typer.Option(help="Passages to retrieve.")] = 5,
    llm: Annotated[str, typer.Option(help="openai, anthropic or ollama.")] = "ollama",
    model: Annotated[str | None, typer.Option(help="Model name for the provider.")] = None,
    embedder: EmbedderOpt = "hashing",
) -> None:
    """Answer a question from the indexed docs, with numbered citations."""
    ws = _open(home, embedder)
    hits = ws.retriever().search(question, k=k)
    answer = answer_question(_make_llm(llm, model), question, hits)
    typer.echo(answer.text)
    if answer.sources:
        typer.echo("\nSources:")
        for s in answer.sources:
            where = s.path + (f", p. {s.page}" if s.page else "")
            typer.echo(f"[{s.number}] {where}: {s.snippet}")


@app.command()
def stats(
    home: Home = Path(".docsage"),
    embedder: EmbedderOpt = "hashing",
) -> None:
    """Show how much is indexed."""
    s = _open(home, embedder).stats()
    typer.echo(f"documents: {s['documents']}")
    typer.echo(f"chunks: {s['chunks']}")


@app.command(name="eval")
def eval_command(
    dataset: Annotated[Path, typer.Argument(help="JSONL file of questions and relevant paths.")],
    home: Home = Path(".docsage"),
    k: Annotated[int, typer.Option(help="Cutoff for recall@k and nDCG@k.")] = 5,
    embedder: EmbedderOpt = "hashing",
) -> None:
    """Measure retrieval quality (recall@k, MRR, nDCG@k) against a labelled dataset."""
    try:
        examples = load_dataset(dataset)
    except (OSError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    report = evaluate(_open(home, embedder).retriever(), examples, k=k)
    typer.echo(f"questions: {report.questions}")
    typer.echo(f"recall@{k}: {report.recall:.3f}")
    typer.echo(f"mrr: {report.mrr:.3f}")
    typer.echo(f"ndcg@{k}: {report.ndcg:.3f}")


@app.command()
def version() -> None:
    """Print the installed docsage version."""
    typer.echo(__version__)
