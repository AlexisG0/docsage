# docsage

Question answering over technical documentation (Markdown, HTML, PDF) with a focus on
retrieval quality you can measure.

## Goals

- **Hybrid retrieval**: BM25 plus dense embeddings, merged with reciprocal rank fusion.
- **Optional reranking** with a cross-encoder.
- **Grounded answers** with inline citations and the source snippets they point to.
- **Pluggable providers** (OpenAI, Anthropic, Ollama, sentence-transformers) behind small
  interfaces, with deterministic fakes so tests never touch the network.
- **A small evaluation suite** (recall@k, MRR, nDCG, faithfulness) to compare chunking and
  retrieval settings.
- Ships as a Python library, a CLI and a FastAPI service.

## Status

Early development. The project skeleton and the document loaders exist so far.

## Loading documents

Loaders return pydantic `Document` models that keep the source metadata needed for
citations (`path`, `title`, `headings`, and `page` for PDFs).

```python
from docsage.loaders import load_file

for doc in load_file("docs/guide.md"):  # .md, .html/.htm and .pdf are supported
    print(doc.title, doc.headings, doc.page)
```

Markdown and HTML produce one document per file; PDFs produce one per non-empty page.

## Chunking

Three strategies turn a `Document` into `Chunk` models that keep the source metadata
and character offsets (`start`/`end`) into the document text.

```python
from docsage.chunking import FixedSizeChunker, RecursiveChunker, SentenceWindowChunker
from docsage.loaders import load_file

doc = load_file("docs/guide.md")[0]
chunks = RecursiveChunker(max_chars=800).chunk(doc)  # splits on headings, keeps the heading path
print(chunks[0].headings, chunks[0].text)

FixedSizeChunker(chunk_size=800, overlap=100).chunk(doc)  # sliding window
SentenceWindowChunker(window=2).chunk(doc)  # one chunk per sentence, `context` has neighbours
```

## Development

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run docsage version
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

Configuration comes from environment variables; see `.env.example`.

## License

MIT
