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

Early development. Only the project skeleton exists so far.

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
