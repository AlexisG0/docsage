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

Early development. What exists today: the project skeleton, a CLI (`ingest`, `ask`, `stats`),
a FastAPI service (`/ingest`, `/query`, `/health`),
document loaders (Markdown, HTML, PDF), three chunking strategies, an embedding interface,
vector stores, a BM25 index, hybrid retrieval with reciprocal rank fusion and optional
reranking, LLM provider interfaces, grounded answer generation and retrieval evaluation. The LanceDB store is not yet tested against a real install; the in-memory
store is what the examples below and the test suite exercise.

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

Only `RecursiveChunker` fills in `headings` per chunk; the other two leave it empty.
`chunk_documents(docs, chunker)` chunks a list of documents (for example all pages of a
PDF) in one call, and chunk `index` restarts for each document. `title` and `page` are
copied from the document by every strategy.

## Embeddings

Embedders implement a two-member `Embedder` protocol (`dimension` and `embed(texts)`) and
return L2-normalised vectors. `HashingEmbedder` is deterministic and needs no model, which
makes it the default for tests and offline demos; the sentence-transformers backend is an
optional extra (`pip install docsage[local]`).

```python
from docsage.embeddings import cosine, embed_query, load_embedder

embedder = load_embedder("hashing", dimension=256)  # or "sentence-transformers"
q = embed_query(embedder, "how do I configure logging?")
d = embedder.embed(["Configure logging in the app"])[0]
print(cosine(q, d))
```

## Vector store

Two stores implement the `VectorStore` protocol:

- `InMemoryVectorStore`: pure Python, nothing persisted. Used in tests and for small corpora.
- `LanceVectorStore`: chunks and vectors in an embedded LanceDB table on disk. It needs the
  optional extra: `uv sync --extra lance` (or `pip install "docsage[lance]"`).

Chunks are identified by `path`, `page` and `index`, so upserting the same document again
replaces its chunks. Searches use cosine similarity and can be restricted with equality
filters on `path`, `title`, `page` and `index`. The tests run against both stores and skip
the LanceDB ones when the extra is not installed.

```python
from docsage.embeddings import HashingEmbedder
from docsage.loaders import load_file
from docsage.chunking import RecursiveChunker
from docsage.vectorstore import LanceVectorStore

embedder = HashingEmbedder()
store = LanceVectorStore(".docsage/index", embedder.dimension)

chunks = RecursiveChunker(max_chars=800).chunk(load_file("docs/guide.md")[0])
store.upsert(chunks, embedder.embed([c.text for c in chunks]))

for hit in store.search(
    embedder.embed(["how do I install it?"])[0], k=3, filters={"path": "docs/guide.md"}
):
    print(round(hit.score, 3), hit.chunk.headings, hit.chunk.text[:60])

store.delete(path="docs/guide.md")  # remove every chunk of a document
```

## Keyword search

`BM25Index` is a pure-Python Okapi BM25 index over chunk text. It uses the same chunk ids and
metadata filters as the vector stores, and is saved to and loaded from a single JSON file.
Exact identifiers such as `max_chars` are tokenized as one term, which is where keyword search
beats embeddings.

```python
from docsage.keyword import BM25Index

index = BM25Index()
index.add(chunks)  # the chunks from the example above
index.save(".docsage/bm25.json")

index = BM25Index.load(".docsage/bm25.json")
for hit in index.search("max_chars", k=3):
    print(round(hit.score, 2), hit.id)

index.delete(path="docs/guide.md")
```

BM25 scores are not comparable with cosine similarities, so the hybrid retriever combines the
two by rank.

## Hybrid retrieval

`HybridRetriever` runs BM25 and vector search, then fuses the two rankings with weighted
reciprocal rank fusion: a chunk scores `sum(weight / (rrf_k + rank))` over the lists it
appears in. Metadata filters apply to both sides. A weight of 0 turns a retriever off.

```python
from docsage.embeddings import HashingEmbedder
from docsage.keyword import BM25Index
from docsage.retrieval import HybridRetriever
from docsage.vectorstore.memory import InMemoryVectorStore

embedder = HashingEmbedder(dimension=256)
store = InMemoryVectorStore(dimension=256)
store.upsert(chunks, embedder.embed([c.text for c in chunks]))
index = BM25Index()
index.add(chunks)

retriever = HybridRetriever(store, index, embedder, keyword_weight=1.0, vector_weight=1.0)
for hit in retriever.search("how do I set max_chars?", k=3):
    print(round(hit.score, 4), hit.id)
```

## Reranking

A `Reranker` scores `(query, passage)` pairs; pass one to `HybridRetriever` and it rescores
the fused candidate pool (`4 * k` by default) before returning the top `k`. The reranked
hits carry the reranker's score instead of the fused RRF score.

```python
from docsage.retrieval import CrossEncoderReranker, HybridRetriever

reranker = CrossEncoderReranker()  # needs `pip install docsage[local]`
retriever = HybridRetriever(store, index, embedder, reranker=reranker)
hits = retriever.search("how do I set max_chars?", k=3)
```

`OverlapReranker` is a deterministic stand-in used in tests. Any object with a
`score(query, passages) -> list[float]` method works.

## LLM providers

Answer generation talks to a small `LLM` protocol with a single `complete(messages)` method.
Backends: `openai` (`pip install docsage[openai]`), `anthropic` (`docsage[anthropic]`) and
`ollama` (plain HTTP, no extra). Keys and host come from `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY` and `OLLAMA_HOST` (see `.env.example`).

```python
from docsage.llm import Message, ScriptedLLM, load_llm

llm = load_llm("ollama", model="llama3.1")  # or "openai" / "anthropic"
# llm.complete([Message("user", "What is RRF?")])

# Tests use a scripted fake that replays replies and records calls.
fake = ScriptedLLM(["Reciprocal rank fusion merges ranked lists."])
print(fake.complete([Message("user", "What is RRF?")]))
```

## Query expansion

`MultiQueryRetriever` wraps any retriever (such as `HybridRetriever`), asks an LLM for alternative
phrasings of the question, searches with each and fuses the rankings with reciprocal rank fusion.
With `rewrite=True` the question is first turned into a standalone search query.

```python
from docsage.retrieval import MultiQueryRetriever

retriever = MultiQueryRetriever(hybrid, llm, n_queries=3, rewrite=True)
hits = retriever.search("how do i make it time out sooner?", k=5)
```

## Answers with citations

`answer_question` numbers the retrieved passages, asks the LLM to cite them as `[n]` and
returns the text with the passages it actually cited. Citations that point at a passage that
was not provided are ignored, and with no retrieved hits the model is not called.

```python
from docsage.answer import answer_question
from docsage.chunking.base import Chunk
from docsage.llm import ScriptedLLM
from docsage.vectorstore.base import SearchHit

chunk = Chunk(text="RRF merges ranked lists.", path="docs/rrf.md", index=0, start=0, end=24)
hits = [SearchHit(id="docs/rrf.md#p0#0", chunk=chunk, score=0.9)]
answer = answer_question(ScriptedLLM(["RRF merges ranked lists [1]."]), "What is RRF?", hits)
print(answer.text, [s.path for s in answer.sources])
```

## Faithfulness check

`check_faithfulness` asks a judge whether each claim in an answer is supported by the
retrieved passages. `LLMJudge` works with any `LLM`; implement the `Judge` protocol to plug
in something else. The report has a `score` (share of supported claims) and the list of
`unsupported` claims. With no passages the judge is not called and the score is 0.

```python
from docsage.faithfulness import LLMJudge, check_faithfulness
from docsage.llm import ScriptedLLM

verdict = '{"claims": [{"claim": "RRF merges ranked lists", "supported": true}]}'
report = check_faithfulness(LLMJudge(ScriptedLLM([verdict])), answer.text, hits)
print(report.score, report.unsupported)
```

## Command line

```sh
docsage ingest docs/ --home .docsage          # files or directories (.md, .html, .pdf)
docsage stats --home .docsage
docsage ask "How do I configure X?" --llm ollama --model llama3 --k 5
```

The index lives in `--home` (default `.docsage`, or `DOCSAGE_HOME`): a BM25 file plus a JSON
file of vectors. Ingestion is incremental: each file is stored with a hash of its content and the
chunker, so re-running `ingest` skips unchanged files, re-indexes edited ones in place and only
embeds chunk texts it has not seen before. Add `--prune` to drop indexed files that no longer
exist under the directories you pass. `--chunker` picks `recursive`,
`fixed` or `sentence`; `--embedder` picks `hashing` (default, offline) or
`sentence-transformers`. Use the same embedder for `ingest` and `ask`. `ask` prints the answer
followed by the cited sources; the LLM needs the matching extra and credentials (see
`.env.example`).

## Evaluating retrieval

A dataset is a JSONL file with one question per line and the documents that answer it:

```json
{"question": "How do I install it?", "relevant": ["a.md"]}
```

An entry matches a retrieved chunk when its path equals the entry or ends with `/<entry>`.
Results are judged per document (repeat chunks of one document count once).

```sh
docsage eval questions.jsonl --home .docsage --k 5
```

This prints recall@k, MRR and nDCG@k averaged over the questions. From Python, use
`docsage.eval.evaluate(retriever, load_dataset(path), k=5)`.

## HTTP API

```sh
uv sync --extra api
uvicorn docsage.api:create_app --factory
```

It uses the same index as the CLI and is configured with `DOCSAGE_HOME`, `DOCSAGE_EMBEDDER`,
`DOCSAGE_LLM` and `DOCSAGE_LLM_MODEL`. Interactive docs, with request examples, are at `/docs`.

```sh
curl -X POST localhost:8000/ingest -H 'content-type: application/json' -d \
  '{"documents": [{"path": "guide/install.md", "content": "# Install\n\nRun pip install docsage.\n"}]}'
curl -X POST localhost:8000/query -H 'content-type: application/json' -d \
  '{"question": "How do I install it?", "k": 5}'
curl localhost:8000/health
```

`/ingest` takes document text in the request body (`.md` or `.html`; the server never reads
paths from its own disk) and replaces anything already indexed under the same `path`. `/query`
returns `{"answer": ..., "sources": [...]}` with the numbered sources the answer cites.

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
