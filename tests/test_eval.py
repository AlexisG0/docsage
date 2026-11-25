import json
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from docsage.chunking.base import Chunk
from docsage.cli import app
from docsage.eval import EvalExample, evaluate, load_dataset, mrr, ndcg_at_k, recall_at_k
from docsage.vectorstore import SearchHit


def test_recall_mrr_ndcg_basic() -> None:
    flags = [False, True, False, True]
    assert recall_at_k(flags, 2, 2) == 0.5
    assert recall_at_k(flags, 2, 4) == 1.0
    assert mrr(flags) == 0.5
    assert mrr([False]) == 0.0
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 + 1 / math.log2(3)
    assert ndcg_at_k(flags, 2, 4) == pytest.approx(dcg / ideal)
    assert ndcg_at_k([True, True], 2, 2) == pytest.approx(1.0)
    assert recall_at_k([], 0, 3) == 0.0


def test_load_dataset_reports_line(tmp_path: Path) -> None:
    f = tmp_path / "d.jsonl"
    good = json.dumps({"question": "q", "relevant": ["a.md"]})
    f.write_text(good + "\n\n{bad\n")
    with pytest.raises(ValueError, match=r"d\.jsonl:3"):
        load_dataset(f)
    f.write_text(json.dumps({"question": "q", "relevant": []}) + "\n")
    with pytest.raises(ValueError, match=r":1"):
        load_dataset(f)
    f.write_text("\n")
    with pytest.raises(ValueError, match="empty"):
        load_dataset(f)


class _Fixed:
    def __init__(self, paths: list[str]) -> None:
        self.paths = paths

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        return [
            SearchHit(
                id=f"{p}:{i}",
                chunk=Chunk(text="t", path=p, index=i, start=0, end=1),
                score=1.0,
            )
            for i, p in enumerate(self.paths[:k])
        ]


def test_evaluate_collapses_duplicate_documents() -> None:
    ex = [EvalExample(question="q", relevant=["a.md", "b.md"])]
    r = evaluate(_Fixed(["docs/a.md", "docs/a.md", "docs/c.md"]), ex, k=3)
    assert r.recall == 0.5
    assert r.mrr == 1.0
    assert 0 < r.ndcg < 1


def test_eval_cli(tmp_path: Path) -> None:
    runner = CliRunner()
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text("# Install\n\nRun pip install docsage to install it.\n")
    (d / "b.md").write_text("# Config\n\nSet DOCSAGE_HOME to change the index location.\n")
    home = tmp_path / "idx"
    assert runner.invoke(app, ["ingest", str(d), "--home", str(home)]).exit_code == 0
    ds = tmp_path / "d.jsonl"
    row = {"question": "pip install docsage", "relevant": ["a.md"]}
    ds.write_text(json.dumps(row) + "\n")
    r = runner.invoke(app, ["eval", str(ds), "--home", str(home), "--k", "2"])
    assert r.exit_code == 0, r.output
    assert "recall@2: 1.000" in r.stdout
    assert "mrr: 1.000" in r.stdout
    bad = runner.invoke(app, ["eval", str(tmp_path / "none.jsonl"), "--home", str(home)])
    assert bad.exit_code == 1
