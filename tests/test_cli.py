from pathlib import Path

import pytest
from typer.testing import CliRunner

from docsage import __version__, cli
from docsage.cli import app
from docsage.llm import ScriptedLLM

runner = CliRunner()


def test_version_command_prints_package_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def _docs(tmp_path: Path) -> Path:
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text("# Install\n\nRun pip install docsage to install it.\n")
    (d / "b.md").write_text("# Config\n\nSet DOCSAGE_HOME to change the index location.\n")
    return d


def test_ingest_stats_ask_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "idx"
    docs = _docs(tmp_path)
    r = runner.invoke(app, ["ingest", str(docs), "--home", str(home)])
    assert r.exit_code == 0, r.output
    assert "2 documents" in r.stdout

    r = runner.invoke(app, ["stats", "--home", str(home)])
    assert "documents: 2" in r.stdout

    fake = ScriptedLLM(["Use pip install docsage [1]."])
    monkeypatch.setattr(cli, "_make_llm", lambda name, model: fake)
    r = runner.invoke(app, ["ask", "how do I install it?", "--home", str(home), "--k", "2"])
    assert r.exit_code == 0, r.output
    assert "Use pip install docsage [1]." in r.stdout
    assert "Sources:" in r.stdout
    assert len(fake.calls) == 1


def test_reingest_replaces_instead_of_duplicating(tmp_path: Path) -> None:
    home = tmp_path / "idx"
    docs = _docs(tmp_path)
    runner.invoke(app, ["ingest", str(docs), "--home", str(home)])
    (docs / "a.md").write_text("# Install\n\nChanged.\n")
    runner.invoke(app, ["ingest", str(docs), "--home", str(home)])
    r = runner.invoke(app, ["stats", "--home", str(home)])
    assert "documents: 2" in r.stdout
    assert "chunks: 2" in r.stdout


def test_ingest_missing_path_fails(tmp_path: Path) -> None:
    r = runner.invoke(app, ["ingest", str(tmp_path / "nope"), "--home", str(tmp_path / "i")])
    assert r.exit_code == 1
