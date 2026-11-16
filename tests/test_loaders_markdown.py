from pathlib import Path

from docsage.loaders.markdown import load_markdown

FIXTURES = Path(__file__).parent / "fixtures"


def test_front_matter_title_and_headings() -> None:
    doc = load_markdown(FIXTURES / "guide.md")
    assert doc.title == "Install Guide"
    assert doc.headings == ["Getting started", "Install", "Configure"]
    assert doc.text.startswith("# Getting started")
    assert "title:" not in doc.text
    assert doc.page is None
    assert doc.path.endswith("guide.md")


def test_title_falls_back_to_h1() -> None:
    assert load_markdown(FIXTURES / "plain.md").title == "Title Here"


def test_title_falls_back_to_file_stem() -> None:
    doc = load_markdown(FIXTURES / "notes.md")
    assert doc.title == "notes"
    assert doc.headings == []
