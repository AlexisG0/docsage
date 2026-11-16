from pathlib import Path

from docsage.loaders.html import load_html

FIXTURES = Path(__file__).parent / "fixtures"


def test_extracts_title_headings_and_text() -> None:
    doc = load_html(FIXTURES / "page.html")
    assert doc.title == "Config & Setup"
    assert doc.headings == ["Configuration", "Advanced options"]
    assert "Set the DOCSAGE_HOME variable." in doc.text
    assert "First\nSecond" in doc.text
    assert doc.page is None


def test_drops_scripts_styles_and_chrome() -> None:
    text = load_html(FIXTURES / "page.html").text
    for noise in ("secret", "color: red", "Home", "Copyright"):
        assert noise not in text


def test_title_falls_back_to_first_heading() -> None:
    assert load_html(FIXTURES / "untitled.html").title == "Only heading"
