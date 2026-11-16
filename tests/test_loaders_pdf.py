from pathlib import Path

import pytest
from reportlab.pdfgen import canvas  # type: ignore[import-untyped]

from docsage.loaders import load_file
from docsage.loaders.pdf import load_pdf


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "manual.pdf"
    c = canvas.Canvas(str(path))
    c.setTitle("User Manual")
    c.drawString(72, 750, "Page one talks about installation.")
    c.showPage()
    c.showPage()  # blank page, should be skipped
    c.drawString(72, 750, "Page three covers troubleshooting.")
    c.showPage()
    c.save()
    return path


def test_one_document_per_non_empty_page(sample_pdf: Path) -> None:
    docs = load_pdf(sample_pdf)
    assert [d.page for d in docs] == [1, 3]
    assert "installation" in docs[0].text
    assert "troubleshooting" in docs[1].text
    assert all(d.title == "User Manual" for d in docs)


def test_load_file_dispatches_by_extension(sample_pdf: Path) -> None:
    fixtures = Path(__file__).parent / "fixtures"
    assert len(load_file(sample_pdf)) == 2
    assert load_file(fixtures / "guide.md")[0].title == "Install Guide"
    assert load_file(fixtures / "page.html")[0].title == "Config & Setup"
    with pytest.raises(ValueError, match="unsupported"):
        load_file(fixtures / "nope.txt")
