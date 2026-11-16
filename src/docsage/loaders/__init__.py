"""Document loaders that keep source metadata."""

from docsage.loaders.base import Document
from docsage.loaders.markdown import load_markdown

__all__ = ["Document", "load_markdown"]
