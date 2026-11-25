"""Evaluation dataset: one JSON object per line, a question and the documents that answer it."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


class EvalExample(BaseModel):
    """A question and the source paths that should be retrieved for it.

    A retrieved chunk counts as relevant when its path equals a ``relevant`` entry or ends
    with ``/<entry>``, so datasets can use short, location-independent names like ``a.md``.
    """

    question: str = Field(min_length=1)
    relevant: list[str] = Field(min_length=1)


def load_dataset(path: str | Path) -> list[EvalExample]:
    """Read a JSONL file. Blank lines are ignored; errors name the offending line."""
    examples: list[EvalExample] = []
    text = Path(path).read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            examples.append(EvalExample.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"{path}:{number}: invalid example: {exc}") from exc
    if not examples:
        raise ValueError(f"{path}: dataset is empty")
    return examples
