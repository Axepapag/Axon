from __future__ import annotations

import pytest

from scripts.apply_canonical_identity import canonical_identity_text


def test_identity_document_extracts_one_exact_nonempty_body() -> None:
    assert canonical_identity_text(
        "heading\n--- BEGIN CANONICAL IDENTITY ---\nI am Axon. 🧠\n"
        "--- END CANONICAL IDENTITY ---\nfooter\n"
    ) == "I am Axon. 🧠"


@pytest.mark.parametrize(
    "document",
    (
        "no markers",
        "--- BEGIN CANONICAL IDENTITY ------ BEGIN CANONICAL IDENTITY ---"
        "x--- END CANONICAL IDENTITY ---",
        "--- BEGIN CANONICAL IDENTITY ---\n\n--- END CANONICAL IDENTITY ---",
    ),
)
def test_identity_document_rejects_missing_duplicate_or_empty_body(document: str) -> None:
    with pytest.raises(ValueError):
        canonical_identity_text(document)
