"""Apply an explicitly ratified Identity document through IDENTITY_STEWARD."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from runtime.field import LogicalRegion
from runtime.heart import HeartHost

ROOT = Path(__file__).resolve().parent.parent
BEGIN = "--- BEGIN CANONICAL IDENTITY ---"
END = "--- END CANONICAL IDENTITY ---"


def canonical_identity_text(document: str) -> str:
    if document.count(BEGIN) != 1 or document.count(END) != 1:
        raise ValueError("identity document requires exactly one marker pair")
    before, remainder = document.split(BEGIN, 1)
    text, after = remainder.split(END, 1)
    if END in before or BEGIN in after:
        raise ValueError("identity document markers are out of order")
    text = text.strip()
    if not text:
        raise ValueError("canonical Identity text must be non-empty")
    return text


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=ROOT / "State")
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--expected-document-sha256", required=True)
    parser.add_argument("--amendment-id", required=True)
    parser.add_argument("--evidence-id", action="append", required=True)
    parser.add_argument("--provenance", required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    data = args.document.resolve().read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != args.expected_document_sha256.lower():
        raise ValueError("ratified Identity document SHA256 mismatch")
    text = canonical_identity_text(data.decode("utf-8"))
    with HeartHost(state_root=args.state_root) as host:
        before = host.coordinator.current_field
        current = before.region(LogicalRegion.IDENTITY).text
        if current == text:
            print(
                json.dumps(
                    {
                        "schema": "axon-canonical-identity-application-v1",
                        "status": "already_applied",
                        "field_id": before.field_id,
                        "identity_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "document_sha256": digest,
                    },
                    sort_keys=True,
                )
            )
            return 0
        commit = host.amend_identity(
            text,
            amendment_id=args.amendment_id,
            evidence_ids=tuple(args.evidence_id),
            provenance=args.provenance,
        )
        print(
            json.dumps(
                {
                    "schema": "axon-canonical-identity-application-v1",
                    "status": "applied",
                    "base_field_id": before.field_id,
                    "successor_field_id": commit.successor.field_id,
                    "delta_id": commit.delta.delta_id,
                    "identity_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "document_sha256": digest,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
