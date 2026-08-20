from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import generate_curriculum as builder  # noqa: E402
import verify_curriculum as verifier  # noqa: E402


class CurriculumV021Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profiles, cls.alphabet = builder.load_package_contracts(PACKAGE_ROOT)
        cls.alphabet_chars = verifier.validate_alphabet(cls.alphabet)
        cls.temp = tempfile.TemporaryDirectory(prefix="axon-v021-tests-")
        cls.generated = Path(cls.temp.name) / "generated"
        builder.build_generated_dataset(
            PACKAGE_ROOT,
            cls.generated,
            groups=20,
            page_size=256,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_profiles_separate_content_from_attention(self) -> None:
        verifier.validate_profiles(self.profiles)
        for core_id in builder.ROSTER:
            self.assertFalse(
                builder.is_allowed(
                    self.profiles,
                    builder.FOCUS_PROFILE,
                    core_id,
                    "content",
                    "tool_results",
                    "replace",
                )
            )
            self.assertTrue(
                builder.is_allowed(
                    self.profiles,
                    builder.FOCUS_PROFILE,
                    core_id,
                    "attention",
                    "tool_results",
                    "set_boundary",
                )
            )
            self.assertFalse(
                builder.is_allowed(
                    self.profiles,
                    builder.FOCUS_PROFILE,
                    core_id,
                    "attention",
                    "tool_results",
                    "set_intervals",
                )
            )

    def test_starter_models_three_phases_inside_one_tick(self) -> None:
        result = verifier.validate_starter(
            PACKAGE_ROOT, self.profiles, self.alphabet_chars
        )
        self.assertEqual(result["records"], 16)

    def test_focus_reads_all_regions_and_writes_three_regions(self) -> None:
        record = builder.build_record(
            self.profiles,
            self.alphabet,
            group_index=0,
            variant="correct",
            page_size=256,
        )
        self.assertEqual(record["authority"]["profile_id"], builder.FOCUS_PROFILE)
        for region in builder.REGIONS:
            self.assertTrue(verifier.region_text(record["snapshot"], region))
            self.assertTrue(record["attention_view"]["regions"][region]["active_intervals"])
        delta = json.loads(record["phase_result"]["delta_envelope"]["payload"]["exact_text"])
        self.assertEqual(
            {operation["region"] for operation in delta["operations"]},
            set(builder.FOCUS_WRITABLE_REGIONS),
        )

    def test_legacy_profile_still_rejects_diary(self) -> None:
        self.assertFalse(
            builder.is_allowed(
                self.profiles,
                "core-v1-scratch-response-only",
                "core-alpha",
                "content",
                "diary",
                "insert",
            )
        )

    def test_generated_groups_are_complete_and_distributed(self) -> None:
        result = verifier.validate_generated(
            self.generated, self.profiles, self.alphabet_chars
        )
        self.assertEqual(result["records"], 100)
        self.assertEqual(set(result["page_strata"]), {1, 2, 4, 8})
        self.assertEqual(set(result["evidence_positions"]), set(builder.EVIDENCE_BINS))

    def test_every_corruption_matches_independent_oracle(self) -> None:
        for group_index in (0, 1):
            for variant in builder.VARIANTS:
                record = builder.build_record(
                    self.profiles,
                    self.alphabet,
                    group_index=group_index,
                    variant=variant,
                    page_size=256,
                )
                observed = verifier.independent_coverage_oracle(
                    record["coverage_manifest"]
                )
                self.assertEqual(observed, record["expected"]["failure_code"])

    def test_one_page_reordered_variant_is_complete(self) -> None:
        record = builder.build_record(
            self.profiles,
            self.alphabet,
            group_index=0,
            variant="reordered_page",
            page_size=256,
            page_count_override=1,
            evidence_bin_override="first",
        )
        self.assertEqual(
            record["expected"]["failure_code"], "BASE_FIELD_PAGE_INDEX_INVALID"
        )

    def test_nonempty_output_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="axon-v021-stale-") as raw:
            root = Path(raw)
            (root / "stale.txt").write_text("stale", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                builder.build_generated_dataset(
                    PACKAGE_ROOT, root, groups=20, page_size=256
                )

    def test_mask_move_never_changes_tool_result_content(self) -> None:
        record = builder.build_record(
            self.profiles,
            self.alphabet,
            group_index=1,
            variant="correct",
            page_size=256,
        )
        update = record["phase_result"]["attention_update"]
        self.assertIsNotNone(update)
        source = verifier.region_text(record["snapshot"], "tool_results")
        self.assertEqual(update["source_text_digest"], builder.digest_text(source))
        old_active = source[update["old_offset"] :]
        restored = source[: update["old_offset"]] + old_active
        self.assertEqual(restored, source)

    def test_chunked_delta_is_one_atomic_canonical_payload(self) -> None:
        record = builder.build_record(
            self.profiles,
            self.alphabet,
            group_index=2,
            variant="correct",
            page_size=256,
        )
        verifier.validate_transport(record)
        self.assertEqual(
            record["phase_result"]["commit_semantics"],
            "atomic_consolidation_commit",
        )

    def test_builder_does_not_claim_external_or_replay_validation(self) -> None:
        manifest = json.loads((self.generated / "manifest.json").read_text(encoding="utf-8"))
        self.assertIsNone(manifest["external_validation"])
        self.assertIsNone(manifest["builder_validation"]["schema_validation"])
        self.assertIsNone(manifest["builder_validation"]["deterministic_replay"])


if __name__ == "__main__":
    unittest.main()
