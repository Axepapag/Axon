from __future__ import annotations

import torch

from training.build_complete_field_r0_curriculum import synthetic_records
from training.complete_field_64d import (
    CompleteField64D,
    CompleteFieldPager,
    REGION_ORDER,
    ReaderConfig,
)


def field_fixture() -> dict[str, str]:
    return {name: (name + " evidence. " if name != "diary" else "") for name in REGION_ORDER}


def test_pages_cover_every_character_and_region_at_multiple_sizes() -> None:
    field = field_fixture()
    expected = sum(map(len, field.values()))
    identities = set()
    for page_size in (7, 16, 64):
        pages, manifest = CompleteFieldPager(page_size).paginate(field)
        assert manifest.complete
        assert manifest.expected_characters == expected
        assert manifest.observed_characters == expected
        assert manifest.visited_regions == REGION_ORDER
        assert any(page.region == "diary" and page.text == "" for page in pages)
        identities.add(manifest.field_sha256)
    assert len(identities) == 1


def test_cpu_transaction_writes_only_scratch_and_response() -> None:
    torch.manual_seed(1)
    model = CompleteField64D(
        ReaderConfig(page_size=32, max_output_chars=80, dropout=0.0)
    ).cpu()
    model.eval()
    result = model.run_transaction(field_fixture())
    assert result["coverage_tick1"]["complete"]
    assert result["coverage_tick2"]["complete"]
    regions = [operation["region"] for operation in result["typed_delta"]["operations"]]
    assert regions == ["scratch", "response_draft"]
    assert "diary" not in regions


def test_teacher_path_accepts_output_longer_than_64_characters() -> None:
    model = CompleteField64D(
        ReaderConfig(page_size=64, max_output_chars=128, dropout=0.0)
    ).cpu()
    target = "A" * 96
    output = model.forward_transaction(field_fixture(), "inspect exact visible evidence.", target)
    assert output["response_targets"].shape[1] == 97
    assert output["response_logits"].shape[1] == 97


def test_scheduled_decoder_matches_teacher_path_at_ratio_one() -> None:
    torch.manual_seed(3)
    model = CompleteField64D(
        ReaderConfig(page_size=64, max_output_chars=128, dropout=0.0)
    ).cpu()
    model.eval()
    state, memory, _ = model.read_field_with_memory(field_fixture())
    teacher_logits, teacher_targets = model.decode_teacher(
        state,
        "Axon is ready.",
        head=1,
        memory=memory,
    )
    scheduled_logits, scheduled_targets = model.decode_scheduled(
        state,
        "Axon is ready.",
        head=1,
        teacher_forcing_ratio=1.0,
        memory=memory,
    )
    assert torch.equal(teacher_targets, scheduled_targets)
    assert torch.equal(teacher_logits, scheduled_logits)


def test_scheduled_decoder_accepts_model_prefixes() -> None:
    torch.manual_seed(4)
    model = CompleteField64D(
        ReaderConfig(page_size=64, max_output_chars=128, dropout=0.0)
    ).cpu()
    state, memory, _ = model.read_field_with_memory(field_fixture())
    logits, targets = model.decode_scheduled(
        state,
        "Use visible evidence.",
        head=0,
        teacher_forcing_ratio=0.0,
        memory=memory,
    )
    assert logits.shape[:2] == targets.shape
    assert torch.isfinite(logits).all()


def test_synthetic_families_have_empty_and_conflicting_scratch_interventions() -> None:
    records = list(synthetic_records(5, seed=7))
    assert len({record["family"] for record in records}) == 5
    for record in records:
        counterfactuals = record["response_counterfactuals"]
        assert [item["variant_id"] for item in counterfactuals] == [
            "empty_scratch",
            "conflicting_scratch",
        ]
        assert counterfactuals[0]["scratch"] == ""
        for counterfactual in counterfactuals:
            assert counterfactual["scratch"] != record["targets"]["scratch"]
            assert counterfactual["response_draft"] != record["targets"]["response_draft"]


def test_addressable_memory_retains_every_page_token() -> None:
    torch.manual_seed(5)
    config = ReaderConfig(page_size=16, max_output_chars=80, dropout=0.0)
    model = CompleteField64D(config).cpu().eval()
    field = field_fixture()
    pages, _ = CompleteFieldPager(config.page_size).paginate(field)

    _, memory, manifest = model.read_field_with_memory(field)

    expected_tokens = sum(max(1, len(page.text)) for page in pages)
    assert manifest.complete
    assert memory.shape == (1, expected_tokens, config.d_model)


def test_decoder_cross_attention_changes_logits() -> None:
    torch.manual_seed(6)
    model = CompleteField64D(
        ReaderConfig(page_size=32, max_output_chars=80, dropout=0.0)
    ).cpu().eval()
    state, memory, _ = model.read_field_with_memory(field_fixture())

    addressed, targets = model.decode_teacher(
        state,
        "Use the requested evidence.",
        head=1,
        memory=memory,
    )
    pooled_only, pooled_targets = model.decode_teacher(
        state,
        "Use the requested evidence.",
        head=1,
    )

    assert torch.equal(targets, pooled_targets)
    assert not torch.equal(addressed, pooled_only)
