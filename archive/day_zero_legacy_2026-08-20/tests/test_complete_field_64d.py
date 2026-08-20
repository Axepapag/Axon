from __future__ import annotations

from argparse import Namespace
import json

import pytest
import torch

from training.build_complete_field_r0_curriculum import (
    synthetic_records,
    v6_alignment_records,
    validate_exact_field_isolation,
)
from training.complete_field_64d import (
    CompleteField64D,
    CompleteFieldPager,
    REGION_ORDER,
    ReaderConfig,
)
from training.train_complete_field_64d import checkpoint_payload, restore


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
            # V6 tests evidence authority rather than obedience to scratch: when
            # immutable evidence resolves the answer, empty/conflicting scratch
            # must preserve the correct response target.
            assert counterfactual["response_draft"] == record["targets"]["response_draft"]
        assert record["alignment"]["schema"] == "axon-r0-source-alignment-v1"


def test_synthetic_fields_have_one_target_and_do_not_cross_splits() -> None:
    records = list(synthetic_records(1000, seed=64018))
    field_targets: dict[str, set[tuple[str, str]]] = {}
    field_splits: dict[str, set[str]] = {}
    for record in records:
        identity = json.dumps(record["field"], sort_keys=True, separators=(",", ":"))
        field_targets.setdefault(identity, set()).add(
            (record["targets"]["scratch"], record["targets"]["response_draft"])
        )
        field_splits.setdefault(identity, set()).add(record["split"])

    assert all(len(targets) == 1 for targets in field_targets.values())
    assert all(len(splits) == 1 for splits in field_splits.values())


def test_exact_copy_uses_heldout_variable_strings() -> None:
    records = list(synthetic_records(1000, seed=64018))
    values_by_split: dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}
    for record in records:
        if record["family"] == "conversation_exact_copy":
            values_by_split[record["split"]].add(record["targets"]["response_draft"])

    assert all(values_by_split.values())
    assert values_by_split["train"].isdisjoint(values_by_split["dev"])
    assert values_by_split["train"].isdisjoint(values_by_split["test"])
    assert values_by_split["dev"].isdisjoint(values_by_split["test"])
    assert not values_by_split["train"].intersection({"Axon", "Jeff", "Council"})


def test_builder_rejects_contradictory_or_cross_split_exact_fields() -> None:
    base = list(synthetic_records(1, seed=9))[0]
    contradictory = dict(base)
    contradictory["example_id"] = "contradictory"
    contradictory["targets"] = dict(base["targets"])
    contradictory["targets"]["response_draft"] = "A different answer."
    try:
        validate_exact_field_isolation([base, contradictory])
    except ValueError as exc:
        assert "contradictory targets" in str(exc)
    else:
        raise AssertionError("contradictory exact field was accepted")

    leaked = dict(base)
    leaked["example_id"] = "leaked"
    leaked["split"] = "dev" if base["split"] != "dev" else "test"
    try:
        validate_exact_field_isolation([base, leaked])
    except ValueError as exc:
        assert "crosses data splits" in str(exc)
    else:
        raise AssertionError("cross-split exact field was accepted")


def test_addressable_memory_retains_every_page_token() -> None:
    torch.manual_seed(5)
    config = ReaderConfig(page_size=16, max_output_chars=80, dropout=0.0)
    model = CompleteField64D(config).cpu().eval()
    field = field_fixture()
    pages, _ = CompleteFieldPager(config.page_size).paginate(field)

    _, memory, manifest = model.read_field_with_memory(field)

    expected_tokens = sum(max(1, len(page.text)) for page in pages)
    expected_char_indices = [
        model.char_to_index[char] if page.text else -1
        for page in pages
        for char in (page.text or "\0")
    ]
    expected_region_ids = [
        page.region_id
        for page in pages
        for _ in (page.text or "\0")
    ]
    expected_region_positions = [
        position if page.text else -1
        for page in pages
        for position in (range(page.region_start, page.region_end) if page.text else (-1,))
    ]
    assert manifest.complete
    assert memory.states.shape == (1, expected_tokens, config.d_model)
    assert memory.char_indices.squeeze(0).tolist() == expected_char_indices
    assert memory.region_ids.squeeze(0).tolist() == expected_region_ids
    assert memory.region_positions.squeeze(0).tolist() == expected_region_positions


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


def test_pointer_distribution_can_copy_exact_source_character() -> None:
    torch.manual_seed(7)
    config = ReaderConfig(page_size=16, max_output_chars=80, dropout=0.0)
    model = CompleteField64D(config).cpu().eval()
    field = {name: "" for name in REGION_ORDER}
    field["user_input"] = "Z"
    _, memory, _ = model.read_field_with_memory(field)
    with torch.no_grad():
        model.copy_gate.weight.zero_()
        model.copy_gate.bias.fill_(-30.0)

    decoder_state = torch.zeros(1, 1, config.d_model)
    logits = model._decoder_logits(decoder_state, memory)

    assert int(logits.argmax(dim=-1).item()) == model.char_to_index["Z"]


def test_v6_alignment_shard_covers_decoys_and_page_boundary() -> None:
    records = list(v6_alignment_records(8, seed=6006, page_size=64))
    assert {record["provenance"]["layout"] for record in records} == {
        "first",
        "middle",
        "page_boundary",
        "last",
    }
    for record in records:
        provenance = record["provenance"]
        token = record["targets"]["response_draft"]
        assert 4 <= len(token) <= 32
        assert token in record["field"]["structured_knowledge"]
        assert provenance["same_region_duplicate"] is True
        assert record["field"]["tool_results"].count(token) >= 2
        assert all(
            counterfactual["response_draft"] == token
            for counterfactual in record["response_counterfactuals"]
        )
        source = record["field"]["tool_results"][
            provenance["source_start"] : provenance["source_end"]
        ]
        assert source == token
        if provenance["layout"] == "page_boundary":
            assert provenance["source_start"] == 62
            assert provenance["source_end"] > 64


def test_v6_alignment_supervision_resolves_exact_duplicate_occurrence() -> None:
    torch.manual_seed(8)
    record = list(v6_alignment_records(3, seed=6010, page_size=64))[-1]
    model = CompleteField64D(
        ReaderConfig(page_size=64, max_output_chars=128, dropout=0.0)
    ).cpu().eval()
    out = model.forward_transaction(
        record["field"],
        record["targets"]["scratch"],
        record["targets"]["response_draft"],
        alignment={
            "schema": record["alignment"]["schema"],
            "scratch": record["alignment"]["scratch"],
            "response_draft": record["alignment"]["response_draft"],
        },
    )
    assert out["alignment"]["copy_positions"] == 2 * len(record["targets"]["response_draft"])
    assert torch.isfinite(out["alignment"]["position_loss"])
    assert torch.isfinite(out["alignment"]["gate_loss"])


def test_v6_checkpoint_restores_exact_model_and_rejects_dataset_mismatch(tmp_path) -> None:
    torch.manual_seed(11)
    config = ReaderConfig(page_size=32, max_output_chars=80, dropout=0.0)
    model = CompleteField64D(config).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    dataset = {"train": "train-sha", "eval": "eval-sha"}
    payload = checkpoint_payload(
        model,
        optimizer,
        scaler,
        step=7,
        config=config,
        args=Namespace(run_dir=tmp_path, device="cpu"),
        baseline={"mean_total_loss": 1.0},
        sampler_state=(1, 2, 3),
        dataset_sha256=dataset,
    )
    assert payload["schema"] == "axon-complete-field-r0-checkpoint-v6"
    path = tmp_path / "ckpt.pt"
    torch.save(payload, path)

    restored = CompleteField64D(config).cpu()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=2e-4)
    restored_scaler = torch.amp.GradScaler("cpu", enabled=False)
    step, baseline, sampler_state = restore(
        path,
        restored,
        restored_optimizer,
        restored_scaler,
        torch.device("cpu"),
        dataset,
    )
    assert step == 7
    assert baseline == {"mean_total_loss": 1.0}
    assert sampler_state == (1, 2, 3)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name]), name

    with pytest.raises(ValueError, match="dataset fingerprint mismatch"):
        restore(
            path,
            restored,
            restored_optimizer,
            restored_scaler,
            torch.device("cpu"),
            {"train": "different", "eval": "eval-sha"},
        )
