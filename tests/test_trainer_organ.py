"""Transport-neutral Trainer organ command surface."""

from __future__ import annotations

from runtime.trainer import (
    TrainerCommandKind,
    TrainerCommandStatus,
    TrainerOrgan,
    TrainerOrganCommand,
)


def test_status_is_read_only_available_and_transport_neutral(tmp_path) -> None:
    organ = TrainerOrgan(state_root=tmp_path / "State")
    command = TrainerOrganCommand(
        kind="status",
        arguments={"detail": "summary"},
        requested_by="test-ui",
    )
    result = organ.dispatch(command)
    assert result.status is TrainerCommandStatus.COMPLETED
    assert result.command_id == command.command_id
    assert "status" in result.payload["available_commands"]
    assert "start" in result.payload["declared_commands"]
    assert result.payload["inspection"]["writer_lease_present"] is False
    assert result.payload["inspection"]["schema"] == (
        "axon-trainer-organ-status-summary-v1"
    )
    assert "latest_telemetry" not in result.payload["inspection"]


def test_status_full_is_explicit_and_invalid_detail_is_rejected(tmp_path) -> None:
    organ = TrainerOrgan(state_root=tmp_path / "State")
    full = organ.dispatch(
        TrainerOrganCommand(kind="status", arguments={"detail": "full"})
    )
    assert full.status is TrainerCommandStatus.COMPLETED
    assert full.payload["inspection"]["schema"] == "axon-trainer-inspection-v1"
    rejected = organ.dispatch(
        TrainerOrganCommand(kind="status", arguments={"detail": "everything"})
    )
    assert rejected.status is TrainerCommandStatus.REJECTED
    assert "summary" in rejected.error


def test_unwired_mutation_fails_closed_without_side_effect(tmp_path) -> None:
    organ = TrainerOrgan(state_root=tmp_path / "State")
    command = TrainerOrganCommand(
        kind=TrainerCommandKind.START,
        arguments={"session_id": "candidate-session"},
    )
    result = organ.dispatch(command)
    assert result.status is TrainerCommandStatus.UNAVAILABLE
    assert result.payload == {}
    assert "no action was taken" in result.error


def test_same_handler_can_back_future_cli_ui_or_slash_command(tmp_path) -> None:
    organ = TrainerOrgan(state_root=tmp_path / "State")
    organ.register_handler(
        TrainerCommandKind.EXPORT_CLOUD_PACKET,
        lambda command: {
            "packet_id": "a" * 64,
            "provider": command.arguments["provider"],
        },
    )
    command = TrainerOrganCommand(
        kind="export_cloud_packet",
        arguments={"provider": "kaggle"},
        requested_by="runtime:/train",
    )
    result = organ.dispatch(command)
    assert result.status is TrainerCommandStatus.COMPLETED
    assert result.payload == {"packet_id": "a" * 64, "provider": "kaggle"}
    assert "export_cloud_packet" in organ.available_commands
