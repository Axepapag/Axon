from runtime.trainer import PromotionGate
from scripts.train_heart_decoder_generalization_smoke import _generalization_gate


def test_generalization_gate_is_complete_and_bound_to_candidate() -> None:
    gate = _generalization_gate("heart64a", "candidate-test")

    assert isinstance(gate, PromotionGate)
    assert gate.module_id == "heart64a"
    assert gate.candidate_generation_id == "candidate-test"
    assert len(gate.requirements) == 14
    assert gate.required_suite_ids == ("heart-decoder-generalization-v1",)
