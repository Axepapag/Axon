"""Compile immutable Trainer sessions from exact Dormant lived experience."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime.dormant import DormantExperienceStore
from runtime.trainer import LivedExperienceSessionCompiler, TrainerControlPlane


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=Path(r"D:\Axon\State"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    experience = DormantExperienceStore(args.state_root)
    compiler = LivedExperienceSessionCompiler(experience)
    heart = compiler.compile_heart_grounding()
    conversation = compiler.compile_observed_conversation()
    with TrainerControlPlane.active(state_root=args.state_root) as trainer:
        heart_path = trainer.publish_session(heart)
        conversation_path = trainer.publish_session(conversation)
    print(
        json.dumps(
            {
                "schema": "axon-lived-experience-session-compilation-result-v1",
                "heart_grounding_session_id": heart.session_id,
                "heart_grounding_examples": len(heart.examples),
                "heart_grounding_path": str(heart_path),
                "conversation_session_id": conversation.session_id,
                "conversation_examples": len(conversation.examples),
                "conversation_path": str(conversation_path),
                "conversation_serving_promotion_eligible": conversation.policy.serving_promotion_eligible,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
