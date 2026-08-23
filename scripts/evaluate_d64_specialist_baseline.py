from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Cortext import evaluate_d64_reranking
from runtime.dormant import DormantEvidenceGenerationStore, forward_semantic_edge_cases


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Axon's grounded D64 structural-lexical reranking baseline"
    )
    parser.add_argument("--state-root", type=Path, default=Path(r"D:\Axon\State"))
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--candidate-multiplier", type=int, default=8)
    parser.add_argument("--seed", default="axon-build-c1-forward-real")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    store = DormantEvidenceGenerationStore(args.state_root)
    descriptor = store.active_descriptor()
    with store.open_active(verify_binding=True) as index:
        cases = forward_semantic_edge_cases(
            index,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        result = evaluate_d64_reranking(
            index,
            cases,
            k=args.k,
            candidate_multiplier=args.candidate_multiplier,
        )

    payload = {
        "active_generation": descriptor.to_canonical_dict(),
        "evaluation": result.to_canonical_dict(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8", newline="\n")

    if args.summary_only:
        print(f"generation_id={descriptor.generation_id}")
        print(f"index_id={result.index_id}")
        print(f"case_count={len(result.cases)}")
        print(f"pool_limit={result.pool_limit}")
        print(f"feature_generation={result.feature_generation}")
        print(f"baseline_generation={result.baseline_generation}")
        print(f"pool_recall={result.pool_recall:.6f}")
        print(f"structural_hit_at_{result.k}={result.structural_hit_at_k:.6f}")
        print(f"structural_mrr={result.structural_mean_reciprocal_rank:.6f}")
        print(f"audited_hit_at_{result.k}={result.audited_hit_at_k:.6f}")
        print(f"audited_mrr={result.audited_mean_reciprocal_rank:.6f}")
        print(f"unsupported_candidate_count={result.unsupported_candidate_count}")
        print(f"evaluation_id={result.evaluation_id}")
        if args.output is not None:
            print(f"output={args.output}")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
