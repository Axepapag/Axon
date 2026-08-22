from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime.dormant import DormantEvidenceGenerationStore
from runtime.dormant.evaluation import (
    evaluate_forward_relevance,
    evaluate_relevance,
    forward_semantic_edge_cases,
    semantic_edge_cases,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Axon dormant relevance on recovered semantic edges")
    parser.add_argument("--state-root", type=Path, default=Path(r"D:\Axon\State"))
    parser.add_argument("--mode", choices=("inverse", "forward"), default="inverse")
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--candidate-multiplier", type=int, default=4)
    parser.add_argument("--seed", default="axon-build-c1")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    store = DormantEvidenceGenerationStore(args.state_root)
    descriptor = store.active_descriptor()
    edges_path = args.state_root / "dormant" / "semantic_edges.jsonl"
    with store.open_active(verify_binding=True) as index:
        if args.mode == "forward":
            cases = forward_semantic_edge_cases(
                index,
                sample_size=args.sample_size,
                seed=args.seed,
            )
            result = evaluate_forward_relevance(
                index,
                cases,
                k=args.k,
                candidate_multiplier=args.candidate_multiplier,
            )
        else:
            cases = semantic_edge_cases(
                edges_path,
                sample_size=args.sample_size,
                seed=args.seed,
            )
            result = evaluate_relevance(
                index,
                cases,
                k=args.k,
                candidate_multiplier=args.candidate_multiplier,
            )
    payload = {
        "active_generation": descriptor.to_canonical_dict(),
        "evaluation_mode": args.mode,
        "evaluation": result.to_canonical_dict(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8", newline="\n")
    if args.summary_only:
        print(f"generation_id={descriptor.generation_id}")
        print(f"index_id={result.index_id}")
        print(f"mode={args.mode}")
        print(f"case_count={len(result.cases)}")
        if args.mode == "forward":
            print(f"pool_limit={result.pool_limit}")
            print(f"pool_recall={result.pool_recall:.6f}")
            print(f"raw_hit_at_{result.k}={result.raw_hit_at_k:.6f}")
            print(f"raw_mrr={result.raw_mean_reciprocal_rank:.6f}")
            print(f"audited_hit_at_{result.k}={result.audited_hit_at_k:.6f}")
            print(f"audited_mrr={result.audited_mean_reciprocal_rank:.6f}")
        else:
            print(f"hit_at_{result.k}={result.hit_at_k:.6f}")
            print(f"mrr={result.mean_reciprocal_rank:.6f}")
        print(f"evaluation_id={result.evaluation_id}")
        if args.output is not None:
            print(f"output={args.output}")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
