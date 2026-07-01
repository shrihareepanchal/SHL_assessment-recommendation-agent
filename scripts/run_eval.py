#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.probes import run_all_probes  # noqa: E402
from app.evaluation.recall import mean_recall_at_k  # noqa: E402
from app.evaluation.replay import replay_trace  # noqa: E402
from app.evaluation.trace_parser import load_all_traces  # noqa: E402

TRACES_DIR = PROJECT_ROOT / "data" / "sample_conversations"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--traces-dir", default=str(TRACES_DIR))
    args = parser.parse_args()

    traces = load_all_traces(args.traces_dir)
    if not traces:
        print(f"No traces found in {args.traces_dir}", file=sys.stderr)
        return 1

    print(f"Replaying {len(traces)} conversation traces against {args.base_url}\n")
    recalls = []
    total_violations = 0
    total_exceeded = 0

    with httpx.Client() as client:
        for trace in traces:
            result = replay_trace(client, args.base_url, trace)
            recalls.append(result.recall_at_10)
            total_violations += len(result.schema_violations)
            total_exceeded += int(result.exceeded_turn_cap)
            status = "OK" if not result.schema_violations else "SCHEMA VIOLATION"
            print(f"  {result.trace_id:>6}  recall@10={result.recall_at_10:.2f}  {status}")
            for v in result.schema_violations:
                print(f"           - {v}")

    print(f"\nMean Recall@10: {mean_recall_at_k(recalls):.3f}")
    print(f"Total schema violations: {total_violations}")
    print(f"Traces exceeding turn cap: {total_exceeded}/{len(traces)}")

    print("\nBehavior probes:")
    probe_results = run_all_probes(args.base_url)
    passed = sum(1 for r in probe_results if r.passed)
    for r in probe_results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"  [{mark}] {r.name}")
    print(f"\nProbe pass-rate: {passed}/{len(probe_results)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
