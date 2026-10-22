"""Record successful offline demonstrations without inventing outcome labels.

An outcomes JSON maps ``/absolute/dataset::episode-N`` to a JSON boolean.
Without it, outcome fields in meta/episodes.jsonl are used. A collection known
to contain only successful demonstrations may instead be explicitly declared
with --declare-success-only --evidence; that assertion stays visible in every
downstream artifact and is not described as simulator-verified success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastwam.zeva.main_data import build_success_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outcomes-json", type=Path)
    parser.add_argument("--declare-success-only", action="store_true")
    parser.add_argument("--evidence", default="")
    args = parser.parse_args()
    outcomes = json.loads(args.outcomes_json.read_text()) if args.outcomes_json else None
    if outcomes is not None and not isinstance(outcomes, dict):
        parser.error("--outcomes-json must map source-qualified episode IDs to JSON booleans")
    payload = build_success_manifest(
        args.dataset_root, outcomes=outcomes,
        declare_success_only=args.declare_success_only, evidence=args.evidence,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    successes = sum(len(source["success_episode_indices"]) for source in payload["sources"])
    total = sum(len(source["episode_indices"]) for source in payload["sources"])
    print(f"Success demonstrations: {successes}/{total}; evidence={payload['evidence']['kind']}")
    print(f"Manifest: {args.output}")


if __name__ == "__main__":
    main()
