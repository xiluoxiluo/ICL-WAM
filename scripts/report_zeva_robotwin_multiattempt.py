"""Report a complete frozen-manifest RoboTwin run, optionally paired PIM on/off."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.robotwin.multiattempt import (
    audit_attempt_videos, load_manifest, paired_comparison, read_attempts, summarize_attempts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--log", help="One complete condition JSONL")
    parser.add_argument("--on", help="Memory-on JSONL, paired with --off")
    parser.add_argument("--off", help="Same-checkpoint memory-off JSONL")
    parser.add_argument("--output")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--metadata-only", action="store_true", help="Skip the video file audit; output is explicitly not marked COMPLETE")
    args = parser.parse_args()
    if bool(args.log) == bool(args.on or args.off) or bool(args.on) != bool(args.off):
        parser.error("Supply either --log or both --on and --off")
    manifest, sha = load_manifest(args.manifest)
    if args.log:
        rows = read_attempts(args.log)
        result = summarize_attempts(manifest, sha, rows)
    else:
        on_rows, off_rows = read_attempts(args.on), read_attempts(args.off)
        rows = on_rows + off_rows
        result = paired_comparison(
            manifest, sha, on_rows, off_rows,
            bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed,
        )
    result["video_audit"] = (
        {"status": "skipped_metadata_only"} if args.metadata_only else audit_attempt_videos(rows)
    )
    result["completion_status"] = "METADATA_ONLY" if args.metadata_only else "COMPLETE"
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
