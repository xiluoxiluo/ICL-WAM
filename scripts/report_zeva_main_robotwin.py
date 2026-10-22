"""Report independent main-observed SR, cumulative success, and replay-aware stability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastwam.zeva_main_evaluation import (
    audit_attempt_videos, load_manifest, paired_comparison, read_attempts, summarize_attempts,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--log")
    parser.add_argument("--on")
    parser.add_argument("--off")
    parser.add_argument("--output")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--metadata-only", action="store_true", help="Skip videos; explicitly not a COMPLETE rollout audit")
    args = parser.parse_args(argv)
    if bool(args.log) == bool(args.on or args.off) or bool(args.on) != bool(args.off):
        parser.error("Supply either --log or both --on and --off")
    manifest, sha = load_manifest(args.manifest, expected_sha256=args.expected_manifest_sha256)
    if args.log:
        rows = read_attempts(args.log)
        report = summarize_attempts(manifest, sha, rows)
    else:
        on, off = read_attempts(args.on), read_attempts(args.off)
        rows = on + off
        report = paired_comparison(manifest, sha, on, off,
                                   bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed)
    report["video_audit"] = {"status": "skipped_metadata_only"} if args.metadata_only else audit_attempt_videos(rows)
    report["completion_status"] = "METADATA_ONLY" if args.metadata_only else "COMPLETE"
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
