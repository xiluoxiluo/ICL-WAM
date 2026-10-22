#!/usr/bin/env bash
set -euo pipefail

DESCRIPTION='Train strict Zeva-Ego Stage2B with prepared PIM history.'
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
iclwam_init "$@"

iclwam_stage2 B
