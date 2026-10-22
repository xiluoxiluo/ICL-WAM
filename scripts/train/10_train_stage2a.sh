#!/usr/bin/env bash
set -euo pipefail

DESCRIPTION='Train strict Zeva-Ego Stage2A with prepared task context.'
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
iclwam_init "$@"

iclwam_stage2 A
