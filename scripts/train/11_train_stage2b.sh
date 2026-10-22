#!/usr/bin/env bash
set -euo pipefail
DESCRIPTION='11: Main Stage2B PIM alignment from the selected Stage2A parent; frozen foundation by default.'
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
iclwam_init "$@"; iclwam_stage2 B
