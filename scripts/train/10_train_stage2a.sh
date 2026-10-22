#!/usr/bin/env bash
set -euo pipefail
DESCRIPTION='10: Main Stage2A global/BIT action conditioning; frozen foundation by default.'
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
iclwam_init "$@"; iclwam_stage2 A
