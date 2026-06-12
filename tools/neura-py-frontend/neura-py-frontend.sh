#!/usr/bin/env bash
# neura-py-frontend — Convenience wrapper for the Python frontend pipeline
#
# This script finds the Python pipeline script relative to the build output
# directory and invokes it with all arguments forwarded.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_SCRIPT="${SCRIPT_DIR}/neura_pipeline.py"

if [[ ! -f "${PIPELINE_SCRIPT}" ]]; then
    echo "Error: neura_pipeline.py not found at ${PIPELINE_SCRIPT}" >&2
    exit 1
fi

exec python3 "${PIPELINE_SCRIPT}" "$@"
