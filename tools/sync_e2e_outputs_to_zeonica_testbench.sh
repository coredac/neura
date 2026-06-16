#!/usr/bin/env bash
set -euo pipefail

# Sync Dataflow e2e outputs into Zeonica_Testbench.
# - Copies generated DFG/instruction artifacts from test/e2e/<kernel>
# - Renames them to Zeonica_Testbench's expected naming
# - Creates missing kernel directories under the submodule

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)"

SRC_E2E="${ROOT}/test/e2e"
DST_TESTBENCH="${ROOT}/test/benchmark/Zeonica_Testbench"
DST_KERNEL="${DST_TESTBENCH}/kernel"

if [[ ! -d "${DST_TESTBENCH}" ]]; then
  echo "ERROR: Zeonica_Testbench submodule not found at: ${DST_TESTBENCH}" >&2
  echo "Hint: run: git submodule update --init --recursive" >&2
  exit 1
fi

mkdir -p "${DST_KERNEL}"

echo "[sync] SRC: ${SRC_E2E}"
echo "[sync] DST: ${DST_KERNEL}"

shopt -s nullglob
for d in "${SRC_E2E}"/*/; do
  k="$(basename "${d}")"

  # Ensure destination exists for every kernel.
  if [[ ! -d "${DST_KERNEL}/${k}" ]]; then
    echo "[create] ${DST_KERNEL}/${k}"
    mkdir -p "${DST_KERNEL}/${k}"
  fi

  echo "[sync] kernel=${k}"

  # 1) Update canonical files expected by Zeonica_Testbench naming.
  [[ -f "${d}/tmp-generated-dfg.dot" ]] && cp -f "${d}/tmp-generated-dfg.dot" "${DST_KERNEL}/${k}/${k}-dfg.dot"
  [[ -f "${d}/tmp-generated-dfg.yaml" ]] && cp -f "${d}/tmp-generated-dfg.yaml" "${DST_KERNEL}/${k}/${k}-dfg.yaml"
  [[ -f "${d}/tmp-generated-instructions.asm" ]] && cp -f "${d}/tmp-generated-instructions.asm" "${DST_KERNEL}/${k}/${k}-instructions.asm"
  [[ -f "${d}/tmp-generated-instructions.yaml" ]] && cp -f "${d}/tmp-generated-instructions.yaml" "${DST_KERNEL}/${k}/${k}-instructions.yaml"
done

# Optional: git commit/push for the submodule (kept commented on purpose).
# if [[ -n "$(git -C "${DST_TESTBENCH}" status --porcelain)" ]]; then
#   echo "[git] changes detected in Zeonica_Testbench"
#   git -C "${DST_TESTBENCH}" add -A
#   git -C "${DST_TESTBENCH}" commit -m "Update kernel outputs ($(date +%Y-%m-%d))"
#   git -C "${DST_TESTBENCH}" push
#   echo "[git] pushed Zeonica_Testbench changes"
# else
#   echo "[git] no changes to push in Zeonica_Testbench"
# fi

echo "[done] synced e2e outputs into Zeonica_Testbench submodule"

