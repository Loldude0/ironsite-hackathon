#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="${FUSION_CONFIG_PATH:-${REPO_ROOT}/fusion/config.yaml}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[fusion] missing config: ${CONFIG_PATH}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
echo "[fusion] starting with config=${CONFIG_PATH}"
exec python -m fusion.server --config "${CONFIG_PATH}"
