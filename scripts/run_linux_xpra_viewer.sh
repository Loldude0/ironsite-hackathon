#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ORB_SCRIPT="${REPO_ROOT}/scripts/run_linux_orbslam3.sh"

XPRA_DISPLAY="${XPRA_DISPLAY:-100}"
XPRA_BIND_ADDR="${XPRA_BIND_ADDR:-0.0.0.0}"
XPRA_PORT="${XPRA_PORT:-14500}"
XPRA_TCP_AUTH="${XPRA_TCP_AUTH:-none}"  # On Tailscale, none is usually fine.

if ! command -v xpra >/dev/null 2>&1; then
  echo "[xpra] xpra is not installed on this Linux machine" >&2
  echo "[xpra] install it, then rerun this script" >&2
  exit 1
fi
if [[ ! -x "${ORB_SCRIPT}" ]]; then
  echo "[xpra] missing executable script: ${ORB_SCRIPT}" >&2
  exit 1
fi

quoted_args=()
for arg in "$@"; do
  quoted_args+=("$(printf '%q' "${arg}")")
done

start_child="${ORB_SCRIPT}"
if [[ ${#quoted_args[@]} -gt 0 ]]; then
  start_child+=" ${quoted_args[*]}"
fi

echo "[xpra] starting display :${XPRA_DISPLAY} on tcp ${XPRA_BIND_ADDR}:${XPRA_PORT}"
echo "[xpra] Windows attach example:"
echo "       xpra attach tcp:<linux-tailscale-ip>:${XPRA_PORT}"

exec xpra start ":${XPRA_DISPLAY}" \
  --daemon=no \
  --exit-with-children=yes \
  --mdns=no \
  --notifications=no \
  --bind-tcp="${XPRA_BIND_ADDR}:${XPRA_PORT}" \
  --tcp-auth="${XPRA_TCP_AUTH}" \
  --start-child="${start_child}"
