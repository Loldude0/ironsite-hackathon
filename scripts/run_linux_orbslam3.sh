#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ORB_SLAM3_ROOT="${ORB_SLAM3_ROOT:-$HOME/Projects/ORB_SLAM3}"
VOCAB_PATH="${VOCAB_PATH:-${ORB_SLAM3_ROOT}/Vocabulary/ORBvoc.txt}"
SETTINGS_TEMPLATE="${SETTINGS_TEMPLATE:-${REPO_ROOT}/configs/iphone_record3d_rgbd.yaml}"

SRC_FILE="${REPO_ROOT}/linux_orbslam3_rgbd_stream.cpp"
BUILD_DIR="${REPO_ROOT}/build"
BIN_FILE="${BUILD_DIR}/orbslam3_rgbd_stream"

ZMQ_ENDPOINT="${ZMQ_ENDPOINT:-tcp://0.0.0.0:5555}"
ZMQ_TOPIC="${ZMQ_TOPIC:-rgbd}"
ZMQ_BIND="${ZMQ_BIND:-1}"            # 1 => receiver binds, 0 => receiver connects
ENABLE_VIEWER="${ENABLE_VIEWER:-1}"  # 1 => Pangolin on, 0 => off
TRAJECTORY_PATH="${TRAJECTORY_PATH:-}"

if [[ "${ENABLE_VIEWER}" == "1" && -z "${DISPLAY:-}" ]]; then
  echo "[run] no DISPLAY detected; disabling Pangolin viewer (set ENABLE_VIEWER=1 once X11/xpra is available)"
  ENABLE_VIEWER="0"
fi

if [[ ! -f "${VOCAB_PATH}" ]]; then
  echo "[run] missing vocabulary file: ${VOCAB_PATH}" >&2
  exit 1
fi
if [[ ! -f "${SETTINGS_TEMPLATE}" ]]; then
  echo "[run] missing settings template: ${SETTINGS_TEMPLATE}" >&2
  exit 1
fi
if [[ ! -f "${SRC_FILE}" ]]; then
  echo "[run] missing source file: ${SRC_FILE}" >&2
  exit 1
fi

if pkg-config --exists opencv4; then
  OPENCV_PKG="opencv4"
elif pkg-config --exists opencv; then
  OPENCV_PKG="opencv"
else
  echo "[run] OpenCV pkg-config package not found" >&2
  exit 1
fi
if ! pkg-config --exists eigen3; then
  echo "[run] eigen3 pkg-config package not found" >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}"

NEED_BUILD=0
if [[ ! -x "${BIN_FILE}" ]]; then
  NEED_BUILD=1
elif [[ "${SRC_FILE}" -nt "${BIN_FILE}" ]]; then
  NEED_BUILD=1
elif [[ "${SETTINGS_TEMPLATE}" -nt "${BIN_FILE}" ]]; then
  NEED_BUILD=1
fi

if [[ "${FORCE_REBUILD:-0}" == "1" ]]; then
  NEED_BUILD=1
fi

if [[ "${NEED_BUILD}" == "1" ]]; then
  echo "[build] compiling ${BIN_FILE}"
  g++ -std=c++17 -O2 "${SRC_FILE}" -o "${BIN_FILE}" \
    -I"${ORB_SLAM3_ROOT}" \
    -I"${ORB_SLAM3_ROOT}/include" \
    -I"${ORB_SLAM3_ROOT}/include/CameraModels" \
    -I"${ORB_SLAM3_ROOT}/Thirdparty/Sophus" \
    $(pkg-config --cflags "${OPENCV_PKG}" libzmq libzstd jsoncpp eigen3) \
    -L"${ORB_SLAM3_ROOT}/lib" \
    -L"${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib" \
    -L"${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib" \
    -Wl,--copy-dt-needed-entries \
    -Wl,--no-as-needed \
    -Wl,-rpath,"${ORB_SLAM3_ROOT}/lib:${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib:${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib" \
    -lORB_SLAM3 -lDBoW2 -lg2o \
    -lGLEW -lGL \
    $(pkg-config --libs "${OPENCV_PKG}" libzmq libzstd jsoncpp) \
    -lpthread
fi

LD_PREFIX="${ORB_SLAM3_ROOT}/lib:${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib:${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${LD_PREFIX}:${LD_LIBRARY_PATH}"
else
  export LD_LIBRARY_PATH="${LD_PREFIX}"
fi

CMD=(
  "${BIN_FILE}"
  --vocab "${VOCAB_PATH}"
  --settings-template "${SETTINGS_TEMPLATE}"
  --endpoint "${ZMQ_ENDPOINT}"
  --topic "${ZMQ_TOPIC}"
)

if [[ "${ZMQ_BIND}" == "0" ]]; then
  CMD+=(--connect)
fi
if [[ "${ENABLE_VIEWER}" == "0" ]]; then
  CMD+=(--no-viewer)
fi
if [[ -n "${TRAJECTORY_PATH}" ]]; then
  CMD+=(--trajectory "${TRAJECTORY_PATH}")
fi

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

echo "[run] starting ORB bridge"
echo "[run] endpoint=${ZMQ_ENDPOINT} topic=${ZMQ_TOPIC} bind=${ZMQ_BIND} viewer=${ENABLE_VIEWER}"
exec "${CMD[@]}"
