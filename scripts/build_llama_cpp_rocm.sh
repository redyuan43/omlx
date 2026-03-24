#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build a HIP-enabled llama.cpp for the local AMD baseline.

Usage:
  bash scripts/build_llama_cpp_rocm.sh [options]

Options:
  --source-dir PATH    Clone/update llama.cpp into PATH.
                       Default: <repo>/.runtime/llama.cpp-src
  --build-dir PATH     Build output directory.
                       Default: <repo>/.runtime/llama.cpp-build
  --ref REF            Git ref to build. Default: master
  --gpu-targets LIST   GPU targets passed to -DGPU_TARGETS=...
                       Default: first gfx target from rocminfo
  --jobs N             Parallel build jobs. Default: nproc or 8
  --skip-update        Do not fetch/update an existing source checkout
  --help               Show this message
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${REPO_ROOT}/.runtime/llama.cpp-src"
BUILD_DIR="${REPO_ROOT}/.runtime/llama.cpp-build"
REF="master"
GPU_TARGETS=""
JOBS=""
SKIP_UPDATE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-dir)
      SOURCE_DIR="$2"
      shift 2
      ;;
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --ref)
      REF="$2"
      shift 2
      ;;
    --gpu-targets)
      GPU_TARGETS="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --skip-update)
      SKIP_UPDATE=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

detect_gpu_target() {
  local detected
  detected="$(rocminfo 2>/dev/null | awk '/Name:[[:space:]]+gfx/{print $2; exit}')"
  if [[ -z "${detected}" ]]; then
    echo "Failed to detect an AMD GPU target from rocminfo. Pass --gpu-targets explicitly." >&2
    exit 1
  fi
  printf '%s\n' "${detected}"
}

resolve_device_lib_path() {
  local rocm_path="$1"
  local candidate
  for candidate in "${rocm_path}/amdgcn/bitcode" "${rocm_path}/amdgcn"; do
    if [[ -f "${candidate}/oclc_abi_version_400.bc" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  echo "Failed to locate ROCm device libraries under ${rocm_path}." >&2
  exit 1
}

require_command git
require_command cmake
require_command hipconfig
require_command rocminfo

if [[ -z "${JOBS}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
  else
    JOBS="8"
  fi
fi

if [[ -z "${GPU_TARGETS}" ]]; then
  GPU_TARGETS="$(detect_gpu_target)"
fi

ROCM_PATH="$(hipconfig -R)"
HIP_CLANG_PATH="$(hipconfig -l)"
HIP_DEVICE_LIB_PATH="$(resolve_device_lib_path "${ROCM_PATH}")"

mkdir -p "$(dirname "${SOURCE_DIR}")" "$(dirname "${BUILD_DIR}")"

if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "${SOURCE_DIR}"
elif [[ "${SKIP_UPDATE}" != "1" ]]; then
  git -C "${SOURCE_DIR}" fetch --tags origin
fi

if [[ "${SKIP_UPDATE}" != "1" ]]; then
  if git -C "${SOURCE_DIR}" rev-parse --verify "${REF}^{commit}" >/dev/null 2>&1; then
    git -C "${SOURCE_DIR}" checkout "${REF}"
  else
    git -C "${SOURCE_DIR}" fetch origin "${REF}"
    git -C "${SOURCE_DIR}" checkout FETCH_HEAD
  fi
fi

echo "Building llama.cpp with HIP"
echo "  source: ${SOURCE_DIR}"
echo "  build:  ${BUILD_DIR}"
echo "  ref:    ${REF}"
echo "  target: ${GPU_TARGETS}"
echo "  rocm:   ${ROCM_PATH}"
echo "  clang:  ${HIP_CLANG_PATH}/clang"
echo "  devlib: ${HIP_DEVICE_LIB_PATH}"

env \
  HIPCXX="${HIP_CLANG_PATH}/clang" \
  HIP_PATH="${ROCM_PATH}" \
  HIP_DEVICE_LIB_PATH="${HIP_DEVICE_LIB_PATH}" \
  cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" \
    -DGGML_HIP=ON \
    -DGPU_TARGETS="${GPU_TARGETS}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_SERVER=ON

cmake --build "${BUILD_DIR}" --config Release --parallel "${JOBS}"

if [[ ! -x "${BUILD_DIR}/bin/llama-server" ]]; then
  echo "Build finished but ${BUILD_DIR}/bin/llama-server was not found." >&2
  exit 1
fi

echo
echo "Built binary:"
echo "  ${BUILD_DIR}/bin/llama-server"
