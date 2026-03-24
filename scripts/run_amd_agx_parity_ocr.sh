#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Start the AMD AGX parity OCR service on omlx_dgx.

Usage:
  bash scripts/run_amd_agx_parity_ocr.sh --manifest /path/to/manifest.json
  bash scripts/run_amd_agx_parity_ocr.sh --ocr-artifact-path /path/to/GLM-OCR-Q4_K_M.gguf --ocr-mmproj-path /path/to/mmproj.gguf
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST=""
OCR_ARTIFACT_PATH=""
OCR_MMPROJ_PATH=""
LAUNCHER_BINARY="${REPO_ROOT}/.runtime/llama.cpp-build/bin/llama-server"
BASE_PATH="${HOME}/.omlx-dgx-amd/agx-ocr"
HOST="127.0.0.1"
CONTROL_PLANE_PORT="8012"
RUNTIME_PORT="30020"
ENABLE_UMA_FALLBACK=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --ocr-artifact-path)
      OCR_ARTIFACT_PATH="$2"
      shift 2
      ;;
    --ocr-mmproj-path)
      OCR_MMPROJ_PATH="$2"
      shift 2
      ;;
    --launcher-binary)
      LAUNCHER_BINARY="$2"
      shift 2
      ;;
    --base-path)
      BASE_PATH="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --control-plane-port)
      CONTROL_PLANE_PORT="$2"
      shift 2
      ;;
    --runtime-port)
      RUNTIME_PORT="$2"
      shift 2
      ;;
    --enable-uma-fallback)
      ENABLE_UMA_FALLBACK=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
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

if [[ -n "${MANIFEST}" ]]; then
  if [[ -z "${OCR_ARTIFACT_PATH}" ]]; then
    OCR_ARTIFACT_PATH="$(python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); print(data["ocr"]["artifact_path"])' "${MANIFEST}")"
  fi
  if [[ -z "${OCR_MMPROJ_PATH}" ]]; then
    OCR_MMPROJ_PATH="$(python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); print(data["ocr"]["mmproj_path"])' "${MANIFEST}")"
  fi
fi

if [[ -z "${OCR_ARTIFACT_PATH}" || -z "${OCR_MMPROJ_PATH}" ]]; then
  echo "ocr artifact and mmproj paths are required" >&2
  exit 1
fi
if [[ ! -f "${OCR_ARTIFACT_PATH}" ]]; then
  echo "ocr model artifact not found: ${OCR_ARTIFACT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${OCR_MMPROJ_PATH}" ]]; then
  echo "ocr mmproj not found: ${OCR_MMPROJ_PATH}" >&2
  exit 1
fi
if [[ ! -x "${LAUNCHER_BINARY}" ]]; then
  echo "llama-server binary not found or not executable: ${LAUNCHER_BINARY}" >&2
  exit 1
fi

COMMAND=(
  python3
  -m
  omlx_dgx.cli
  serve
  --base-path
  "${BASE_PATH}"
  --backend-kind
  llama_cpp
  --backend-url
  "http://${HOST}:${RUNTIME_PORT}"
  --host
  "${HOST}"
  --port
  "${CONTROL_PLANE_PORT}"
  --model-id
  "ocr-lite"
  --model-alias
  "ocr-lite"
  --primary-service
  "ocr"
  --quant-mode
  gguf_experimental
  --model-source
  gguf
  --artifact-path
  "${OCR_ARTIFACT_PATH}"
  --mmproj-path
  "${OCR_MMPROJ_PATH}"
  --launcher-binary
  "${LAUNCHER_BINARY}"
  --serving-preset
  "single_session_low_latency"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'PYTHONPATH=%q' "${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  if [[ "${ENABLE_UMA_FALLBACK}" == "1" ]]; then
    printf ' GGML_CUDA_ENABLE_UNIFIED_MEMORY=1'
  fi
  printf ' '
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "${ENABLE_UMA_FALLBACK}" == "1" ]]; then
  export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
fi

cd "${REPO_ROOT}"
exec "${COMMAND[@]}"
