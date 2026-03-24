#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Start the AMD AGX parity main service on omlx_dgx.

Usage:
  bash scripts/run_amd_agx_parity_main.sh --manifest /path/to/manifest.json
  bash scripts/run_amd_agx_parity_main.sh --main-artifact-path /path/to/Qwen3.5-35B-A3B-Q4_K_M.gguf --main-mmproj-path /path/to/mmproj.gguf
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST=""
MAIN_ARTIFACT_PATH=""
MAIN_MMPROJ_PATH=""
LAUNCHER_BINARY="${REPO_ROOT}/.runtime/llama.cpp-build/bin/llama-server"
BASE_PATH="${HOME}/.omlx-dgx-amd/agx-main"
HOST="127.0.0.1"
CONTROL_PLANE_PORT="8008"
RUNTIME_PORT="30000"
ENABLE_UMA_FALLBACK=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --main-artifact-path)
      MAIN_ARTIFACT_PATH="$2"
      shift 2
      ;;
    --main-mmproj-path)
      MAIN_MMPROJ_PATH="$2"
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
  if [[ -z "${MAIN_ARTIFACT_PATH}" ]]; then
    MAIN_ARTIFACT_PATH="$(python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); print(data["chat"]["artifact_path"])' "${MANIFEST}")"
  fi
  if [[ -z "${MAIN_MMPROJ_PATH}" ]]; then
    MAIN_MMPROJ_PATH="$(python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); print(data["chat"]["mmproj_path"])' "${MANIFEST}")"
  fi
fi

if [[ -z "${MAIN_ARTIFACT_PATH}" || -z "${MAIN_MMPROJ_PATH}" ]]; then
  echo "main artifact and mmproj paths are required" >&2
  exit 1
fi
if [[ ! -f "${MAIN_ARTIFACT_PATH}" ]]; then
  echo "main model artifact not found: ${MAIN_ARTIFACT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${MAIN_MMPROJ_PATH}" ]]; then
  echo "main mmproj not found: ${MAIN_MMPROJ_PATH}" >&2
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
  "qwen35-35b"
  --model-alias
  "qwen35-35b"
  --primary-service
  "chat"
  --quant-mode
  gguf_experimental
  --model-source
  gguf
  --artifact-path
  "${MAIN_ARTIFACT_PATH}"
  --mmproj-path
  "${MAIN_MMPROJ_PATH}"
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
