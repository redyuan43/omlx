#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Start the AMD Qwen3.5-4B baseline on omlx_dgx.

Usage:
  bash scripts/run_amd_qwen35_baseline.sh --artifact-path /path/to/Qwen3.5-4B-Q4_K_M.gguf [options]

Options:
  --artifact-path PATH       Local GGUF path for the main Qwen3.5-4B Q4_K_M model. Required.
  --launcher-binary PATH     llama-server binary. Default: <repo>/.runtime/llama.cpp-build/bin/llama-server
  --preset NAME              single_session_low_latency or mixed_traffic.
                             Default: single_session_low_latency
  --base-path PATH           DGX state root. Default depends on preset.
  --control-plane-port PORT  Default: 8010 (single) or 8020 (mixed)
  --runtime-port PORT        Default: 31000 (single) or 31200 (mixed)
  --host HOST                Default: 127.0.0.1
  --model-id ID              Default: qwen35-4b
  --model-alias ALIAS        Default: qwen35-4b
  --enable-uma-fallback      Export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 before launch
  --dry-run                  Print the command without starting the server
  --help                     Show this message
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_PATH=""
LAUNCHER_BINARY="${REPO_ROOT}/.runtime/llama.cpp-build/bin/llama-server"
PRESET="single_session_low_latency"
BASE_PATH=""
CONTROL_PLANE_PORT=""
RUNTIME_PORT=""
HOST="127.0.0.1"
MODEL_ID="qwen35-4b"
MODEL_ALIAS="qwen35-4b"
ENABLE_UMA_FALLBACK=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifact-path)
      ARTIFACT_PATH="$2"
      shift 2
      ;;
    --launcher-binary)
      LAUNCHER_BINARY="$2"
      shift 2
      ;;
    --preset)
      PRESET="$2"
      shift 2
      ;;
    --base-path)
      BASE_PATH="$2"
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
    --host)
      HOST="$2"
      shift 2
      ;;
    --model-id)
      MODEL_ID="$2"
      shift 2
      ;;
    --model-alias)
      MODEL_ALIAS="$2"
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

if [[ -z "${ARTIFACT_PATH}" ]]; then
  echo "--artifact-path is required." >&2
  exit 1
fi

if [[ ! -f "${ARTIFACT_PATH}" ]]; then
  echo "Main model artifact not found: ${ARTIFACT_PATH}" >&2
  exit 1
fi

if [[ ! -x "${LAUNCHER_BINARY}" ]]; then
  echo "llama-server binary not found or not executable: ${LAUNCHER_BINARY}" >&2
  exit 1
fi

case "${PRESET}" in
  single_session_low_latency)
    : "${CONTROL_PLANE_PORT:=8010}"
    : "${RUNTIME_PORT:=31000}"
    : "${BASE_PATH:=${HOME}/.omlx-dgx-amd/qwen35-4b-single}"
    ;;
  mixed_traffic)
    : "${CONTROL_PLANE_PORT:=8020}"
    : "${RUNTIME_PORT:=31200}"
    : "${BASE_PATH:=${HOME}/.omlx-dgx-amd/qwen35-4b-mixed}"
    ;;
  *)
    echo "Unsupported preset: ${PRESET}" >&2
    exit 1
    ;;
esac

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
  "${MODEL_ID}"
  --model-alias
  "${MODEL_ALIAS}"
  --quant-mode
  gguf_experimental
  --model-source
  gguf
  --artifact-path
  "${ARTIFACT_PATH}"
  --launcher-binary
  "${LAUNCHER_BINARY}"
  --serving-preset
  "${PRESET}"
)

echo "Launching AMD baseline"
echo "  preset:            ${PRESET}"
echo "  state:             ${BASE_PATH}"
echo "  control plane:     http://${HOST}:${CONTROL_PLANE_PORT}"
echo "  runtime:           http://${HOST}:${RUNTIME_PORT}"
echo "  launcher:          ${LAUNCHER_BINARY}"
echo "  model artifact:    ${ARTIFACT_PATH}"
if [[ "${ENABLE_UMA_FALLBACK}" == "1" ]]; then
  echo "  unified memory:    enabled"
else
  echo "  unified memory:    disabled"
fi

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
