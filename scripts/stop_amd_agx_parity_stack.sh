#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Stop the background AMD AGX parity stack started by run_amd_agx_parity_stack.sh.

Usage:
  bash "scripts/stop_amd_agx_parity_stack.sh"
  bash "scripts/stop_amd_agx_parity_stack.sh" --state-dir ".runtime/amd-agx-parity-stack"
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${REPO_ROOT}/.runtime/amd-agx-parity-stack"
MAIN_BASE_PATH="${HOME}/.omlx-dgx-amd/agx-main"
OCR_BASE_PATH="${HOME}/.omlx-dgx-amd/agx-ocr"
WAIT_TIMEOUT_SECONDS="30"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --state-dir)
      STATE_DIR="$2"
      shift 2
      ;;
    --main-base-path)
      MAIN_BASE_PATH="$2"
      shift 2
      ;;
    --ocr-base-path)
      OCR_BASE_PATH="$2"
      shift 2
      ;;
    --wait-timeout-seconds)
      WAIT_TIMEOUT_SECONDS="$2"
      shift 2
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

stop_pid_file() {
  local name="$1"
  local pid_file="$2"
  if [[ ! -f "${pid_file}" ]]; then
    echo "${name}: no pid file"
    return 0
  fi
  local pid
  pid="$(cat "${pid_file}")"
  if [[ -z "${pid}" ]]; then
    rm -f "${pid_file}"
    echo "${name}: empty pid file removed"
    return 0
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    rm -f "${pid_file}"
    echo "${name}: stale pid file removed"
    return 0
  fi
  kill "${pid}"
  local deadline
  deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
  while kill -0 "${pid}" 2>/dev/null; do
    if (( SECONDS >= deadline )); then
      echo "${name}: process ${pid} did not exit after ${WAIT_TIMEOUT_SECONDS}s" >&2
      return 1
    fi
    sleep 1
  done
  rm -f "${pid_file}"
  echo "${name}: stopped pid ${pid}"
}

stop_pid_file "ocr-runtime" "${OCR_BASE_PATH}/llama_cpp_model_pool/ocr-lite/runtime/llama.pid"
stop_pid_file "rerank-runtime" "${MAIN_BASE_PATH}/llama_cpp_model_pool/rerank-qwen/runtime/llama.pid"
stop_pid_file "main-runtime" "${MAIN_BASE_PATH}/llama_cpp_model_pool/qwen35-35b/runtime/llama.pid"
stop_pid_file "ocr" "${STATE_DIR}/ocr.pid"
stop_pid_file "main" "${STATE_DIR}/main.pid"
