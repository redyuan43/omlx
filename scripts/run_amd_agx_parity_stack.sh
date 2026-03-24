#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Start the full AMD AGX parity stack in the background.

Usage:
  bash "scripts/run_amd_agx_parity_stack.sh" --manifest ".runtime/amd-agx-parity-manifest.json"
  bash "scripts/run_amd_agx_parity_stack.sh" --manifest ".runtime/amd-agx-parity-manifest.json" --bootstrap-missing
  bash "scripts/run_amd_agx_parity_stack.sh" --status
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${REPO_ROOT}/.runtime/amd-agx-parity-manifest.json"
STATE_DIR="${REPO_ROOT}/.runtime/amd-agx-parity-stack"
MAIN_BASE_PATH="${HOME}/.omlx-dgx-amd/agx-main"
OCR_BASE_PATH="${HOME}/.omlx-dgx-amd/agx-ocr"
WAIT_TIMEOUT_SECONDS="180"
BOOTSTRAP_MISSING=0
PULL_EMBEDDING=0
ENABLE_UMA_FALLBACK=0
FORCE_RESTART=0
SKIP_SMOKE=0
STATUS_ONLY=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
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
    --bootstrap-missing)
      BOOTSTRAP_MISSING=1
      shift
      ;;
    --pull-embedding)
      PULL_EMBEDDING=1
      shift
      ;;
    --enable-uma-fallback)
      ENABLE_UMA_FALLBACK=1
      shift
      ;;
    --force-restart)
      FORCE_RESTART=1
      shift
      ;;
    --skip-smoke)
      SKIP_SMOKE=1
      shift
      ;;
    --status)
      STATUS_ONLY=1
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

mkdir -p "${STATE_DIR}"

MAIN_PID_FILE="${STATE_DIR}/main.pid"
OCR_PID_FILE="${STATE_DIR}/ocr.pid"
MAIN_LOG="${STATE_DIR}/main.log"
OCR_LOG="${STATE_DIR}/ocr.log"
REGISTER_LOG="${STATE_DIR}/register.log"
SMOKE_JSON="${STATE_DIR}/smoke.json"
STACK_STATUS_JSON="${STATE_DIR}/status.json"
MAIN_RUNTIME_PID_FILE="${MAIN_BASE_PATH}/llama_cpp_model_pool/qwen35-35b/runtime/llama.pid"
RERANK_RUNTIME_PID_FILE="${MAIN_BASE_PATH}/llama_cpp_model_pool/rerank-qwen/runtime/llama.pid"
OCR_RUNTIME_PID_FILE="${OCR_BASE_PATH}/llama_cpp_model_pool/ocr-lite/runtime/llama.pid"

health_url() {
  local url="$1"
  python3 - "$url" <<'PY'
import json
import sys
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=2) as response:
        payload = json.loads(response.read().decode("utf-8"))
except Exception:
    print("0")
else:
    is_ok = False
    if response.status == 200:
        if payload.get("status") == "ok":
            is_ok = True
        elif payload.get("ok") is True:
            is_ok = True
    print("1" if is_ok else "0")
PY
}

pid_alive() {
  local pid_file="$1"
  if [[ ! -f "${pid_file}" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "${pid_file}")"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

print_status_json() {
  python3 - \
    "${MAIN_PID_FILE}" \
    "${OCR_PID_FILE}" \
    "${MAIN_RUNTIME_PID_FILE}" \
    "${RERANK_RUNTIME_PID_FILE}" \
    "${OCR_RUNTIME_PID_FILE}" \
    "${SMOKE_JSON}" \
    "${STACK_STATUS_JSON}" <<'PY'
import json
import os
import sys
import urllib.request

(
    main_pid_file,
    ocr_pid_file,
    main_runtime_pid_file,
    rerank_runtime_pid_file,
    ocr_runtime_pid_file,
    smoke_path,
    status_path,
) = sys.argv[1:]

def pid_info(path: str) -> dict:
    if not os.path.exists(path):
        return {"pid_file": path, "present": False, "pid": None, "alive": False}
    raw = open(path, encoding="utf-8").read().strip()
    pid = int(raw) if raw else None
    alive = False
    if pid:
        try:
            os.kill(pid, 0)
        except OSError:
            alive = False
        else:
            alive = True
    return {"pid_file": path, "present": True, "pid": pid, "alive": alive}

def health(url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
            ok = False
            if response.status == 200:
                if payload.get("status") == "ok":
                    ok = True
                elif payload.get("ok") is True:
                    ok = True
            return {"url": url, "ok": ok}
    except Exception as exc:
        return {"url": url, "ok": False, "error": str(exc)}

payload = {
    "main": pid_info(main_pid_file),
    "ocr": pid_info(ocr_pid_file),
    "runtimes": {
        "main": pid_info(main_runtime_pid_file),
        "rerank": pid_info(rerank_runtime_pid_file),
        "ocr": pid_info(ocr_runtime_pid_file),
    },
    "health": {
        "main": health("http://127.0.0.1:8008/health"),
        "ocr": health("http://127.0.0.1:8012/health"),
    },
}
if os.path.exists(smoke_path):
    payload["smoke_json"] = smoke_path
open(status_path, "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False, indent=2))
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
}

wait_health() {
  local name="$1"
  local url="$2"
  local deadline
  deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
  until [[ "$(health_url "${url}")" == "1" ]]; do
    if (( SECONDS >= deadline )); then
      echo "${name} health check timed out: ${url}" >&2
      return 1
    fi
    sleep 1
  done
}

start_bg() {
  local name="$1"
  local pid_file="$2"
  local log_file="$3"
  shift 3
  if pid_alive "${pid_file}"; then
    echo "${name} already managed by stack launcher: pid $(cat "${pid_file}")"
    return 0
  fi
  nohup "$@" >"${log_file}" 2>&1 &
  echo "$!" >"${pid_file}"
  echo "started ${name}: pid $(cat "${pid_file}")"
}

if [[ "${STATUS_ONLY}" == "1" ]]; then
  print_status_json
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "manifest=${MANIFEST}"
  echo "state_dir=${STATE_DIR}"
  echo "main_base_path=${MAIN_BASE_PATH}"
  echo "ocr_base_path=${OCR_BASE_PATH}"
  echo "wait_timeout_seconds=${WAIT_TIMEOUT_SECONDS}"
  echo "bootstrap_missing=${BOOTSTRAP_MISSING}"
  echo "pull_embedding=${PULL_EMBEDDING}"
  echo "enable_uma_fallback=${ENABLE_UMA_FALLBACK}"
  echo "force_restart=${FORCE_RESTART}"
  echo "skip_smoke=${SKIP_SMOKE}"
  exit 0
fi

if [[ "${FORCE_RESTART}" == "1" ]]; then
  bash "${REPO_ROOT}/scripts/stop_amd_agx_parity_stack.sh" --state-dir "${STATE_DIR}" || true
fi

if [[ "${BOOTSTRAP_MISSING}" == "1" ]]; then
  BOOTSTRAP_ARGS=(python3 "${REPO_ROOT}/scripts/bootstrap_amd_agx_parity.py")
  if [[ "${PULL_EMBEDDING}" == "1" ]]; then
    BOOTSTRAP_ARGS+=(--pull-embedding)
  fi
  BOOTSTRAP_ARGS+=(--download-missing --output "${MANIFEST}")
  (
    cd "${REPO_ROOT}"
    "${BOOTSTRAP_ARGS[@]}"
  )
fi

if [[ ! -f "${MANIFEST}" ]]; then
  echo "manifest not found: ${MANIFEST}" >&2
  exit 1
fi

if [[ "$(health_url "http://127.0.0.1:8008/health")" == "1" && ! -f "${MAIN_PID_FILE}" ]]; then
  echo "main control plane is already running on 8008 but is not managed by ${MAIN_PID_FILE}" >&2
  exit 1
fi
if [[ "$(health_url "http://127.0.0.1:8012/health")" == "1" && ! -f "${OCR_PID_FILE}" ]]; then
  echo "ocr control plane is already running on 8012 but is not managed by ${OCR_PID_FILE}" >&2
  exit 1
fi

MAIN_COMMAND=(
  bash
  "${REPO_ROOT}/scripts/run_amd_agx_parity_main.sh"
  --manifest
  "${MANIFEST}"
  --base-path
  "${MAIN_BASE_PATH}"
)
OCR_COMMAND=(
  bash
  "${REPO_ROOT}/scripts/run_amd_agx_parity_ocr.sh"
  --manifest
  "${MANIFEST}"
  --base-path
  "${OCR_BASE_PATH}"
)
if [[ "${ENABLE_UMA_FALLBACK}" == "1" ]]; then
  MAIN_COMMAND+=(--enable-uma-fallback)
  OCR_COMMAND+=(--enable-uma-fallback)
fi

start_bg "main" "${MAIN_PID_FILE}" "${MAIN_LOG}" "${MAIN_COMMAND[@]}"
wait_health "main" "http://127.0.0.1:8008/health"

(
  cd "${REPO_ROOT}"
  python3 "${REPO_ROOT}/scripts/register_amd_agx_parity_models.py" --manifest "${MANIFEST}"
) >"${REGISTER_LOG}" 2>&1

start_bg "ocr" "${OCR_PID_FILE}" "${OCR_LOG}" "${OCR_COMMAND[@]}"
wait_health "ocr" "http://127.0.0.1:8012/health"

if [[ "${SKIP_SMOKE}" == "0" ]]; then
  (
    cd "${REPO_ROOT}"
    python3 "${REPO_ROOT}/scripts/smoke_amd_agx_parity.py"
  ) >"${SMOKE_JSON}"
  cat "${SMOKE_JSON}"
fi

print_status_json > /dev/null
echo "stack launcher state written to ${STATE_DIR}"
