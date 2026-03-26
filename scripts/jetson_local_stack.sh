#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/venvs/omlx-dgx/bin/python}"
MAIN_BASE_PATH="${MAIN_BASE_PATH:-/data/omlx-dgx-35b-ctx32768}"
OCR_BASE_PATH="${OCR_BASE_PATH:-/data/omlx-dgx-ocr-lite-gguf}"
MAIN_URL="${MAIN_URL:-http://127.0.0.1:8008}"
OCR_URL="${OCR_URL:-http://127.0.0.1:8012}"
STATE_DIR="${STATE_DIR:-/data/omlx-local-stack}"
LOG_DIR="$STATE_DIR/logs"
PID_DIR="$STATE_DIR/pids"
MAIN_PIDFILE="$PID_DIR/main-control-plane.pid"
OCR_PIDFILE="$PID_DIR/ocr-control-plane.pid"

readonly TINY_PNG_DATA_URL='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO9W7x8AAAAASUVORK5CYII='

mkdir -p "$LOG_DIR" "$PID_DIR"

usage() {
  cat <<'EOF'
Usage:
  scripts/jetson_local_stack.sh start
  scripts/jetson_local_stack.sh stop
  scripts/jetson_local_stack.sh restart
  scripts/jetson_local_stack.sh status

Environment overrides:
  PYTHON_BIN        Python used to run omlx_dgx.cli
  MAIN_BASE_PATH    Base path for main 35B service
  OCR_BASE_PATH     Base path for OCR service
  MAIN_URL          Main control plane URL
  OCR_URL           OCR control plane URL
  STATE_DIR         PID/log directory root
EOF
}

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "[$(timestamp)] $*"
}

read_pid() {
  local pidfile="$1"
  [[ -f "$pidfile" ]] || return 1
  local pid
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  echo "$pid"
}

is_pid_running() {
  local pid="$1"
  kill -0 "$pid" 2>/dev/null
}

port_ready() {
  local url="$1"
  curl -fsS "$url" >/dev/null 2>&1
}

wait_url() {
  local url="$1"
  local attempts="${2:-120}"
  local delay="${3:-1}"
  local i
  for ((i=1; i<=attempts; i++)); do
    if port_ready "$url"; then
      return 0
    fi
    sleep "$delay"
  done
  return 1
}

start_control_plane() {
  local name="$1"
  local base_path="$2"
  local health_url="$3"
  local pidfile="$4"
  local logfile="$5"

  local existing_pid=""
  if existing_pid="$(read_pid "$pidfile")" && is_pid_running "$existing_pid"; then
    log "$name already running with PID $existing_pid"
    return 0
  fi

  rm -f "$pidfile"
  log "starting $name"
  nohup "$PYTHON_BIN" -m omlx_dgx.cli serve --base-path "$base_path" >"$logfile" 2>&1 &
  local pid=$!
  echo "$pid" > "$pidfile"

  if ! wait_url "$health_url" 120 1; then
    log "$name failed to become ready; see $logfile"
    return 1
  fi
  log "$name ready on $health_url (PID $pid)"
}

json_post() {
  local url="$1"
  local payload="$2"
  curl -fsS \
    -H 'Content-Type: application/json' \
    -d "$payload" \
    "$url"
}

warm_main_service() {
  log "warming qwen35-35b"
  json_post "$MAIN_URL/v1/chat/completions" '{
    "model":"qwen35-35b",
    "stream":false,
    "messages":[{"role":"user","content":"Reply with exactly PONG."}],
    "extra_body":{"think":false},
    "max_tokens":8
  }' >/dev/null

  log "warming rerank-qwen"
  json_post "$MAIN_URL/v1/rerank" '{
    "model":"rerank-qwen",
    "query":"Which sentence is about weather?",
    "documents":["The sky is cloudy today.","This document is about taxes."]
  }' >/dev/null
}

warm_ocr_service() {
  log "warming ocr-lite"
  json_post "$OCR_URL/v1/chat/completions" "$(cat <<JSON
{
  "model":"ocr-lite",
  "stream":false,
  "ocr":true,
  "messages":[
    {
      "role":"user",
      "content":[
        {"type":"text","text":"Please OCR this image and output the text only."},
        {"type":"image_url","image_url":{"url":"$TINY_PNG_DATA_URL"}}
      ]
    }
  ],
  "temperature":0,
  "max_tokens":32
}
JSON
)" >/dev/null || true
}

stop_by_pidfile() {
  local name="$1"
  local pidfile="$2"
  local base_path="$3"
  local pid=""

  if pid="$(read_pid "$pidfile")" && is_pid_running "$pid"; then
    log "stopping $name control plane PID $pid"
    kill "$pid" 2>/dev/null || true
    for _ in {1..20}; do
      if ! is_pid_running "$pid"; then
        break
      fi
      sleep 1
    done
    if is_pid_running "$pid"; then
      log "force-killing $name control plane PID $pid"
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$pidfile"

  local pattern="$base_path/llama_cpp_model_pool"
  if pgrep -f "$pattern" >/dev/null 2>&1; then
    log "stopping orphan runtimes matching $pattern"
    pkill -TERM -f "$pattern" || true
    sleep 2
    pkill -KILL -f "$pattern" || true
  fi
}

status_one() {
  local name="$1"
  local pidfile="$2"
  local health_url="$3"
  local pid=""

  if pid="$(read_pid "$pidfile")" && is_pid_running "$pid"; then
    echo "$name: running (PID $pid)"
  else
    echo "$name: stopped"
  fi

  if curl -fsS "$health_url/v1/models" >/dev/null 2>&1; then
    echo "  api: ready at $health_url"
  else
    echo "  api: not ready"
  fi
}

status_stack() {
  status_one "main-control-plane" "$MAIN_PIDFILE" "$MAIN_URL"
  status_one "ocr-control-plane" "$OCR_PIDFILE" "$OCR_URL"
  echo "runtime processes:"
  ps -eo pid,ppid,etime,cmd | egrep 'omlx_dgx\.cli serve|llama-server' | grep -v egrep || true
}

start_stack() {
  start_control_plane \
    "main-control-plane" \
    "$MAIN_BASE_PATH" \
    "$MAIN_URL/v1/models" \
    "$MAIN_PIDFILE" \
    "$LOG_DIR/main-control-plane.log"

  start_control_plane \
    "ocr-control-plane" \
    "$OCR_BASE_PATH" \
    "$OCR_URL/v1/models" \
    "$OCR_PIDFILE" \
    "$LOG_DIR/ocr-control-plane.log"

  warm_main_service
  warm_ocr_service
  status_stack
}

stop_stack() {
  stop_by_pidfile "main-control-plane" "$MAIN_PIDFILE" "$MAIN_BASE_PATH"
  stop_by_pidfile "ocr-control-plane" "$OCR_PIDFILE" "$OCR_BASE_PATH"
  status_stack
}

action="${1:-}"
case "$action" in
  start)
    start_stack
    ;;
  stop)
    stop_stack
    ;;
  restart)
    stop_stack
    start_stack
    ;;
  status)
    status_stack
    ;;
  *)
    usage
    exit 1
    ;;
esac
