#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODEX_HOME_DIR="${CODEX_HOME:-$HOME/.codex}"
GLOBAL_SKILL_PATH="${CODEX_HOME_DIR}/skills/omlx-dgx-phase-executor/SKILL.md"
REPO_SKILL_PATH="${ROOT_DIR}/codex_skills/omlx-dgx-phase-executor/SKILL.md"
PHASE_TARGET="${1:-}"
shift || true

SANDBOX_MODE="dangerous"
OUTPUT_FILE=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_codex_phase.sh <phase-number|phase-file> [options]

Examples:
  bash scripts/run_codex_phase.sh 1
  bash scripts/run_codex_phase.sh plans/codex-phase-2.md --sandbox dangerous
  bash scripts/run_codex_phase.sh 3 --sandbox read-only --dry-run

Options:
  --sandbox <full-auto|dangerous|read-only>
  --output <file>
  --dry-run
EOF
}

if [[ -z "${PHASE_TARGET}" ]]; then
  usage
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sandbox)
      SANDBOX_MODE="${2:-}"
      shift 2
      ;;
    --output)
      OUTPUT_FILE="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "${PHASE_TARGET}" in
  [1-5]|phase-[1-5])
    PHASE_NUMBER="${PHASE_TARGET##*-}"
    PHASE_DOC="${ROOT_DIR}/plans/codex-phase-${PHASE_NUMBER}.md"
    ;;
  *)
    if [[ "${PHASE_TARGET}" = /* ]]; then
      PHASE_DOC="${PHASE_TARGET}"
    else
      PHASE_DOC="${ROOT_DIR}/${PHASE_TARGET}"
    fi
    ;;
esac

if [[ ! -f "${PHASE_DOC}" ]]; then
  echo "Phase document not found: ${PHASE_DOC}" >&2
  exit 1
fi

if [[ -f "${GLOBAL_SKILL_PATH}" ]]; then
  SKILL_PATH="${GLOBAL_SKILL_PATH}"
else
  SKILL_PATH="${REPO_SKILL_PATH}"
fi

if [[ ! -f "${SKILL_PATH}" ]]; then
  echo "Skill file not found: ${SKILL_PATH}" >&2
  exit 1
fi

if ! command -v codex >/dev/null 2>&1; then
  echo "codex CLI not found in PATH" >&2
  exit 1
fi

PROMPT_FILE="$(mktemp)"
trap 'rm -f "${PROMPT_FILE}"' EXIT

cat > "${PROMPT_FILE}" <<EOF
You are Codex CLI working inside ${ROOT_DIR}.

Before editing anything, read and follow this repo-specific skill:
${SKILL_PATH}

Then execute exactly one phase document:
${PHASE_DOC}

Hard rules:
- Execute only this phase.
- Stay within the allowed scope from the phase document.
- Run the required tests and benchmarks.
- End with exactly these sections:
  1. Changes
  2. Benchmarks
  3. Known Issues
  4. Next-Phase Recommendation
- After the four sections, append one final line exactly:
  Phase Status: PASS
  or
  Phase Status: FAIL

Below is the phase document content:

EOF

cat "${PHASE_DOC}" >> "${PROMPT_FILE}"

case "${SANDBOX_MODE}" in
  full-auto)
    CMD=(codex exec --full-auto -C "${ROOT_DIR}")
    ;;
  dangerous)
    CMD=(codex exec --dangerously-bypass-approvals-and-sandbox -C "${ROOT_DIR}")
    ;;
  read-only)
    CMD=(codex exec -s read-only -C "${ROOT_DIR}")
    ;;
  *)
    echo "Unsupported sandbox mode: ${SANDBOX_MODE}" >&2
    exit 1
    ;;
esac

if [[ -n "${OUTPUT_FILE}" ]]; then
  CMD+=(-o "${OUTPUT_FILE}")
fi

CMD+=(-)

echo "Skill: ${SKILL_PATH}"
echo "Phase: ${PHASE_DOC}"
echo "Sandbox: ${SANDBOX_MODE}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf 'Command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}" < "${PROMPT_FILE}"
