#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_ROOT="${ROOT_DIR}/.runtime/codex-phase-runs/$(date +%Y%m%d-%H%M%S)"
PHASES=(1 2 3 4 5)
SUMMARY_FILE="${REPORT_ROOT}/summary.txt"

mkdir -p "${REPORT_ROOT}"

echo "Codex phase run root: ${REPORT_ROOT}"
echo "Codex phase run root: ${REPORT_ROOT}" > "${SUMMARY_FILE}"
echo >> "${SUMMARY_FILE}"

for phase in "${PHASES[@]}"; do
  PHASE_DOC="${ROOT_DIR}/plans/codex-phase-${phase}.md"
  PHASE_OUTPUT="${REPORT_ROOT}/phase-${phase}.md"

  echo "=== Running phase ${phase} ==="
  echo "=== Running phase ${phase} ===" >> "${SUMMARY_FILE}"
  echo "Phase document: ${PHASE_DOC}" >> "${SUMMARY_FILE}"
  echo "Output file: ${PHASE_OUTPUT}" >> "${SUMMARY_FILE}"

  if ! bash "${ROOT_DIR}/scripts/run_codex_phase.sh" "${phase}" --output "${PHASE_OUTPUT}"; then
    echo "Phase ${phase} execution command failed." | tee -a "${SUMMARY_FILE}" >&2
    exit 1
  fi

  if [[ ! -s "${PHASE_OUTPUT}" ]]; then
    echo "Phase ${phase} produced no output: ${PHASE_OUTPUT}" | tee -a "${SUMMARY_FILE}" >&2
    exit 1
  fi

  STATUS_LINE="$(grep -E '^Phase Status: (PASS|FAIL)$' "${PHASE_OUTPUT}" | tail -n 1 || true)"
  if [[ -z "${STATUS_LINE}" ]]; then
    echo "Phase ${phase} did not emit a machine-readable status line." | tee -a "${SUMMARY_FILE}" >&2
    exit 1
  fi

  echo "${STATUS_LINE}" | tee -a "${SUMMARY_FILE}"

  if [[ "${STATUS_LINE}" != "Phase Status: PASS" ]]; then
    echo "Stopping after phase ${phase} because it did not pass." | tee -a "${SUMMARY_FILE}" >&2
    exit 1
  fi

  echo >> "${SUMMARY_FILE}"
done

echo "All five phases completed successfully." | tee -a "${SUMMARY_FILE}"
echo "Reports saved under: ${REPORT_ROOT}" | tee -a "${SUMMARY_FILE}"
