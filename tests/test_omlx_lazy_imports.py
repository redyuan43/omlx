# SPDX-License-Identifier: Apache-2.0
"""Lazy import guards for Linux-compatible package entrypoints."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_package_import_keeps_mlx_unloaded():
    result = _run_python(
        """
import json
import sys

import omlx

payload = {
    "request_export_loaded": hasattr(omlx, "Request"),
    "mlx_core_loaded": "mlx.core" in sys.modules,
}
print(json.dumps(payload))
"""
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip())
    assert payload["request_export_loaded"] is True
    assert payload["mlx_core_loaded"] is False


def test_api_models_import_without_mlx_runtime():
    result = _run_python(
        """
from omlx.api.anthropic_models import MessagesRequest

print(MessagesRequest.__name__)
"""
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "MessagesRequest"
