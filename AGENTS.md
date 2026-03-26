# Repository Guidelines

## Project Structure & Module Organization
`omlx/` contains the main server and runtime code: `api/`, `engine/`, `cache/`, `models/`, `integrations/`, `utils/`, and the FastAPI admin UI under `admin/`. `omlx_dgx/` is the experimental DGX/Jetson runtime and control plane. `tests/` holds unit coverage, with server and model-flow cases under `tests/integration/`. Use `packaging/` for the macOS menubar app, `docs/` for user and contributor documentation, `scripts/` for benchmarks and utilities, and `plans/` for DGX roadmap notes.

## Build, Test, and Development Commands
`pip install -e ".[dev]"` installs the editable package with pytest, Black, Ruff, and mypy.

`omlx serve --model-dir ~/models` starts the local OpenAI-compatible server on port `8000`.

`pytest` runs the default fast suite. `pytest tests/test_config.py -v` is the standard way to target one module.

`pytest -m slow` runs model-loading tests. `pytest -m integration` runs server-backed tests.

`black . && ruff check . && mypy omlx` formats and validates Python changes.

`python omlx/admin/build_css.py` rebuilds Tailwind output after editing `omlx/admin/src/input.css`.

`cd packaging && python build.py --skip-venv` rebuilds the macOS app bundle from existing venvstacks layers.

## Coding Style & Naming Conventions
Use 4-space indentation, Python 3.10+ syntax, and type hints for new or changed code. Black and Ruff enforce an 88-character line target. Keep modules and functions in `snake_case`, classes in `PascalCase`, and CLI or config names descriptive and explicit. Match the existing SPDX header in source files:

```python
# SPDX-License-Identifier: Apache-2.0
```

## Testing Guidelines
Name tests `tests/test_<module>.py` and keep integration coverage in `tests/integration/`. Add or update tests whenever behavior changes in the scheduler, cache layers, API schemas, or model routing. Prefer focused unit tests first; use `slow` and `integration` markers only when the path depends on live models or a running server.

## Commit & Pull Request Guidelines
Follow the existing history: short, imperative subjects such as `Add benchmark for PDF route strategy` or `Stabilize Jetson 35B checkpoint`. Keep one logical change per commit. Pull requests should target `main`, explain what changed and why, list the validation commands you ran, and include screenshots for `omlx/admin/` or `packaging/omlx_app/` UI changes.

## Platform Notes
The main path targets Apple Silicon on macOS 15+, while `omlx_dgx/` is experimental and should stay isolated from Mac-specific packaging changes. Keep packaging work under `packaging/` and DGX runtime changes under `omlx_dgx/`.
