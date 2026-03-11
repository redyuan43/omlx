# SPDX-License-Identifier: Apache-2.0
"""CLI for the experimental DGX runtime/control plane."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from omlx_dgx.config import DGXSettingsManager, ModelProfile
from omlx_dgx.control_plane.app import create_app


def serve_command(args: argparse.Namespace) -> None:
    settings = DGXSettingsManager(Path(args.base_path))
    if args.backend_kind:
        settings.config.backend.kind = args.backend_kind
    if args.backend_url:
        settings.config.backend.base_url = args.backend_url
    if args.host:
        settings.config.control_plane.host = args.host
    if args.port:
        settings.config.control_plane.port = args.port
    if args.runtime_python:
        settings.config.backend.runtime_python = args.runtime_python
    if args.model_repo_id:
        settings.config.backend.model_repo_id = args.model_repo_id
    if args.model_id:
        settings.ensure_model(
            ModelProfile(
                model_id=args.model_id,
                model_alias=args.model_alias,
                is_default=True,
            )
        )
    else:
        settings.save()

    app = create_app(base_path=args.base_path, settings_manager=settings)
    uvicorn.run(
        app,
        host=settings.config.control_plane.host,
        port=settings.config.control_plane.port,
        log_level="info",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experimental DGX control plane")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Start the DGX control plane")
    serve.add_argument("--base-path", default="~/.omlx-dgx")
    serve.add_argument("--backend-kind", default=None)
    serve.add_argument("--backend-url", default=None)
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--runtime-python", default=None)
    serve.add_argument("--model-id", default="")
    serve.add_argument("--model-alias", default=None)
    serve.add_argument("--model-repo-id", default="")
    serve.set_defaults(func=serve_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
