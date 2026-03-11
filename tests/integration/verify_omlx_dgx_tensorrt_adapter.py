# SPDX-License-Identifier: Apache-2.0
"""Run a real TensorRT-LLM direct adapter smoke test from a file path."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from omlx_dgx.config import BackendConfig
from omlx_dgx.runtime.tensorrt_llm import TensorRTLLMBackendAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model repo id or local engine dir")
    parser.add_argument("--prompt", default="Reply with exactly one word: pong")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="omlx-dgx-adapter-") as temp_dir:
        config = BackendConfig(
            kind="tensorrt_llm",
            base_url="http://127.0.0.1:65530",
            model_repo_id=args.model,
            direct_api_enabled=True,
        )
        adapter = TensorRTLLMBackendAdapter.from_backend_config(config, Path(temp_dir))
        started = adapter.start_runtime()
        response = adapter.proxy(
            "POST",
            "v1/chat/completions",
            json={
                "model": args.model,
                "messages": [{"role": "user", "content": args.prompt}],
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "stream": False,
            },
        )
        payload = response.json()
        stopped = adapter.stop_runtime()
        print(
            json.dumps(
                {
                    "started": started,
                    "content": payload["choices"][0]["message"]["content"],
                    "finish_reason": payload["choices"][0]["finish_reason"],
                    "usage": payload["usage"],
                    "stopped": stopped,
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
