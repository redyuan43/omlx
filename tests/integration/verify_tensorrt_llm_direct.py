# SPDX-License-Identifier: Apache-2.0
"""Standalone TensorRT-LLM direct API smoke test.

This must be executed as a real file path instead of stdin so mpi4py worker
processes can re-import the main module.
"""

from __future__ import annotations

import argparse
import json
import time

import tensorrt_llm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model repo id or local path")
    parser.add_argument(
        "--backend",
        default="",
        help="Optional TensorRT-LLM backend override, e.g. pytorch",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--prompt", default="Reply with exactly one word: pong")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    llm = None
    try:
        llm_kwargs = {
            "model": args.model,
            "dtype": args.dtype,
            "tensor_parallel_size": args.tensor_parallel_size,
            "trust_remote_code": args.trust_remote_code,
        }
        if args.backend == "pytorch":
            llm_cls = tensorrt_llm.LLM
            llm_kwargs["backend"] = "pytorch"
        else:
            from tensorrt_llm._tensorrt_engine import LLM as TensorRTLLM
            llm_cls = TensorRTLLM

        llm = llm_cls(**llm_kwargs)
        ready_seconds = time.time() - start
        sampling_params = tensorrt_llm.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        output = llm.generate(args.prompt, sampling_params=sampling_params, use_tqdm=False)
        generated = output.outputs[0]
        payload = {
            "backend": args.backend or "tensorrt",
            "model": args.model,
            "dtype": args.dtype,
            "ready_seconds": round(ready_seconds, 2),
            "generated_text": generated.text,
            "finish_reason": generated.finish_reason,
            "prompt_tokens": (
                len(output.prompt_token_ids)
                if output.prompt_token_ids is not None
                else None
            ),
            "completion_tokens": (
                len(generated.token_ids)
                if generated.token_ids is not None
                else None
            ),
        }
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    finally:
        if llm is not None:
            llm.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
