# SPDX-License-Identifier: Apache-2.0
"""Low-level OpenAI-compatible LLM client for duplex sessions."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List

import requests


@dataclass(frozen=True)
class LlmResult:
    text: str
    latency_ms: int
    raw: Dict[str, Any]


class LLMChatClient:
    def __init__(self, base_url: str, *, timeout: float = 180.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def health(self) -> Dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/health",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def models(self) -> Dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/v1/models",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def chat(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        conversation_id: str,
        system_prompt: str = "",
    ) -> LlmResult:
        request_started = time.perf_counter()
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 192,
            "chat_template_kwargs": {"enable_thinking": False},
            "enableThinking": False,
            "reasoning": False,
            "reasoning_budget": 0,
            "reasoning_format": "none",
            "metadata": {
                "conversation_id": conversation_id,
                "omlx_duplex": True,
            },
        }
        if system_prompt:
            payload["metadata"]["system_prompt_hash"] = system_prompt
        response = self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        text = ""
        for choice in body.get("choices", []):
            message = choice.get("message", {})
            text = str(message.get("content", "") or "").strip()
            if text:
                break
        return LlmResult(
            text=text,
            latency_ms=int((time.perf_counter() - request_started) * 1000),
            raw=body,
        )

    def chat_stream(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        conversation_id: str,
        system_prompt: str = "",
    ) -> Iterator[str]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 192,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "enableThinking": False,
            "reasoning": False,
            "reasoning_budget": 0,
            "reasoning_format": "none",
            "metadata": {
                "conversation_id": conversation_id,
                "omlx_duplex": True,
            },
        }
        if system_prompt:
            payload["metadata"]["system_prompt_hash"] = system_prompt
        with self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout,
            stream=True,
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = str(raw_line).strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for choice in payload.get("choices", []):
                    delta = choice.get("delta", {})
                    content = str(delta.get("content", "") or "")
                    if content:
                        yield content
