from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from typing import AsyncIterator

import httpx

from app.core.config import AppConfig


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_message_tokens(messages: list[dict[str, str]]) -> int:
    total = 0
    for row in messages:
        total += estimate_text_tokens(row.get("content", ""))
        total += 4
    return total


@dataclass
class LlmResponse:
    text: str | None
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: list["LlmToolCall"] = field(default_factory=list)


@dataclass
class LlmToolCall:
    tool_name: str
    arguments: dict[str, object]


class LlmProvider(ABC):
    name: str = "llm"

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, object]] | None = None,
        model: str | None = None,
    ) -> LlmResponse:
        raise NotImplementedError

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        # Streaming is optional for providers; default falls back to non-streaming.
        response = await self.chat(messages)
        if response.text:
            yield response.text

    def health_meta(self) -> dict[str, str | float | bool]:
        return {"provider": self.name}


class OllamaProvider(LlmProvider):
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout_seconds: float, temperature: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature

    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, object]] | None = None,
        model: str | None = None,
    ) -> LlmResponse:
        selected_model = (model or self.model).strip() or self.model
        payload = {
            "model": selected_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if tools:
            payload["tools"] = tools
        prompt_tokens = estimate_message_tokens(messages)
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(f"{self.base_url}/api/chat", json=payload)
                if resp.status_code >= 400:
                    return LlmResponse(text=None, error=f"ollama_status_{resp.status_code}")
                data = resp.json()
        except Exception as ex:
            return LlmResponse(text=None, error=f"ollama_unreachable: {ex}")

        if not isinstance(data, dict):
            return LlmResponse(text=None, error="ollama_invalid_payload")

        message = data.get("message")
        content = ""
        parsed_tool_calls: list[LlmToolCall] = []
        if isinstance(message, dict):
            content = str(message.get("content", "")).strip()
            parsed_tool_calls = parse_ollama_tool_calls(message)
        if not content:
            content = str(data.get("response", "")).strip()
        if not content and not parsed_tool_calls:
            return LlmResponse(text=None, error="ollama_empty_response")

        completion_tokens = int(data.get("eval_count", 0) or 0)
        if completion_tokens <= 0:
            completion_tokens = estimate_text_tokens(content)
        prompt_eval = int(data.get("prompt_eval_count", 0) or 0)
        if prompt_eval > 0:
            prompt_tokens = prompt_eval

        return LlmResponse(
            text=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tool_calls=parsed_tool_calls,
        )

    def health_meta(self) -> dict[str, str | float | bool]:
        return {
            "provider": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
        }


class OpenAiCompatibleProvider(LlmProvider):
    name = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        chat_path: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
        temperature: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.chat_path = chat_path if chat_path.startswith("/") else f"/{chat_path}"
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature

    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, object]] | None = None,
        model: str | None = None,
    ) -> LlmResponse:
        selected_model = (model or self.model).strip() or self.model
        payload = {
            "model": selected_model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        prompt_tokens = estimate_message_tokens(messages)
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(f"{self.base_url}{self.chat_path}", headers=headers, json=payload)
                if resp.status_code >= 400:
                    return LlmResponse(text=None, error=f"openai_compatible_status_{resp.status_code}")
                data = resp.json()
        except Exception as ex:
            return LlmResponse(text=None, error=f"openai_compatible_unreachable: {ex}")

        if not isinstance(data, dict):
            return LlmResponse(text=None, error="openai_compatible_invalid_payload")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return LlmResponse(text=None, error="openai_compatible_empty_choices")

        first = choices[0]
        if not isinstance(first, dict):
            return LlmResponse(text=None, error="openai_compatible_invalid_choice")
        message = first.get("message")
        content = extract_openai_message_text(message)
        parsed_tool_calls = parse_openai_tool_calls(message)
        if not content and not parsed_tool_calls:
            return LlmResponse(text=None, error="openai_compatible_empty_response")

        usage = data.get("usage", {})
        prompt_used = int(usage.get("prompt_tokens", 0) or 0) if isinstance(usage, dict) else 0
        completion_used = int(usage.get("completion_tokens", 0) or 0) if isinstance(usage, dict) else 0
        if prompt_used <= 0:
            prompt_used = prompt_tokens
        if completion_used <= 0:
            completion_used = estimate_text_tokens(content)
        return LlmResponse(
            text=content,
            prompt_tokens=prompt_used,
            completion_tokens=completion_used,
            tool_calls=parsed_tool_calls,
        )

    def health_meta(self) -> dict[str, str | float | bool]:
        return {
            "provider": self.name,
            "base_url": self.base_url,
            "chat_path": self.chat_path,
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "has_api_key": bool(self.api_key),
        }


def build_llm_provider(config: AppConfig) -> LlmProvider | None:
    if not config.llm_enabled:
        return None
    provider = config.llm_provider
    if provider == "ollama":
        return OllamaProvider(
            base_url=config.ollama_base_url,
            model=config.llm_model,
            timeout_seconds=config.llm_timeout_seconds,
            temperature=config.llm_temperature,
        )
    if provider in {"openai", "openai_compatible", "openai-compatible"}:
        return OpenAiCompatibleProvider(
            base_url=config.openai_base_url,
            chat_path=config.openai_chat_path,
            api_key=config.openai_api_key,
            model=config.llm_model,
            timeout_seconds=config.llm_timeout_seconds,
            temperature=config.llm_temperature,
        )
    return None


def extract_openai_message_text(message: object) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for row in content:
            if not isinstance(row, dict):
                continue
            text = row.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
        return "\n".join(chunks).strip()
    return ""


def parse_openai_tool_calls(message: object) -> list[LlmToolCall]:
    if not isinstance(message, dict):
        return []
    raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, list):
        return []
    parsed: list[LlmToolCall] = []
    for row in raw_calls:
        if not isinstance(row, dict):
            continue
        function = row.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name", "")).strip()
        if not name:
            continue
        arguments = parse_arguments(function.get("arguments"))
        parsed.append(LlmToolCall(tool_name=name, arguments=arguments))
    return parsed


def parse_ollama_tool_calls(message: dict[str, object]) -> list[LlmToolCall]:
    raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, list):
        return []
    parsed: list[LlmToolCall] = []
    for row in raw_calls:
        if not isinstance(row, dict):
            continue
        function = row.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name", "")).strip()
        if not name:
            continue
        arguments = parse_arguments(function.get("arguments"))
        parsed.append(LlmToolCall(tool_name=name, arguments=arguments))
    return parsed


def parse_arguments(raw: object) -> dict[str, object]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except Exception:
            return {}
        if isinstance(data, dict):
            return data
    return {}
