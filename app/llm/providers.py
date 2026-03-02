from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from typing import Any, AsyncIterator

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
                    compat = await self._tool_router_compat_if_needed(
                        client=client,
                        status_code=resp.status_code,
                        messages=messages,
                        tools=tools,
                        model=selected_model,
                    )
                    if compat is not None:
                        return compat
                    return LlmResponse(text=None, error=f"ollama_status_{resp.status_code}")
                data = resp.json()
        except httpx.TimeoutException as ex:
            return LlmResponse(text=None, error=f"ollama_timeout: {ex}")
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
        if not parsed_tool_calls and content:
            parsed_tool_calls = parse_tool_calls_from_text(content)
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

    async def _tool_router_compat_if_needed(
        self,
        *,
        client: httpx.AsyncClient,
        status_code: int,
        messages: list[dict[str, str]],
        tools: list[dict[str, object]] | None,
        model: str,
    ) -> LlmResponse | None:
        # Some local models reject native tools payload with HTTP 400.
        # In that case, keep routing strictly LLM-driven by asking for JSON tool_calls.
        if status_code != 400 or not tools:
            return None

        compat_messages = build_ollama_tool_router_compat_messages(messages, tools)
        compat_payload = {
            "model": model,
            "messages": compat_messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        try:
            compat_resp = await client.post(f"{self.base_url}/api/chat", json=compat_payload)
        except httpx.TimeoutException as ex:
            return LlmResponse(text=None, error=f"ollama_timeout: {ex}")
        except Exception as ex:
            return LlmResponse(text=None, error=f"ollama_unreachable: {ex}")

        if compat_resp.status_code >= 400:
            return None

        data = compat_resp.json()
        if not isinstance(data, dict):
            return LlmResponse(text=None, error="ollama_invalid_payload")

        content = ""
        parsed_tool_calls: list[LlmToolCall] = []
        message = data.get("message")
        if isinstance(message, dict):
            content = str(message.get("content", "")).strip()
            parsed_tool_calls = parse_ollama_tool_calls(message)
        if not content:
            content = str(data.get("response", "")).strip()
        if not parsed_tool_calls and content:
            parsed_tool_calls = parse_tool_calls_from_text(content)
        if not content and not parsed_tool_calls:
            return LlmResponse(text=None, error="ollama_empty_response")

        completion_tokens = int(data.get("eval_count", 0) or 0)
        if completion_tokens <= 0:
            completion_tokens = estimate_text_tokens(content)
        prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
        if prompt_tokens <= 0:
            prompt_tokens = estimate_message_tokens(compat_messages)

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
        except httpx.TimeoutException as ex:
            return LlmResponse(text=None, error=f"openai_compatible_timeout: {ex}")
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


def build_ollama_tool_router_compat_messages(
    messages: list[dict[str, str]],
    tools: list[dict[str, object]],
) -> list[dict[str, str]]:
    allowed_tools: list[dict[str, Any]] = []
    for row in tools:
        if not isinstance(row, dict):
            continue
        function = row.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name", "")).strip()
        if not name:
            continue
        parameters = function.get("parameters", {})
        allowed_arguments: list[str] = []
        required_arguments: list[str] = []
        if isinstance(parameters, dict):
            props = parameters.get("properties")
            if isinstance(props, dict):
                allowed_arguments = sorted(str(key) for key in props.keys() if isinstance(key, str))
            required = parameters.get("required")
            if isinstance(required, list):
                required_arguments = [str(item) for item in required if isinstance(item, str)]
        allowed_tools.append(
            {
                "tool_name": name,
                "description": str(function.get("description", "")).strip(),
                "parameters": parameters,
                "allowed_arguments": allowed_arguments,
                "required_arguments": required_arguments,
            }
        )

    compat_instruction = (
        "Native tool calling is unavailable for this model. "
        "You must output strict JSON only, no markdown. "
        'Schema: {"tool_calls":[{"tool_name":"<allowed_tool_name>","arguments":{...}}]}. '
        "For each selected tool, arguments must use only keys in allowed_arguments. "
        "Never emit unknown keys. "
        "Never include internal mapping fields such as area_entity_map. "
        'If no tool applies, output {"tool_calls":[]}.'
    )

    compat_tools = "Allowed tools JSON:\n" + json.dumps(
        allowed_tools,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return [
        {"role": "system", "content": compat_instruction},
        {"role": "system", "content": compat_tools},
        *messages,
    ]


def parse_tool_calls_from_text(text: str) -> list[LlmToolCall]:
    raw = text.strip()
    if not raw:
        return []

    candidates: list[str] = [raw]
    unfenced = _strip_markdown_fence(raw)
    if unfenced and unfenced not in candidates:
        candidates.append(unfenced)

    for chunk in (raw, unfenced):
        extracted = _extract_json_slice(chunk)
        if extracted and extracted not in candidates:
            candidates.append(extracted)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        tool_calls = _parse_tool_calls_from_payload(parsed)
        if tool_calls:
            return tool_calls
    return []


def _parse_tool_calls_from_payload(payload: object) -> list[LlmToolCall]:
    if isinstance(payload, dict):
        if "tool_calls" in payload:
            return _parse_tool_calls_from_payload(payload.get("tool_calls"))
        call = _parse_single_tool_call(payload)
        return [call] if call is not None else []

    if isinstance(payload, list):
        parsed: list[LlmToolCall] = []
        for row in payload:
            call = _parse_single_tool_call(row)
            if call is not None:
                parsed.append(call)
        return parsed

    return []


def _parse_single_tool_call(payload: object) -> LlmToolCall | None:
    if not isinstance(payload, dict):
        return None

    name = str(payload.get("tool_name", "")).strip()
    arguments: dict[str, object] = parse_arguments(payload.get("arguments"))

    if not name:
        name = str(payload.get("name", "")).strip()
    if not name:
        function = payload.get("function")
        if isinstance(function, dict):
            name = str(function.get("name", "")).strip()
            if not arguments:
                arguments = parse_arguments(function.get("arguments"))

    if not name:
        return None
    return LlmToolCall(tool_name=name, arguments=arguments)


def _strip_markdown_fence(text: str) -> str:
    raw = text.strip()
    if not raw.startswith("```"):
        return raw
    lines = raw.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return raw


def _extract_json_slice(text: str) -> str | None:
    if not text:
        return None
    object_start = text.find("{")
    object_end = text.rfind("}")
    if object_start >= 0 and object_end > object_start:
        return text[object_start : object_end + 1].strip()

    array_start = text.find("[")
    array_end = text.rfind("]")
    if array_start >= 0 and array_end > array_start:
        return text[array_start : array_end + 1].strip()
    return None
