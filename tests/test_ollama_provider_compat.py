from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.llm.providers import OllamaProvider


class TestOllamaToolRouterCompat(unittest.IsolatedAsyncioTestCase):
    @patch("app.llm.providers.httpx.AsyncClient")
    async def test_tools_http_400_falls_back_to_json_router(self, async_client_cls) -> None:
        client = MagicMock()
        async_client_cls.return_value.__aenter__.return_value = client

        first_response = MagicMock()
        first_response.status_code = 400

        second_response = MagicMock()
        second_response.status_code = 200
        second_response.json.return_value = {
            "message": {
                "content": (
                    '{"tool_calls":[{"tool_name":"home.lights.on","arguments":{"area":"dining_room"}}]}'
                )
            },
            "prompt_eval_count": 25,
            "eval_count": 9,
        }

        client.post = AsyncMock(side_effect=[first_response, second_response])

        provider = OllamaProvider(
            base_url="http://127.0.0.1:11434",
            model="deepseek-r1:8b-0528-qwen3-q8_0",
            timeout_seconds=30.0,
            temperature=0.2,
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "home.lights.on",
                    "description": "Turn on lights by area",
                    "parameters": {
                        "type": "object",
                        "properties": {"area": {"type": "string"}},
                        "required": ["area"],
                    },
                },
            }
        ]

        response = await provider.chat(
            messages=[{"role": "user", "content": "打开餐厅的灯"}],
            tools=tools,
        )

        self.assertIsNone(response.error)
        self.assertEqual(1, len(response.tool_calls))
        self.assertEqual("home.lights.on", response.tool_calls[0].tool_name)
        self.assertEqual("dining_room", response.tool_calls[0].arguments.get("area"))
        self.assertEqual(2, client.post.await_count)

        second_payload = client.post.await_args_list[1].kwargs["json"]
        self.assertNotIn("tools", second_payload)
        self.assertEqual("json", second_payload.get("format"))

    @patch("app.llm.providers.httpx.AsyncClient")
    async def test_tools_http_400_and_second_failure_keeps_original_error(self, async_client_cls) -> None:
        client = MagicMock()
        async_client_cls.return_value.__aenter__.return_value = client

        first_response = MagicMock()
        first_response.status_code = 400
        second_response = MagicMock()
        second_response.status_code = 400
        client.post = AsyncMock(side_effect=[first_response, second_response])

        provider = OllamaProvider(
            base_url="http://127.0.0.1:11434",
            model="deepseek-r1:8b-0528-qwen3-q8_0",
            timeout_seconds=30.0,
            temperature=0.2,
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "home.lights.on",
                    "description": "Turn on lights by area",
                    "parameters": {"type": "object", "properties": {"area": {"type": "string"}}},
                },
            }
        ]

        response = await provider.chat(
            messages=[{"role": "user", "content": "打开餐厅的灯"}],
            tools=tools,
        )
        self.assertEqual("ollama_status_400", response.error)


if __name__ == "__main__":
    unittest.main()
