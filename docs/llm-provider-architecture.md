# LLM Provider Architecture (v0.4)

## Goal
Keep the orchestration flow provider-agnostic while preserving a single interface.

## Interface
- `LlmProvider.chat(messages, tools=None) -> LlmResponse`
- `LlmProvider.stream(messages)` (optional)

`LlmResponse` contains:
- `text`
- `tool_calls`
- `error`
- `prompt_tokens`
- `completion_tokens`

## Implemented Providers
- `ollama`
- `openai_compatible`

## Orchestration Usage
`app/runtime/agent_service.py` uses LLM in two phases:
1. routing phase: LLM may return tool calls from provided schemas
2. feedback phase: after tool execution, LLM builds user-facing response

## Config
- Shared:
  - `LLM_ENABLED`
  - `LLM_PROVIDER`
  - `LLM_MODEL`
  - `LLM_TIMEOUT_SECONDS`
  - `LLM_MAX_HISTORY_TURNS`
  - `LLM_SYSTEM_PROMPT`
- Ollama:
  - `OLLAMA_BASE_URL`
- OpenAI-compatible:
  - `OPENAI_BASE_URL`
  - `OPENAI_API_KEY`
  - `OPENAI_CHAT_PATH`
  - `OPENAI_TEMPERATURE`

## Extension Steps
1. Implement a new provider in `app/llm/providers.py`
2. Register it in `build_llm_provider`
3. Add env vars in `.env.example`
4. Update docs
