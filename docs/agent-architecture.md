# Agent Architecture (v0.4)

## Closed Loop
1. Perception
2. Decision / Planning
3. Action
4. Feedback

## Layer Mapping
- LLM: `app/llm/`
- Memory: `app/memory/`
- Tool Use: `app/tools/`
- Planning: `app/planning/`
- Action: `app/action/`
- Runtime / Security / Observability: `app/runtime/`, `app/core/`

## Key Design Choices
- No external agent framework; native function-calling orchestration
- Strict LLM-first routing: no rule-based tool fallback for `/v1/agent/respond`
- Tool catalog source is HA Bridge only; no hardcoded local catalog fallback
- `metadata.route` override is disabled; route comes from LLM intent classifier only
- Agent mode requires valid `tool_calls`; plain-text-only router output is a hard failure
- Tool arguments are strict whitelist (`additionalProperties=false`), validated before execution
- Short-term memory: bounded window + token budget
- Long-term memory backend is fixed:
  - FAISS index: `IndexIDMap2(IndexFlatIP)`
  - SQLite metadata store with vector backup for rebuild
- Session-priority recall:
  - recall from same session first
  - then fill from global results
- Recovery:
  - load FAISS index if present and valid
  - rebuild from SQLite vectors if index missing/corrupted
  - if FAISS unavailable, disable long-term memory without stopping service
- Security:
  - tool whitelist
  - role-based permission checks
  - prompt injection guard
- Failure semantics:
  - LLM unreachable -> HTTP 503
  - LLM timeout -> HTTP 504
  - LLM invalid/parse/routing failure -> HTTP 502
- Prompting:
  - Intent/tool/chat prompts are modularized in `app/runtime/prompt_templates.py`
  - Tool router prompt explicitly forbids unknown args and default area guessing
- Observability:
  - traces for decision/action path
  - token usage
  - memory recall metrics and remember failure counters
