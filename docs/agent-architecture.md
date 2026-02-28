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
- Observability:
  - traces for decision/action path
  - token usage
  - memory recall metrics and remember failure counters
