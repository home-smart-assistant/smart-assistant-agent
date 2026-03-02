# smart_assistant_agent

FastAPI-based smart home agent with a non-framework loop.

Strict behavior:
1. LLM classifies route and generates tool calls (agent mode).
2. Agent validates and executes tool calls.
3. Tool catalog comes from HA Bridge only; if unavailable, routing fails explicitly.
4. `agent` mode requires valid `tool_calls`; text-only LLM output is treated as routing failure.
5. `metadata.route` override is disabled; route is decided by LLM intent classification.
6. If LLM is unreachable/timeout/invalid, API fails explicitly.
7. Short-term + long-term memory are updated.

## Tech Stack
- Python
- FastAPI / Uvicorn
- httpx
- pydantic
- FAISS (fixed long-term memory backend)
- SQLite (metadata + vector backup)

## Architecture Layers
- LLM layer: `app/llm/`
- Memory layer: `app/memory/`
  - short-term window memory
  - long-term memory: FAISS (`IndexFlatIP + IndexIDMap2`) + SQLite
- Tool layer: `app/tools/`
- Planning layer: `app/planning/`
- Action layer: `app/action/`
- Runtime orchestration: `app/runtime/agent_service.py`

## APIs
- `POST /v1/agent/respond`
- `GET /v1/agent/session/{session_id}`
- `DELETE /v1/agent/session/{session_id}`
- `GET /v1/agent/context/ha`
- `GET /v1/agent/tools`
- `GET /v1/agent/trace/{trace_id}`
- `GET /v1/agent/traces`
- `GET /v1/agent/metrics`
- `GET /v1/agent/models`
- `GET /ui` (lightweight ChatGPT-style web chat UI for calling `/v1/agent/respond`)
  - UI supports `mode=agent|chat` (default `agent`), and optional per-request `metadata.llm_model`.

`POST /v1/agent/respond` error semantics in strict mode:
- `503`: `llm_unreachable`
- `504`: `llm_timeout`
- `502`: `llm_bad_response | llm_routing_failed | llm_intent_failed`

`agent` mode strict semantics:
- candidate tools are built from Bridge catalog only (no hardcoded fallback)
- tool arguments are strict whitelist (`additionalProperties=false` per tool schema)
- unknown keys or missing required arguments fail before execution
- when no valid `tool_calls` are generated, request fails instead of returning chat text

## Long-term Memory (FAISS fixed)
- No vector DB backend switching
- Index file: `AGENT_FAISS_INDEX_PATH` (default `./data/memory/faiss.index`)
- Metadata DB: `AGENT_FAISS_META_DB_PATH` (default `./data/memory/memory.db`)
- Similarity threshold: `AGENT_FAISS_MIN_SCORE`
- Session-priority recall: same session first, then global fill
- If FAISS import/load fails: long-term memory is disabled, service still runs

## Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8091
```

## Docker
- Image installs all dependencies from `requirements.txt`
- Data directory is prepared at `/app/data/memory` for index/db persistence

## Docs
- `docs/openapi/agent.openapi.yaml`
- `docs/llm-provider-architecture.md`
- `docs/agent-architecture.md`
- `app/runtime/prompt_templates.py` (intent/tool/chat prompt templates)

## Encoding Policy
- All source files and docs use UTF-8.
- Before routing/execution, request text and metadata are normalized to repair common mojibake.
- If text cannot be repaired reliably and `TEXT_ENCODING_STRICT=true`, API returns `400` with `error_code=invalid_text_encoding`.

## CI/CD (Deploy to 192.168.3.103)
- Workflow: `.github/workflows/cicd-deploy.yml`
- Trigger: push to `main` or manual `workflow_dispatch`
- Output image:
  - `ghcr.io/home-smart-assistant/smart-assistant-agent:main`
  - `ghcr.io/home-smart-assistant/smart-assistant-agent:<commit_sha>`
- Deploy target service: `smart_assistant_agent` in `/opt/smart-assistant/docker-compose.yml`
- Runner labels:
  - Build: `[self-hosted, Linux, X64, builder-linux-11]` (recommended on `192.168.3.11` via WSL2 Ubuntu runner)
  - Deploy: `[self-hosted, Linux, X64, deploy-linux]` (recommended on `192.168.3.103`)
- Optional config:
  - Repository variable `DEPLOY_PATH` (default `/opt/smart-assistant`)
  - `GHCR_USERNAME` + `GHCR_TOKEN` only if your package policy requires explicit login
