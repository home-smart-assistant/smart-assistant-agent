from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_local_env(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        os.environ.setdefault(key, value)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE_PATH = PROJECT_ROOT / ".env"
load_local_env(ENV_FILE_PATH)


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_csv(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    values = [item.strip() for item in raw.split(",")]
    return [item for item in values if item]


@dataclass(frozen=True)
class AppConfig:
    app_name: str = "smart_assistant_agent"
    app_version: str = "0.4.0"
    ha_bridge_url: str = "http://localhost:8092"

    # Agent core
    agent_memory_max_turns: int = 12
    agent_tool_auto_execute: bool = True
    agent_action_timeout_seconds: float = 6.0
    agent_token_budget: int = 1600
    agent_multi_step_enabled: bool = True
    agent_max_plan_steps: int = 5
    agent_rollback_on_failure: bool = True
    agent_use_llm_tool_router: bool = True
    agent_use_llm_intent_router: bool = True
    agent_intent_min_confidence: float = 0.60
    agent_candidate_tool_limit: int = 20
    agent_runtime_env: str = "home"
    agent_default_role: str = "operator"
    agent_trace_max_items: int = 200
    text_encoding_strict: bool = True
    agent_tool_whitelist: tuple[str, ...] = (
        "home.lights.on",
        "home.lights.off",
        "home.scene.activate",
        "home.climate.turn_on",
        "home.climate.turn_off",
        "home.climate.set_temperature",
        "home.areas.sync",
        "home.areas.audit",
        "home.areas.assign",
    )

    # Prompt security
    prompt_injection_guard_enabled: bool = True
    prompt_injection_patterns: tuple[str, ...] = (
        "ignore previous instructions",
        "system prompt",
        "developer prompt",
        "越狱",
        "请忽略之前指令",
        "执行任意命令",
        "泄露提示词",
    )

    # HA context
    ha_context_enabled: bool = True
    ha_context_path: str = "/v1/context/summary"
    ha_context_ttl_seconds: float = 30.0
    ha_context_timeout_seconds: float = 6.0
    ha_context_max_service_domains: int = 12
    ha_context_max_chars: int = 2400

    # LLM
    llm_enabled: bool = True
    llm_provider: str = "ollama"
    llm_model: str = "deepseek-r1:1.5b"
    llm_timeout_seconds: float = 20.0
    llm_max_history_turns: int = 6
    llm_system_prompt: str = "你是家庭语音助手。回答简洁直接，优先给出可执行建议，不要编造事实。"
    llm_temperature: float = 0.3

    # Ollama provider
    ollama_base_url: str = "http://localhost:11434"

    # OpenAI-compatible provider
    openai_base_url: str = "https://api.openai.com"
    openai_api_key: str = ""
    openai_chat_path: str = "/v1/chat/completions"

    # Long-term memory (FAISS fixed backend)
    long_term_memory_enabled: bool = True
    long_term_memory_top_k: int = 3
    long_term_memory_limit: int = 300
    agent_faiss_index_path: str = "./data/memory/faiss.index"
    agent_faiss_meta_db_path: str = "./data/memory/memory.db"
    agent_faiss_min_score: float = 0.10
    agent_faiss_auto_flush_every: int = 1

    # Embedding provider (for vector generation only)
    embedding_provider: str = "hash"
    embedding_model: str = "text-embedding-3-small"
    openai_embedding_path: str = "/v1/embeddings"
    embedding_timeout_seconds: float = 12.0

    @classmethod
    def from_env(cls) -> "AppConfig":
        legacy_llm_enabled = env_bool("OLLAMA_ENABLED", True)
        legacy_model = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")
        legacy_timeout = env_float("OLLAMA_TIMEOUT_SECONDS", 20.0)
        legacy_max_turns = env_int("OLLAMA_MAX_HISTORY_TURNS", 6)

        llm_enabled = env_bool("LLM_ENABLED", legacy_llm_enabled)
        llm_model = os.getenv("LLM_MODEL", legacy_model)
        llm_timeout = env_float("LLM_TIMEOUT_SECONDS", legacy_timeout)
        llm_max_turns = env_int("LLM_MAX_HISTORY_TURNS", legacy_max_turns)
        llm_provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

        whitelist = tuple(
            env_csv(
                "AGENT_TOOL_WHITELIST",
                [
                    "home.lights.on",
                    "home.lights.off",
                    "home.scene.activate",
                    "home.climate.turn_on",
                    "home.climate.turn_off",
                    "home.climate.set_temperature",
                    "home.areas.sync",
                    "home.areas.audit",
                    "home.areas.assign",
                ],
            )
        )

        patterns = tuple(
            env_csv(
                "AGENT_PROMPT_INJECTION_PATTERNS",
                [
                    "ignore previous instructions",
                    "system prompt",
                    "developer prompt",
                    "越狱",
                    "请忽略之前指令",
                    "执行任意命令",
                    "泄露提示词",
                ],
            )
        )

        return cls(
            app_name=os.getenv("APP_NAME", "smart_assistant_agent"),
            app_version=os.getenv("APP_VERSION", "0.4.0"),
            ha_bridge_url=os.getenv("HA_BRIDGE_URL", "http://localhost:8092").rstrip("/"),
            agent_memory_max_turns=env_int("AGENT_MEMORY_MAX_TURNS", 12),
            agent_tool_auto_execute=env_bool("AGENT_TOOL_AUTO_EXECUTE", True),
            agent_action_timeout_seconds=env_float("AGENT_ACTION_TIMEOUT_SECONDS", 6.0),
            agent_token_budget=env_int("AGENT_TOKEN_BUDGET", 1600),
            agent_multi_step_enabled=env_bool("AGENT_MULTI_STEP_ENABLED", True),
            agent_max_plan_steps=env_int("AGENT_MAX_PLAN_STEPS", 5),
            agent_rollback_on_failure=env_bool("AGENT_ROLLBACK_ON_FAILURE", True),
            agent_use_llm_tool_router=env_bool("AGENT_USE_LLM_TOOL_ROUTER", True),
            agent_use_llm_intent_router=env_bool("AGENT_USE_LLM_INTENT_ROUTER", True),
            agent_intent_min_confidence=env_float("AGENT_INTENT_MIN_CONFIDENCE", 0.60),
            agent_candidate_tool_limit=env_int("AGENT_CANDIDATE_TOOL_LIMIT", 20),
            agent_runtime_env=os.getenv("AGENT_RUNTIME_ENV", "home").strip().lower() or "home",
            agent_default_role=os.getenv("AGENT_DEFAULT_ROLE", "operator").strip().lower(),
            agent_trace_max_items=env_int("AGENT_TRACE_MAX_ITEMS", 200),
            text_encoding_strict=env_bool("TEXT_ENCODING_STRICT", True),
            agent_tool_whitelist=whitelist,
            prompt_injection_guard_enabled=env_bool("AGENT_PROMPT_INJECTION_GUARD_ENABLED", True),
            prompt_injection_patterns=patterns,
            ha_context_enabled=env_bool("HA_CONTEXT_ENABLED", True),
            ha_context_path=os.getenv("HA_CONTEXT_PATH", "/v1/context/summary"),
            ha_context_ttl_seconds=env_float("HA_CONTEXT_TTL_SECONDS", 30.0),
            ha_context_timeout_seconds=env_float("HA_CONTEXT_TIMEOUT_SECONDS", 6.0),
            ha_context_max_service_domains=env_int("HA_CONTEXT_MAX_SERVICE_DOMAINS", 12),
            ha_context_max_chars=env_int("HA_CONTEXT_MAX_CHARS", 2400),
            llm_enabled=llm_enabled,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_timeout_seconds=llm_timeout,
            llm_max_history_turns=llm_max_turns,
            llm_system_prompt=os.getenv(
                "LLM_SYSTEM_PROMPT",
                "你是家庭语音助手。回答简洁直接，优先给出可执行建议，不要编造事实。",
            ),
            llm_temperature=env_float("OPENAI_TEMPERATURE", 0.3),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_chat_path=os.getenv("OPENAI_CHAT_PATH", "/v1/chat/completions"),
            long_term_memory_enabled=env_bool("AGENT_LONG_TERM_MEMORY_ENABLED", True),
            long_term_memory_top_k=env_int("AGENT_LONG_TERM_MEMORY_TOP_K", 3),
            long_term_memory_limit=env_int("AGENT_LONG_TERM_MEMORY_LIMIT", 300),
            agent_faiss_index_path=os.getenv("AGENT_FAISS_INDEX_PATH", "./data/memory/faiss.index"),
            agent_faiss_meta_db_path=os.getenv("AGENT_FAISS_META_DB_PATH", "./data/memory/memory.db"),
            agent_faiss_min_score=env_float("AGENT_FAISS_MIN_SCORE", 0.10),
            agent_faiss_auto_flush_every=env_int("AGENT_FAISS_AUTO_FLUSH_EVERY", 1),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "hash").strip().lower(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            openai_embedding_path=os.getenv("OPENAI_EMBEDDING_PATH", "/v1/embeddings"),
            embedding_timeout_seconds=env_float("EMBEDDING_TIMEOUT_SECONDS", 12.0),
        )
