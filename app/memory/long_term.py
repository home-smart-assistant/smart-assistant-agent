from __future__ import annotations

import math
import re
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

from app.core.config import AppConfig
from app.memory.faiss_store import FaissIndexStore
from app.memory.sqlite_store import SqliteMemoryStore


TOKEN_PATTERN = re.compile(r"[\w一-龿]+")


def tokenize(text: str) -> list[str]:
    return [item.lower() for item in TOKEN_PATTERN.findall(text)]


class EmbeddingProvider(ABC):
    name: str = "embedding"

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class HashEmbeddingProvider(EmbeddingProvider):
    name = "hash"

    def __init__(self, dim: int = 128) -> None:
        self._dim = max(32, dim)

    async def embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        tokens = tokenize(text)
        if not tokens:
            return vec
        for token in tokens:
            idx = hash(token) % self._dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(value * value for value in vec))
        if norm > 0:
            vec = [value / norm for value in vec]
        return vec


class OpenAiCompatibleEmbeddingProvider(EmbeddingProvider):
    name = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        path: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._path = path if path.startswith("/") else f"/{path}"
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds

    async def embed(self, text: str) -> list[float]:
        payload = {"model": self._model, "input": text}
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{self._base_url}{self._path}", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError("embedding_invalid_payload")
        rows = data.get("data", [])
        if not isinstance(rows, list) or not rows:
            raise RuntimeError("embedding_empty_payload")
        first = rows[0]
        if not isinstance(first, dict):
            raise RuntimeError("embedding_invalid_row")
        embedding = first.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError("embedding_invalid_vector")
        return [float(x) for x in embedding]


@dataclass
class MemoryDocument:
    memory_id: int
    doc_id: str
    session_id: str
    text: str
    metadata: dict[str, Any]
    created_at: float
    score: float | None = None


class LongTermMemoryService:
    def __init__(self, config: AppConfig, provider: EmbeddingProvider) -> None:
        self._provider = provider
        self._top_k = max(1, config.long_term_memory_top_k)
        self._max_items = max(1, config.long_term_memory_limit)
        self._min_score = max(-1.0, min(1.0, config.agent_faiss_min_score))
        self._enabled_by_config = bool(config.long_term_memory_enabled)
        self._index_path = config.agent_faiss_index_path
        self._meta_db_path = config.agent_faiss_meta_db_path
        self._lock = threading.Lock()

        self._enabled = False
        self._disabled_reason: str | None = None
        self._sqlite: SqliteMemoryStore | None = None
        self._faiss: FaissIndexStore | None = None

        if not self._enabled_by_config:
            self._disabled_reason = "disabled_by_config"
            return

        try:
            self._sqlite = SqliteMemoryStore(self._meta_db_path)
            self._faiss = FaissIndexStore(
                index_path=self._index_path,
                auto_flush_every=config.agent_faiss_auto_flush_every,
            )
            if not self._faiss.available:
                self._disabled_reason = self._faiss.disabled_reason or "faiss_unavailable"
                return
            self._bootstrap_index()
            self._enabled = True
        except Exception as ex:
            self._enabled = False
            self._disabled_reason = f"long_term_init_failed: {ex}"

    async def remember(self, session_id: str, text: str, metadata: dict[str, Any]) -> None:
        if not self._enabled:
            return
        if not self._sqlite or not self._faiss:
            return

        normalized = text.strip()
        if not normalized:
            return

        vector = await self._provider.embed(normalized)
        if not vector:
            return
        vector_np = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector_np.size == 0:
            return
        if float(np.linalg.norm(vector_np)) <= 0:
            return

        created_at = time.time()
        memory_id: int | None = None
        try:
            with self._lock:
                memory_id = self._sqlite.insert(
                    doc_id=uuid.uuid4().hex,
                    session_id=session_id,
                    text=normalized,
                    metadata=dict(metadata),
                    created_at=created_at,
                    vector=vector_np.tolist(),
                )
                self._faiss.add(memory_id=memory_id, vector=vector_np)
                self._enforce_limit_locked()
        except Exception as ex:
            if memory_id is not None:
                self._sqlite.delete_by_ids([memory_id])
            raise

    async def recall(self, session_id: str, query: str) -> list[MemoryDocument]:
        if not self._enabled:
            return []
        if not self._sqlite or not self._faiss:
            return []

        normalized = query.strip()
        if not normalized:
            return []

        vector = await self._provider.embed(normalized)
        if not vector:
            return []
        vector_np = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector_np.size == 0:
            return []
        if float(np.linalg.norm(vector_np)) <= 0:
            return []

        with self._lock:
            total_docs = self._sqlite.count()
            if total_docs <= 0:
                return []

            probe_k = min(total_docs, max(self._top_k * 8, 32))
            scored_ids = self._faiss.search(vector=vector_np, top_k=probe_k)
            if not scored_ids:
                return []

            filtered = [(memory_id, score) for memory_id, score in scored_ids if score >= self._min_score]
            if not filtered:
                return []

            ordered_ids = [memory_id for memory_id, _ in filtered]
            rows = self._sqlite.get_by_ids(ordered_ids, include_vector=False)
            row_map = {row.memory_id: row for row in rows}

            selected: list[MemoryDocument] = []
            seen_ids: set[int] = set()

            for memory_id, score in filtered:
                row = row_map.get(memory_id)
                if row is None or row.memory_id in seen_ids:
                    continue
                if row.session_id != session_id:
                    continue
                selected.append(self._to_memory_doc(row, score))
                seen_ids.add(row.memory_id)
                if len(selected) >= self._top_k:
                    return selected

            for memory_id, score in filtered:
                row = row_map.get(memory_id)
                if row is None or row.memory_id in seen_ids:
                    continue
                selected.append(self._to_memory_doc(row, score))
                seen_ids.add(row.memory_id)
                if len(selected) >= self._top_k:
                    break

            return selected

    def health_meta(self) -> dict[str, Any]:
        docs = 0
        index_total = 0
        if self._sqlite is not None:
            docs = self._sqlite.count()
        if self._faiss is not None:
            index_total = self._faiss.ntotal
        return {
            "backend": "faiss",
            "enabled": self._enabled,
            "disabled_reason": self._disabled_reason,
            "provider": self._provider.name,
            "docs": docs,
            "top_k": self._top_k,
            "index_ntotal": index_total,
            "index_path": self._index_path,
            "meta_db_path": self._meta_db_path,
            "min_score": self._min_score,
        }

    def _bootstrap_index(self) -> None:
        if not self._sqlite or not self._faiss:
            raise RuntimeError("memory_store_not_ready")

        loaded, load_error = self._faiss.load()
        if loaded:
            return

        vectors = self._sqlite.list_vectors()
        if vectors:
            self._faiss.rebuild(vectors)
            return

        if load_error:
            self._faiss.rebuild([])

    def _enforce_limit_locked(self) -> None:
        if not self._sqlite or not self._faiss:
            return
        current_count = self._sqlite.count()
        overflow = current_count - self._max_items
        if overflow <= 0:
            return
        deleted_ids = self._sqlite.delete_oldest(overflow)
        if not deleted_ids:
            return
        # Rebuild after cleanup to keep FAISS ids fully aligned with SQLite source of truth.
        vectors = self._sqlite.list_vectors()
        self._faiss.rebuild(vectors)

    def _disable(self, reason: str) -> None:
        self._enabled = False
        self._disabled_reason = reason

    def close(self) -> None:
        if self._sqlite is not None:
            self._sqlite.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _to_memory_doc(row: Any, score: float) -> MemoryDocument:
        return MemoryDocument(
            memory_id=row.memory_id,
            doc_id=row.doc_id,
            session_id=row.session_id,
            text=row.text,
            metadata=row.metadata,
            created_at=row.created_at,
            score=score,
        )


def build_embedding_provider(config: AppConfig) -> EmbeddingProvider:
    provider = config.embedding_provider
    if provider in {"openai", "openai_compatible", "openai-compatible"}:
        return OpenAiCompatibleEmbeddingProvider(
            base_url=config.openai_base_url,
            path=config.openai_embedding_path,
            api_key=config.openai_api_key,
            model=config.embedding_model,
            timeout_seconds=config.embedding_timeout_seconds,
        )
    return HashEmbeddingProvider()
