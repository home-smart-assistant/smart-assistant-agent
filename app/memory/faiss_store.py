from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np


def try_import_faiss() -> tuple[Any | None, str | None]:
    try:
        import faiss  # type: ignore

        return faiss, None
    except Exception as ex:
        return None, f"faiss_import_error: {ex}"


class FaissIndexStore:
    def __init__(self, index_path: str, auto_flush_every: int = 1) -> None:
        self.index_path = os.path.abspath(index_path)
        self.auto_flush_every = max(1, int(auto_flush_every))
        self._lock = threading.Lock()
        self._faiss, self._import_error = try_import_faiss()
        self._index: Any | None = None
        self._dim: int | None = None
        self._pending_writes = 0
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

    @property
    def available(self) -> bool:
        return self._faiss is not None

    @property
    def disabled_reason(self) -> str | None:
        return self._import_error

    @property
    def ntotal(self) -> int:
        with self._lock:
            if self._index is None:
                return 0
            return int(getattr(self._index, "ntotal", 0) or 0)

    @property
    def dim(self) -> int | None:
        with self._lock:
            return self._dim

    def load(self) -> tuple[bool, str | None]:
        if not self.available:
            return False, self._import_error
        if not os.path.exists(self.index_path):
            return False, None

        with self._lock:
            try:
                index = self._faiss.read_index(self.index_path)
            except Exception as ex:
                return False, f"faiss_load_error: {ex}"

            if not hasattr(index, "add_with_ids"):
                return False, "faiss_invalid_index_type_no_ids"

            dim = int(getattr(index, "d", 0) or 0)
            if dim <= 0:
                return False, "faiss_invalid_index_dimension"

            self._index = index
            self._dim = dim
            self._pending_writes = 0
            return True, None

    def save(self, force: bool = False) -> None:
        if not self.available:
            return
        with self._lock:
            self._save_locked(force=force)

    def add(self, memory_id: int, vector: np.ndarray | list[float]) -> None:
        if not self.available:
            raise RuntimeError(self._import_error or "faiss_unavailable")

        normalized = self._normalize(vector)
        with self._lock:
            self._ensure_index_locked(normalized.size)
            vec = normalized.reshape(1, -1).astype(np.float32, copy=False)
            ids = np.asarray([int(memory_id)], dtype=np.int64)
            self._index.add_with_ids(vec, ids)
            self._pending_writes += 1
            if self._pending_writes >= self.auto_flush_every:
                self._save_locked(force=True)

    def search(self, vector: np.ndarray | list[float], top_k: int) -> list[tuple[int, float]]:
        if not self.available:
            return []
        normalized = self._normalize(vector)
        with self._lock:
            if self._index is None:
                return []
            total = int(getattr(self._index, "ntotal", 0) or 0)
            if total <= 0:
                return []
            k = min(max(1, int(top_k)), total)
            vec = normalized.reshape(1, -1).astype(np.float32, copy=False)
            scores, ids = self._index.search(vec, k)

        results: list[tuple[int, float]] = []
        for raw_id, raw_score in zip(ids[0], scores[0]):
            current_id = int(raw_id)
            if current_id < 0:
                continue
            results.append((current_id, float(raw_score)))
        return results

    def remove_ids(self, ids: list[int]) -> int:
        if not self.available or not ids:
            return 0
        with self._lock:
            if self._index is None:
                return 0
            id_array = np.asarray([int(row) for row in ids], dtype=np.int64)
            try:
                removed = int(self._index.remove_ids(id_array))
            except Exception:
                selector = self._faiss.IDSelectorBatch(id_array.size, self._faiss.swig_ptr(id_array))
                removed = int(self._index.remove_ids(selector))
            if removed > 0:
                self._pending_writes += 1
                if self._pending_writes >= self.auto_flush_every:
                    self._save_locked(force=True)
            return removed

    def rebuild(self, rows: list[tuple[int, np.ndarray]]) -> None:
        if not self.available:
            raise RuntimeError(self._import_error or "faiss_unavailable")
        with self._lock:
            valid_rows: list[tuple[int, np.ndarray]] = []
            for memory_id, vector in rows:
                normalized = self._normalize(vector)
                valid_rows.append((int(memory_id), normalized))

            if not valid_rows:
                self._index = None
                self._dim = None
                self._pending_writes = 0
                if os.path.exists(self.index_path):
                    os.remove(self.index_path)
                return

            dim = valid_rows[0][1].size
            vectors = np.vstack([vector for _, vector in valid_rows]).astype(np.float32, copy=False)
            ids = np.asarray([memory_id for memory_id, _ in valid_rows], dtype=np.int64)
            index = self._faiss.IndexIDMap2(self._faiss.IndexFlatIP(dim))
            index.add_with_ids(vectors, ids)
            self._index = index
            self._dim = dim
            self._pending_writes = 0
            self._save_locked(force=True)

    def _save_locked(self, force: bool = False) -> None:
        if self._index is None:
            return
        if self._pending_writes <= 0 and not force:
            return
        self._faiss.write_index(self._index, self.index_path)
        self._pending_writes = 0

    def _ensure_index_locked(self, dim: int) -> None:
        if self._index is None:
            self._index = self._faiss.IndexIDMap2(self._faiss.IndexFlatIP(dim))
            self._dim = dim
            return
        if self._dim != dim:
            raise RuntimeError(f"faiss_dimension_mismatch: current={self._dim}, incoming={dim}")

    @staticmethod
    def _normalize(vector: np.ndarray | list[float]) -> np.ndarray:
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            raise RuntimeError("empty_vector")
        norm = float(np.linalg.norm(vec))
        if norm <= 0:
            raise RuntimeError("zero_norm_vector")
        return vec / norm
