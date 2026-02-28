from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SqliteMemoryRow:
    memory_id: int
    doc_id: str
    session_id: str
    text: str
    metadata: dict[str, Any]
    created_at: float
    vector: np.ndarray | None = None


class SqliteMemoryStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = os.path.abspath(db_path)
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_docs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    vector_blob BLOB NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_docs_session ON memory_docs(session_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_docs_created ON memory_docs(created_at)"
            )
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS cnt FROM memory_docs").fetchone()
            return int(row["cnt"] if row else 0)

    def insert(
        self,
        doc_id: str,
        session_id: str,
        text: str,
        metadata: dict[str, Any],
        created_at: float,
        vector: list[float],
    ) -> int:
        metadata_json = json.dumps(metadata, ensure_ascii=False)
        vector_np = np.asarray(vector, dtype=np.float32)
        vector_blob = sqlite3.Binary(vector_np.tobytes())
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO memory_docs(doc_id, session_id, text, metadata_json, created_at, vector_blob)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (doc_id, session_id, text, metadata_json, created_at, vector_blob),
            )
            self._conn.commit()
            return int(cursor.lastrowid)

    def delete_by_ids(self, ids: list[int]) -> None:
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            self._conn.execute(f"DELETE FROM memory_docs WHERE id IN ({placeholders})", tuple(ids))
            self._conn.commit()

    def delete_oldest(self, limit: int) -> list[int]:
        if limit <= 0:
            return []
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id FROM memory_docs
                ORDER BY created_at ASC, id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            ids = [int(row["id"]) for row in rows]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                self._conn.execute(f"DELETE FROM memory_docs WHERE id IN ({placeholders})", tuple(ids))
                self._conn.commit()
            return ids

    def get_by_ids(self, ids: list[int], include_vector: bool = False) -> list[SqliteMemoryRow]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        fields = "id, doc_id, session_id, text, metadata_json, created_at"
        if include_vector:
            fields += ", vector_blob"
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {fields} FROM memory_docs WHERE id IN ({placeholders})",
                tuple(ids),
            ).fetchall()

        mapping: dict[int, SqliteMemoryRow] = {}
        for row in rows:
            vector = None
            if include_vector:
                blob = row["vector_blob"]
                vector = np.frombuffer(blob, dtype=np.float32).copy() if isinstance(blob, (bytes, bytearray)) else None
            metadata = self._parse_metadata(row["metadata_json"])
            mapping[int(row["id"])] = SqliteMemoryRow(
                memory_id=int(row["id"]),
                doc_id=str(row["doc_id"]),
                session_id=str(row["session_id"]),
                text=str(row["text"]),
                metadata=metadata,
                created_at=float(row["created_at"]),
                vector=vector,
            )
        return [mapping[memory_id] for memory_id in ids if memory_id in mapping]

    def list_vectors(self) -> list[tuple[int, np.ndarray]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, vector_blob FROM memory_docs ORDER BY id ASC"
            ).fetchall()
        results: list[tuple[int, np.ndarray]] = []
        for row in rows:
            blob = row["vector_blob"]
            if not isinstance(blob, (bytes, bytearray)):
                continue
            vector = np.frombuffer(blob, dtype=np.float32).copy()
            results.append((int(row["id"]), vector))
        return results

    @staticmethod
    def _parse_metadata(raw: object) -> dict[str, Any]:
        if not isinstance(raw, str):
            return {}
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
