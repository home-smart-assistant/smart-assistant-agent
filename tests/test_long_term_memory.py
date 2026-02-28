from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from dataclasses import replace
from unittest import mock

from app.core.config import AppConfig
from app.memory.faiss_store import try_import_faiss
from app.memory.long_term import EmbeddingProvider, LongTermMemoryService


def run_async(coro):
    return asyncio.run(coro)


class DummyEmbeddingProvider(EmbeddingProvider):
    name = "dummy"

    async def embed(self, text: str) -> list[float]:
        if "卧室" in text or "bedroom" in text:
            return [1.0, 0.0, 0.0]
        if "空调" in text or "climate" in text:
            return [0.0, 1.0, 0.0]
        if "天气" in text or "weather" in text:
            return [0.0, 0.0, 1.0]
        if "灯" in text or "light" in text:
            return [0.8, 0.2, 0.0]
        return [0.2, 0.3, 0.5]


FAISS_MODULE, _ = try_import_faiss()
HAS_FAISS = FAISS_MODULE is not None


@unittest.skipUnless(HAS_FAISS, "faiss-cpu not installed in current environment")
class TestFaissLongTermMemory(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.index_path = os.path.join(self.tempdir.name, "faiss.index")
        self.db_path = os.path.join(self.tempdir.name, "memory.db")
        self.provider = DummyEmbeddingProvider()

    def make_service(
        self,
        *,
        top_k: int = 3,
        limit: int = 300,
        min_score: float = 0.0,
    ) -> LongTermMemoryService:
        config = replace(
            AppConfig(),
            long_term_memory_enabled=True,
            long_term_memory_top_k=top_k,
            long_term_memory_limit=limit,
            agent_faiss_index_path=self.index_path,
            agent_faiss_meta_db_path=self.db_path,
            agent_faiss_min_score=min_score,
            agent_faiss_auto_flush_every=1,
        )
        service = LongTermMemoryService(config=config, provider=self.provider)
        self.addCleanup(service.close)
        return service

    def test_remember_and_recall_hit(self) -> None:
        service = self.make_service(top_k=2)
        run_async(service.remember("s1", "打开卧室灯", {"role": "user"}))
        docs = run_async(service.recall("s1", "卧室灯"))
        self.assertGreaterEqual(len(docs), 1)
        self.assertEqual("s1", docs[0].session_id)
        self.assertIn("卧室", docs[0].text)

    def test_session_priority_then_global_fill(self) -> None:
        service = self.make_service(top_k=2)
        run_async(service.remember("s2", "打开卧室灯", {"source": "global"}))
        run_async(service.remember("s1", "把卧室灯打开", {"source": "session"}))
        docs = run_async(service.recall("s1", "卧室灯"))
        self.assertEqual(2, len(docs))
        self.assertEqual("s1", docs[0].session_id)
        self.assertEqual("s2", docs[1].session_id)

    def test_limit_cleanup_keeps_docs_within_bound(self) -> None:
        service = self.make_service(top_k=3, limit=2)
        run_async(service.remember("s1", "打开卧室灯", {}))
        run_async(service.remember("s1", "空调调到25度", {}))
        run_async(service.remember("s1", "今天天气如何", {}))
        meta = service.health_meta()
        self.assertLessEqual(meta["docs"], 2)
        self.assertEqual(meta["docs"], meta["index_ntotal"])

    def test_rebuild_when_index_file_missing(self) -> None:
        service = self.make_service(top_k=2)
        run_async(service.remember("s1", "打开卧室灯", {}))
        self.assertTrue(os.path.exists(self.index_path))

        os.remove(self.index_path)
        recovered = self.make_service(top_k=2)
        docs = run_async(recovered.recall("s1", "卧室灯"))
        self.assertGreaterEqual(len(docs), 1)
        self.assertEqual("s1", docs[0].session_id)


class TestFaissImportFailureFallback(unittest.TestCase):
    def test_disable_long_term_memory_if_faiss_missing(self) -> None:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)

        config = replace(
            AppConfig(),
            long_term_memory_enabled=True,
            agent_faiss_index_path=os.path.join(tempdir.name, "faiss.index"),
            agent_faiss_meta_db_path=os.path.join(tempdir.name, "memory.db"),
        )

        with mock.patch("app.memory.faiss_store.try_import_faiss", return_value=(None, "mocked_missing_faiss")):
            service = LongTermMemoryService(config=config, provider=DummyEmbeddingProvider())
            self.addCleanup(service.close)
            meta = service.health_meta()
            self.assertFalse(meta["enabled"])
            self.assertIn("mocked_missing_faiss", str(meta["disabled_reason"]))

            run_async(service.remember("s1", "打开卧室灯", {}))
            docs = run_async(service.recall("s1", "卧室灯"))
            self.assertEqual([], docs)
