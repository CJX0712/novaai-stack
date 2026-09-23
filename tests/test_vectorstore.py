import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from novaai.embed.hash_embedder import HashEmbedder
from novaai.types import Chunk
from novaai.vectorstore.memory_store import MemoryVectorStore


def _chunk(cid, doc_id, text):
    return Chunk(chunk_id=cid, doc_id=doc_id, text=text, title="t", start=0, end=len(text))


class TestMemoryVectorStore(unittest.TestCase):
    def setUp(self):
        self.store = MemoryVectorStore()
        self.emb = HashEmbedder(dim=64)
        texts = {"a": "苹果 公司 发布 新手机", "b": "香蕉 水果 富含 钾", "c": "火箭 发射 进入 轨道"}
        chunks = [_chunk(k, k, v) for k, v in texts.items()]
        vecs = self.emb.embed([v for v in texts.values()])
        self.store.add(list(texts.keys()), vecs, chunks)

    def test_search_returns_relevant(self):
        q = self.emb.embed_one("苹果 手机 新品")
        res = self.store.search(q, top_k=3)
        self.assertEqual(res[0].chunk.chunk_id, "a")

    def test_scores_in_range(self):
        q = self.emb.embed_one("火箭 轨道")
        for rc in self.store.search(q, top_k=3):
            self.assertGreaterEqual(rc.score, -1.0)
            self.assertLessEqual(rc.score, 1.0)

    def test_drop_document(self):
        self.store.drop_document("a")
        q = self.emb.embed_one("苹果 手机")
        res = self.store.search(q, top_k=3)
        self.assertNotIn("a", [r.chunk.chunk_id for r in res])


if __name__ == "__main__":
    unittest.main()
