import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from novaai.rerank.lexical_reranker import LexicalReranker
from novaai.types import Chunk, RetrievedChunk


def _chunk(cid):
    return Chunk(chunk_id=cid, doc_id=cid, text="内容 " + cid, title="t", start=0, end=10)


class TestLexicalReranker(unittest.TestCase):
    def setUp(self):
        self.r = LexicalReranker()

    def test_fusion_top_k(self):
        c1, c2, c3 = _chunk("a"), _chunk("b"), _chunk("c")
        cands = [
            RetrievedChunk(c1, 0.9, "dense"),
            RetrievedChunk(c2, 0.8, "dense"),
            RetrievedChunk(c1, 0.5, "sparse"),
            RetrievedChunk(c3, 0.4, "sparse"),
        ]
        out = self.r.rerank("q", cands, top_k=3)
        self.assertEqual(len(out), 3)
        # a 同时出现在两路，RRF 融合分最高，排第一
        self.assertEqual(out[0].chunk.chunk_id, "a")
        self.assertEqual(out[0].method, "rerank")

    def test_deterministic_tiebreak(self):
        c1, c2 = _chunk("a"), _chunk("b")
        cands = [RetrievedChunk(c1, 0.5, "dense"), RetrievedChunk(c2, 0.5, "dense")]
        out1 = self.r.rerank("q", cands, top_k=2)
        out2 = self.r.rerank("q", cands, top_k=2)
        self.assertEqual([c.chunk.chunk_id for c in out1], [c.chunk.chunk_id for c in out2])


if __name__ == "__main__":
    unittest.main()
