import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from novaai.lexical.bm25 import Bm25Retriever
from novaai.types import Chunk


def _chunk(cid, doc_id, text):
    return Chunk(chunk_id=cid, doc_id=doc_id, text=text, title="t", start=0, end=len(text))


class TestBm25(unittest.TestCase):
    def setUp(self):
        self.docs = {
            "a": "苹果公司发布了新款手机，搭载自研芯片。",
            "b": "香蕉是一种富含钾元素的热带水果。",
            "c": "火箭技术用于将卫星送入地球轨道。",
            "d": "深度学习模型需要大量标注数据进行训练。",
            "e": "深圳的台风季节通常在夏秋之交。",
        }
        self.bm = Bm25Retriever()
        chunks = [_chunk(k, k, v) for k, v in self.docs.items()]
        self.bm.index(chunks)

    def test_search_relevant(self):
        res = self.bm.search("火箭 卫星 轨道", top_k=3)
        self.assertEqual(res[0].chunk.chunk_id, "c")

    def test_method_label(self):
        res = self.bm.search("火箭", top_k=1)
        self.assertEqual(res[0].method, "sparse")

    def test_idf_positive_small_corpus(self):
        # 仅出现在 1 篇的词的 IDF 必为正（规避 rank_bm25 负值问题）
        self.assertGreater(self.bm._idf("火箭"), 0.0)
        # 出现在所有文档的词 IDF 仍为正（不会因 ln(1)=0 造成反转）
        common = Bm25Retriever()
        common.index([_chunk("x", "x", "系统 测试 系统"), _chunk("y", "y", "系统 学习 系统"), _chunk("z", "z", "系统 模型 系统")])
        self.assertGreater(common._idf("系统"), 0.0)


if __name__ == "__main__":
    unittest.main()
