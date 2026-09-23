import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from novaai.ingest.chunker import SemanticChunker
from novaai.ingest.loader import LocalTextLoader, make_document
from novaai.types import Document


def _long_doc() -> Document:
    para = "NovaAI 是一个模块化检索增强推理系统，默认实现均为零依赖的真实算法。" * 12
    text = (
        "# 第一章 架构\n" + para + "\n\n"
        "# 第二章 部署\n" + para + "\n\n"
        "# 第三章 编排\n" + para
    )
    return Document(doc_id="long", title="长文档", text=text)


class TestChunker(unittest.TestCase):
    def test_multi_chunk(self):
        chunks = SemanticChunker(max_chars=400, overlap_chars=80).chunk(_long_doc())
        self.assertGreater(len(chunks), 1)

    def test_heading_inheritance(self):
        chunks = SemanticChunker(max_chars=400, overlap_chars=80).chunk(_long_doc())
        # 首个块继承其前标题「第一章 架构」
        self.assertTrue(any("第一章 架构" in p for p in chunks[0].heading_path))
        # 含「第三章」的块应继承到第三章
        hit = [c for c in chunks if any("第三章" in p for p in c.heading_path)]
        self.assertTrue(hit)

    def test_chunk_ids_unique(self):
        chunks = SemanticChunker().chunk(_long_doc())
        ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_short_doc_single_chunk(self):
        doc = make_document("s", "短", "仅一句话的文档内容。")
        chunks = SemanticChunker().chunk(doc)
        self.assertEqual(len(chunks), 1)


class TestLoader(unittest.TestCase):
    def test_make_document(self):
        d = make_document("x", "标题", "正文")
        self.assertEqual(d.doc_id, "x")
        self.assertEqual(d.title, "标题")


if __name__ == "__main__":
    unittest.main()
