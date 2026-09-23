import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from novaai.pipeline.pipeline import build_pipeline
from novaai.types import Document

CORPUS = [
    Document("d1", "架构", "# 架构\nNovaAI 支持稠密向量召回与稀疏 BM25 召回两条通道。\n两条通道经 RRF 融合重排后送入生成模块。\n"),
    Document("d2", "部署", "# 部署\nNovaAI 默认零依赖，仅需 Python 3.10 以上即可运行服务。\n"),
    Document("d3", "编排", "# 编排\nNovaAI 的 Agent 基于 ReAct 范式进行推理与行动。\n"),
]


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.pipe = build_pipeline()
        for d in CORPUS:
            self.pipe.ingest_document(d)

    def test_ingest_and_retrieve(self):
        res = self.pipe.ask("NovaAI 支持哪些检索通道？")
        self.assertTrue(res.grounded)
        self.assertTrue(res.contexts)

    def test_reingest_shorter_replaces(self):
        r = self.pipe.ingest_text("d1", "架构", "短内容。")
        self.assertEqual(r.chunk_count, 1)

    def test_deterministic_answer(self):
        a1 = self.pipe.ask("NovaAI 依赖什么运行？").answer
        a2 = self.pipe.ask("NovaAI 依赖什么运行？").answer
        self.assertEqual(a1, a2)

    def test_arithmetic_via_pipeline(self):
        res = self.pipe.ask("请计算 15 * 6")
        self.assertEqual(res.answer, "90")


if __name__ == "__main__":
    unittest.main()
