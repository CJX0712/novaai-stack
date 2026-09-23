import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from novaai.agent.react import ReActAgent
from novaai.agent.tools import calc, detect_arithmetic, detect_date
from novaai.llm.mock_llm import MockLLM
from novaai.types import Chunk, RetrievedChunk

_CTX = "深圳位于中国南部沿海，是科技创新之城。"


def _fake_retriever(query, top_k=5):
    c = Chunk(chunk_id="r1", doc_id="d1", text=_CTX, title="地理", start=0, end=len(_CTX))
    return [RetrievedChunk(c, 0.9, "dense")]


class TestTools(unittest.TestCase):
    def test_calc_basic(self):
        self.assertEqual(calc("12 * (3 + 4)"), "84")

    def test_calc_division(self):
        self.assertEqual(calc("10 / 4"), "2.5")

    def test_calc_fullwidth(self):
        self.assertEqual(calc("１２×（３＋４）"), "84")

    def test_calc_zero_division(self):
        with self.assertRaises(ValueError):
            calc("1 / 0")

    def test_detect_arithmetic(self):
        self.assertEqual(detect_arithmetic("请计算 12*(3+4)"), "12*(3+4)")

    def test_detect_date(self):
        self.assertIn("星期", detect_date("今天是星期几") or "")


class TestReActAgent(unittest.TestCase):
    def setUp(self):
        self.agent = ReActAgent(retriever=_fake_retriever, llm=MockLLM())

    def test_arithmetic_route(self):
        res = self.agent.run("12 * (3 + 4) 等于多少？")
        self.assertEqual(res.answer, "84")
        self.assertEqual(res.tool_calls[0]["tool"], "calculator")

    def test_date_route(self):
        res = self.agent.run("今天是星期几？")
        self.assertIn("星期", res.answer)
        self.assertEqual(res.tool_calls[0]["tool"], "date")

    def test_search_route_grounded(self):
        res = self.agent.run("深圳在哪个位置？")
        self.assertEqual(res.tool_calls[0]["tool"], "search")
        self.assertTrue(res.grounded)
        self.assertIn("深圳", res.answer)


if __name__ == "__main__":
    unittest.main()
