import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from novaai.llm.mock_llm import MockLLM
from novaai.llm.ollama_llm import OllamaLLM
from novaai.llm.openai_llm import OpenAILLM


class TestMockLLM(unittest.TestCase):
    def setUp(self):
        self.llm = MockLLM()

    def test_extractive_answer(self):
        ctx = "深圳位于中国南部沿海。上海是国际金融中心。北京是首都。"
        ans = self.llm.generate("深圳在哪里？", context=ctx)
        self.assertIn("深圳", ans)
        self.assertIn("南部", ans)

    def test_strip_citation(self):
        ctx = "[id#0] 深圳位于中国南部沿海。其他无关内容。"
        ans = self.llm.generate("深圳在哪里？", context=ctx)
        self.assertNotIn("[id#0]", ans)

    def test_fallback_no_context(self):
        ans = self.llm.generate("任意问题", context="")
        self.assertIn("未检索", ans)


class TestAdaptersImportable(unittest.TestCase):
    def test_ollama_returns_string(self):
        llm = OllamaLLM(model="qwen2.5:0.5b")
        out = llm.generate("你好", context="")
        self.assertIsInstance(out, str)

    def test_openai_without_key(self):
        llm = OpenAILLM()
        out = llm.generate("你好", context="")
        self.assertIn("NOVAAI_OPENAI_KEY", out)


if __name__ == "__main__":
    unittest.main()
