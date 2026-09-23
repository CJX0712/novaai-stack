import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from novaai.embed.hash_embedder import HashEmbedder


class TestHashEmbedder(unittest.TestCase):
    def setUp(self):
        self.e = HashEmbedder(dim=128)

    def test_dim(self):
        self.assertEqual(self.e.dim, 128)

    def test_deterministic(self):
        a = self.e.embed_one("人工智能 系统")
        b = self.e.embed_one("人工智能 系统")
        self.assertEqual(a, b)

    def test_normalized(self):
        v = self.e.embed_one("深度学习 模型")
        norm = math.sqrt(sum(x * x for x in v))
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_different_texts_differ(self):
        a = self.e.embed_one("苹果 公司 发布 新品")
        b = self.e.embed_one("香蕉 水果 富含 钾元素")
        self.assertNotEqual(a, b)

    def test_chinese_bigram_influence(self):
        # 中文相近文本应有较高余弦相似度
        a = self.e.embed_one("深圳 天气 今天 晴朗")
        b = self.e.embed_one("深圳 天气 今天 多云")
        dot = sum(x * y for x, y in zip(a, b))
        self.assertGreater(dot, 0.0)


if __name__ == "__main__":
    unittest.main()
