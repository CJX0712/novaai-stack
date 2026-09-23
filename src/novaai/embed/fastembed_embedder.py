"""NovaAI Stack — 向量化生产适配器（embed）。

作者：晨星
FastEmbedEmbedder：复用 Qdrant 开源的 fastembed 模型（如 bge-small-zh），
通过环境变量 NOVAAI_EMBED_MODEL 选择。未安装 fastembed 时按需报错。
这是「复用业界领先开源成果」的生产路径；默认运行不依赖它。
"""

from __future__ import annotations

from typing import List

from .hash_embedder import HashEmbedder


class FastEmbedEmbedder:
    """fastembed 适配实现（可选依赖）。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        try:
            from fastembed import TextEmbedding  # type: ignore
        except Exception as exc:  # pragma: no cover - 取决于运行环境
            raise RuntimeError("FastEmbedEmbedder 需要可选依赖 fastembed：pip install fastembed") from exc
        self._model = TextEmbedding(model_name=model_name)
        # 探测维度
        sample = list(self._model.embed(["维度探测"]))[0]
        self.dim = len(sample)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [list(v) for v in self._model.embed(texts)]

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


def build_embedder(use_fastembed: bool = False, model_name: str = "BAAI/bge-small-zh-v1.5"):
    """工厂：默认返回零依赖 HashEmbedder，生产可切换 fastembed。"""
    if use_fastembed:
        return FastEmbedEmbedder(model_name=model_name)
    return HashEmbedder()
