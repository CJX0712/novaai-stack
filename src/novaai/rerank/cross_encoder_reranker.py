"""NovaAI Stack — 重排生产适配器（rerank）。

作者：晨星
CrossEncoderReranker：复用 FlagEmbedding 开源的 BGE 重排模型（如 bge-reranker-base），
对 query–chunk 对打分重排，精度优于 lexical RRF。未安装 flagembedding 时按需报错。
这是生产路径；默认运行使用 LexicalReranker。
"""

from __future__ import annotations

from typing import List

from ..types import RetrievedChunk
from .lexical_reranker import LexicalReranker


class CrossEncoderReranker:
    """BGE cross-encoder 重排（可选依赖 flagembedding）。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        try:
            from FlagEmbedding import FlagReranker  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "CrossEncoderReranker 需要可选依赖 flagembedding：pip install flagembedding"
            ) from exc
        self._model = FlagReranker(model_name, use_fp16=False)

    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [[query, c.chunk.text] for c in candidates]
        scores = self._model.compute_score(pairs, normalize=True)
        scored = sorted(
            zip(candidates, scores), key=lambda t: (-t[1], t[0].chunk.chunk_id)
        )[:top_k]
        return [RetrievedChunk(c.chunk, float(s), "rerank") for c, s in scored]


def build_reranker(use_cross_encoder: bool = False):
    """工厂：默认 LexicalReranker，生产可切换 cross-encoder。"""
    if use_cross_encoder:
        return CrossEncoderReranker()
    return LexicalReranker()
