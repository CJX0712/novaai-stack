"""NovaAI Stack — 稠密向量索引模块（vectorstore）。

作者：晨星
默认实现 MemoryVectorStore：零依赖、进程内的精确余弦相似度索引。
提供真实可用的稠密召回；生产可替换为 FAISS（Meta 开源）适配器。
upsert 语义由调用方保证（摄入前先 drop_document），本实现按 chunk_id 去重。
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from ..types import Chunk, RetrievedChunk


def _cosine(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class MemoryVectorStore:
    """进程内精确余弦向量索引（默认实现）。"""

    def __init__(self) -> None:
        self._ids: List[str] = []
        self._vecs: List[List[float]] = []
        self._chunks: Dict[str, Chunk] = {}

    def add(self, chunk_ids: List[str], vectors: List[List[float]], chunks: List[Chunk]) -> None:
        for cid, vec, chk in zip(chunk_ids, vectors, chunks):
            if cid in self._chunks:
                self._remove(cid)
            self._ids.append(cid)
            self._vecs.append(vec)
            self._chunks[cid] = chk

    def _remove(self, cid: str) -> None:
        if cid not in self._chunks:
            return
        idx = self._ids.index(cid)
        self._ids.pop(idx)
        self._vecs.pop(idx)
        self._chunks.pop(cid)

    def drop_document(self, doc_id: str) -> None:
        for cid in [i for i, c in self._chunks.items() if c.doc_id == doc_id]:
            self._remove(cid)

    def _ranked(self, vector: List[float]) -> List[Tuple[str, float]]:
        scored = [(cid, _cosine(vector, self._vecs[i])) for i, cid in enumerate(self._ids)]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored

    def search(self, vector: List[float], top_k: int) -> List[RetrievedChunk]:
        ranked = self._ranked(vector)[:top_k]
        return [RetrievedChunk(self._chunks[cid], score, "dense") for cid, score in ranked]
