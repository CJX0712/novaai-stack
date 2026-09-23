"""NovaAI Stack — 多路融合重排模块（rerank）。

作者：晨星
默认实现 LexicalReranker：零依赖的 Reciprocal Rank Fusion（RRF）。
将稠密召回与稀疏召回各自排序后，按 1/(k+rank) 融合，得到稳定的跨通道排序。
对并列得分用 chunk_id 做确定性 tie-break，保证可复现。
"""

from __future__ import annotations

from typing import Dict, List

from ..types import Chunk, RetrievedChunk

RRF_K = 60


class LexicalReranker:
    """RRF 多路融合重排（默认实现）。"""

    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        dense = sorted(
            [c for c in candidates if c.method == "dense"],
            key=lambda c: (-c.score, c.chunk.chunk_id),
        )
        sparse = sorted(
            [c for c in candidates if c.method == "sparse"],
            key=lambda c: (-c.score, c.chunk.chunk_id),
        )
        fused: Dict[str, Dict] = {}
        for method, lst in (("dense", dense), ("sparse", sparse)):
            for rank, c in enumerate(lst):
                entry = fused.setdefault(
                    c.chunk.chunk_id, {"chunk": c.chunk, "score": 0.0}
                )
                entry["score"] += 1.0 / (RRF_K + rank + 1)
        ranked = sorted(
            fused.values(), key=lambda d: (-d["score"], d["chunk"].chunk_id)
        )[:top_k]
        return [RetrievedChunk(d["chunk"], d["score"], "rerank") for d in ranked]
