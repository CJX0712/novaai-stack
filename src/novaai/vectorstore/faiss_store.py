"""NovaAI Stack — 稠密向量索引生产适配器（vectorstore）。

作者：晨星
FaissVectorStore：复用 Meta 开源的 FAISS 近似/精确索引，适合百万级向量。
未安装 faiss-cpu 时按需报错。这是生产路径；默认运行使用 MemoryVectorStore。
"""

from __future__ import annotations

from typing import Dict, List

from ..types import Chunk, RetrievedChunk
from .memory_store import _cosine


class FaissVectorStore:
    """FAISS 适配实现（可选依赖 faiss-cpu）。"""

    def __init__(self, metric: str = "cosine") -> None:
        try:
            import faiss  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("FaissVectorStore 需要可选依赖 faiss-cpu：pip install faiss-cpu") from exc
        import numpy as np  # 延迟导入，避免默认路径强依赖 numpy
        self._np = np
        self._faiss = faiss
        self._index = None
        self._dim = 0
        self._ids: List[str] = []
        self._chunks: Dict[str, Chunk] = {}
        self._metric = metric

    def add(self, chunk_ids: List[str], vectors: List[List[float]], chunks: List[Chunk]) -> None:
        arr = self._np.asarray(vectors, dtype="float32")
        self._dim = arr.shape[1]
        if self._metric == "cosine":
            self._faiss.normalize_L2(arr)
        if self._index is None:
            self._index = self._faiss.IndexFlatIP(self._dim)
        self._index.add(arr)
        for cid, chk in zip(chunk_ids, chunks):
            self._ids.append(cid)
            self._chunks[cid] = chk

    def drop_document(self, doc_id: str) -> None:
        keep = [(cid, chk) for cid, chk in self._chunks.items() if chk.doc_id != doc_id]
        self._chunks = {}
        self._ids = []
        self._index = None
        self._dim = 0
        if keep:
            ids = [cid for cid, _ in keep]
            chs = [chk for _, chk in keep]
            # 从默认实现借余弦重算向量不可行（FAISS 不存原始向量），故要求调用方在 drop 后重新 add。
            # 为可用性，这里仅保留 chunk 元数据；真正重建由 pipeline 重新摄入完成。
            for cid, chk in keep:
                self._chunks[cid] = chk
                self._ids.append(cid)

    def search(self, vector: List[float], top_k: int) -> List[RetrievedChunk]:
        q = self._np.asarray([vector], dtype="float32")
        if self._metric == "cosine":
            self._faiss.normalize_L2(q)
        k = min(top_k, len(self._ids))
        if k == 0:
            return []
        scores, idxs = self._index.search(q, k)
        out: List[RetrievedChunk] = []
        for score, ix in zip(scores[0], idxs[0]):
            if ix < 0:
                continue
            cid = self._ids[ix]
            out.append(RetrievedChunk(self._chunks[cid], float(score), "dense"))
        return out
