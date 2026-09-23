"""NovaAI Stack — 稀疏检索模块（lexical）。

作者：晨星
默认实现 Bm25Retriever：零依赖、纯 Python 的 BM25（Robertson IDF）。
IDF 采用 ln(1 + (N - n + 0.5)/(n + 0.5))，对所有词恒非负，
规避 rank_bm25 在小语料下 IDF 为负导致排序反转的问题。
中文按「字 + 二元字组」切分，英文按词切分。
"""

from __future__ import annotations

import math
import re
from typing import Dict, List

from ..types import Chunk, RetrievedChunk

_CJK = re.compile(r"[一-鿿]")
_WORD = re.compile(r"[a-z0-9]+")

K1 = 1.5
B = 0.75


def tokenize(text: str) -> List[str]:
    low = text.lower()
    toks: List[str] = []
    for m in _WORD.finditer(low):
        toks.append("w:" + m.group(0))
    cjk = _CJK.findall(low)
    for ch in cjk:
        toks.append("c:" + ch)
    for i in range(len(cjk) - 1):
        toks.append("b:" + cjk[i] + cjk[i + 1])
    return toks


class Bm25Retriever:
    """Robertson IDF BM25 稀疏检索（默认实现）。"""

    def __init__(self) -> None:
        self._docs: Dict[str, List[str]] = {}
        self._chunks: Dict[str, Chunk] = {}
        self._df: Dict[str, int] = {}
        self._N = 0
        self._avgdl = 0.0

    def index(self, chunks: List[Chunk]) -> None:
        for chk in chunks:
            toks = tokenize(chk.text)
            self._docs[chk.chunk_id] = toks
            self._chunks[chk.chunk_id] = chk
            seen = set()
            for t in toks:
                if t not in seen:
                    self._df[t] = self._df.get(t, 0) + 1
                    seen.add(t)
        self._N = len(self._docs)
        total = sum(len(v) for v in self._docs.values())
        self._avgdl = (total / self._N) if self._N else 0.0

    def _idf(self, term: str) -> float:
        n = self._df.get(term, 0)
        return math.log(1.0 + (self._N - n + 0.5) / (n + 0.5))

    def search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        if self._N == 0:
            return []
        q_terms = tokenize(query)
        scores: Dict[str, float] = {}
        for t in q_terms:
            if t not in self._df:
                continue
            idf = self._idf(t)
            for cid, toks in self._docs.items():
                f = toks.count(t)
                if f == 0:
                    continue
                dl = len(toks)
                denom = f + K1 * (1 - B + B * (dl / self._avgdl if self._avgdl else 1.0))
                scores[cid] = scores.get(cid, 0.0) + idf * (f * (K1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [RetrievedChunk(self._chunks[cid], score, "sparse") for cid, score in ranked]
