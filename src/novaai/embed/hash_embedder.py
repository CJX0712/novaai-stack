"""NovaAI Stack — 向量化模块（embed）。

作者：晨星
默认实现 HashEmbedder：零依赖的确定性哈希嵌入。对中文采用「字 + 二元字组」，
对英文/数字采用词元，经 BLAKE2b 散列到固定维度并带符号累加，最后 L2 归一化。
相似文本的余弦相似度与其词元重叠正相关，可作为生产嵌入（fastembed）的可复现替身。
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List

_CJK = re.compile(r"[一-鿿]")
_WORD = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """确定性哈希嵌入（默认实现，零依赖）。"""

    def __init__(self, dim: int = 256, seed: int = 0) -> None:
        self.dim = dim
        self._salt = seed.to_bytes(4, "big")

    def _tokens(self, text: str) -> List[str]:
        low = text.lower()
        tokens: List[str] = []
        for m in _WORD.finditer(low):
            tokens.append("w:" + m.group(0))
        cjk = _CJK.findall(low)
        for ch in cjk:
            tokens.append("c:" + ch)
        for i in range(len(cjk) - 1):
            tokens.append("b:" + cjk[i] + cjk[i + 1])
        return tokens

    def embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for tok in self._tokens(text):
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8, salt=self._salt).digest()
            h = int.from_bytes(digest[:8], "big")
            idx = h % self.dim
            sign = -1.0 if (h >> 63) & 1 else 1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            inv = 1.0 / norm
            vec = [v * inv for v in vec]
        return vec

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_one(t) for t in texts]
