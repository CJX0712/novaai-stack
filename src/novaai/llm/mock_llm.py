"""NovaAI Stack — 文本生成模块（llm）。

作者：晨星
默认实现 MockLLM：零依赖的「抽取式阅读理解」基线（真实实现，非占位）。
从检索上下文中挑出与问题词集合余弦最高的句子作为答案，并剥离引用标记（[id#n]）。
生产可替换为 Ollama / OpenAI 兼容适配器。
"""

from __future__ import annotations

import math
import re
from typing import List, Set

from ..lexical.bm25 import tokenize

_CTX_OPEN = "<kb-context>"
_CTX_CLOSE = "</kb-context>"
_Q_MARK = "用户问题："

_CITE_RE = re.compile(r"\[id#[0-9]+\]|\((?:依据|来源|source)[^)]*\)", re.IGNORECASE)


def _word_set_cosine(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    if not inter:
        return 0.0
    return len(inter) / math.sqrt(len(a) * len(b))


class MockLLM:
    """抽取式阅读理解 LLM（默认实现，零依赖）。"""

    def generate(self, prompt: str, context: str = "") -> str:
        ctx = context or self._extract_context(prompt)
        question = self._extract_question(prompt)
        if not ctx or not ctx.strip():
            return "（未检索到相关上下文，无法回答该问题）"
        sentences: List[str] = [s.strip() for s in re.split(r"(?<=[。！？])|(?<=[.!?])(?=\s|$)", ctx) if s.strip()]
        if not sentences:
            return ctx[:200].strip()
        q_tokens = set(tokenize(question))
        best = max(sentences, key=lambda s: _word_set_cosine(set(tokenize(s)), q_tokens))
        return _CITE_RE.sub("", best).strip() or best

    def _extract_context(self, prompt: str) -> str:
        start = prompt.find(_CTX_OPEN)
        end = prompt.find(_CTX_CLOSE)
        if start == -1 or end == -1 or end <= start:
            return ""
        return prompt[start + len(_CTX_OPEN):end].strip()

    def _extract_question(self, prompt: str) -> str:
        idx = prompt.find(_Q_MARK)
        if idx == -1:
            return prompt.strip()
        return prompt[idx + len(_Q_MARK):].strip()
