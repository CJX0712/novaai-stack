"""NovaAI Stack — 文本分块模块（ingest）。

作者：晨星
职责：将文档切分为语义块，并继承所属标题路径（heading inheritance）。
默认实现为零依赖的段落聚合分块：按段落累积至字符预算后切分，块起始位置之前的
最近标题决定该块的 heading_path，保证首个块也能继承其前的标题。
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..types import Chunk, Document

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


class SemanticChunker:
    """段落聚合 + 标题继承的语义分块器。"""

    def __init__(self, max_chars: int = 800, overlap_chars: int = 160, min_chars: int = 80) -> None:
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.min_chars = min_chars

    def _split_paragraphs(self, text: str) -> List[Tuple[int, int, bool, str]]:
        """返回 (start, end, is_heading, heading_title)。按空行分段。"""
        out: List[Tuple[int, int, bool, str]] = []
        idx = 0
        for raw in text.split("\n\n"):
            stripped = raw.strip()
            if not stripped:
                idx += len(raw) + 2
                continue
            start = idx
            end = idx + len(raw)
            first = stripped.split("\n", 1)[0].strip()
            m = _HEADING_RE.match(first)
            if m:
                # 标题行单独成块（仅更新标题栈，不进入正文缓冲）
                hlen = len(first)
                out.append((start, start + hlen, True, m.group(2).strip()))
                rest = raw[hlen:].strip()
                if rest:
                    out.append((start + hlen, end, False, ""))
            else:
                out.append((start, end, False, ""))
            idx = end + 2
        return out

    def chunk(self, doc: Document) -> List[Chunk]:
        paragraphs = self._split_paragraphs(doc.text)
        chunks: List[Chunk] = []
        path_stack: List[Tuple[int, str]] = []
        current_path: List[str] = []
        buf: List[str] = []
        buf_start: Optional[int] = None
        buf_len = 0
        buf_path: List[str] = []
        counter = 0

        def flush(keep_last: Optional[str] = None, keep_start: Optional[int] = None) -> None:
            nonlocal buf, buf_start, buf_len, buf_path, counter
            if not buf:
                return
            body = "\n\n".join(buf)
            title = buf_path[-1] if buf_path else doc.title
            cid = f"{doc.doc_id}#c{counter}"
            counter += 1
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    doc_id=doc.doc_id,
                    text=body,
                    title=title,
                    heading_path=list(buf_path),
                    start=buf_start if buf_start is not None else 0,
                    end=(buf_start if buf_start is not None else 0) + len(body),
                )
            )
            buf = []
            buf_start = None
            buf_len = 0
            buf_path = []
            if keep_last is not None and keep_start is not None:
                buf = [keep_last]
                buf_start = keep_start
                buf_len = len(keep_last)
                buf_path = list(current_path)

        for start, end, is_heading, title in paragraphs:
            if is_heading:
                # 标题前的缓冲沿用旧路径先切分，保证标题继承正确
                if buf:
                    flush()
                fl = doc.text[start:end].split("\n", 1)[0].strip()
                level = len(fl) - len(fl.lstrip("#"))
                while path_stack and path_stack[-1][0] >= level:
                    path_stack.pop()
                path_stack.append((level, title))
                current_path = [t for _, t in path_stack]
                continue
            para_text = doc.text[start:end]
            if buf_start is None:
                buf_start = start
                buf_path = list(current_path)
            if buf_len > 0 and buf_len + len(para_text) > self.max_chars and buf_len >= self.min_chars:
                flush(keep_last=para_text, keep_start=start)
                continue
            buf.append(para_text)
            buf_len += len(para_text)
        flush()
        return chunks
