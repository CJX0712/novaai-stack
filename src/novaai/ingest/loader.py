"""NovaAI Stack — 文档加载模块（ingest）。

作者：晨星
职责：将本地文件加载为 Document。默认支持 txt / md（零依赖）；
可选支持 PDF（若已安装 pypdf，则自动启用，否则跳过）。
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import List

from ..types import Document


def _slug(text: str, max_len: int = 48) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text[:max_len] if text else "untitled"


class LocalTextLoader:
    """加载本地纯文本 / Markdown 文件（零依赖）。"""

    def _doc_id(self, path: str) -> str:
        key = os.path.abspath(path)
        return "doc-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]

    def _derive_title(self, text: str, path: str) -> str:
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("#"):
                return s.lstrip("#").strip()
        return _slug(os.path.splitext(os.path.basename(path))[0])

    def load(self, path: str) -> Document:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        return Document(
            doc_id=self._doc_id(path),
            title=self._derive_title(text, path),
            text=text,
            source=os.path.abspath(path),
        )


class PdfLoader:
    """加载 PDF 文件（可选依赖 pypdf，未安装则报错提示）。"""

    def __init__(self) -> None:
        try:
            import pypdf  # type: ignore
            self._have = True
        except Exception:
            self._have = False

    def load(self, path: str) -> Document:
        if not self._have:
            raise RuntimeError("PdfLoader 需要可选依赖 pypdf：pip install pypdf")
        from pypdf import PdfReader

        reader = PdfReader(path)
        parts: List[str] = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        text = "\n\n".join(parts)
        doc_id = "doc-" + hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
        return Document(doc_id=doc_id, title=os.path.splitext(os.path.basename(path))[0], text=text, source=path)


def make_document(doc_id: str, title: str, text: str, source: str = "") -> Document:
    """从原始文本直接构造 Document（供测试与 API 使用）。"""
    return Document(doc_id=doc_id, title=title, text=text, source=source)
