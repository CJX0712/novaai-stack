"""NovaAI Stack — 共享数据类型定义。

作者：晨星
本文件定义跨模块流转的不可变数据契约（dataclass），所有模块均依赖这些类型，
不依赖任何具体实现，从而保证各模块可独立验证、可经依赖注入替换为 fake。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Document:
    """一份原始文档。"""

    doc_id: str
    title: str
    text: str
    source: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """文档切分后的语义块，继承所属标题路径（heading inheritance）。"""

    chunk_id: str
    doc_id: str
    text: str
    title: str
    heading_path: List[str] = field(default_factory=list)
    start: int = 0
    end: int = 0


@dataclass
class RetrievedChunk:
    """带召回来源与得分的检索结果。method 标记召回通道。"""

    chunk: Chunk
    score: float
    method: str = "dense"  # dense | sparse | rerank


@dataclass
class QueryResult:
    """一次查询的完整结果，包含答案、证据与可观测链路。"""

    query: str
    answer: str
    contexts: List[RetrievedChunk] = field(default_factory=list)
    tool_calls: List[Dict[str, str]] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)
    grounded: bool = False


@dataclass
class IngestResult:
    """摄入文档后的统计结果。"""

    doc_id: str
    chunk_count: int
    status: str = "ok"
