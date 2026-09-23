"""NovaAI Stack — 模块接口契约（Protocol 定义）。

作者：晨星
每个外部能力都定义为 Protocol（结构化子类型），运行时通过依赖注入注入具体实现。
默认注入零依赖真实实现；生产环境可注入 Ollama / FAISS / fastembed 等开源适配器。
这样每个模块可独立单测（用 fake 实现），又能协同组成完整可运行链路。
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from .types import Chunk, Document, RetrievedChunk


@runtime_checkable
class DocumentLoader(Protocol):
    """文档加载端口。"""

    def load(self, path: str) -> Document:
        """从路径加载一份文档。"""
        ...


@runtime_checkable
class Chunker(Protocol):
    """文本分块端口。"""

    def chunk(self, doc: Document) -> List[Chunk]:
        """将文档切分为语义块（含标题继承）。"""
        ...


@runtime_checkable
class Embedder(Protocol):
    """文本向量化端口。"""

    dim: int

    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量向量化，返回与输入等长的向量列表。"""
        ...

    def embed_one(self, text: str) -> List[float]:
        """单条向量化。"""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """稠密向量索引端口。"""

    def add(self, chunk_ids: List[str], vectors: List[List[float]], chunks: List[Chunk]) -> None:
        ...

    def search(self, vector: List[float], top_k: int) -> List[RetrievedChunk]:
        ...

    def drop_document(self, doc_id: str) -> None:
        ...


@runtime_checkable
class LexicalRetriever(Protocol):
    """稀疏检索端口。"""

    def index(self, chunks: List[Chunk]) -> None:
        ...

    def search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        ...


@runtime_checkable
class Reranker(Protocol):
    """多路融合重排端口。"""

    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        ...


@runtime_checkable
class LLM(Protocol):
    """文本生成端口。"""

    def generate(self, prompt: str, context: str = "") -> str:
        """基于 prompt 与可选上下文生成文本。"""
        ...


@runtime_checkable
class Agent(Protocol):
    """推理编排端口。"""

    def run(self, query: str) -> "object":
        """执行一次推理，返回 QueryResult。"""
        ...
