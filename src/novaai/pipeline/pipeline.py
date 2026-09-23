"""NovaAI Stack — 组合根 / 依赖注入（pipeline）。

作者：晨星
Pipeline 是系统的组合根：将 ingest / embed / vectorstore / lexical / rerank / llm / agent
通过构造函数注入装配，使各模块可独立替换为 fake 并单测。
对外暴露 ingest_document / retrieve / ask 三个入口，串起完整可运行链路。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from ..agent.react import ReActAgent
from ..embed.fastembed_embedder import build_embedder
from ..embed.hash_embedder import HashEmbedder
from ..ingest.chunker import SemanticChunker
from ..ingest.loader import LocalTextLoader, PdfLoader
from ..lexical.bm25 import Bm25Retriever
from ..llm.mock_llm import MockLLM
from ..llm.ollama_llm import OllamaLLM
from ..llm.openai_llm import OpenAILLM
from ..rerank.cross_encoder_reranker import build_reranker
from ..rerank.lexical_reranker import LexicalReranker
from ..types import Chunk, Document, IngestResult, QueryResult, RetrievedChunk
from ..vectorstore.memory_store import MemoryVectorStore


class Pipeline:
    """系统组合根。"""

    def __init__(
        self,
        embedder,
        vectorstore,
        lexical,
        reranker,
        llm,
        chunker=None,
        dense_top_k: int = 10,
        sparse_top_k: int = 10,
        rerank_top_k: int = 6,
    ) -> None:
        self._embedder = embedder
        self._vectorstore = vectorstore
        self._lexical = lexical
        self._reranker = reranker
        self._llm = llm
        self._chunker = chunker or SemanticChunker()
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.rerank_top_k = rerank_top_k
        self._chunks_by_doc: Dict[str, List[Chunk]] = {}
        self._agent = ReActAgent(retriever=self.retrieve, llm=self._llm, retrieve_top_k=rerank_top_k)

    # ---- 摄入 ----
    def ingest_document(self, doc: Document) -> IngestResult:
        chunks = self._chunker.chunk(doc)
        self._chunks_by_doc[doc.doc_id] = chunks
        self._vectorstore.drop_document(doc.doc_id)
        if chunks:
            vecs = self._embedder.embed([c.text for c in chunks])
            self._vectorstore.add([c.chunk_id for c in chunks], vecs, chunks)
        self._reindex_lexical()
        return IngestResult(doc.doc_id, len(chunks))

    def ingest_text(self, doc_id: str, title: str, text: str) -> IngestResult:
        return self.ingest_document(Document(doc_id=doc_id, title=title, text=text))

    def ingest_file(self, path: str) -> IngestResult:
        if path.lower().endswith(".pdf"):
            loader: object = PdfLoader()
        else:
            loader = LocalTextLoader()
        return self.ingest_document(loader.load(path))  # type: ignore[attr-defined]

    def _reindex_lexical(self) -> None:
        all_chunks = [c for cs in self._chunks_by_doc.values() for c in cs]
        self._lexical.index(all_chunks)

    # ---- 检索 ----
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        k = top_k or self.rerank_top_k
        qv = self._embedder.embed_one(query)
        dense = self._vectorstore.search(qv, self.dense_top_k)
        sparse = self._lexical.search(query, self.sparse_top_k)
        return self._reranker.rerank(query, dense + sparse, k)

    # ---- 问答 ----
    def ask(self, query: str) -> QueryResult:
        return self._agent.run(query)


def build_pipeline(
    use_fastembed: bool = False,
    use_faiss: bool = False,
    use_cross_encoder: bool = False,
    llm_kind: str = "mock",
    embed_dim: int = 256,
    embed_model: str = "BAAI/bge-small-zh-v1.5",
) -> Pipeline:
    """工厂：装配默认零依赖实现，或按环境变量/参数切换生产适配器。"""
    embedder = build_embedder(use_fastembed, embed_model) if use_fastembed else HashEmbedder(dim=embed_dim)
    if use_faiss:
        from ..vectorstore.faiss_store import FaissVectorStore
        vectorstore = FaissVectorStore()
    else:
        vectorstore = MemoryVectorStore()
    lexical = Bm25Retriever()
    reranker = build_reranker(use_cross_encoder)
    if llm_kind == "ollama":
        llm = OllamaLLM()
    elif llm_kind == "openai":
        llm = OpenAILLM()
    else:
        llm = MockLLM()
    return Pipeline(embedder, vectorstore, lexical, reranker, llm)


def build_pipeline_from_env() -> Pipeline:
    """按环境变量装配：NOVAAI_LLM={mock|ollama|openai}，NOVAAI_BACKEND={default|fastembed|faiss}。"""
    backend = os.environ.get("NOVAAI_BACKEND", "default")
    llm_kind = os.environ.get("NOVAAI_LLM", "mock")
    return build_pipeline(
        use_fastembed=(backend in ("fastembed", "full")),
        use_faiss=(backend in ("faiss", "full")),
        use_cross_encoder=(backend in ("full",)),
        llm_kind=llm_kind,
    )
