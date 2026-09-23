"""NovaAI Stack — 模块化检索增强推理系统。

作者：晨星
版本：1.0.0
"""

__version__ = "1.0.0"
__author__ = "晨星"

from .ports import (
    Agent,
    Chunker,
    DocumentLoader,
    Embedder,
    LexicalRetriever,
    LLM,
    Reranker,
    VectorStore,
)
from .pipeline.pipeline import Pipeline, build_pipeline, build_pipeline_from_env
from .types import Chunk, Document, IngestResult, QueryResult, RetrievedChunk

__all__ = [
    "Document", "Chunk", "RetrievedChunk", "QueryResult", "IngestResult",
    "DocumentLoader", "Chunker", "Embedder", "VectorStore", "LexicalRetriever",
    "Reranker", "LLM", "Agent", "Pipeline", "build_pipeline", "build_pipeline_from_env",
]
