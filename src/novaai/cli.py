"""NovaAI Stack — 命令行入口（cli）。

作者：晨星
提供 ingest / query / serve / demo 四个子命令。零依赖即可运行：
  python -m novaai.cli demo
  python -m novaai.cli serve --port 8000
  python -m novaai.cli query "NovaAI 支持哪些检索通道？"
生产后端通过环境变量 NOVAAI_LLM / NOVAAI_BACKEND 切换。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from .pipeline.pipeline import build_pipeline_from_env
from .types import Document

DEMO_DOCS: List[Document] = [
    Document(
        doc_id="kb-arch",
        title="NovaAI 系统架构",
        text=(
            "# NovaAI 系统架构\n"
            "NovaAI 是一个模块化检索增强推理系统。\n"
            "核心由摄入、向量化、稠密向量索引、稀疏检索、重排、生成与编排七个模块组成。\n"
            "每个模块通过 Protocol 定义接口，运行时依赖注入具体实现。\n"
            "默认实现均为零依赖的真实算法，保证在干净环境中一键复现。\n"
            "# 检索通道\n"
            "系统同时提供稠密向量召回与稀疏 BM25 召回两条通道。\n"
            "两条通道的结果经 RRF 融合重排后送入生成模块。\n"
        ),
    ),
    Document(
        doc_id="kb-deploy",
        title="NovaAI 部署指南",
        text=(
            "# 部署指南\n"
            "NovaAI 默认零依赖，仅需 Python 3.10 以上即可运行。\n"
            "启动服务：python -m novaai.cli serve --port 8000。\n"
            "生产环境可切换 Ollama 本地大模型或 OpenAI 兼容接口。\n"
            "也可启用 fastembed 与 FAISS 提升向量化与检索性能。\n"
        ),
    ),
    Document(
        doc_id="kb-agent",
        title="NovaAI 编排能力",
        text=(
            "# 编排能力\n"
            "NovaAI 的 Agent 基于 ReAct 范式进行推理与行动。\n"
            "默认走确定性意图路由：算术问题交给计算器，时间问题交给日期工具。\n"
            "其余问题进入检索增强生成链路，先召回再生成答案。\n"
        ),
    ),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="novaai", description="NovaAI Stack — 模块化检索增强推理系统")
    sub = parser.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser("ingest", help="摄入一个本地文件（txt/md/pdf）")
    p_ingest.add_argument("path", help="文件路径")

    p_query = sub.add_parser("query", help="对知识库提问")
    p_query.add_argument("question", help="问题")
    p_query.add_argument("--corpus", help="可选：先摄入该目录/文件作为知识库")
    p_query.add_argument("--json", action="store_true", help="以 JSON 输出")

    p_serve = sub.add_parser("serve", help="启动 HTTP 服务")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    sub.add_parser("demo", help="摄入内置示例知识库并演示问答")
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 0

    if args.cmd == "ingest":
        pipe = build_pipeline_from_env()
        res = pipe.ingest_file(args.path)
        print(json.dumps({"doc_id": res.doc_id, "chunk_count": res.chunk_count, "status": res.status}, ensure_ascii=False))
        return 0

    if args.cmd == "demo":
        pipe = build_pipeline_from_env()
        for doc in DEMO_DOCS:
            pipe.ingest_document(doc)
        sample = "NovaAI 支持哪些检索通道？"
        result = pipe.ask(sample)
        print(f"问题：{sample}")
        print(f"答案：{result.answer}")
        print(f"证据块数：{len(result.contexts)}  接地：{result.grounded}")
        return 0

    if args.cmd == "query":
        pipe = build_pipeline_from_env()
        if args.corpus:
            if os.path.isdir(args.corpus):
                for name in sorted(os.listdir(args.corpus)):
                    if name.lower().endswith((".txt", ".md", ".pdf")):
                        pipe.ingest_file(os.path.join(args.corpus, name))
            else:
                pipe.ingest_file(args.corpus)
        else:
            for doc in DEMO_DOCS:
                pipe.ingest_document(doc)
        result = pipe.ask(args.question)
        if args.json:
            print(json.dumps({
                "answer": result.answer, "grounded": result.grounded,
                "tool_calls": result.tool_calls, "trace": result.trace,
                "contexts": [{"title": rc.chunk.title, "score": round(rc.score, 4)} for rc in result.contexts],
            }, ensure_ascii=False, indent=2))
        else:
            print(f"答案：{result.answer}")
            print(f"接地：{result.grounded}  工具：{[t['tool'] for t in result.tool_calls]}")
        return 0

    if args.cmd == "serve":
        from .api.server import run
        pipe = build_pipeline_from_env()
        for doc in DEMO_DOCS:
            pipe.ingest_document(doc)
        print(f"NovaAI 服务已启动：http://{args.host}:{args.port}  (Ctrl+C 停止)")
        run(pipe, host=args.host, port=args.port)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
