"""NovaAI Stack — HTTP 服务模块（api）。

作者：晨星
默认使用 Python 标准库 http.server 提供零依赖 REST 服务（保证一键可复现）。
生产可改用 FastAPI 适配器（见 docs/SPEC.md）。提供 /health /ingest /query 三端点。
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional


def _make_handler(pipeline):
    class Handler(BaseHTTPRequestHandler):
        _pipeline = pipeline

        def _send(self, code: int, obj) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:  # 静默访问日志
            return

        def do_GET(self) -> None:
            if self.path.split("?")[0] == "/health":
                self._send(200, {"status": "ok", "service": "novaai", "docs": len(self._pipeline._chunks_by_doc)})
            else:
                self._send(404, {"error": "not_found"})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except Exception:
                self._send(400, {"error": "invalid_json"})
                return
            route = self.path.split("?")[0]
            if route == "/ingest":
                self._ingest(payload)
            elif route == "/query":
                self._query(payload)
            else:
                self._send(404, {"error": "not_found"})

        def _ingest(self, payload) -> None:
            if "path" in payload:
                res = self._pipeline.ingest_file(payload["path"])
            elif "text" in payload:
                res = self._pipeline.ingest_text(
                    payload.get("doc_id", "doc-api"), payload.get("title", "API 文档"), payload["text"]
                )
            else:
                self._send(400, {"error": "需要 path 或 text"})
                return
            self._send(200, {"doc_id": res.doc_id, "chunk_count": res.chunk_count, "status": res.status})

        def _query(self, payload) -> None:
            query = payload.get("query", "")
            if not query:
                self._send(400, {"error": "需要 query"})
                return
            result = self._pipeline.ask(query)
            self._send(200, {
                "query": result.query,
                "answer": result.answer,
                "grounded": result.grounded,
                "tool_calls": result.tool_calls,
                "trace": result.trace,
                "contexts": [
                    {"chunk_id": rc.chunk.chunk_id, "title": rc.chunk.title, "score": round(rc.score, 4),
                     "text": rc.chunk.text[:300]}
                    for rc in result.contexts
                ],
            })

    return Handler


def run(pipeline, host: str = "127.0.0.1", port: int = 8000) -> None:
    """启动 HTTP 服务（阻塞）。"""
    server = ThreadingHTTPServer((host, port), _make_handler(pipeline))
    server.serve_forever()


def create_app(pipeline):
    """返回 (handler_class, runnable) 供测试拉起独立端口。"""
    return _make_handler(pipeline)
