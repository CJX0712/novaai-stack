import json
import os
import sys
import threading
import time
import unittest
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from novaai.api.server import create_app
from novaai.pipeline.pipeline import build_pipeline
from novaai.types import Document

# 客户端使用无代理 opener，避免本机流量被系统 SOCKS/HTTP 代理劫持
_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _http(method, url, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with _NO_PROXY.open(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


class TestApiE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pipe = build_pipeline()
        for d in [
            Document("a", "架构", "# 架构\nNovaAI 支持稠密向量召回与稀疏 BM25 召回两条通道。\n"),
            Document("b", "部署", "# 部署\nNovaAI 仅需 Python 3.10 以上即可运行服务。\n"),
        ]:
            pipe.ingest_document(d)
        cls.pipe = pipe
        cls.server = create_app(pipe)
        import socketserver
        from http.server import ThreadingHTTPServer
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), cls.server)
        cls.port = httpd.server_address[1]
        cls.httpd = httpd
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        # 就绪轮询
        for _ in range(50):
            try:
                _http("GET", f"http://127.0.0.1:{cls.port}/health")
                break
            except Exception:
                time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def test_health(self):
        status, body = _http("GET", f"http://127.0.0.1:{self.port}/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")

    def test_ingest(self):
        status, body = _http("POST", f"http://127.0.0.1:{self.port}/ingest",
                             {"doc_id": "api1", "title": "测试", "text": "人工智能 改变 世界。深度学习 推动 进步。"})
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")

    def test_query(self):
        _http("POST", f"http://127.0.0.1:{self.port}/ingest",
              {"doc_id": "api2", "title": "召回", "text": "NovaAI 提供稠密向量召回与稀疏 BM25 召回。"})
        status, body = _http("POST", f"http://127.0.0.1:{self.port}/query",
                             {"query": "NovaAI 提供哪些召回方式？"})
        self.assertEqual(status, 200)
        self.assertIn("召回", body["answer"])
        self.assertTrue(body["contexts"])


if __name__ == "__main__":
    unittest.main()
