"""NovaAI Stack — 文本生成生产适配器（llm）。

作者：晨星
OllamaLLM：复用开源 Ollama 本地大模型运行时（零 GPU 也可跑量化模型）。
使用标准库 urllib 直连 127.0.0.1，避免系统代理劫持本机流量。未启动 Ollama 时返回可读错误。
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Optional


class OllamaLLM:
    """Ollama 本地推理适配（可选依赖：本地需运行 ollama serve）。"""

    def __init__(self, model: str = "qwen2.5:7b", base_url: str = "http://127.0.0.1:11434") -> None:
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, context: str = "") -> str:
        full = (context + "\n\n" + prompt) if context else prompt
        payload = json.dumps({"model": self.model, "prompt": full, "stream": False}).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        opener = urllib.request.build_opener(urllib.request.HTTPHandler())
        try:
            with opener.open(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return (data.get("response") or "").strip()
        except Exception as exc:  # pragma: no cover - 取决于本地 Ollama
            return f"（Ollama 调用失败：{exc}）"
