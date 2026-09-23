"""NovaAI Stack — 文本生成生产适配器（llm）。

作者：晨星
OpenAILLM：兼容 OpenAI / 任意 OpenAI-compatible 网关（如 vLLM、LocalAI）。
使用标准库 urllib；API Key 取自环境变量 NOVAAI_OPENAI_KEY。未配置时返回可读错误。
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import List


class OpenAILLM:
    """OpenAI 兼容推理适配（可选：需配置 API Key 与 base_url）。"""

    def __init__(self, model: str = "gpt-4o-mini", base_url: str = "https://api.openai.com/v1") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, context: str = "") -> str:
        key = os.environ.get("NOVAAI_OPENAI_KEY", "")
        if not key:
            return "（未配置 NOVAAI_OPENAI_KEY，无法调用 OpenAI 兼容接口）"
        messages: List[dict] = []
        if context:
            messages.append({"role": "system", "content": "参考上下文：\n" + context})
        messages.append({"role": "user", "content": prompt})
        payload = json.dumps({"model": self.model, "messages": messages, "temperature": 0.0}).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # pragma: no cover - 取决于网关可用性
            return f"（OpenAI 兼容接口调用失败：{exc}）"
