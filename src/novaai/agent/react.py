"""NovaAI Stack — 推理编排模块（agent）。

作者：晨星
ReActAgent：基于 ReAct（推理-行动）范式构建的编排器。
默认走「确定性意图路由」以保证离线可复现：算术→calculator，时间→date，其余→检索+生成；
生产接入 Ollama / OpenAI 后即为真正的 LLM 驱动 ReAct 循环。
全程记录 tool_calls 与 trace，保证链路可观测、可独立验证。
"""

from __future__ import annotations

from typing import Callable, List, Optional

from ..llm.mock_llm import _CTX_CLOSE, _CTX_OPEN, _Q_MARK
from ..types import QueryResult, RetrievedChunk
from .tools import calc, detect_arithmetic, detect_date

_FALLBACK = "（未检索到相关上下文，无法回答该问题）"


class ReActAgent:
    """检索增强推理编排器。"""

    def __init__(
        self,
        retriever: Callable[[str, int], List[RetrievedChunk]],
        llm,
        retrieve_top_k: int = 6,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._top_k = retrieve_top_k

    def run(self, query: str) -> QueryResult:
        trace: List[str] = []
        tool_calls: List[dict] = []

        arith = detect_arithmetic(query)
        if arith is not None:
            try:
                result = calc(arith)
                tool_calls.append({"tool": "calculator", "input": arith, "output": result})
                trace.append(f"路由→calculator：{arith} = {result}")
                return QueryResult(query=query, answer=result, tool_calls=tool_calls, trace=trace, grounded=False)
            except ValueError as exc:
                trace.append(f"calculator 失败：{exc}")
                return QueryResult(query=query, answer=f"无法计算：{exc}", tool_calls=tool_calls, trace=trace, grounded=False)

        date_ans = detect_date(query)
        if date_ans is not None:
            tool_calls.append({"tool": "date", "input": query, "output": date_ans})
            trace.append("路由→date")
            return QueryResult(query=query, answer=date_ans, tool_calls=tool_calls, trace=trace, grounded=False)

        contexts = self._retriever(query, self._top_k)
        tool_calls.append({"tool": "search", "input": query, "output": f"{len(contexts)} chunks"})
        trace.append(f"路由→search：召回 {len(contexts)} 块")
        if not contexts:
            answer = self._llm.generate(query, context="")
            return QueryResult(query=query, answer=answer, contexts=contexts, tool_calls=tool_calls, trace=trace, grounded=False)

        block_lines = []
        for i, rc in enumerate(contexts):
            flat = rc.chunk.text.replace("\n", " ")
            block_lines.append(f"[id#{i}] {rc.chunk.title}：{flat}")
        context_block = "\n".join(block_lines)
        prompt = (
            "你是企业知识库助手。请仅依据以下上下文回答用户问题；"
            "若上下文不含答案，请如实说明。\n"
            f"{_CTX_OPEN}\n{context_block}\n{_CTX_CLOSE}\n"
            f"{_Q_MARK}{query}"
        )
        answer = self._llm.generate(prompt, context=context_block)
        grounded = bool(answer) and answer != _FALLBACK
        trace.append("路由→llm.generate（抽取式生成）")
        return QueryResult(query=query, answer=answer, contexts=contexts, tool_calls=tool_calls, trace=trace, grounded=grounded)
