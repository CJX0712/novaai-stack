# ADR-002: 默认实现策略 — 真实算法而非 stub

- **状态**：Accepted（2026-09-24）
- **作者**：晨星

## Background
验证类系统常见缺陷：默认实现是空 stub，导致"测试全绿但系统从未真跑过"。
用户要求系统"稳定、可部署、可复现"。

## Decision
所有默认实现必须是**真实可用的算法**：哈希嵌入（BLAKE2b + 中文 bigram + L2 归一）、
精确余弦内存索引、Robertson IDF 的 BM25、RRF 融合重排、抽取式阅读理解 LLM、
确定性意图路由的 ReAct Agent。stub 仅在生产适配器缺失时返回可读错误。

## Consequences
- 正面：`git clone && python scripts/verify.py` 在裸 CPython 上即验证完整链路。
- 负面：默认精度有限，需要生产 LLM/嵌入才能用于真实业务。
- 约束：新增默认实现必须自带可验证的不变量（如余弦∈[-1,1]、IDF≥0、答案由上下文支撑）。
