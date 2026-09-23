# ADR-001: 架构形态 — 接口驱动 + 默认零依赖

- **状态**：Accepted（2026-09-24）
- **作者**：晨星

## Background
用户要求"世界顶级、可复现、可发布的 AI 系统"，且"优先整合复用开源成果、避免从零自研"。

## Decision
采用接口驱动（Protocol + 依赖注入）的模块化架构：每个能力定义 `Protocol`，
运行时注入实现。默认注入零依赖的真实算法实现；开源成果（FAISS / Ollama / fastembed / BGE）
以适配器形式存在，通过环境变量切换。

## Consequences
- 正面：默认路径在干净 Python 环境一键复现，CI/verify 离线全绿；生产可平滑升级。
- 负面：默认实现（哈希嵌入、抽取式 LLM）精度低于生产模型，仅作占位与验证。
- 约束：所有生产适配器必须延迟导入其重依赖（numpy / faiss / flagembedding），不污染默认路径。
