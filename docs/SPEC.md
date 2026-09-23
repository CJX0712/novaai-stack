# Spec — NovaAI Stack v1.0.0

> 生成日期：2026-09-24
> 基于：需求澄清（用户：构建世界顶级、可复现、可发布的模块化 AI 系统）
> 状态：已确认（自主推进实现）

---

## 1. 产品定义

- **一句话描述**：一个模块化、接口驱动、零依赖可复现的检索增强推理（RAG + ReAct Agent）系统。
- **目标用户**：需要在本地/私有环境快速搭建可运行 AI 检索问答能力的开发者与团队。
- **核心问题**：现有 AI 系统要么强依赖重型栈难以复现，要么从零自研成本高；NovaAI 用"接口 + 默认零依赖真实实现 + 开源适配器"兼顾二者。

---

## 2. MVP 范围（锁定）

| 优先级 | 功能 | 验收标准摘要 |
|--------|------|--------------|
| P0 | 文档摄入与语义分块（标题继承） | 标题行独立成块，内容继承所属标题路径 |
| P0 | 稠密 + 稀疏双通道检索 + RRF 重排 | 两路结果融合，返回 top-k |
| P0 | 抽取式生成 + 确定性意图路由 | 算术/日期走工具，其余走检索增强生成 |
| P0 | HTTP 服务（/health /ingest /query） | 离线、零依赖可启动并应答 |
| P0 | 八阶段自检 + 一键复现 | `verify.py` 离线全绿 |
| P1 | 生产适配器（Ollama/fastembed/FAISS/BGE） | 环境变量切换，无需改业务代码 |

---

## 3. 明确不做（Out-of-Scope）

| 不做的功能 | 原因 | 何时考虑 |
|------------|------|----------|
| 多模态（图片/音视频）检索 | MVP 阶段 ROI 不足 | v2.0 用户反馈后 |
| 用户认证与多租户持久化 | 默认零依赖目标，避免引入数据库 | 生产部署需要时 |
| 前端 Web UI | 后端系统优先，CLI/API 已覆盖交互 | 有 UI 需求时 |

---

## 4. 技术架构（锁定）

| 层 | 技术 | 默认版本 | 锁定原因 |
|----|------|----------|----------|
| 语言 | Python | 3.10+ | 标准库即够，保证复现 |
| 向量化 | HashEmbedder（默认）/ fastembed | 零依赖 / BAAI/bge-small-zh-v1.5 | 默认可复现，生产可升级 |
| 稠密索引 | MemoryVectorStore（默认）/ FAISS | 零依赖 / faiss-cpu | 同上 |
| 稀疏检索 | BM25（Robertson IDF） | 零依赖 | 恒非负 IDF，小语料稳定 |
| 重排 | LexicalReranker（默认）/ BGE | 零依赖 / bge-reranker-base | 默认 RRF，生产 cross-encoder |
| 生成 | MockLLM（默认）/ Ollama / OpenAI | 零依赖 / qwen2.5 / gpt-4o-mini | 默认抽取式基线，生产接 LLM |
| 服务 | 标准库 http.server（默认）/ FastAPI | 零依赖 / FastAPI | 默认零依赖，生产可换 |
| 测试 | unittest + 自研 verify | 标准库 | 无第三方依赖 |

---

## 5. API 端点清单（锁定）

| Method | Path | 功能 | 认证 | 请求体 | 响应体 |
|--------|------|------|------|--------|--------|
| GET | `/health` | 健康检查 | 无 | — | `{status, service, docs}` |
| POST | `/ingest` | 摄入文档 | 无 | `{path}` 或 `{doc_id,title,text}` | `{doc_id, chunk_count, status}` |
| POST | `/query` | 问答 | 无 | `{query}` | `{query, answer, grounded, tool_calls, trace, contexts}` |

完整契约见 [openapi.yaml](openapi.yaml)。

---

## 6. 页面清单

本系统为后端/CLI 系统，无 Web 页面。交互入口为：
- CLI：`python -m novaai.cli {demo|serve|ingest|query}`
- HTTP API：见第 5 节

---

## 7. 设计 Token

后端系统，无 UI 设计 Token。命名与日志风格统一：
- 日志/输出：中文为主，关键数据保留英文（如模型名、指标名）。
- 状态标记：使用文本而非 emoji（遵循 P0 绝对规则）。

---

## 8. 验收标准（EARS）

| 编号 | 功能 | EARS 格式验收标准 | 优先级 |
|------|------|-------------------|--------|
| AC-01 | 摄入 | While 用户摄入合法文档，系统必须返回 chunk_count≥1 的 IngestResult | P0 |
| AC-02 | 检索 | When 用户提问，系统必须在 top-k 内返回相关块且 doc_hit_rate≥1.0（评测集） | P0 |
| AC-03 | 生成 | When 问题可由上下文回答，系统必须返回 grounded=true 的答案 | P0 |
| AC-04 | 算术 | If 提问含算术表达式，系统必须路由 calculator 并返回正确结果 | P0 |
| AC-05 | 日期 | If 提问含日期意图，系统必须路由 date 并返回人话时间 | P0 |
| AC-06 | 去重 | When 重复摄入更短文档，系统必须使该文档分块数降为 1 | P0 |
| AC-07 | 服务 | While 服务运行，GET /health 必须返回 200 | P0 |

---

## 9. 边界与约束

- 仅支持 Python 3.10+；不支持 IE（无前端）。
- 默认路径不依赖网络、API Key、数据库。
- 中文按"字 + 二元字组"切分；英文按词切分。
- 分块默认 800 字符、重叠 160、最小 80（可在 `SemanticChunker` 构造调整）。

---

## 10. 内嵌已知坑

| 坑 | 技术栈指纹 | 根因 | 修法 |
|----|------------|------|------|
| 版本号句号被当句末 | MockLLM | `3.10` 中的 `.` 误判 | 句号后接空白/中文才切分 |
| 标题继承错位 | chunker | 路径取 flush 时刻 | 起始快照 buf_path |
| 小语料 BM25 反转 | lexical | IDF 可为负 | Robertson IDF 恒非负 |
| 全角算式失败 | agent.tools | 未归一化全角 | `_FW_MAP` 覆盖全角数字 |

---

## 11. 端到端验证步骤

```bash
# 1. 模块导入
python -c "import novaai"

# 2. 八阶段自检（离线全绿即证明系统可运行）
python scripts/verify.py

# 3. 核心成功流
python -m novaai.cli demo
# 断言：输出含"检索通道"且 grounded=true

# 4. 关键错误流（无上下文）
python -m novaai.cli query "与知识库无关的问题"
# 断言：返回"未检索到相关上下文"之类可读提示，不崩溃
```

---

## 12. 变更记录

| 日期 | 变更内容 | 原因 | 影响范围 |
|------|----------|------|----------|
| 2026-09-24 | v1.0.0 初始交付 | 用户需求：可复现模块化 AI 系统 | 全系统 |

---

## 13. ADR 索引

- [ADR-001 架构形态：接口驱动 + 默认零依赖](decisions/ADR-001-architecture.md)
- [ADR-002 默认实现策略：真实算法而非 stub](decisions/ADR-002-default-implementations.md)
- [ADR-003 API 层：标准库优先](decisions/ADR-003-api-layer.md)
