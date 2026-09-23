# NovaAI Stack

> 模块化检索增强推理（RAG + ReAct Agent）系统 —— 默认零依赖，一键可复现。

- **作者**：晨星
- **版本**：1.0.0
- **许可证**：MIT
- **语言**：Python 3.10+（核心零第三方依赖）

NovaAI 是一个遵循单一职责、接口驱动、可独立验证、可协同运行的模块化 AI 系统。
它**优先复用业界领先的开源 AI 成果**（RAG 范式、FAISS、Ollama、fastembed、BGE 重排、ReAct 范式），
同时提供**零依赖的真实算法实现**作为默认路径，保证在干净环境中无需联网、无需 API Key、无需数据库即可一键运行与复现。

---

## 特性

- **模块化**：ingest / embed / vectorstore / lexical / rerank / llm / agent / pipeline / api / cli 十个职责清晰的模块。
- **接口驱动**：每个能力定义为 `Protocol`，运行时依赖注入具体实现；默认注入零依赖实现，可无缝替换为生产适配器。
- **可独立验证**：每个模块可单独单测、可用 fake 替换；`pytest`/`unittest` 与 E2E 在无网络下全绿。
- **可复现**：核心路径仅依赖 Python 标准库，干净环境 `git clone && python scripts/verify.py` 全绿。
- **生产就绪**：通过环境变量切换 Ollama 本地大模型、OpenAI 兼容接口、fastembed 向量化、FAISS 索引、BGE 跨编码器重排。

---

## 系统架构

```
                 ┌────────────┐
   用户问题 ───► │   CLI/API   │  (cli.py / api/server.py)
                 └─────┬──────┘
                       ▼
                 ┌────────────┐
                 │   Agent    │  ReActAgent：确定性意图路由 (calculator / date / search)
                 └─────┬──────┘
            ┌──────────┼──────────┐
            ▼          ▼          ▼
      ┌──────────┐ ┌────────┐ ┌──────────┐
      │ calculator│ │  date  │ │  search  │
      └──────────┘ └────────┘ └────┬─────┘
                                    ▼
                          ┌──────────────────┐
                          │     Pipeline      │  组合根 (依赖注入)
                          └──┬───┬───┬───┬───┘
            ┌────────────────┘   │   └────────────────┐
            ▼                    ▼                    ▼
      ┌──────────┐        ┌────────────┐       ┌────────────┐
      │  lexical │        │ vectorstore │       │   rerank   │  LexicalReranker(RRF)
      │  BM25    │        │  Memory /  │       └─────┬──────┘
      └────┬─────┘        │  FAISS     │             │
           │              └─────┬──────┘             │
           │     ┌──────────────┼──────────────┐     │
           │     ▼              ▼              ▼     │
           │  ┌──────┐    ┌──────────┐   ┌─────────┐ │
           └─►│ 合并 │◄───┤  embed   │   │         │─┘
               └──────┘    │ Hash/   │   │  LLM    │
                           │ fastembed│  │ Mock/   │
                           └──────────┘  │ Ollama/ │
                                         │ OpenAI  │
                                         └─────────┘
           摄入侧：DocumentLoader → SemanticChunker(标题继承) → embed → 写入 vectorstore + lexical
```

---

## 模块清单

| 模块 | 职责 | 默认实现（零依赖） | 生产适配器（开源复用） |
|------|------|-------------------|------------------------|
| `ingest` | 文档加载 + 语义分块（标题继承） | `LocalTextLoader` / `SemanticChunker` | `PdfLoader`（pypdf） |
| `embed` | 文本向量化 | `HashEmbedder`（BLAKE2b + 中文 bigram） | `FastEmbedEmbedder`（Qdrant fastembed） |
| `vectorstore` | 稠密向量索引 | `MemoryVectorStore`（精确余弦） | `FaissVectorStore`（Meta FAISS） |
| `lexical` | 稀疏检索 | `Bm25Retriever`（Robertson IDF） | — |
| `rerank` | 多路融合重排 | `LexicalReranker`（RRF） | `CrossEncoderReranker`（BGE reranker） |
| `llm` | 文本生成 | `MockLLM`（抽取式阅读理解） | `OllamaLLM` / `OpenAILLM` |
| `agent` | 推理编排 | `ReActAgent`（确定性意图路由） | 同左（接入 LLM 即真实 ReAct） |
| `pipeline` | 组合根（依赖注入） | `Pipeline` / `build_pipeline` | 同左 |
| `api` | HTTP 服务 | 标准库 `http.server` | FastAPI（可选，见 SPEC） |
| `cli` | 命令行 | `novaai.cli` | 同左 |

---

## 快速开始

```bash
# 1. 克隆
git clone https://github.com/CJX0712/<repo>.git
cd <repo>

# 2. 运行自检（离线、无 Key、无数据库，全绿即通过）
python scripts/verify.py

# 3. 内置示例演示
python -m novaai.cli demo

# 4. 启动服务
python -m novaai.cli serve --port 8000
# 浏览器/客户端访问：GET /health  POST /ingest  POST /query
```

无第三方依赖即可完成上述全部步骤。核心仅使用 Python 标准库。

---

## 部署指南

### 默认（零依赖）
```bash
python -m novaai.cli serve --host 127.0.0.1 --port 8000
```
生产反向代理（nginx / caddy）将 `/` 转发到该端口即可。

### 生产后端切换（环境变量）
```bash
# 使用本地 Ollama 大模型 + fastembed 向量化 + FAISS 索引 + BGE 重排
export NOVAAI_LLM=ollama
export NOVAAI_BACKEND=full
export NOVAAI_EMBED_MODEL=BAAI/bge-small-zh-v1.5
pip install fastembed faiss-cpu flagembedding
python -m novaai.cli serve --port 8000
```
- `NOVAAI_LLM`：`mock`（默认）| `ollama` | `openai`
- `NOVAAI_BACKEND`：`default`（默认）| `fastembed` | `faiss` | `full`
- `NOVAAI_OPENAI_KEY`：OpenAI 兼容接口密钥（可选）

---

## 接口说明

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查，返回 `{status, service, docs}` |
| POST | `/ingest` | 摄入文档，body：`{path}` 或 `{doc_id,title,text}` |
| POST | `/query` | 问答，body：`{query}`；返回 `{answer, grounded, tool_calls, trace, contexts}` |

`contexts` 含召回证据块（标题、得分、文本摘要），`grounded` 标记答案是否由检索上下文支撑。

完整契约见 [`docs/openapi.yaml`](docs/openapi.yaml)。

---

## 测试与自检

```bash
python scripts/verify.py     # 八阶段：P0扫描→导入→单测→HTTP E2E→评分卡→运行时不变式→确定性
python scripts/scan_emoji.py # P0 字符门禁（emoji 不得作为功能图标）
```
自检报告写入 `scripts/verify_report.json`。

---

## 文档索引

- [ARCHITECTURE.md](ARCHITECTURE.md) — 架构、接口契约、失败模式与守护测试
- [docs/SPEC.md](docs/SPEC.md) — 系统规格即契约（模块/API/页面/Token/验收/端到端验证）
- [docs/openapi.yaml](docs/openapi.yaml) — HTTP 接口契约
- [docs/decisions/](docs/decisions/) — 架构决策记录（ADR）

---

## 许可证

MIT —— 作者：晨星。详见 [LICENSE](LICENSE)。
