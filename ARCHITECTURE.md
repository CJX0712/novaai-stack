# NovaAI Stack — 架构文档

- **作者**：晨星
- **版本**：1.0.0

本文档说明 NovaAI 的系统架构、接口契约与设计取舍，并列出已踩过的坑与对应守护测试，使"踩过的坑"变成"不会再犯"。

---

## 1. 设计原则

1. **单一职责**：每个模块只解决一个问题（加载、分块、向量化、索引、稀疏检索、重排、生成、编排）。
2. **接口驱动**：所有外部能力定义为 `Protocol`（`src/novaai/ports.py`），运行时依赖注入实现。
3. **默认即可运行**：默认实现全部为零依赖的真实算法，保证干净环境一键复现。
4. **生产可切换**：开源成果以适配器形式存在，环境变量切换，不污染默认路径。

---

## 2. 组合根与依赖注入

`Pipeline`（`src/novaai/pipeline/pipeline.py`）是组合根，构造函数接收所有组件：
`embedder / vectorstore / lexical / reranker / llm / chunker`。
`build_pipeline()` 与 `build_pipeline_from_env()` 是工厂，按参数/环境变量装配。

注入关系：
```
Pipeline
 ├─ ingest_document(doc) ──► Chunker → Embedder → VectorStore.add + Lexical.index
 ├─ retrieve(query)      ──► Embedder.embed_one → VectorStore.search(dense)
 │                          + Lexical.search(sparse) → Reranker.rerank
 └─ ask(query)           ──► Agent.run → (calculator|date|search) → LLM.generate
```

---

## 3. 接口契约（Protocol）

| 接口 | 方法 | 契约要点 |
|------|------|----------|
| `DocumentLoader` | `load(path)` | 返回 `Document` |
| `Chunker` | `chunk(doc)` | 返回 `List[Chunk]`，含标题继承 `heading_path` |
| `Embedder` | `embed(texts)` / `embed_one(text)` | 固定 `dim`，输出 L2 归一化向量 |
| `VectorStore` | `add` / `search` / `drop_document` | 稠密余弦召回；按 `chunk_id` 去重 |
| `LexicalRetriever` | `index` / `search` | Robertson IDF BM25；IDF 恒非负 |
| `Reranker` | `rerank` | RRF 融合；并列用 `chunk_id` 确定性 tie-break |
| `LLM` | `generate(prompt, context)` | 抽取式基线返回上下文最高重叠句 |
| `Agent` | `run(query)` | 确定性意图路由，返回 `QueryResult` |

---

## 4. 失败模式与守护测试

| 症状 | 根因 | 修法 | 守护测试 |
|------|------|------|----------|
| 评分卡 grounding 偏低（答案漏关键词） | `MockLLM` 句子切分把版本号 `3.10` 中的英文句号当句末，截断句子 | 句子切分正则仅当句号后接空白/结尾/中文才切分 | `test_llm` 抽取式答案含上下文关键词 |
| 首个分块标题继承错误 | 标题路径取 flush 时刻而非 chunk 起始时刻 | 缓冲起始即快照 `buf_path`；遇标题先 flush 旧缓冲 | `test_ingest::test_heading_inheritance` |
| 标题行后的内容被整块丢弃 | 标题与同行内容被判为同一"标题段落" | 标题行单独成块，后续内容另成块并继承标题 | `test_ingest::test_multi_chunk` |
| 小语料 BM25 排序反转 | rank_bm25 的 IDF 在词恰好出现在半数文档时 = ln(1)=0 | 改用 Robertson IDF `ln(1+(N-n+0.5)/(n+0.5))`，恒非负 | `test_lexical::test_idf_positive_small_corpus` |
| 全角算式无法计算 | 全角数字/运算符未归一化 | `_FW_MAP` 覆盖全角数字 ０-９ 与运算符，`_OP_CHARS` 不含运算符字符 | `test_agent::test_calc_fullwidth` |
| 含中文的问句抽不出算式 | `detect_arithmetic` 对整句 `ast.parse` 失败 | 在问句中抽取"数字+运算符"连续片段再校验 | `test_agent::test_detect_arithmetic` |
| 重摄入更短文档后旧分块残留 | upsert 语义未落实 | `ingest_document` 先 `drop_document` 再写入，并按文档重建 lexical 索引 | `test_pipeline::test_reingest_shorter_replaces` |
| 默认路径强依赖 numpy | `faiss_store` 顶层 `import numpy` | numpy 改为方法内延迟导入；`FaissVectorStore` 仅在 `use_faiss` 时导入 | `verify::模块导入` 默认路径不引 numpy |

---

## 5. 复现性约定

- 核心运行仅需 Python 3.10+ 标准库，`requirements.txt` 顶层为空（零第三方依赖）。
- 生产依赖集中在 `requirements-dev.txt`，按需安装；锁版清单见 `requirements.lock.txt`。
- `scripts/verify.py` 在裸 CPython 上即可全绿，作为 CI 的可复现证据。

---

## 6. 未来扩展

- 接入真实 ReAct：将 `ReActAgent` 的确定性路由替换为 LLM 驱动的 Thought/Action 循环（已预留 `LLM` 端口）。
- 多租户与持久化：将 `MemoryVectorStore` / `Bm25Retriever` 替换为 Postgres + Qdrant 适配器。
- 评估基准：将 `verify.py` 的评分卡扩展为可配置的检索/生成评测集。
