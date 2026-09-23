# ADR-003: API 层 — 标准库优先

- **状态**：Accepted（2026-09-24）
- **作者**：晨星

## Background
用户要求"干净环境一键复现"。引入 FastAPI/uvicorn 会增加依赖与安装风险（镜像源劫持、构建脚本受限等）。

## Decision
默认 API 层使用 Python 标准库 `http.server`（零依赖），暴露 `/health` `/ingest` `/query`。
同时提供 FastAPI 适配器说明（见 README 部署指南）作为生产可选路径，不进入默认依赖。

## Consequences
- 正面：默认 `python -m novaai.cli serve` 任何 Python 3.10+ 即可运行，无需安装任何包。
- 负面：标准库实现无自动 OpenAPI 文档；已用手写 `docs/openapi.yaml` 弥补契约。
- 约束：若改用 FastAPI，须保证三个端点语义与本文档契约一致。
