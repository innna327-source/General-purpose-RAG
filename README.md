# 白盒可视化通用 RAG 框架

一个面向中文文档的本地 RAG 示例框架，覆盖 PDF 加载、文本清洗、层级分块、BM25 + FAISS 混合检索、语义图谱增强、Cross-Encoder 重排、LLM 生成、评估报告、Streamlit 白盒看板和 MCP 风格 HTTP 服务。

这个仓库是公开发布版：不会提交 PDF、索引、日志、模型权重、真实 API key 或运行输出。你可以把自己的 PDF 放到本地目录中运行，生成的运行产物会被 `.gitignore` 排除。

## 核心能力

- PDF 文本抽取：基于 PyMuPDF，必要时可接入 Tesseract OCR。
- 文本预处理：段落清洗、规范化、文件级 hash 去重、语义去重。
- 层级分块：先构建父块，再用滑动窗口生成子块，兼顾上下文完整性和召回精度。
- 混合检索：BM25 与 FAISS 向量检索双路召回，按权重融合排序。
- 图谱增强：用 spaCy 与 LLM 抽取实体关系，支持 JSON 图谱和可选 Neo4j 存储。
- 重排优化：可选 Cross-Encoder rerank，默认模型为 `BAAI/bge-reranker-v2-m3`。
- 生成控制：OpenAI-compatible API，内置“仅基于上下文回答”的防幻觉约束。
- Answerability Gate：生成前检查召回证据是否足够、分数是否可信、是否存在 direct evidence；生成后回查答案 claim 是否被上下文支持。
- 白盒看板：Streamlit 展示分块统计、检索分数、召回链路、生成结果和评估指标。
- MCP HTTP 服务：提供 `POST /v1/completions`，方便接入 Agent 或其他客户端。

## 架构概览

```text
PDF
  -> loader/             PyMuPDF / optional OCR
  -> preprocess/         clean, normalize, deduplicate, hash tracking
  -> chunk/              hierarchical parent chunks + sliding child chunks
  -> graph/              entity graph extraction, JSON or optional Neo4j
  -> retrieval/          BM25 + FAISS + graph recall + rerank
  -> generation/         OpenAI-compatible LLM answer generation + answerability gate
  -> evaluation/         retrieval and generation evaluation reports
  -> streamlit_app.py    white-box dashboard
  -> mcp/                HTTP completion service
```

## 环境准备

建议使用 Python 3.10 或 3.11。

```bash
git clone <your-repo-url>
cd RAG
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements_streamlit.txt
python -m spacy download zh_core_web_sm
```

如果你的环境无法访问 Hugging Face，可以先配置镜像或提前缓存模型。默认配置会使用 `HF_ENDPOINT=https://hf-mirror.com`，也可以手动设置：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

## 配置 LLM

项目读取环境变量，不会自动读取 `.env` 文件。你可以参考 `.env.example`，然后在终端或 IDE 中设置环境变量。

OpenAI 示例：

```powershell
$env:OPENAI_API_KEY="your-api-key-here"
$env:LLM_MODEL="gpt-4o-mini"
$env:GENERATION_LLM_MODEL="gpt-4o-mini"
$env:LLM_BASE_URL="https://api.openai.com/v1"
```

Ollama / 本地 OpenAI-compatible 服务示例：

```powershell
$env:OPENAI_API_KEY="ollama"
$env:LLM_MODEL="qwen2.5:7b"
$env:GENERATION_LLM_MODEL="qwen2.5:7b"
$env:LLM_BASE_URL="http://localhost:11434/v1"
```

如果不设置 `OPENAI_API_KEY`，生成阶段会降级，不会调用外部 LLM。

常用可选配置：

```powershell
$env:RAG_USE_RERANK="true"
$env:RAG_RERANK_MODEL="BAAI/bge-reranker-v2-m3"
$env:RAG_GRAPH_STORE="json"
```

如需使用 Neo4j：

```powershell
$env:RAG_GRAPH_STORE="neo4j"
$env:NEO4J_URI="bolt://localhost:7687"
$env:NEO4J_USER="neo4j"
$env:NEO4J_PASSWORD="change-me"
$env:NEO4J_DATABASE="neo4j"
```

## 快速开始

1. 放入 PDF

把你的 PDF 放到：

```text
data/raw_pdf/your.pdf
```

2. 构建索引并运行测试查询

```bash
python main.py --mode test --file your.pdf
```

运行成功后，日志中会出现类似：

```text
索引构建完成 file_hash=<hash>
```

请记下这个 `file_hash`，启动 MCP 服务时会用到。

3. 生成评估报告

```bash
python main.py --mode test --file your.pdf --eval
```

输出文件默认写入 `output/test_report.json`。更完整的检索与生成评估可以使用：

```bash
python run_evaluation.py --file your.pdf
```

4. 启动 Streamlit 看板

```bash
python -m streamlit run streamlit_app.py
```

浏览器打开：

```text
http://localhost:8501
```

看板会展示分块统计、测试报告、Top-K 召回结果、BM25/向量/融合分数、rerank 状态和 LLM 生成答案。

## Answerability Gate

项目在 LLM 生成前后增加了轻量规则 gate，用来降低“召回结果看起来相关，但其实证据不足仍然硬答”的风险。

当前实现分为四层：

1. Retrieval Sufficiency Gate：判断有没有基本证据。检查 TopK 是否为空、上下文长度是否过短、query 关键词覆盖率是否低、是否只有 related_context 而没有 direct evidence。不满足时直接返回“根据现有资料无法回答该问题”。

2. Retrieval Confidence Gate：判断证据是否可靠。综合 `rerank_score`、`final_score`、Top1/Top2 分差、direct evidence 占比、related_context 占比、来源/父块集中度，生成 `retrieval_confidence`。置信度过低时拒答。

3. Direct Evidence Gate：判断召回片段是否覆盖问题里的关键对象和关键维度。比如比较类问题会检查两个比较对象是否同时出现在直接证据窗口里；成本、风险、指标、年份等关键维度缺失时，普通问答默认继续生成但加提示。

4. Answer Support Gate：生成后检查答案是否越界。基础版本做答案关键词覆盖率检查；金融/高风险场景会进一步抽取公司名、指标、年份、金额、百分比和风险项，回查这些 claim 是否出现在召回上下文中。

普通问答默认策略是 `warn`：当召回内容与问题主题相关但关键维度不完整时，允许生成，但会在答案前提示“仅基于现有片段，可能不完整”。高风险场景可以通过 `high_risk=True` 或配置切到更严格策略，证据不完整时直接拒答。`enable_high_risk_llm_judge` 目前保留为报告生成等高价值输出的兜底开关，默认不走 LLM judge，避免把所有请求都变成高成本路径。

主要配置项在 `config/settings.py`：

```python
enable_answerability_gate = True
answerability_min_context_chars = 80
answerability_min_retrieval_confidence = 0.35
enable_direct_evidence_answerability_gate = True
answerability_uncertain_normal_policy = "warn"  # warn / reject
enable_high_risk_llm_judge = False
```

## MCP HTTP 服务

先确保目标 PDF 已构建索引，然后启动服务：

```bash
python main.py --mode mcp --index-hash <file_hash>
```

默认监听 `0.0.0.0:8080`。请求示例：

```bash
curl -X POST http://localhost:8080/v1/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"你的问题\", \"top_k\": 5, \"max_tokens\": 600}"
```

响应格式：

```json
{
  "id": "mcp-rag-...",
  "model": "deep_rag",
  "object": "text_completion",
  "created": 1710000000,
  "choices": [
    {
      "index": 0,
      "text": "回答内容",
      "context": ["chunk_id_1", "chunk_id_2"]
    }
  ]
}
```

## 测试查询与评估数据

测试集位于 `tests/`。推荐使用 dict 格式：

```json
[
  {
    "query": "这里写问题",
    "expected_keywords": ["关键词1", "关键词2"],
    "has_answer_in_doc": true
  }
]
```

评估会输出 Recall@5、MRR、命中详情，以及有答案问题的召回率。生成评估会进一步检查答案是否忠实于检索上下文。

## 项目结构

```text
config/              全局配置：模型、路径、检索权重、MCP、Neo4j
loader/              PDF 加载与 OCR 辅助
preprocess/          清洗、去重、hash/version 记录
chunk/               层级分块与 chunk 元数据
graph/               语义图谱构建、JSON/Neo4j 存取
retrieval/           BM25、FAISS、混合检索、rerank、retriever factory
generation/          LLM 调用、防幻觉提示、答案门控
evaluation/          检索评估与生成评估
mcp/                 Flask HTTP 服务与请求/响应协议
scripts/             调参、图谱重建、Neo4j 迁移、流程图生成脚本
tests/               测试查询样例
frontend/ backend/   预留的前后端版本结构
main.py              CLI 入口：test / mcp
streamlit_app.py     可视化看板
run_evaluation.py    综合评估入口
```

## 运行产物与开源安全

以下内容默认不会提交：

- `data/raw_pdf/*.pdf`
- `data/hash_db.json`
- `data/version_log.json`
- `index/*.faiss`
- `index/*.bm25.pkl`
- `index/*.chunks.jsonl`
- `graph/*_semantic_graph.json`
- `logs/*.log`、`logs/*.jsonl`
- `output/*`
- `models/`
- `.env`、`.env.*`，但保留 `.env.example`

发布前建议再跑一次：

```bash
git status --short
git ls-files | rg "(\.pdf$|^logs/|^index/|^models/|_semantic_graph\.json$)"
rg -n "sk-[A-Za-z0-9_-]{12,}|AKIA[0-9A-Z]{16}|password|secret|token" -g "!**/.git/**"
```

## 常见问题

### 找不到 `zh_core_web_sm`

运行：

```bash
python -m spacy download zh_core_web_sm
```

### 首次运行很慢

首次运行会下载 embedding、tokenizer、reranker 等模型。网络慢时可以配置 `HF_ENDPOINT`，或提前下载模型后设置：

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
```

### 不想启用 rerank

```powershell
$env:RAG_USE_RERANK="false"
```

### 修改分块或 embedding 后结果不更新

修改 `chunk_size`、`overlap`、`embedding_model_name`、rerank/图谱相关配置后，建议删除旧运行产物并重建：

```text
data/hash_db.json
index/<hash>.*
graph/<hash>_semantic_graph.json
output/*.json
```

然后重新运行：

```bash
python main.py --mode test --file your.pdf
```

## 适合的使用场景

- 学习和演示 RAG 全链路。
- 分析单篇或少量中文专业 PDF。
- 对比 BM25、向量检索、图谱召回、重排和生成效果。
- 作为更大知识库系统的原型骨架。

如果要用于生产级多文档知识库，建议进一步引入集合级索引管理、权限控制、异步任务队列、外部向量数据库、结构化监控和更完整的测试集。
