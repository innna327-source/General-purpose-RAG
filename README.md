# White-Box RAG

一个面向真实落地场景设计的**通用可扩展 RAG 框架**。

区别于大多数 RAG 教程和工具包，本项目的核心目标不是"能跑通"，而是让检索过程完全**可观测、可调试、可扩展**。每一步数据流都有结构化日志，每一个 chunk 的 BM25 分数和向量分数都在前端实时透视，每一次查询扩展的来源都可追溯。

---

## 核心亮点

### 1. 层级分块 + 语义降级

`chunk/hierarchical_chunk.py` 实现了双策略分块器：

- **一级策略（标题正则）**：识别文档的显式章节结构，切出语义完整的"父块"，再对每个父块做滑动窗口生成子块。适合有明确标题层级的文档（技术文档、年报、合同）。
- **二级策略（语义聚类，自动降级）**：当正则切出的父块数量过少时（如 PDF 无文字标题的文档），自动对所有段落做 embedding，在余弦相似度的"低谷"处划定主题边界，生成语义化父块。

两种策略无缝切换，无需人工判断文档类型。

### 2. 语义图谱驱动的查询扩展

`graph/semantic_graph.py` 用 spaCy NER 从文档的 chunk 中提取实体，构建**共现关系图**（节点 = 实体，边权重 = 共现次数）。

检索时，系统自动沿图的边展开：

```
用户提问："RLHF的训练成本"
  ↓ 图邻居扩展
  → "强化学习"、"人类反馈"（高共现邻居）
  ↓ 同义词注入（可选）
  → "奖励模型"、"PPO"
  ↓ BM25 使用扩展后的 query 召回，向量检索保持原 query
```

**效果**：对口语化、模糊、缩写的问题有显著召回增益，无需大模型参与扩展。

### 3. 两级幂等去重

- **文件级**：SHA256 哈希，已处理的 PDF 跳过，增量入库。
- **Chunk 级**：基于 embedding 的语义相似度去重（默认阈值 0.85），过滤扫描件中的重复段落、页眉页脚等噪声。

### 4. 混合检索 + Min-Max 归一化融合

BM25 和 FAISS 双路各召回 top-20，再用 Min-Max 归一化后加权融合，避免两路分数量纲不一致导致一路失效：

```
final_score = bm25_weight × norm(bm25) + vector_weight × norm(vector)
```

融合权重在 `config/settings.py` 中一行调整，领域切换无需改代码。

### 5. 全链路白盒可视化

Streamlit 前端（`streamlit_app.py`）实时展示：
- 每个 chunk 的 `bm25_score`、`vector_score`、`final_score` 三列数值
- 查询扩展后的实际 query（可验证图谱扩展效果）
- 逐 chunk 文本内容对比

所有中间结果同步写入 `logs/debug_YYYYMMDD_HHMMSS.jsonl`，可离线复现任意一次检索过程。

### 6. 零基础设施依赖

- **无 GPU**：embedding 和检索全程 CPU 可运行
- **无外部服务**：FAISS 本地文件索引，不依赖向量数据库
- **无训练**：直接使用预训练模型，开箱即用
- **无 Docker**：`pip install` 后即可启动

适合在笔记本上做原型验证，也支持平滑迁移到生产基础设施。

---

## 快速开始

### 环境准备

```bash
# 1. 克隆项目
git clone <repo-url>
cd white-box-rag

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 下载 spaCy 中文模型
python -m spacy download zh_core_web_sm

# 5. 配置 LLM（复制并填写 .env）
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY、LLM_MODEL、LLM_BASE_URL
```

> **不配置 LLM 也能跑**：系统在 `llm_api_key` 为空时自动降级，直接返回检索结果，跳过生成步骤。

### 构建索引

将 PDF 放入 `data/raw_pdf/`，运行：

```bash
python main.py --mode test --file your_document.pdf
```

成功后控制台输出：

```
索引构建完成 file_hash=d1257ca2...
```

记录此 hash，MCP 服务启动时需要。

### 启动可视化前端

```bash
python -m streamlit run streamlit_app.py
```

访问 `http://localhost:8501`，从下拉框选择 query，实时查看双路检索分数和生成结果。

### 批量评估（可选）

在 `tests/test_queries.json` 中编写查询和预期关键词，运行：

```bash
python main.py --mode test --file your_document.pdf --eval
```

评估报告写入 `output/test_report.json`，包含 Recall@5 和 MRR 指标。

### 启动 MCP HTTP 服务（可选）

```bash
python main.py --mode mcp --index-hash <your_file_hash>
```

服务监听 `0.0.0.0:8080`，暴露 OpenAI-compatible 接口：

```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"query": "你的问题"}'
```

---

## 可扩展点

### 替换向量数据库

当前用 FAISS（单机本地文件）。`VectorStore` 是一个三方法接口：

```python
# retrieval/vector_store.py
class VectorStore:
    def add(self, chunk_id: str, text: str) -> None: ...
    def search(self, query: str, top_k: int) -> List[dict]: ...
    def chunk_text_by_id(self) -> Dict[str, str]: ...
```

迁移到 Milvus：新建 `retrieval/milvus_store.py` 实现这三个方法，在 `main.py` 中替换实例化一行，检索逻辑和上层代码零修改。

Qdrant、Chroma、Weaviate 同理，接口契约不变。

### 替换 Embedding 模型

`config/settings.py` 中修改一行：

```python
# 中文大模型，精度更高
embedding_model_name = "BAAI/bge-large-zh-v1.5"

# 英文场景
embedding_model_name = "BAAI/bge-small-en-v1.5"

# 多语言
embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

> 换模型后需重建索引（旧向量维度/空间不兼容）：删除 `index/` 和 `graph/` 下对应 hash 的文件，重跑 `main.py`。

### 替换 LLM

三个环境变量控制，支持任意 OpenAI-compatible 接口：

```bash
# OpenAI
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o
LLM_BASE_URL=https://api.openai.com/v1

# 本地 Ollama（完全离线）
OPENAI_API_KEY=ollama
LLM_MODEL=qwen2.5:7b
LLM_BASE_URL=http://localhost:11434/v1

# DeepSeek
OPENAI_API_KEY=sk-...
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com/v1
```

### 扩展文档类型

`main.py` 的 `_get_loader()` 是文件类型注册表：

```python
def _get_loader(file_path: Path) -> BaseLoader:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return PDFLoader()
    # 新增：
    if suffix == ".docx":
        return DocxLoader()
    if suffix in (".html", ".htm"):
        return HTMLLoader()
    raise ValueError(f"不支持的文件类型：{suffix}")
```

只需继承 `loader/base_loader.py` 的 `BaseLoader`，实现 `load(path) -> str` 方法即可。

### 适配专业领域

**调整检索权重**：专业术语密集场景（金融报告、法律文书）精确匹配更重要，调高 BM25 权重：

```python
# config/settings.py
bm25_weight   = 0.8   # 默认 0.7
vector_weight = 0.2   # 默认 0.3
```

**注入领域同义词表**：提升专业术语的查询扩展召回：

```python
synonym_dict = {
    # 金融
    "净利润": ["归母净利润", "净利", "税后利润"],
    "市值":   ["总市值", "流通市值"],
    # 法律
    "违约":   ["违反合同", "不履行义务", "违反约定"],
    # 医疗
    "高血压": ["血压升高", "血压偏高"],
}
```

**自定义标题正则**：适配非标准章节格式：

```python
import re
title_patterns = [
    re.compile(r"^\s*[一二三四五六七八九十]+[、.]"),   # 一、二、三、
    re.compile(r"^\s*第\s*\d+\s*条"),                  # 第X条（法律文书）
    re.compile(r"^\s*【.+?】"),                         # 【章节名】格式
]
```

**扩展语义图谱术语表**：`graph/semantic_graph.py` 的 `DEFAULT_TECH_TERMS` 决定 NER 之外额外识别哪些领域词汇：

```python
DEFAULT_TECH_TERMS = [
    # 金融
    "市盈率", "净资产收益率", "资产负债率", "EBITDA",
    # 医疗
    "高血压", "糖尿病", "心肌梗死", "靶向治疗",
    # 法律
    "诉讼时效", "管辖权", "不可抗力",
]
```

### 接入 Agent / MCP 生态

MCP 服务遵循 OpenAI-compatible 格式，可直接对接：
- **LangChain**：作为自定义 LLM 包装（`langchain_community.llms.openai.OpenAI(base_url=...)`）
- **自定义 Agent**：工具描述 + `POST /v1/completions` 调用
- **任意支持自定义 endpoint 的客户端**：Claude Desktop、Continue.dev 等

---

## 项目结构

```
├── config/settings.py          # 全局配置（LLM、Embedding、分块参数、路径）
│
├── loader/                     # 文档加载层
│   ├── pdf_loader.py           # PyMuPDF 文本提取 + Tesseract OCR 回退
│   ├── ocr_utils.py            # OCR 工具函数
│   └── base_loader.py          # 抽象基类（扩展新文档类型的入口）
│
├── preprocess/                 # 文本预处理层
│   ├── cleaner.py              # 正则降噪、段落规整
│   ├── deduplicator.py         # 语义相似度去重（embedding-based）
│   └── hash_checker.py         # 文件级 SHA256 幂等校验
│
├── chunk/                      # 分块层
│   ├── hierarchical_chunk.py   # HierarchicalChunker：标题正则 + 语义聚类双策略
│   └── base_chunker.py         # 抽象基类
│
├── graph/                      # 语义图谱层
│   └── semantic_graph.py       # spaCy NER 实体提取 + 共现图构建
│
├── retrieval/                  # 检索层
│   ├── hybrid.py               # 混合检索：图谱扩展 + BM25 + FAISS + 融合排序
│   ├── bm25.py                 # BM25 索引构建、查询
│   └── vector_store.py         # FAISS 向量索引（可替换接口）
│
├── generation/
│   └── llm.py                  # LLM 生成封装（OpenAI-compatible，支持流式）
│
├── mcp/                        # MCP 协议层
│   ├── server.py               # Flask HTTP 服务
│   ├── handler.py              # 请求路由与处理
│   └── protocol.py             # 协议数据结构
│
├── backend/
│   └── server.py               # FastAPI 后端（React 前端对接）
│
├── frontend/                   # React + TypeScript 前端
│
├── streamlit_app.py            # 白盒可视化看板（Streamlit）
├── main.py                     # CLI 入口（--mode test/mcp）
│
├── data/raw_pdf/               # 放置待索引的 PDF（不入 Git）
├── index/                      # 运行时 BM25 + FAISS 索引文件（不入 Git）
├── graph/                      # 运行时语义图谱 JSON（不入 Git）
├── logs/                       # 结构化调试日志 JSONL（不入 Git）
├── output/                     # 评估报告（不入 Git）
└── tests/                      # 评估查询集（JSON 格式）
```

---

## 注意事项

### 索引重建

修改 `chunk_size`、`overlap`、`embedding_model_name` 后，旧索引与新参数不兼容，必须重建：

1. 清空 `data/hash_db.json`（设为 `{}`）
2. 删除 `index/` 下对应 hash 的三个文件（`.faiss`、`.bm25.pkl`、`.chunks.jsonl`）
3. 删除 `graph/` 下对应的 `_semantic_graph.json`
4. 重跑 `python main.py --mode test --file your_document.pdf`

快捷方式：将 PDF 改名（如加 `_v2`），系统视为新文件自动全量重建。

### 性能边界

| 场景 | 建议 |
|------|------|
| 单文档原型验证 | 默认配置即可，FAISS IndexFlatIP 暴力搜索 |
| 文档 > 10 万 chunk | 改用 FAISS IndexIVFFlat 或外部向量库 |
| NER 精度不足 | 换 `zh_core_web_lg` 或接入专业 NER 服务 |
| 多文档并行检索 | 在 `HybridRetriever` 层实现跨索引聚合，或迁移至 Milvus/Qdrant |

### 依赖说明

OCR 功能（扫描版 PDF）需要额外安装 [Tesseract](https://github.com/tesseract-ocr/tesseract)，并配置 `TESSDATA_PREFIX` 环境变量。纯文字 PDF 无需此步骤。
