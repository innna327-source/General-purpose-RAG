# 白盒可视通用 RAG 框架

针对知识库搭建"重复造轮子"问题设计的通用可扩展 RAG 框架，覆盖个人技术知识库至企业级专业知识库全场景。核心亮点：语义图谱驱动的查询扩展 + 层级分块 + 全链路白盒可视化，可无缝适配金融、法律等多领域文档检索需求。

**技术栈：** Python · BM25 · FAISS · spaCy · PyMuPDF · MCP · Streamlit · Flask · OpenAI-compatible API

---

## 架构概览

```
PDF 文档
   ↓
[Loader]        PyMuPDF 提取文本 / Tesseract OCR（可选）
   ↓
[Preprocess]    正则降噪 → 语义去重（0.85阈值）→ 段落规整
                两级幂等去重：文件级 SHA256 + chunk 级语义相似度
   ↓
[Chunk]         HierarchicalChunker：父块（标题边界）→ 子块（滑动窗口）
                build_semantic_graph：spaCy NER 构建实体语义图谱
   ↓
[Index]         BM25 倒排索引（.bm25.pkl）
                FAISS 向量索引（.faiss）+ BAAI/bge-small-zh-v1.5
   ↓
[Retrieval]     查询扩展（图谱实体 + 同义词）→ BM25 + FAISS 双路召回
                Min-Max 归一化加权融合（BM25:0.7 / Vector:0.3）
   ↓
[Generation]    OpenAI-compatible LLM + System Prompt 防幻觉约束
   ↓
[Frontend]      Streamlit 白盒看板：双路分数透视 + 逐 chunk 追踪
[MCP Service]   Flask HTTP 服务（OpenAI-compatible API，可选）
```

---

## 快速启动

### 第 0 步：安装依赖

```bash
# 激活虚拟环境（Windows）
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

# 下载 spaCy 中文模型
python -m spacy download zh_core_web_sm
```

### 第 1 步：配置 LLM

打开 `config/settings.py`，修改以下三个字段：

```python
llm_model   = "your-model-name"          # 模型名称
llm_api_key = "your-api-key"             # 若为空则读 OPENAI_API_KEY 环境变量
llm_base_url = "https://your-api.com/v1" # 只填到 /v1，不加 /chat/completions
```

> **降级策略：** `llm_api_key` 为空且环境变量未设置时，系统自动返回检索原文，不影响检索链路运行。

### 第 2 步：构建索引

将 PDF 放入 `data/raw_pdf/`，然后运行：

```bash
# 国内网络设置 HuggingFace 镜像
$env:HF_ENDPOINT="https://hf-mirror.com"

python main.py --mode test --file your_document.pdf
```

运行成功后控制台输出 `file_hash`，例如：

```
索引构建完成 file_hash=d1257ca2...
```

**请记录此 hash**，启动 MCP 服务时需要。

### 第 3 步：批量评估（可选）

```bash
python main.py --mode test --file your_document.pdf --eval
```

- 评估报告：`output/test_report.json`（Hit Rate@K、MRR 等指标）
- 调试日志：`logs/debug_YYYYMMDD_HHMMSS.jsonl`

### 第 4 步：启动前端

```bash
.\.venv\Scripts\Activate.ps1
python -m streamlit run streamlit_app.py
```

浏览器访问 `http://localhost:8501`，从下拉框选择测试 query，页面实时展示：
- 双路召回分数（bm25_score / vector_score / final_score）
- Top-K 检索结果
- LLM 生成答案

### （可选）启动 MCP HTTP 服务

```bash
python main.py --mode mcp --index-hash <your_file_hash>
```

服务监听 `0.0.0.0:8080`，验证：

```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"你的问题\"}"
```

---

## 可扩展点

### 1. 替换向量数据库

当前使用 FAISS（本地文件，零基础设施依赖）。生产环境可替换为 Milvus、Qdrant 等，只需实现 `retrieval/vector_store.py` 中的接口：

```python
class VectorStore:
    def add(self, chunk_id: str, text: str) -> None: ...
    def search(self, query: str, top_k: int) -> List[dict]: ...
    def chunk_text_by_id(self) -> Dict[str, str]: ...
```

替换步骤：新建 `retrieval/milvus_store.py` 实现上述接口 → 在 `main.py` 中替换实例化调用，无需修改检索逻辑。

### 2. 替换 Embedding 模型

在 `config/settings.py` 修改：

```python
embedding_model_name = "BAAI/bge-large-zh-v1.5"   # 换更大的中文模型
# 或英文场景
embedding_model_name = "BAAI/bge-small-en-v1.5"
```

> 注意：换模型后需删除旧索引重新构建（旧向量与新模型维度/空间不兼容）。

### 3. 替换 LLM

只需修改 `config/settings.py` 中的三个字段，支持任意 OpenAI-compatible 接口：

```python
llm_model    = "gpt-4o"
llm_base_url = "https://api.openai.com/v1"
# 或本地 Ollama
llm_model    = "qwen2.5:7b"
llm_base_url = "http://localhost:11434/v1"
```

### 4. 适配专业领域

**调整检索权重：** 专业术语密集场景（金融报告、法律文书）适当调高 BM25 权重：

```python
# config/settings.py
bm25_weight   = 0.8   # 默认 0.7
vector_weight = 0.2   # 默认 0.3
```

**注入领域同义词表：** 提升专业术语的查询扩展效果：

```python
synonym_dict = {
    "净利润": ["归母净利润", "净利", "税后利润"],
    "违约": ["违反合同", "不履行义务"],
}
```

**自定义标题正则：** 适配不同文档结构（如金融年报章节格式）：

```python
import re
title_patterns = [
    re.compile(r"^\s*[一二三四五六七八九十]+[、.]"),  # 一、二、三、
    re.compile(r"^\s*第\s*\d+\s*条"),               # 第X条（法律文书）
]
```

### 5. 扩展语义图谱术语表

`graph/semantic_graph.py` 中的 `DEFAULT_TECH_TERMS` 为预置技术术语，可按领域扩展：

```python
DEFAULT_TECH_TERMS = [
    # 金融领域
    "市盈率", "净资产收益率", "资产负债率",
    # 医疗领域
    "高血压", "糖尿病", "心肌梗死",
    # 替换为你的领域术语
]
```

### 6. 接入 MCP Client

MCP 服务（`POST /v1/completions`）遵循 OpenAI-compatible 格式，可直接接入：
- 自定义 Agent 的工具调用
- LangChain 的 OpenAI-compatible LLM 包装
- 任意支持自定义 API endpoint 的客户端

---

## 注意事项

### 索引管理

**何时需要重建索引：** 修改 `chunk_size`、`overlap`、`embedding_model_name` 后，旧索引与新参数不兼容，必须重建。

**重建步骤：**
1. 删除 `data/hash_db.json` 中对应记录（或清空为 `{}`）
2. 删除 `index/` 下对应的 `.faiss`、`.bm25.pkl`、`.chunks.jsonl`
3. 删除 `graph/` 下对应的 `_semantic_graph.json`
4. 重新运行 `python main.py --mode test --file your.pdf`

**快捷方式：** 将 PDF 重命名（如 `your_document_v2.pdf`），系统视为新文件自动重处理。

### 性能说明

- **CPU 可运行**，无 GPU 依赖，无需训练
- FAISS IndexFlatIP 暴力搜索，适合单文档或小规模（< 10 万 chunk）场景；大规模生产建议换 IndexIVFFlat 或外部向量库
- spaCy `zh_core_web_sm` 为轻量中文模型，NER 精度有限；精度要求高的场景可换 `zh_core_web_lg` 或接入专业 NER 服务

### OCR 依赖（可选）

处理扫描版 PDF 需额外安装 Tesseract：
1. 安装 [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) 并加入系统 PATH
2. Python 依赖 `pytesseract` 已在 requirements 中

未安装不影响主流程，扫描版 PDF 会提示"无可提取文本内容"。

### 多文档场景

当前架构以 `file_hash` 为索引单元，多文档并行检索需在 `HybridRetriever` 层做聚合。如需管理多个文档集合，建议：
- 为每个文档/集合维护独立索引
- 在检索层实现跨索引聚合逻辑
- 或迁移至支持 Collection 管理的向量库（Milvus、Qdrant）

---

## 项目结构

```
├── config/settings.py          # 全局配置（LLM、Embedding、分块参数、路径）
├── loader/                     # PDF 加载（PyMuPDF + OCR）
├── preprocess/                 # 文本清洗、去重、哈希校验
├── chunk/                      # 层级分块（HierarchicalChunker）
├── graph/                      # 语义图谱构建（spaCy NER）
├── retrieval/                  # BM25 索引、FAISS 索引、混合检索
├── generation/                 # LLM 生成（OpenAI-compatible）
├── mcp/                        # MCP HTTP 服务（Flask）
├── backend/                    # 后端服务
├── streamlit_app.py            # 白盒可视化前端
├── main.py                     # 入口（--mode test/mcp）
├── data/                       # 原始 PDF、处理记录（hash_db.json）
├── index/                      # BM25 + FAISS 索引文件
├── graph/                      # 语义图谱 JSON
├── output/                     # 评估报告
└── logs/                       # 结构化调试日志（JSONL）
```
