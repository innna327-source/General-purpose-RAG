# Deep RAG Demo

本项目是一个**最小可运行**的 Deep-RAG Demo，包含文档解析、混合检索（BM25 + 向量）、语义图谱，以及接入 LLM 生成答案的完整流程。不使用 GPU，不含训练过程。

## 核心功能

- **文档解析**：支持 PDF 文本提取，集成 OCR 功能（可选）处理扫描版 PDF。
- **文本分块**：实现层级化文本分块（Hierarchical Chunking）。
- **混合检索**：结合 BM25（关键词检索）和 FAISS（向量检索）的混合检索策略。
- **语义图谱**：构建文档内容的语义图谱，辅助实体扩展和检索。
- **LLM 生成**：检索到相关段落后，调用 LLM 基于原文生成答案，并通过 System Prompt 约束模型只用检索内容回答，防止胡编。
- **MCP 服务**：提供兼容 OpenAI 接口格式的 API 服务，方便集成。

---

## 运行流程总览

```
第 1 步  配置 LLM（一次性）
第 2 步  构建索引          python main.py --mode test --file demo.pdf
第 3 步  批量评估（可选）   python main.py --mode test --file demo.pdf --eval
第 4 步  启动前端          python -m streamlit run streamlit_app.py   ← LLM 在这里调用
```

> **LLM 在哪一步调用，问题从哪里来？**
> 在第 4 步（Streamlit 运行期间）。前端下拉框里选一条测试 query → 自动做混合检索 → 把检索到的段落 + 这条 query 一起发给 LLM → 在页面上显示生成答案。
> 问题直接来自 `tests/test_queries.json`，不需要另外输入。第 2、3 步（构建索引 / 批量评估）**不调用 LLM**。
>
> **MCP 服务是否必要？**
> 对于 Streamlit Demo 不需要。MCP 服务仅当你需要把 RAG 能力作为 HTTP API 提供给外部系统时才启动。

---

## 第 0 步：安装依赖

```bash
# 激活虚拟环境（Windows PowerShell）
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

---

## 第 1 步：配置 LLM

打开 `config/settings.py`，修改以下三个字段：

`
> **注意**：`llm_base_url` 只填到 `/v1`，不要加 `/chat/completions`。SDK 会自动拼接完整路径。
>
> 若不填 `llm_api_key`，也可以设置环境变量 `OPENAI_API_KEY`。若两者都为空，服务会自动降级为直接返回检索原文，不影响运行。

---

## 第 2 步：构建索引

将 PDF 文件放入 `data/raw_pdf/` 目录，然后运行：

```bash
# 设置 HuggingFace 镜像（国内网络）
$env:HF_ENDPOINT="https://hf-mirror.com"

python main.py --mode test --file demo.pdf
```

运行成功后，控制台会打印该文件的 `file_hash`，例如：

```
索引构建完成 file_hash=d1257ca2ac08ed674fe3315319dd64b423e51991f2b3932a5c8fc697a1da97a3
```

**请记录这个 hash**，后续步骤需要用到。

---

## 第 3 步：批量评估（可选）

评估检索质量（不调用 LLM），读取 `tests/test_queries.json` 中的问题：

```bash
python main.py --mode test --file demo.pdf --eval
```

- 评估报告输出至：`output/test_report.json`
- 详细调试日志输出至：`logs/debug_YYYYMMDD_HHMMSS.jsonl`

---

## 第 4 步：启动前端（**含 LLM 生成**）

```bash
.\.venv\Scripts\Activate.ps1
python -m streamlit run streamlit_app.py
```

浏览器访问 `http://localhost:8501`。

页面上的"召回链路透视表"下方有下拉框，从 100 条测试 query 中选一条：
1. 自动展示检索到的 Top 5 分块及评分
2. 自动调用 LLM，将检索段落 + 问题一起发送，生成答案显示在蓝色框中

---

## （可选）启动 MCP HTTP 服务

如需将 RAG 能力作为 HTTP API 对外提供，才需要启动 MCP 服务：

```bash
python main.py --mode mcp --index-hash d1257ca2ac08ed674fe3315319dd64b423e51991f2b3932a5c8fc697a1da97a3
```

服务默认监听 `0.0.0.0:8080`，运行日志输出至 `logs/service.log`。

验证：

```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"你好\"}"
```

---

## OCR 依赖（可选）

如果需要处理扫描版 PDF，需安装 Tesseract OCR：
- 安装 Tesseract 软件并将其添加到系统环境变量。
- Python 依赖 `pytesseract` 已在 requirements 中。
- *未安装不影响主流程，但扫描版 PDF 可能提示"无可提取文本内容"。*

---

## 重建索引

修改了 `config/settings.py` 中的分块参数（如 `chunk_size`、`overlap`）后，需要清理旧索引：

1. 打开 `data/hash_db.json`，删除 `demo.pdf` 对应的记录（或清空为 `{}`）。
2. 删除 `index/` 目录下对应的 `.faiss`、`.bm25.pkl`、`.chunks.jsonl` 文件。
3. 删除 `graph/` 目录下对应的 `_semantic_graph.json` 文件。

**快捷方法**：将 PDF 重命名（如 `demo_v2.pdf`），系统会当作新文件重新处理。

重建完成后：
1. 使用新的 `file_hash` 重新启动 MCP 服务。
2. 在 Streamlit 前端点击左侧边栏的 **"重新加载数据"** 按钮刷新看板。
