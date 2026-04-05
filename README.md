# General-purpose-RAG
 基于 GraphRAG 架构的本地文档问答系统，支持 PDF 文档的向量检索 + BM25 混合召回，并通过 MCP 协议对外提供检索服务。

  ---

  ## 目前支持的能力

  **文档格式**
  - PDF（含 OCR 兜底，扫描件也能用）

  **分块策略**
  - 层级式滑动窗口分块：先按标题行切成父块，再对每个父块做 token 级滑动窗口切子块，子块之间有重叠避免语义断裂

  **检索**
  - 双路混合召回：BM25（精确词匹配）+ FAISS 向量检索（语义匹配），结果做 min-max 归一化后加权融合
  - 语义图谱辅助：对图谱中识别到的实体做同义词扩展，增强 BM25 召回率

  **接口**
  - MCP 协议服务（`--mode mcp`），可对接外部系统
  - Streamlit 可视化界面，支持上传文档、提问、查看检索分数分布
  - OpenAI 兼容 LLM 接口，检索结果可直接送入大模型生成回答

  ---

  ## 快速开始

  ```bash
  # 安装依赖
  pip install -r requirements.txt

  # 构建索引并测试检索（把你的 PDF 放到 data/raw_pdf/ 下）
  python main.py --mode test --file 你的文件.pdf

  # 启动 Streamlit 界面
  streamlit run streamlit_app.py

  # 启动 MCP 服务（需要先跑过 test 模式拿到 index_hash）
  python main.py --mode mcp --index-hash <hash值>
  ```

  ---

  ## 如何扩展

  ### 换领域（金融 / 法律 / 企业内部文档）

  只需修改 `config/settings.py` 里的两个字段，不改任何业务代码：

  ```python
  # 1. 换标题识别规则（控制父块切分边界）
  #    默认支持"第X章/节/条"等通用中文结构
  #    金融年报示例：
  import re
  title_patterns = [
      re.compile(r"^\s*第[一二三四五六七八九十]+节\s*"),
      re.compile(r"^\s*[（(]\s*[一二三四五六七八九十]+\s*[）)]\s*"),
  ]

  # 2. 加入领域同义词表（用于 BM25 查询扩展）
  #    金融示例：
  synonym_dict = {
      "净利润": ["归母净利润", "净利", "税后利润"],
      "营收": ["营业收入", "总收入"],
  }
  #    法律示例：
  synonym_dict = {
      "违约": ["违反合同", "不履行义务", "逾期"],
      "赔偿": ["损害赔偿", "赔付"],
  }
  ```

  ### 新增文件格式（Word / HTML / Markdown）

  在 `loader/` 目录下新建一个 Loader 类，继承 `BaseLoader`，实现 `load()` 方法返回纯文本字符串：

  ```python
  # loader/docx_loader.py
  from loader.base_loader import BaseLoader

  class DocxLoader(BaseLoader):
      def load(self, file_path: str) -> str:
          # 用 python-docx 读取内容
          ...
          return text
  ```

  然后在 `main.py` 的 `_get_loader()` 工厂函数里注册一行：

  ```python
  if suffix == ".docx":
      return DocxLoader()
  ```

  其他流程（分块、建索引、检索）完全不需要改。

  ### 换分块策略（语义分块 / 句子级分块）

  继承 `BaseChunker`，实现 `chunk()` 方法：

  ```python
  # chunk/semantic_chunker.py
  from chunk.base_chunker import BaseChunker, ChunkResult

  class SemanticChunker(BaseChunker):
      def chunk(self, paragraphs: list[str]) -> ChunkResult:
          # 自定义分块逻辑
          ...
          return ChunkResult(chunks=..., parent_child_map=..., stats=...)
  ```

  在 `main.py` 的 `_build_all()` 里把 `build_hierarchical_chunks(...)` 替换成新的 chunker 实例调用即可。

  ### 调整检索融合权重

  在 `config/settings.py` 里直接改数字：

  ```python
  bm25_weight: float = 0.7   # 精确匹配权重，术语密集的领域适合调高
  vector_weight: float = 0.3  # 语义匹配权重
  ```

  ### 接入大模型

  在 `config/settings.py` 里填入兼容 OpenAI 接口的配置：

  ```python
  llm_model: str = "你的模型名"
  llm_api_key: str = "你的 API Key"
  llm_base_url: str = "https://你的接口地址/v1"
  ```

  支持 SiliconFlow、DeepSeek、本地 Ollama 等任意 OpenAI 兼容接口。

  ---

  ## 项目结构

  ```
  ├── chunk/          # 分块策略（BaseChunker + HierarchicalChunker）
  ├── loader/         # 文档加载（BaseLoader + PDFLoader）
  ├── preprocess/     # 文本清洗、去重
  ├── retrieval/      # BM25 索引、向量索引、混合检索器
  ├── graph/          # 语义图谱构建
  ├── generation/     # LLM 生成
  ├── mcp/            # MCP 协议服务
  ├── backend/        # HTTP 后端
  ├── frontend/       # React 前端
  ├── config/         # 全局配置（settings.py）
  ├── utils/          # 日志、路径、哈希工具
  ├── data/raw_pdf/   # 放你的 PDF 文档（不提交到 git）
  └── index/          # 构建产物（不提交到 git）
  ```
