from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# 设置 Hugging Face 镜像源，防止国内下载模型超时
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
# 模型已缓存，禁止网络请求（无网络环境下防止超时报错）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from utils.paths import project_root


@dataclass(frozen=True)
class Settings:
    # Embedding / tokenizer
    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"
    tokenizer_model_name: str = "BAAI/bge-small-zh-v1.5"

    # Chunking
    chunk_size: int = 220
    overlap: int = 50
    # 领域定制标题正则（None = 使用 HierarchicalChunker 内置默认值）
    # 示例：金融年报可传 [re.compile(r"^\s*[一二三四五六七八九十]+[、.]")]
    # 在此处设为 None，由 chunk 层负责应用默认值，保持 settings 不依赖 re 模块
    title_patterns: object = None
    # 语义父块切分阈值（相邻段落余弦相似度低于此值视为主题边界）
    # None = 自适应（均值 - 0.5×标准差）；调小→父块更少；调大→父块更多
    semantic_threshold: object = None

    # Deduplication
    dedup_similarity_threshold: float = 0.85

    # BM25
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # 领域同义词表：{"规范词": ["同义词1", ...]}
    # 空 dict = 不做查询扩展（默认）；按领域在此处或外部 JSON 填入
    synonym_dict: object = None  # type: dict | None

    # 混合检索融合权重（BM25 : 向量）；专业术语密集领域可调高 bm25_weight
    bm25_weight: float = 0.7
    vector_weight: float = 0.3

    # Paths (relative to project root)
    root: Path = project_root()
    raw_pdf_dir: Path = root / "data" / "raw_pdf"
    processed_dir: Path = root / "data" / "processed"
    hash_db_path: Path = root / "data" / "hash_db.json"
    index_dir: Path = root / "index"
    graph_dir: Path = root / "graph"
    logs_dir: Path = root / "logs"
    output_dir: Path = root / "output"

    # LLM generation（OpenAI 兼容接口）
    # 支持任意 OpenAI-compatible 接口，留空则降级为直接返回检索结果
    llm_model: str = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    llm_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    llm_base_url: str = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")  # 填基础地址，不要加 /chat/completions（SDK 会自动拼接）

    # MCP server config
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8080
    mcp_service_name: str = "deep_rag"
    mcp_protocol_version: str = "1.0"
    mcp_model_type: str = "retrieval"
    mcp_capabilities: list[str] = None  # set in __post_init__ style below

    def with_defaults(self) -> "Settings":
        # keep dataclass frozen while still giving a stable default list
        if self.mcp_capabilities is None:
            return Settings(
                embedding_model_name=self.embedding_model_name,
                tokenizer_model_name=self.tokenizer_model_name,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
                title_patterns=self.title_patterns,
                semantic_threshold=self.semantic_threshold,
                dedup_similarity_threshold=self.dedup_similarity_threshold,
                bm25_k1=self.bm25_k1,
                bm25_b=self.bm25_b,
                synonym_dict=self.synonym_dict if self.synonym_dict is not None else {},
                bm25_weight=self.bm25_weight,
                vector_weight=self.vector_weight,
                root=self.root,
                raw_pdf_dir=self.raw_pdf_dir,
                processed_dir=self.processed_dir,
                hash_db_path=self.hash_db_path,
                index_dir=self.index_dir,
                graph_dir=self.graph_dir,
                logs_dir=self.logs_dir,
                output_dir=self.output_dir,
                mcp_host=self.mcp_host,
                mcp_port=self.mcp_port,
                mcp_service_name=self.mcp_service_name,
                mcp_protocol_version=self.mcp_protocol_version,
                mcp_model_type=self.mcp_model_type,
                mcp_capabilities=["query", "health"],
                llm_model=self.llm_model,
                llm_api_key=self.llm_api_key,
                llm_base_url=self.llm_base_url,
            )
        return self


SETTINGS = Settings().with_defaults()

