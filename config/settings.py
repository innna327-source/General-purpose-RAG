from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# 设置 Hugging Face 镜像源，防止国内下载模型超时
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
# 模型已缓存，禁止网络请求（无网络环境下防止超时报错）
os.environ.setdefault("HF_HUB_OFFLINE", os.environ.get("HF_HUB_OFFLINE", "0"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", os.environ.get("TRANSFORMERS_OFFLINE", "0"))

from utils.paths import project_root


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
    # Parent split fallback threshold.
    # None = adaptive: compare regex title splitting with semantic boundary
    # detection derived from this document's adjacent-paragraph similarity
    # distribution. Set an integer to force a fixed minimum parent count.
    min_parent_count: int | None = None

    # Deduplication
    dedup_similarity_threshold: float = 0.85

    # BM25
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # 领域同义词表：{"规范词": ["同义词1", ...]}
    # 空 dict = 不做查询扩展（默认）；按领域在此处或外部 JSON 填入
    synonym_dict: dict | None = None

    # 混合检索融合权重（BM25 : 向量）；专业术语密集领域可调高 bm25_weight
    bm25_weight: float = 0.7
    vector_weight: float = 0.3
    # 每一路召回进入融合/重排前的候选数量；应通过评估集调参，而不是视为固定最优值
    candidate_top_k: int = 50

    # 重排配置（Cross-Encoder）
    use_rerank: bool = _env_bool("RAG_USE_RERANK", True)  # 是否启用 Cross-Encoder 重排
    rerank_model: str = os.environ.get("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")  # 中文重排模型
    rerank_top_k: int = 50  # 重排前召回数量
    rerank_batch_size: int = 16  # 批处理大小

    # 图谱检索配置
    use_graph_retrieve: bool = True  # 是否启用图谱增强检索
    graph_hop_depth: int = 2  # 图谱遍历深度（1-3跳），默认2跳
    graph_entity_recall: bool = True  # 是否直接从 entity_chunks 召回相关实体块
    graph_boost_weight: float = 0.3  # 图谱召回结果的权重提升

    # Keep entity quota higher than parent quota so parent diversity can take effect.
    graph_max_chunks_per_entity: int = 3
    graph_max_chunks_per_parent: int = 2
    # Graph storage. Neo4j is the primary graph store; JSON remains an export/fallback
    # so the framework can still run when a local Neo4j service is not started.
    graph_store_backend: str = os.environ.get("RAG_GRAPH_STORE", "json")
    neo4j_uri: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password: str = os.environ.get("NEO4J_PASSWORD", "")
    neo4j_database: str = os.environ.get("NEO4J_DATABASE", "neo4j")
    neo4j_enabled_fallback: bool = True

    # Answerability gate: reject when retrieved context cannot support an answer.
    enable_answerability_gate: bool = True
    answerability_min_query_coverage: float = 0.30
    answerability_min_answer_coverage: float = 0.50
    answerability_min_context_chars: int = 80
    answerability_min_retrieval_confidence: float = 0.35
    answerability_min_top1_rerank_score: float = -3.0
    answerability_min_top1_final_score: float = 0.05
    answerability_max_related_context_ratio: float = 0.80
    enable_direct_evidence_answerability_gate: bool = True
    answerability_min_direct_sentence_coverage: float = 0.35
    answerability_min_direct_term_hits: int = 2
    answerability_direct_window_chars: int = 420
    answerability_uncertain_normal_policy: str = "warn"  # warn / reject
    enable_high_risk_llm_judge: bool = False

    # Paths (relative to project root)
    root: Path = field(default_factory=project_root)
    raw_pdf_dir: Path = None  # set in __post_init__
    processed_dir: Path = None
    hash_db_path: Path = None
    version_log_path: Path = None
    version_retention_days: int = 7
    index_dir: Path = None
    graph_dir: Path = None
    logs_dir: Path = None
    output_dir: Path = None

    # LLM generation（OpenAI 兼容接口）
    # 图谱抽取：需要强推理能力抽取实体关系，使用重量级模型
    llm_model: str = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    # 问答生成：上下文已由检索给好，按材料作答即可，轻量模型足够
    generation_llm_model: str = os.environ.get("GENERATION_LLM_MODEL", os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    llm_api_key: str = os.environ.get("OPENAI_API_KEY", "")  # 留空则降级为直接返回检索结果
    llm_base_url: str = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")  # OpenAI-compatible base URL

    # MCP server config
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8080
    mcp_service_name: str = "deep_rag"
    mcp_protocol_version: str = "1.0"
    mcp_model_type: str = "retrieval"
    mcp_capabilities: list[str] = None

    def __post_init__(self):
        root = self.root
        _path_defaults = {
            "raw_pdf_dir": root / "data" / "raw_pdf",
            "processed_dir": root / "data" / "processed",
            "hash_db_path": root / "data" / "hash_db.json",
            "version_log_path": root / "data" / "version_log.json",
            "index_dir": root / "index",
            "graph_dir": root / "graph",
            "logs_dir": root / "logs",
            "output_dir": root / "output",
        }
        for attr, value in _path_defaults.items():
            if getattr(self, attr) is None:
                object.__setattr__(self, attr, value)

        if self.synonym_dict is None:
            object.__setattr__(self, "synonym_dict", {})
        if self.mcp_capabilities is None:
            object.__setattr__(self, "mcp_capabilities", ["query", "health"])


SETTINGS = Settings()
