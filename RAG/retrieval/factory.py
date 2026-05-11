"""
检索器工厂：统一 HybridRetriever 的构建逻辑。

消除 main.py / run_evaluation.py / streamlit_app.py 三处重复代码。
"""

from __future__ import annotations

from pathlib import Path

from config.settings import SETTINGS
from retrieval import bm25 as bm25_mod
from retrieval.bm25 import BM25Index
from retrieval.hybrid import HybridRetriever
from retrieval.vector_store import VectorStore, load_index as load_faiss_index


def load_retriever(file_hash: str) -> HybridRetriever:
    """根据 file_hash 加载 BM25 + FAISS + 图谱 + 重排模型，构建 HybridRetriever。"""
    bm25_index: BM25Index = bm25_mod.load_index(file_hash)
    vector_index: VectorStore = load_faiss_index(file_hash, SETTINGS.embedding_model_name)
    graph_path = SETTINGS.graph_dir / f"{file_hash}_semantic_graph.json"
    return HybridRetriever(
        file_hash=file_hash,
        bm25=bm25_index,
        vector=vector_index,
        graph_path=graph_path,
        synonym_dict=SETTINGS.synonym_dict or {},
        bm25_weight=SETTINGS.bm25_weight,
        vector_weight=SETTINGS.vector_weight,
        candidate_top_k=getattr(SETTINGS, "candidate_top_k", 50),
        use_rerank=SETTINGS.use_rerank,
        rerank_model=SETTINGS.rerank_model,
        rerank_top_k=SETTINGS.rerank_top_k,
        rerank_batch_size=SETTINGS.rerank_batch_size,
        use_graph_retrieve=getattr(SETTINGS, "use_graph_retrieve", True),
        graph_hop_depth=getattr(SETTINGS, "graph_hop_depth", 2),
        graph_entity_recall=getattr(SETTINGS, "graph_entity_recall", True),
        graph_boost_weight=getattr(SETTINGS, "graph_boost_weight", 0.3),
        graph_max_chunks_per_entity=getattr(SETTINGS, "graph_max_chunks_per_entity", 3),
        graph_max_chunks_per_parent=getattr(SETTINGS, "graph_max_chunks_per_parent", 2),
    )
