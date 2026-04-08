from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from chunk.hierarchical_chunk import build_hierarchical_chunks
from config.settings import SETTINGS
from graph.semantic_graph import build_semantic_graph
from loader.base_loader import BaseLoader
from loader.pdf_loader import PDFLoader
from mcp.handler import MCPHandler
from mcp.server import run_server
from preprocess.cleaner import clean_to_paragraphs, normalize_for_match
from preprocess.deduplicator import deduplicate_paragraphs
from preprocess.hash_checker import is_processed, record_hash
from retrieval import bm25 as bm25_mod
from retrieval.bm25 import BM25Index
from retrieval.hybrid import HybridRetriever
from retrieval.vector_store import VectorStore
from retrieval.vector_store import build_index as build_faiss_index
from retrieval.vector_store import load_index as load_faiss_index
from utils import logger
from utils.paths import ensure_runtime_dirs


def _get_loader(file_path: Path) -> BaseLoader:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return PDFLoader()
    raise ValueError(f"不支持的文件类型：{suffix}，目前支持：.pdf")


def _index_paths_exist(file_hash: str) -> bool:
    faiss_p = SETTINGS.index_dir / f"{file_hash}.faiss"
    bm25_p = SETTINGS.index_dir / f"{file_hash}.bm25.pkl"
    chunks_p = SETTINGS.index_dir / f"{file_hash}.chunks.jsonl"
    graph_p = SETTINGS.graph_dir / f"{file_hash}_semantic_graph.json"
    return faiss_p.exists() and bm25_p.exists() and chunks_p.exists() and graph_p.exists()


def _load_retriever(file_hash: str) -> HybridRetriever:
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
    )


def _build_all(file_path: Path, file_hash: str, mode: str, run_timestamp: str) -> HybridRetriever:
    loader = _get_loader(file_path)
    raw_text = loader.load(str(file_path))

    paragraphs = clean_to_paragraphs(raw_text)
    if len(paragraphs) > 2000:
        logger.warning(f"段落数过多（{len(paragraphs)}），将截断至 2000 以避免 O(N^2) 去重过慢。")
        paragraphs = paragraphs[:2000]

    paragraphs = deduplicate_paragraphs(
        paragraphs,
        model_name=SETTINGS.embedding_model_name,
        similarity_threshold=SETTINGS.dedup_similarity_threshold,
    )

    chunk_res = build_hierarchical_chunks(
        paragraphs=paragraphs,
        tokenizer_model_name=SETTINGS.tokenizer_model_name,
        chunk_size=SETTINGS.chunk_size,
        overlap=SETTINGS.overlap,
        title_patterns=SETTINGS.title_patterns,  # None = 使用默认值
    )

    if chunk_res.stats.get("used_fallback_tokenizer"):
        logger.warning("AutoTokenizer 加载失败，已进入 fallback（字符计数）分块模式。")

    logger.log_step_data("chunk_stats", chunk_res.stats, mode=mode, run_timestamp=run_timestamp)

    # 语义图谱
    graph_path = build_semantic_graph(chunk_res.chunks, file_hash=file_hash, graph_dir=SETTINGS.graph_dir)

    # 保存 chunks 顺序映射（pos 对齐）
    bm25_mod.save_chunks_jsonl(chunk_res.chunks, file_hash=file_hash)

    # BM25
    bm25_mod.build_index(
        chunk_res.chunks,
        file_hash=file_hash,
        k1=SETTINGS.bm25_k1,
        b=SETTINGS.bm25_b,
    )

    # FAISS
    build_faiss_index(
        chunk_res.chunks,
        file_hash=file_hash,
        model_name=SETTINGS.embedding_model_name,
    )

    # 仅在索引构建成功后写入 hash_db
    record_hash(
        str(file_path),
        file_hash=file_hash,
        metadata_dict={
            "file_name": file_path.name,
            "index_path": f"index/{file_hash}.faiss",
            "bm25_path": f"index/{file_hash}.bm25.pkl",
            "graph_path": f"graph/{file_hash}_semantic_graph.json",
        },
    )

    logger.info(f"索引构建完成 file_hash={file_hash}")
    return _load_retriever(file_hash)


def _run_queries(retriever: HybridRetriever, queries: list[dict], mode: str, run_timestamp: str) -> None:
    for q in queries:
        query = q["query"]
        results, debug_data = retriever.hybrid_retrieve(query, top_k=5, return_debug_info=True)
        assert debug_data is not None

        # 白盒：检索分数分布（BM25 + 向量）
        logger.log_step_data(
            "retrieve_score_distribution",
            {
                "query": query,
                "bm25_scores": [float(d["bm25_score"]) for d in debug_data],
                "vector_scores": [float(d["vector_score"]) for d in debug_data],
            },
            mode=mode,
            run_timestamp=run_timestamp,
        )

        if debug_data and debug_data[0].get("graph_fallback_used"):
            logger.warning("semantic_graph.json 缺失/损坏或为空，已降级为空图谱（不影响主流程）。")

        logger.info(f'Query="{query}" Top5:')
        for i, r in enumerate(results, start=1):
            snippet = r["text"][:120].replace("\n", " ")
            logger.info(
                f"  {i}. chunk_id={r['chunk_id']} final={r['final_score']:.4f} "
                f"bm25={r['bm25_score']:.4f} vec={r['vector_score']:.4f} text={snippet}..."
            )


def _eval(retriever: HybridRetriever, test_queries_path: Path, out_path: Path) -> None:
    with open(test_queries_path, "r", encoding="utf-8") as f:
        data = json.load(f) or []
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
            queries = [{"query": q} for q in data]
        else:
            queries = data

    details = []
    hits = 0
    rr_sum = 0.0

    for item in queries:
        query = item["query"]
        expected_keywords = item.get("expected_keywords", [])
        expected_chunk_ids = item.get("expected_chunk_ids", [])
        _ = normalize_for_match(query)  # 按要求做同口径清洗（用于评估一致性）

        results, _ = retriever.hybrid_retrieve(query, top_k=5, return_debug_info=False)

        norm_keywords = [normalize_for_match(k) for k in expected_keywords if isinstance(k, str) and k.strip()]
        hit_rank = 0
        
        for rank, r in enumerate(results, start=1):
            chunk_norm = normalize_for_match(r["text"])
            # 检查关键词命中
            if hit_rank == 0 and any(k and (k in chunk_norm) for k in norm_keywords):
                hit_rank = rank

        if hit_rank > 0:
            hits += 1
            rr = 1.0 / hit_rank
            is_hit = True
        else:
            rr = 0.0
            is_hit = False

        rr_sum += rr
        details.append({
            "query": query, 
            "hit_rank": hit_rank, 
            "reciprocal_rank": rr,
            "is_keyword_hit": is_hit,
            "expected_keywords": expected_keywords
        })

    total = len(queries)
    
    # 计算基于关键词的通过率 (等同于 Recall@5，但为了前端展示保留字段)
    passed_queries = hits
    pass_rate = (passed_queries / total) if total > 0 else 0.0

    report = {
        "total_queries": total,
        "recall_at_5": pass_rate,
        "mrr": (rr_sum / total) if total else 0.0,
        "pass_rate": pass_rate,
        "passed_queries": passed_queries,
        "total_with_expected_ids": total,  # 保持字段兼容前端
        "details": details,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"评估完成，报告已写入：{out_path}")


def _load_test_queries() -> list[dict]:
    p = SETTINGS.root / "tests" / "test_queries.json"
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f) or []
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
            return [{"query": q} for q in data]
        return data


def main() -> int:
    ensure_runtime_dirs()

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["test", "mcp"])
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--index-hash", type=str, default=None)
    args = parser.parse_args()

    mode = args.mode

    # 启动时先解析 mode，再检查索引存在性
    if mode == "mcp":
        if not args.index_hash:
            raise ValueError("--mode mcp 时必须提供 --index-hash")
        if not _index_paths_exist(args.index_hash):
            raise ValueError(f"索引文件不存在或不完整：{args.index_hash}（禁止在 mcp 模式重建索引）")

        retriever = _load_retriever(args.index_hash)
        handler = MCPHandler(retriever=retriever, service_name=SETTINGS.mcp_service_name)
        logger.info(f"MCP 服务启动：{SETTINGS.mcp_host}:{SETTINGS.mcp_port} index_hash={args.index_hash}")
        run_server(handler, host=SETTINGS.mcp_host, port=SETTINGS.mcp_port)
        return 0

    # test 模式
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.file:
        raise ValueError("--mode test 时必须提供 --file（仅文件名，位于 data/raw_pdf/）")

    file_name = args.file
    if not file_name.lower().endswith(".pdf"):
        raise ValueError("仅支持 pdf 文件类型")
    file_path = SETTINGS.raw_pdf_dir / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"未找到文件：{file_path}")

    processed, file_hash = is_processed(str(file_path))
    logger.info(f"test 模式 file={file_name} processed={processed} file_hash={file_hash}")

    if processed:
        if not _index_paths_exist(file_hash):
            raise ValueError(f"hash_db 中存在但索引文件缺失/不完整：{file_hash}")
        retriever = _load_retriever(file_hash)
    else:
        retriever = _build_all(file_path, file_hash=file_hash, mode=mode, run_timestamp=run_timestamp)

    if args.eval:
        _eval(
            retriever,
            test_queries_path=SETTINGS.root / "tests" / "test_queries.json",
            out_path=SETTINGS.output_dir / "test_report.json",
        )
        return 0

    # 查询测试（使用 tests/test_queries.json）
    queries = _load_test_queries()
    _run_queries(retriever, queries=queries, mode=mode, run_timestamp=run_timestamp)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)

