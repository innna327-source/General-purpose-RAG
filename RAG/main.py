from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from chunk.hierarchical_chunk import build_hierarchical_chunks
from config.settings import SETTINGS
from graph.semantic_graph import build_semantic_graph
from evaluation.retrieval_eval import RetrievalEvaluator
from loader import get_loader
from mcp.handler import MCPHandler
from mcp.server import run_server
from preprocess.cleaner import clean_to_paragraphs
from preprocess.deduplicator import deduplicate_paragraphs
from preprocess.hash_checker import get_version_history, is_processed, purge_old_versions, record_hash, record_version
from retrieval.factory import load_retriever
from retrieval.vector_store import build_index as build_faiss_index
from retrieval import bm25 as bm25_mod
from utils import logger
from utils.index_paths import all_index_files_exist, index_paths
from utils.paths import ensure_runtime_dirs
from utils.queries import load_test_queries


def _cleanup_index(file_hash: str) -> None:
    """删除该 hash 对应的所有索引/图谱文件，并清除 hash_db 记录，允许干净重建。"""
    for p in index_paths(file_hash):
        if p.exists():
            p.unlink()
            logger.info(f"已清理残留文件：{p.name}")

    db_path = SETTINGS.hash_db_path
    if db_path.exists():
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                db = json.load(f) or {}
            if file_hash in db:
                del db[file_hash]
                with open(db_path, "w", encoding="utf-8") as f:
                    json.dump(db, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


def _build_all(file_path: Path, file_hash: str, mode: str, run_timestamp: str):
    loader = get_loader(file_path)
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
        title_patterns=SETTINGS.title_patterns,
        embedding_model_name=SETTINGS.embedding_model_name,
        semantic_threshold=SETTINGS.semantic_threshold,
        min_parent_count=SETTINGS.min_parent_count,
    )

    if chunk_res.stats.get("used_fallback_tokenizer"):
        logger.warning("AutoTokenizer 加载失败，已进入 fallback（字符计数）分块模式。")

    logger.log_step_data("chunk_stats", chunk_res.stats, mode=mode, run_timestamp=run_timestamp)

    # 语义图谱
    build_semantic_graph(chunk_res.chunks, file_hash=file_hash, graph_dir=SETTINGS.graph_dir)

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
    metadata_dict = {
        "file_name": file_path.name,
        "index_path": f"index/{file_hash}.faiss",
        "bm25_path": f"index/{file_hash}.bm25.pkl",
        "graph_path": f"graph/{file_hash}_semantic_graph.json",
    }
    record_hash(str(file_path), file_hash=file_hash, metadata_dict=metadata_dict)
    record_version(file_path.name, file_hash=file_hash, metadata_dict=metadata_dict)
    purge_old_versions(file_path.name)

    logger.info(f"索引构建完成 file_hash={file_hash}")
    return load_retriever(file_hash)


def _run_queries(retriever, queries: list[dict], mode: str, run_timestamp: str) -> None:
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


def _eval(retriever, out_path: Path) -> None:
    """使用 RetrievalEvaluator 进行评估，输出兼容 streamlit 的报告格式。"""
    queries = load_test_queries()
    evaluator = RetrievalEvaluator(retriever)
    raw = evaluator.evaluate_batch(queries)

    # 从 details 中统计有答案问题的命中数
    hits_with_answer = sum(
        1 for d in raw["details"] if d["is_topic_hit"] and d.get("has_answer_in_doc") is True
    )

    # 转为 streamlit_app 兼容的嵌套结构
    report = {
        "evaluation_type": "区分检索和生成评估",
        "total_queries": raw["total_queries"],
        "queries_with_answer_in_doc": raw["queries_with_answer_in_doc"],
        "retrieval_metrics": {
            "recall_at_5": raw["recall_at_5"],
            "mrr": raw["mrr"],
            "topic_hit_count": raw["topic_hit_count"],
        },
        "answer_metrics": {
            "answer_recall": raw["answer_recall"],
            "queries_with_answer": raw["queries_with_answer_in_doc"],
            "hits_with_answer": hits_with_answer,
        },
        "pass_rate": raw["recall_at_5"],
        "passed_queries": raw["topic_hit_count"],
        "total_with_expected_ids": raw["total_queries"],
        "details": [
            {**d, "is_keyword_hit": d["is_topic_hit"]}
            for d in raw["details"]
        ],
    }

    evaluator.save_report(report, out_path)
    logger.info(f"评估完成，报告已写入：{out_path}")
    logger.info(
        f"【检索评估】话题命中率 Recall@5: {raw['recall_at_5']:.2%}, "
        f"MRR: {raw['mrr']:.3f}"
    )
    if raw["answer_recall"] is not None:
        logger.info(
            f"【答案评估】有答案问题召回率: {raw['answer_recall']:.2%} "
            f"({hits_with_answer}/{raw['queries_with_answer_in_doc']})"
        )


def main() -> int:
    ensure_runtime_dirs()

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["test", "mcp"])
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--index-hash", type=str, default=None)
    parser.add_argument("--rollback-to", type=str, default=None, help="回滚至指定时间点，格式：YYYY-MM-DD HH:MM:SS")
    args = parser.parse_args()

    mode = args.mode

    # 启动时先解析 mode，再检查索引存在性
    if mode == "mcp":
        if not args.index_hash:
            raise ValueError("--mode mcp 时必须提供 --index-hash")
        if not all_index_files_exist(args.index_hash):
            raise ValueError(f"索引文件不存在或不完整：{args.index_hash}（禁止在 mcp 模式重建索引）")

        retriever = load_retriever(args.index_hash)
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

    if args.rollback_to:
        history = get_version_history(file_name)
        target_ts = args.rollback_to
        candidates = [v for v in history if v["timestamp"] <= target_ts]
        if not candidates:
            logger.error(f"未找到 {file_name} 在 {target_ts} 之前的版本记录，请检查 --rollback-to 格式（YYYY-MM-DD HH:MM:SS）")
            return 1
        target_version = candidates[-1]
        rollback_hash = target_version["hash"]
        print(f"已回滚至版本 {target_version['timestamp']}（hash={rollback_hash}）")
        retriever = load_retriever(rollback_hash)
        if args.eval:
            _eval(retriever, out_path=SETTINGS.output_dir / "test_report.json")
            return 0
        queries = load_test_queries()
        _run_queries(retriever, queries=queries, mode=mode, run_timestamp=run_timestamp)
        return 0

    processed, file_hash = is_processed(str(file_path))
    logger.info(f"test 模式 file={file_name} processed={processed} file_hash={file_hash}")

    if processed:
        if not all_index_files_exist(file_hash):
            logger.warning(f"索引文件不完整，清理残留后重新构建：{file_hash}")
            _cleanup_index(file_hash)
            retriever = _build_all(file_path, file_hash=file_hash, mode=mode, run_timestamp=run_timestamp)
        else:
            retriever = load_retriever(file_hash)
    else:
        # hash_db 无记录：清理可能存在的残留文件（如上次中途失败遗留），再全量构建
        _cleanup_index(file_hash)
        retriever = _build_all(file_path, file_hash=file_hash, mode=mode, run_timestamp=run_timestamp)

    if args.eval:
        _eval(retriever, out_path=SETTINGS.output_dir / "test_report.json")
        return 0

    # 查询测试（使用 tests/test_queries.json）
    queries = load_test_queries()
    _run_queries(retriever, queries=queries, mode=mode, run_timestamp=run_timestamp)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        # 顶层捕获：记录 error 日志，退出码 1
        logger.error(str(e))
        sys.exit(1)
