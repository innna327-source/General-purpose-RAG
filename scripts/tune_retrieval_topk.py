from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chunk.hierarchical_chunk import build_hierarchical_chunks
from config.settings import SETTINGS
from evaluation.retrieval_eval import RetrievalEvaluator
from loader import get_loader
from preprocess.cleaner import clean_to_paragraphs
from preprocess.deduplicator import deduplicate_paragraphs
from retrieval import bm25 as bm25_mod
from retrieval.factory import load_retriever
from retrieval.vector_store import build_index as build_faiss_index
from utils.hash_utils import sha256_file
from utils.queries import load_test_queries


def _parse_topks(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(max(1, int(part)))
    return sorted(set(values))


def _index_files_exist(file_hash: str) -> bool:
    return (
        (SETTINGS.index_dir / f"{file_hash}.chunks.jsonl").exists()
        and (SETTINGS.index_dir / f"{file_hash}.bm25.pkl").exists()
        and (SETTINGS.index_dir / f"{file_hash}.faiss").exists()
    )


def _build_retrieval_indexes(file_path: Path, file_hash: str) -> None:
    loader = get_loader(file_path)
    raw_text = loader.load(str(file_path))

    paragraphs = clean_to_paragraphs(raw_text)
    if len(paragraphs) > 2000:
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

    bm25_mod.save_chunks_jsonl(chunk_res.chunks, file_hash=file_hash)
    bm25_mod.build_index(
        chunk_res.chunks,
        file_hash=file_hash,
        k1=SETTINGS.bm25_k1,
        b=SETTINGS.bm25_b,
    )
    build_faiss_index(
        chunk_res.chunks,
        file_hash=file_hash,
        model_name=SETTINGS.embedding_model_name,
    )


def tune(
    file_path: Path,
    queries_path: Path,
    candidate_topks: list[int],
    final_top_k: int,
    use_rerank: bool,
    use_graph: bool,
    rebuild: bool,
) -> dict:
    file_hash = sha256_file(file_path)
    if rebuild or not _index_files_exist(file_hash):
        _build_retrieval_indexes(file_path, file_hash)

    retriever = load_retriever(file_hash)
    retriever.use_rerank = use_rerank
    retriever.use_graph_retrieve = use_graph
    retriever.graph_entity_recall = use_graph

    test_queries = load_test_queries(queries_path)
    rows: list[dict] = []

    for candidate_top_k in candidate_topks:
        retriever.candidate_top_k = candidate_top_k
        evaluator = RetrievalEvaluator(retriever)

        started = time.perf_counter()
        report = evaluator.evaluate_batch(test_queries, top_k=final_top_k)
        elapsed = time.perf_counter() - started

        rows.append(
            {
                "candidate_top_k": candidate_top_k,
                "final_top_k": final_top_k,
                "recall_at_final_top_k": report["recall_at_5"],
                "mrr": report["mrr"],
                "answer_recall": report["answer_recall"],
                "topic_hit_count": report["topic_hit_count"],
                "total_queries": report["total_queries"],
                "elapsed_seconds": round(elapsed, 3),
                "avg_ms_per_query": round(elapsed * 1000 / max(1, report["total_queries"]), 2),
            }
        )

    best = max(rows, key=lambda r: (r["recall_at_final_top_k"], r["mrr"], -r["avg_ms_per_query"]))
    return {
        "file": str(file_path),
        "file_hash": file_hash,
        "queries": str(queries_path),
        "use_rerank": use_rerank,
        "use_graph": use_graph,
        "best": best,
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune BM25/vector candidate top-k for hybrid retrieval.")
    parser.add_argument("--file", default=str(SETTINGS.raw_pdf_dir / "your.pdf"))
    parser.add_argument("--queries", default=str(SETTINGS.root / "tests" / "test_queries_v2.json"))
    parser.add_argument("--candidate-topks", default="5,10,20,30,50,80")
    parser.add_argument("--final-top-k", type=int, default=5)
    parser.add_argument("--use-rerank", action="store_true")
    parser.add_argument("--use-graph", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--out", default=str(SETTINGS.output_dir / "tune_retrieval_topk.json"))
    args = parser.parse_args()

    report = tune(
        file_path=Path(args.file),
        queries_path=Path(args.queries),
        candidate_topks=_parse_topks(args.candidate_topks),
        final_top_k=args.final_top_k,
        use_rerank=args.use_rerank,
        use_graph=args.use_graph,
        rebuild=args.rebuild,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("candidate_top_k\trecall@{}\tmrr\tanswer_recall\tavg_ms/query".format(args.final_top_k))
    for row in report["results"]:
        answer_recall = row["answer_recall"]
        answer_recall_text = "-" if answer_recall is None else f"{answer_recall:.4f}"
        print(
            f"{row['candidate_top_k']}\t"
            f"{row['recall_at_final_top_k']:.4f}\t"
            f"{row['mrr']:.4f}\t"
            f"{answer_recall_text}\t"
            f"{row['avg_ms_per_query']:.2f}"
        )
    print(f"best_candidate_top_k={report['best']['candidate_top_k']}")
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
