"""
检索评估模块

评估指标：
- Recall@5: 前5个结果中话题命中率（关键词匹配）
- MRR: 平均倒数排名
- Topic Hit Rate: 检索结果是否命中了相关话题

注意：检索分数高只代表找到了相关话题，不代表文档中一定有完整答案
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from retrieval.hybrid import HybridRetriever


class RetrievalEvaluator:
    """检索质量评估器"""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def evaluate_single(
        self,
        query: str,
        expected_keywords: list[str],
        top_k: int = 5,
    ) -> dict:
        """
        评估单个查询的检索效果

        返回：
        - hit_rank: 命中排名（0表示未命中）
        - reciprocal_rank: 倒数排名分数
        - is_topic_hit: 是否命中话题（关键词匹配）
        - retrieved_chunks: 检索到的文本片段列表
        - top_scores: Top结果的分数分布
        """
        from preprocess.cleaner import normalize_for_match

        results, debug_info = self.retriever.hybrid_retrieve(
            query, top_k=top_k, return_debug_info=True
        )

        norm_keywords = [
            normalize_for_match(k) for k in expected_keywords
            if isinstance(k, str) and k.strip()
        ]

        hit_rank = 0
        for rank, r in enumerate(results, start=1):
            chunk_norm = normalize_for_match(r["text"])
            if hit_rank == 0 and any(k and (k in chunk_norm) for k in norm_keywords):
                hit_rank = rank

        reciprocal_rank = 1.0 / hit_rank if hit_rank > 0 else 0.0

        return {
            "hit_rank": hit_rank,
            "reciprocal_rank": reciprocal_rank,
            "is_topic_hit": hit_rank > 0,
            "retrieved_chunks": [r["text"] for r in results],
            "top_scores": [
                {
                    "chunk_id": r["chunk_id"],
                    "final_score": r["final_score"],
                    "bm25_score": r["bm25_score"],
                    "vector_score": r["vector_score"],
                }
                for r in results
            ],
        }

    def evaluate_batch(
        self,
        test_queries: list[dict],
        top_k: int = 5,
    ) -> dict:
        """
        批量评估检索效果

        输入格式：
        [
            {
                "query": "问题文本",
                "expected_keywords": ["关键词1", "关键词2"],
                "has_answer_in_doc": true/false,  # 可选，标注文档是否有答案
            }
        ]
        """
        details = []
        hits = 0
        rr_sum = 0.0
        queries_with_answer = 0
        hits_with_answer = 0

        for item in test_queries:
            query = item["query"]
            expected_keywords = item.get("expected_keywords", [])
            has_answer_in_doc = item.get("has_answer_in_doc", None)

            result = self.evaluate_single(query, expected_keywords, top_k)

            if result["is_topic_hit"]:
                hits += 1
                rr_sum += result["reciprocal_rank"]
                if has_answer_in_doc is True:
                    hits_with_answer += 1

            if has_answer_in_doc is True:
                queries_with_answer += 1

            details.append({
                "query": query,
                "hit_rank": result["hit_rank"],
                "reciprocal_rank": result["reciprocal_rank"],
                "is_topic_hit": result["is_topic_hit"],
                "expected_keywords": expected_keywords,
                "has_answer_in_doc": has_answer_in_doc,
                "top_final_score": result["top_scores"][0]["final_score"] if result["top_scores"] else 0,
            })

        total = len(test_queries)
        recall_at_5 = hits / total if total > 0 else 0.0
        mrr = rr_sum / total if total > 0 else 0.0

        # 新增：针对"文档中有答案"的问题的召回率
        answer_recall = hits_with_answer / queries_with_answer if queries_with_answer > 0 else None

        return {
            "evaluation_type": "retrieval",
            "total_queries": total,
            "recall_at_5": recall_at_5,
            "mrr": mrr,
            "topic_hit_count": hits,
            "queries_with_answer_in_doc": queries_with_answer,
            "answer_recall": answer_recall,  # 有答案问题的召回率
            "details": details,
        }

    def save_report(self, report: dict, out_path: Path) -> None:
        """保存评估报告"""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)