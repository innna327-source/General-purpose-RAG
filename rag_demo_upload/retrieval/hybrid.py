from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from retrieval.bm25 import BM25Index
from retrieval.vector_store import VectorStore


def _minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        if all((v == 0 or v is None) for v in values):
            return [0.0 for _ in values]
        return [1.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def _safe_load_graph(path: Path) -> tuple[Dict[str, List[str]], bool]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return (json.load(f) or {}), False
    except Exception:
        return {}, True


@dataclass
class HybridRetriever:
    """
    BM25 + 向量双路混合检索器。

    synonym_dict：领域同义词表，用于 BM25 查询扩展。
                  格式：{"规范词": ["同义词1", "同义词2"]}
                  默认为空（不扩展），由调用方按领域注入：
                    - 医疗：{"高血压": ["血压升高"], ...}
                    - 金融：{"净利润": ["归母净利润", "净利"], ...}
                    - 法律：{"违约": ["违反合同", "不履行"], ...}
                  也可从外部 JSON 文件加载后传入，避免硬编码。
    bm25_weight / vector_weight：融合权重，默认 0.7/0.3（BM25 偏重精确匹配）。
                  专业术语密集的领域（金融、法律）可适当调高 bm25_weight。
    """

    file_hash: str
    bm25: BM25Index
    vector: VectorStore
    graph_path: Path
    synonym_dict: Dict[str, List[str]] = field(default_factory=dict)
    bm25_weight: float = 0.7
    vector_weight: float = 0.3

    def hybrid_retrieve(
        self, query: str, top_k: int = 5, return_debug_info: bool = False
    ) -> Tuple[List[dict], List[dict] | None]:
        graph, graph_fallback_used = _safe_load_graph(self.graph_path)

        # 查询扩展：图谱实体命中 + 同义词替换（BM25 用扩展后的 query，向量保持原 query）
        expanded_query = query
        for ent in graph.keys():
            if ent and ent in query:
                extra = self.synonym_dict.get(ent, [])
                if extra:
                    expanded_query = (query + " " + " ".join(extra)).strip()

        bm25_hits = self.bm25.search(expanded_query, top_k=20)
        vec_hits = self.vector.search(query, top_k=20)

        cand: Dict[str, dict] = {}
        text_map = self.bm25.chunk_text_by_id
        for h in bm25_hits:
            cid = h["chunk_id"]
            cand[cid] = {
                "chunk_id": cid,
                "text": text_map.get(cid, ""),
                "bm25_score": float(h["score"]),
                "vector_score": 0.0,
            }
        for h in vec_hits:
            cid = h["chunk_id"]
            text = text_map.get(cid, self.vector.chunk_text_by_id.get(cid, ""))
            if cid not in cand:
                cand[cid] = {
                    "chunk_id": cid,
                    "text": text,
                    "bm25_score": 0.0,
                    "vector_score": float(h["score"]),
                }
            else:
                cand[cid]["vector_score"] = float(h["score"])

        candidates = list(cand.values())[:40]
        norm_bm25 = _minmax_norm([float(c["bm25_score"]) for c in candidates])
        norm_vec = _minmax_norm([float(c["vector_score"]) for c in candidates])

        debug_data: list[dict] = []
        for c, nb, nv in zip(candidates, norm_bm25, norm_vec):
            final = self.bm25_weight * nb + self.vector_weight * nv
            row = {
                "chunk_id": c["chunk_id"],
                "bm25_score": float(c["bm25_score"]),
                "vector_score": float(c["vector_score"]),
                "norm_bm25": float(nb),
                "norm_vector": float(nv),
                "final_score": float(final),
                "graph_fallback_used": bool(graph_fallback_used),
            }
            debug_data.append(row)
            c["final_score"] = float(final)

        results = sorted(candidates, key=lambda x: float(x.get("final_score", 0.0)), reverse=True)[:top_k]
        results_out = [
            {
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "bm25_score": float(r["bm25_score"]),
                "vector_score": float(r["vector_score"]),
                "final_score": float(r["final_score"]),
            }
            for r in results
        ]

        if return_debug_info:
            return results_out, debug_data
        return results_out, None
