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


def _safe_load_graph(path: Path) -> tuple[dict, bool]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return (json.load(f) or {}), False
    except Exception:
        return {}, True


def _graph_neighbor_expand(
    query: str, graph_data: dict, top_n: int = 2
) -> List[str]:
    """沿图的边找邻居，按边权重降序取 top-n 不在 query 里的实体词（词长 ≥ 2）。

    兼容两种格式：
    - 新格式：nodes 为 [{id, label, ...}]，edges 的 from/to 为节点 ID
    - 旧格式：nodes 为 [str]，edges 的 from/to 为实体标签
    """
    nodes_raw: list = graph_data.get("nodes", [])
    edges: List[dict] = graph_data.get("edges", [])

    if not nodes_raw:
        return []

    is_new_format = isinstance(nodes_raw[0], dict)

    if is_new_format:
        id_to_label: Dict[str, str] = {}
        matched_ids: set[str] = set()
        for node in nodes_raw:
            nid = node.get("id", "")
            label = node.get("label", "")
            if nid and label:
                id_to_label[nid] = label
                if len(label) >= 2 and label in query:
                    matched_ids.add(nid)

        if not matched_ids:
            return []

        neighbor_scores: Dict[str, int] = {}
        for edge in edges:
            src_id = edge.get("from", "")
            dst_id = edge.get("to", "")
            w = int(edge.get("weight", 0))
            dst_label = id_to_label.get(dst_id, "")
            src_label = id_to_label.get(src_id, "")
            if src_id in matched_ids and len(dst_label) >= 2 and dst_label not in query:
                neighbor_scores[dst_label] = max(neighbor_scores.get(dst_label, 0), w)
            elif dst_id in matched_ids and len(src_label) >= 2 and src_label not in query:
                neighbor_scores[src_label] = max(neighbor_scores.get(src_label, 0), w)

        ranked = sorted(neighbor_scores.items(), key=lambda x: x[1], reverse=True)
        return [label for label, _ in ranked[:top_n]]

    # 旧格式：nodes 是字符串列表，edges 的 from/to 是 label
    nodes_str: List[str] = nodes_raw
    matched_labels: set[str] = {n for n in nodes_str if n and len(n) >= 2 and n in query}
    if not matched_labels:
        return []

    neighbor_scores_old: Dict[str, int] = {}
    for edge in edges:
        src = edge.get("from", "")
        dst = edge.get("to", "")
        w = int(edge.get("weight", 0))
        if src in matched_labels and len(dst) >= 2 and dst not in query:
            neighbor_scores_old[dst] = max(neighbor_scores_old.get(dst, 0), w)
        elif dst in matched_labels and len(src) >= 2 and src not in query:
            neighbor_scores_old[src] = max(neighbor_scores_old.get(src, 0), w)

    ranked_old = sorted(neighbor_scores_old.items(), key=lambda x: x[1], reverse=True)
    return [e for e, _ in ranked_old[:top_n]]


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
        graph_data, graph_fallback_used = _safe_load_graph(self.graph_path)

        # 查询扩展：图邻居 + synonym_dict 合并（BM25 用扩展后的 query，向量保持原 query）
        expanded_query = query
        extra_terms: List[str] = _graph_neighbor_expand(query, graph_data)
        for node in graph_data.get("nodes", []):
            # 兼容新格式（dict）和旧格式（str）
            label = node.get("label", "") if isinstance(node, dict) else node
            if label and label in query:
                extra_terms.extend(self.synonym_dict.get(label, []))
        # 去重并过滤已在 query 中的词
        seen: set[str] = set()
        deduped: List[str] = []
        for t in extra_terms:
            if t and t not in seen and t not in query:
                seen.add(t)
                deduped.append(t)
        if deduped:
            expanded_query = (query + " " + " ".join(deduped)).strip()

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
