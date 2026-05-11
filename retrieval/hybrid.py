from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from graph.semantic_graph import load_semantic_graph
from retrieval.bm25 import BM25Index
from retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


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
        graph_data = load_semantic_graph(path)
        return graph_data, not bool(graph_data.get("nodes") or graph_data.get("entity_chunks"))
    except Exception:
        return {}, True


def _graph_multi_hop_expand(
    query: str, graph_data: dict, hop_depth: int = 2, top_n_per_hop: int = 3
) -> Dict[str, float]:
    """多跳遍历图谱，返回相关实体及其得分。

    参数：
        query: 查询文本
        graph_data: 图谱数据
        hop_depth: 遍历深度（1-3跳）
        top_n_per_hop: 每跳保留的top邻居数

    返回：{实体标签: 综合得分}
    得分 = 路径权重累积 × 距离衰减（每跳衰减0.7）
    """
    nodes_raw: list = graph_data.get("nodes", [])
    edges: List[dict] = graph_data.get("edges", [])

    if not nodes_raw or not edges or hop_depth < 1:
        return {}

    is_new_format = isinstance(nodes_raw[0], dict)

    if is_new_format:
        id_to_label: Dict[str, str] = {}
        label_to_id: Dict[str, str] = {}
        for node in nodes_raw:
            nid = node.get("id", "")
            label = node.get("label", "")
            if nid and label:
                id_to_label[nid] = label
                label_to_id[label] = nid

        # 构建邻接表：{节点id: [(邻居id, 边权重, 关系)]}
        adj: Dict[str, List[Tuple[str, int, str]]] = {}
        for edge in edges:
            src_id = edge.get("from", "")
            dst_id = edge.get("to", "")
            w = int(edge.get("weight", 0))
            rel = edge.get("relation", "")
            if src_id and dst_id:
                adj.setdefault(src_id, []).append((dst_id, w, rel))
                adj.setdefault(dst_id, []).append((src_id, w, rel))

        # 找到查询中的实体节点
        matched_ids: set[str] = set()
        for label in label_to_id:
            if len(label) >= 2 and label in query:
                matched_ids.add(label_to_id[label])

        if not matched_ids:
            return {}

        # BFS 多跳遍历
        from collections import deque

        visited: set[str] = set(matched_ids)
        entity_scores: Dict[str, float] = {}
        queue = deque([(nid, 1.0, 0) for nid in matched_ids])  # (节点id, 当前得分, 跳数)

        while queue:
            curr_id, curr_score, hop = queue.popleft()
            if hop >= hop_depth:
                continue

            for neighbor_id, weight, _ in adj.get(curr_id, []):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)

                # 得分 = 当前得分 × 边权重 × 衰减因子
                decay = 0.7 ** hop
                new_score = curr_score * weight * decay
                neighbor_label = id_to_label.get(neighbor_id, "")

                if neighbor_label and len(neighbor_label) >= 2 and neighbor_label not in query:
                    entity_scores[neighbor_label] = max(entity_scores.get(neighbor_label, 0), new_score)
                    queue.append((neighbor_id, new_score, hop + 1))

        # 按得分排序返回
        return dict(sorted(entity_scores.items(), key=lambda x: x[1], reverse=True))

    # 旧格式兼容逻辑
    nodes_str: List[str] = nodes_raw
    adj_old: Dict[str, List[Tuple[str, int]]] = {}
    for edge in edges:
        src = edge.get("from", "")
        dst = edge.get("to", "")
        w = int(edge.get("weight", 0))
        if src and dst:
            adj_old.setdefault(src, []).append((dst, w))
            adj_old.setdefault(dst, []).append((src, w))

    matched_labels: set[str] = {n for n in nodes_str if n and len(n) >= 2 and n in query}
    if not matched_labels:
        return {}

    visited_old: set[str] = set(matched_labels)
    entity_scores_old: Dict[str, float] = {}
    queue_old = deque([(label, 1.0, 0) for label in matched_labels])

    while queue_old:
        curr_label, curr_score, hop = queue_old.popleft()
        if hop >= hop_depth:
            continue

        for neighbor, weight in adj_old.get(curr_label, []):
            if neighbor in visited_old:
                continue
            visited_old.add(neighbor)

            decay = 0.7 ** hop
            new_score = curr_score * weight * decay

            if neighbor and len(neighbor) >= 2 and neighbor not in query:
                entity_scores_old[neighbor] = max(entity_scores_old.get(neighbor, 0), new_score)
                queue_old.append((neighbor, new_score, hop + 1))

    return dict(sorted(entity_scores_old.items(), key=lambda x: x[1], reverse=True))


def _rank_chunk_ids(chunk_ids: List[str], chunk_pos: Optional[Dict[str, int]] = None) -> List[str]:
    seen: set[str] = set()
    deduped: List[str] = []
    for cid in chunk_ids:
        if cid and cid not in seen:
            seen.add(cid)
            deduped.append(cid)

    if not chunk_pos:
        return sorted(deduped)
    return sorted(deduped, key=lambda cid: (chunk_pos.get(cid, 10**9), cid))


def _apply_graph_recall_limits(
    chunks_by_entity: List[List[str]],
    chunk_to_parent: Optional[Dict[str, str]] = None,
    chunk_pos: Optional[Dict[str, int]] = None,
    max_chunks_per_entity: int = 3,
    max_chunks_per_parent: int = 2,
) -> List[str]:
    selected: List[str] = []
    selected_set: set[str] = set()
    parent_counts: Dict[str, int] = {}

    for entity_chunks in chunks_by_entity:
        taken_for_entity = 0
        for cid in _rank_chunk_ids(entity_chunks, chunk_pos):
            if max_chunks_per_entity > 0 and taken_for_entity >= max_chunks_per_entity:
                break
            if cid in selected_set:
                continue

            parent_id = (chunk_to_parent or {}).get(cid, "")
            if parent_id and max_chunks_per_parent > 0:
                if parent_counts.get(parent_id, 0) >= max_chunks_per_parent:
                    continue

            selected.append(cid)
            selected_set.add(cid)
            taken_for_entity += 1
            if parent_id:
                parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1

    return selected


def _graph_entity_recall(
    query: str,
    graph_data: dict,
    chunk_to_parent: Optional[Dict[str, str]] = None,
    chunk_pos: Optional[Dict[str, int]] = None,
    max_chunks_per_entity: int = 3,
    max_chunks_per_parent: int = 2,
) -> List[str]:
    """直接从 entity_chunks 中召回相关实体所在的 chunks。

    返回：chunk_id 列表
    """
    entity_chunks: Dict[str, List[str]] = graph_data.get("entity_chunks", {})
    nodes_raw: list = graph_data.get("nodes", [])
    if not nodes_raw or not entity_chunks:
        return []

    is_new_format = isinstance(nodes_raw[0], dict)

    chunks_by_entity: List[List[str]] = []

    if is_new_format:
        for node in nodes_raw:
            label = node.get("label", "")
            if label and len(label) >= 2 and label in query:
                nid = node.get("id", "")
                if nid in entity_chunks:
                    chunks_by_entity.append(entity_chunks[nid])
    else:
        nodes_str: List[str] = nodes_raw
        for node_label in nodes_str if isinstance(nodes_raw, list) else []:
            if node_label and len(node_label) >= 2 and node_label in query:
                if node_label in entity_chunks:
                    chunks_by_entity.append(entity_chunks[node_label])

    return _apply_graph_recall_limits(
        chunks_by_entity,
        chunk_to_parent=chunk_to_parent,
        chunk_pos=chunk_pos,
        max_chunks_per_entity=max_chunks_per_entity,
        max_chunks_per_parent=max_chunks_per_parent,
    )


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


def _rerank(
    query: str,
    candidates: List[dict],
    model_name: str = "BAAI/bge-reranker-v2-m3",
    batch_size: int = 16,
) -> List[dict]:
    """使用 Cross-Encoder 重排序候选结果。

    参数：
    - query: 用户查询
    - candidates: 候选结果列表，每个元素包含 chunk_id, text 等
    - model_name: 重排模型名称
    - batch_size: 批处理大小

    返回：重排序后的候选结果列表（新增 rerank_score 字段）
    """
    from sentence_transformers import CrossEncoder

    # 懒加载：函数级缓存模型
    if not hasattr(_rerank, "_model_cache"):
        _rerank._model_cache = {}
    if model_name not in _rerank._model_cache:
        _rerank._model_cache[model_name] = CrossEncoder(model_name, device="cpu")

    model = _rerank._model_cache[model_name]

    # 构造 (query, doc) 对
    pairs = [(query, c["text"]) for c in candidates]

    # 批量推理
    scores = model.predict(pairs, batch_size=batch_size)

    # 重新排序并添加 rerank_score
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    results = []
    for c, score in reranked:
        c_copy = c.copy()
        c_copy["rerank_score"] = float(score)
        results.append(c_copy)

    return results


def _rerank_safe(
    query: str,
    candidates: List[dict],
    model_name: str = "BAAI/bge-reranker-v2-m3",
    batch_size: int = 16,
) -> List[dict]:
    try:
        results = _rerank(
            query,
            candidates,
            model_name=model_name,
            batch_size=batch_size,
        )
        for r in results:
            r.setdefault("rerank_used", True)
        return results
    except Exception as exc:
        logger.warning(
            "Cross-Encoder rerank unavailable, using fused retrieval order: %s",
            exc,
        )
        out = []
        for c in candidates:
            c_copy = c.copy()
            c_copy["rerank_score"] = 0.0
            c_copy["rerank_used"] = False
            c_copy["rerank_error"] = str(exc)
            out.append(c_copy)
        return out


@dataclass
class HybridRetriever:
    """
    BM25 + 向量双路混合检索器 + 图谱增强 + Cross-Encoder 重排。

    synonym_dict：领域同义词表，用于 BM25 查询扩展。
                  格式：{"规范词": ["同义词1", "同义词2"]}
                  默认为空（不扩展），由调用方按领域注入：
                    - 医疗：{"高血压": ["血压升高"], ...}
                    - 金融：{"净利润": ["归母净利润", "净利"], ...}
                    - 法律：{"违约": ["违反合同", "不履行"], ...}
                  也可从外部 JSON 文件加载后传入，避免硬编码。
    bm25_weight / vector_weight：融合权重，默认 0.7/0.3（BM25 偏重精确匹配）。
                  专业术语密集的领域（金融、法律）可适当调高 bm25_weight。
    use_rerank：是否启用 Cross-Encoder 重排，默认启用。
                 重排在融合分数排序后进行，对 top_k 候选进行精细打分重新排序。
    rerank_model：重排模型名称，默认 BAAI/bge-reranker-v2-m3（中文友好）。
    rerank_top_k：重排前召回数量，默认 50。
    rerank_batch_size：重排批处理大小，默认 16（影响推理速度和内存）。

    use_graph_retrieve：是否启用图谱增强检索，默认启用。
    graph_hop_depth：图谱遍历深度（1-3跳），默认2跳。
    graph_entity_recall：是否直接从 entity_chunks 召回相关实体块，默认启用。
    graph_boost_weight：图谱召回结果的权重提升，默认0.3。
    """

    file_hash: str
    bm25: BM25Index
    vector: VectorStore
    graph_path: Path
    synonym_dict: Dict[str, List[str]] = field(default_factory=dict)
    bm25_weight: float = 0.7
    vector_weight: float = 0.3
    candidate_top_k: int = 50
    use_rerank: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_k: int = 50
    rerank_batch_size: int = 16
    use_graph_retrieve: bool = True
    graph_hop_depth: int = 2
    graph_entity_recall: bool = True
    graph_boost_weight: float = 0.3
    graph_max_chunks_per_entity: int = 3
    graph_max_chunks_per_parent: int = 2

    def hybrid_retrieve(
        self, query: str, top_k: int = 5, return_debug_info: bool = False, use_parent_context: bool = False
    ) -> Tuple[List[dict], List[dict] | None]:
        """
        混合检索：BM25 + 向量双路召回 + 图谱增强

        参数：
        - query: 用户查询
        - top_k: 返回结果数量
        - return_debug_info: 是否返回调试信息
        - use_parent_context: 是否使用窗口扩展（推荐）

        当 use_parent_context=True 时：
        - 检索阶段：用短chunk进行精确匹配
        - 返回阶段：返回命中chunk + 前后各2个相邻chunk（避免信息切断，又不会太多无关内容）

        图谱增强（当 use_graph_retrieve=True 时）：
        - 查询扩展：多跳图邻居 + 同义词扩展
        - 实体召回：直接从 entity_chunks 召回相关实体块
        - 权重提升：图谱召回结果获得 graph_boost_weight 的分数加成
        """
        graph_data, graph_fallback_used = _safe_load_graph(self.graph_path)

        # 查询扩展：多跳图邻居 + synonym_dict 合并（BM25 用扩展后的 query，向量保持原 query）
        expanded_query = query
        extra_terms: List[str] = []

        if self.use_graph_retrieve and not graph_fallback_used:
            # 多跳图扩展
            multi_hop_entities = _graph_multi_hop_expand(query, graph_data, hop_depth=self.graph_hop_depth)
            if multi_hop_entities:
                extra_terms.extend(multi_hop_entities.keys())

        # 同义词扩展
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

        route_top_k = max(1, int(self.candidate_top_k or top_k))
        bm25_hits = self.bm25.search(expanded_query, top_k=route_top_k)
        vec_hits = self.vector.search(query, top_k=route_top_k)

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

        # 图谱召回：直接从 entity_chunks 召回相关实体所在的 chunks
        if self.use_graph_retrieve and self.graph_entity_recall and not graph_fallback_used:
            chunk_pos = {
                c["chunk_id"]: int(c.get("pos", idx))
                for idx, c in enumerate(self.bm25.chunks_by_pos)
                if c.get("chunk_id")
            }
            graph_chunks = _graph_entity_recall(
                query,
                graph_data,
                chunk_to_parent=self.bm25.chunk_to_parent,
                chunk_pos=chunk_pos,
                max_chunks_per_entity=self.graph_max_chunks_per_entity,
                max_chunks_per_parent=self.graph_max_chunks_per_parent,
            )
            for cid in graph_chunks:
                text = text_map.get(cid, "")
                if not text:
                    continue
                if cid not in cand:
                    cand[cid] = {
                        "chunk_id": cid,
                        "text": text,
                        "bm25_score": 0.0,
                        "vector_score": 0.0,
                        "graph_boost": True,  # 标记为图谱召回
                    }
                else:
                    cand[cid]["graph_boost"] = True  # 已存在，标记图谱召回

        candidates = list(cand.values())[:50]  # 增加候选数量以容纳图谱召回
        norm_bm25 = _minmax_norm([float(c["bm25_score"]) for c in candidates])
        norm_vec = _minmax_norm([float(c["vector_score"]) for c in candidates])

        debug_data: list[dict] = []
        for c, nb, nv in zip(candidates, norm_bm25, norm_vec):
            base_score = self.bm25_weight * nb + self.vector_weight * nv
            final = base_score
            # 图谱召回的块应用权重提升
            if c.get("graph_boost", False):
                final *= (1 + self.graph_boost_weight)

            row = {
                "chunk_id": c["chunk_id"],
                "bm25_score": float(c["bm25_score"]),
                "vector_score": float(c["vector_score"]),
                "norm_bm25": float(nb),
                "norm_vector": float(nv),
                "base_score": float(base_score),
                "final_score": float(final),
                "graph_boost": c.get("graph_boost", False),
                "graph_fallback_used": bool(graph_fallback_used),
            }
            debug_data.append(row)
            c["base_score"] = float(base_score)
            c["final_score"] = float(final)

        # 根据融合分数排序，取前 N 个候选用于重排
        sorted_candidates = sorted(candidates, key=lambda x: float(x.get("final_score", 0.0)), reverse=True)

        # 可选重排：对召回的 top N 候选进行 Cross-Encoder 重排序
        if self.use_rerank:
            rerank_candidates = sorted_candidates[:self.rerank_top_k]
            results = _rerank_safe(
                query,
                rerank_candidates,
                model_name=self.rerank_model,
                batch_size=self.rerank_batch_size,
            )[:top_k]
            # 标记使用了重排
            for r in results:
                r.setdefault("rerank_used", True)
        else:
            results = sorted_candidates[:top_k]

        # 如果使用窗口扩展，合并前后相邻chunk（只扩展同父文档内的chunk）
        if use_parent_context:
            expanded_results = []
            for r in results:
                chunk_id = r["chunk_id"]
                expanded_text = self.bm25.expand_chunk_window(chunk_id, window_size=2)
                chunk_text = r["text"]  # 原始chunk文本

                expanded_results.append({
                    "chunk_id": chunk_id,
                    "text": expanded_text,  # 扩展后的文本
                    "chunk_text": chunk_text,  # 原始chunk（用于对比）
                    "bm25_score": float(r["bm25_score"]),
                    "vector_score": float(r["vector_score"]),
                    "base_score": float(r.get("base_score", r["final_score"])),
                    "final_score": float(r["final_score"]),
                    "rerank_score": float(r.get("rerank_score", 0.0)),
                    "rerank_used": r.get("rerank_used", False),
                })
            results_out = expanded_results
        else:
            results_out = [
                {
                    "chunk_id": r["chunk_id"],
                    "text": r["text"],
                    "bm25_score": float(r["bm25_score"]),
                    "vector_score": float(r["vector_score"]),
                    "base_score": float(r.get("base_score", r["final_score"])),
                    "final_score": float(r["final_score"]),
                    "rerank_score": float(r.get("rerank_score", 0.0)),
                    "rerank_used": r.get("rerank_used", False),
                }
                for r in results
            ]

        if return_debug_info:
            return results_out, debug_data
        return results_out, None
