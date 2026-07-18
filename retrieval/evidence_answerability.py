from __future__ import annotations

import math
import re
from typing import Any

from config.settings import SETTINGS
from retrieval.retrieval_gate import content_terms, is_direct_evidence, normalize_text, term_coverage


_FILLER_TERMS = {
    "如果",
    "为了",
    "一个",
    "这个",
    "那个",
    "请问",
    "能否",
    "能不能",
    "是不是",
    "是否",
    "哪个",
    "哪些",
    "什么",
    "怎么",
    "如何",
    "多少",
    "到底",
    "一下",
    "可以",
    "能够",
    "进行",
    "介绍",
    "说明",
    "讲讲",
    "举个",
    "例子",
    "请",
}

_COMPARISON_MARKERS = (
    "哪个",
    "哪一个",
    "还是",
    "相比",
    "比较",
    "对比",
    "区别",
    "不同",
    "差异",
    "更",
    "好",
    "优于",
    "高于",
    "低于",
    "快",
    "慢",
)

_ASPECT_TERMS = (
    "成本",
    "维护",
    "更新",
    "动态知识",
    "响应速度",
    "速度",
    "延迟",
    "幻觉",
    "准确性",
    "可靠",
    "适用场景",
    "场景",
    "效果",
    "优点",
    "缺点",
    "风险",
    "流程",
    "步骤",
    "原理",
    "作用",
    "用途",
    "指标",
    "评估",
    "忠实度",
    "召回",
    "召回率",
    "准确率",
    "上下文",
    "嵌入",
    "编码",
    "检索",
    "生成",
    "图谱",
    "关系",
    "实体",
    "金额",
    "收入",
    "利润",
    "毛利率",
    "现金流",
)

_KNOWN_OBJECT_TERMS = (
    "rag",
    "graphrag",
    "ragas",
    "bert",
    "openai",
    "bm25",
    "faiss",
    "mcp",
    "llm",
    "微调",
    "向量",
    "大模型",
    "知识图谱",
    "图数据库",
)


def _clean_terms(terms: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for term in terms:
        t = term.strip().lower()
        if not t or t in _FILLER_TERMS:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def extract_query_facets(query: str) -> dict[str, Any]:
    norm_query = normalize_text(query)
    terms = _clean_terms(content_terms(query))
    object_terms: list[str] = []
    aspect_terms: list[str] = []
    hard_aspect_terms: list[str] = []

    for known in _KNOWN_OBJECT_TERMS:
        norm_known = normalize_text(known)
        if re.fullmatch(r"[a-z0-9][a-z0-9_.+-]*", norm_known):
            present = re.search(rf"(?<![a-z0-9]){re.escape(norm_known)}(?![a-z0-9])", norm_query) is not None
        else:
            present = norm_known in norm_query
        if present and known not in object_terms:
            object_terms.append(known)

    for term in terms:
        if re.fullmatch(r"[a-z0-9][a-z0-9_.+-]*", term) and term not in object_terms:
            object_terms.append(term)

    for aspect in _ASPECT_TERMS:
        if normalize_text(aspect) in norm_query and aspect not in aspect_terms:
            aspect_terms.append(aspect)
            hard_aspect_terms.append(aspect)

    for term in terms:
        if term in object_terms or term in aspect_terms:
            continue
        # Keep important Chinese query facets even when they are not in the
        # curated list, but avoid turning every generic verb into a hard facet.
        if len(term) >= 2 and not re.fullmatch(r"[a-z0-9][a-z0-9_.+-]*", term):
            aspect_terms.append(term)

    is_comparison = any(marker in norm_query for marker in _COMPARISON_MARKERS)
    return {
        "terms": terms,
        "object_terms": object_terms,
        "aspect_terms": aspect_terms,
        "hard_aspect_terms": hard_aspect_terms,
        "is_comparison": is_comparison,
    }


def split_evidence_windows(
    retrieval_results: list[dict[str, Any]] | None = None,
    context_text: str = "",
) -> list[dict[str, Any]]:
    max_chars = int(getattr(SETTINGS, "answerability_direct_window_chars", 420))
    stride = max(120, int(max_chars * 0.65))
    rows = retrieval_results or [{"text": context_text, "evidence_type": "direct"}]
    windows: list[dict[str, Any]] = []
    for row in rows:
        if retrieval_results is not None and not is_direct_evidence(row):
            continue
        text = str(row.get("text") or row.get("chunk_text") or "")
        parts = [
            part.strip()
            for part in re.split(r"(?<=[。！？!?；;])|\n+", text)
            if part and part.strip()
        ]
        if not parts and text.strip():
            parts = [text.strip()]

        bounded_parts: list[str] = []
        for part in parts:
            if len(part) <= max_chars:
                bounded_parts.append(part)
                continue
            start = 0
            while start < len(part):
                segment = part[start : start + max_chars].strip()
                if segment:
                    bounded_parts.append(segment)
                if start + max_chars >= len(part):
                    break
                start += stride
        parts = bounded_parts

        for idx, part in enumerate(parts):
            prev_part = parts[idx - 1] if idx > 0 else ""
            next_part = parts[idx + 1] if idx + 1 < len(parts) else ""
            window = " ".join(p for p in (prev_part, part, next_part) if p).strip()
            if len(window) > max_chars * 2:
                window = window[: max_chars * 2]
            windows.append(
                {
                    "text": window,
                    "chunk_id": row.get("chunk_id", ""),
                    "source_file": row.get("source_file") or row.get("file_hash") or "",
                    "parent_id": row.get("parent_id", ""),
                }
            )
    return windows


def _hit_terms(terms: list[str], text: str) -> list[str]:
    norm_text = normalize_text(text)
    return [term for term in terms if normalize_text(term) in norm_text]


def check_direct_answer_evidence(
    query: str,
    retrieval_results: list[dict[str, Any]] | None = None,
    context_text: str = "",
) -> tuple[bool, dict[str, Any]]:
    facets = extract_query_facets(query)
    query_terms = facets["terms"]
    object_terms = facets["object_terms"]
    aspect_terms = facets["aspect_terms"]
    hard_aspect_terms = facets["hard_aspect_terms"]
    windows = split_evidence_windows(retrieval_results, context_text)

    if not windows:
        return False, {
            "gate": "direct_answer_evidence",
            "passed": False,
            "reasons": ["no_direct_evidence_windows"],
            "facets": facets,
        }

    min_coverage = float(getattr(SETTINGS, "answerability_min_direct_sentence_coverage", 0.50))
    min_hits = int(getattr(SETTINGS, "answerability_min_direct_term_hits", 2))
    if query_terms:
        min_hits = max(min_hits, math.ceil(len(query_terms) * min_coverage))

    best: dict[str, Any] | None = None
    for window in windows:
        text = window["text"]
        term_hits = _hit_terms(query_terms, text)
        object_hits = _hit_terms(object_terms, text)
        aspect_hits = _hit_terms(aspect_terms, text)
        hard_aspect_hits = _hit_terms(hard_aspect_terms, text)
        score = (
            len(term_hits)
            + 1.5 * len(object_hits)
            + 1.25 * len(aspect_hits)
            + 1.75 * len(hard_aspect_hits)
            + term_coverage(query_terms, text)
        )
        row = {
            **window,
            "query_coverage": round(term_coverage(query_terms, text), 4),
            "term_hits": term_hits,
            "object_hits": object_hits,
            "aspect_hits": aspect_hits,
            "hard_aspect_hits": hard_aspect_hits,
            "score": round(score, 4),
        }
        if best is None or score > best["score"]:
            best = row

    assert best is not None
    reasons: list[str] = []
    if len(best["term_hits"]) < min_hits:
        reasons.append("low_direct_term_hits")

    if facets["is_comparison"] and len(object_terms) >= 2:
        required_objects = min(2, len(object_terms))
        if len(best["object_hits"]) < required_objects:
            reasons.append("missing_comparison_objects")

    if hard_aspect_terms and len(best["hard_aspect_hits"]) == 0:
        reasons.append("missing_question_aspect")

    return not reasons, {
        "gate": "direct_answer_evidence",
        "passed": not reasons,
        "reasons": reasons,
        "facets": facets,
        "min_direct_term_hits": min_hits,
        "min_direct_sentence_coverage": min_coverage,
        "best_window": best,
    }
